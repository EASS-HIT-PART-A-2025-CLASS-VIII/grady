import json
import os
import re
from collections import deque
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logfire
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.anthropic import AnthropicModel

load_dotenv()

logfire_key = os.getenv("LOGFIRE_KEY")
if logfire_key and not os.getenv("LOGFIRE_TOKEN"):
    os.environ["LOGFIRE_TOKEN"] = logfire_key

logfire.configure()

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
BYPASS_OUTPUT_VALIDATION = True
MODEL_SETTINGS = {"max_tokens": 8192}

SYSTEM_PROMPT = (
    "You are an automated grading assistant.\n"
    "You will receive two raw text inputs: (1) a guide that contains questions and grading expectations, and (2) a student's answer text.\n"
    "The texts may have inconsistent or missing formatting; do not rely on numbering or delimiters. Infer question boundaries and answer alignment from meaning and context.\n"
    "Return ONLY a JSON object with exactly three keys: titles, grades, and comments.\n"
    "titles must be an object keyed by sequential question numbers as strings (\"1\", \"2\", ...) with short, appropriate titles for each question.\n"
    "grades must be an object keyed by the same question numbers with integer grades from 0 to 10 as values.\n"
    "comments must be an object keyed by the same question numbers with feedback no more than 10 words per item.\n"
    "Do NOT include any other keys or nested objects (no separated_questions, student_answers, or grades_for_each_question).\n"
    "Keep the question order as it appears in the guide. If an answer is missing, output 0 and a short comment.\n"
    "Do not output any extra keys, metadata, markdown, or explanation - only JSON."
)

USER_PROMPT_TEMPLATE = (
    "GUIDE (raw text):\n"
    "{guide}\n\n"
    "STUDENT ANSWERS (raw text):\n"
    "{student_answers}"
)


class GradeRequest(BaseModel):
    guide: str
    student_answers: str


class GradesWithComments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    titles: dict[str, str]
    grades: dict[str, int]
    comments: dict[str, str]

    @model_validator(mode="after")
    def validate_payload(self) -> "GradesWithComments":
        normalized_titles: dict[str, str] = {}
        normalized_grades: dict[str, int] = {}
        normalized_comments: dict[str, str] = {}
        numeric_keys: list[int] = []

        for key, value in self.titles.items():
            key_str = str(key)
            title = str(value).strip()
            if re.fullmatch(r"[1-9]\d*", key_str) is None:
                raise ValueError(f"Invalid title key: {key_str}")
            if not title:
                raise ValueError(f"Empty title for {key_str}")
            normalized_titles[key_str] = title
            numeric_keys.append(int(key_str))

        for key, value in self.grades.items():
            key_str = str(key)
            if re.fullmatch(r"[1-9]\d*", key_str) is None:
                raise ValueError(f"Invalid question key: {key_str}")
            if not 0 <= value <= 10:
                raise ValueError(f"Invalid grade for {key_str}: {value}")
            normalized_grades[key_str] = int(value)

        for key, value in self.comments.items():
            key_str = str(key)
            comment = str(value).strip()
            if re.fullmatch(r"[1-9]\d*", key_str) is None:
                raise ValueError(f"Invalid comment key: {key_str}")
            if len(comment.split()) > 10:
                raise ValueError(f"Comment too long for {key_str}")
            normalized_comments[key_str] = comment

        numeric_keys.sort()
        if numeric_keys and numeric_keys != list(range(1, len(numeric_keys) + 1)):
            raise ValueError("Question keys must be consecutive starting at 1")

        if set(normalized_grades.keys()) != set(normalized_comments.keys()):
            raise ValueError("grades and comments keys must match")
        if set(normalized_titles.keys()) != set(normalized_comments.keys()):
            raise ValueError("titles and comments keys must match")

        self.titles = normalized_titles
        self.grades = normalized_grades
        self.comments = normalized_comments
        return self


class RawOutputResponse(BaseModel):
    raw_output: str


_agent: Agent | None = None
_raw_outputs: dict[str, str] = {}
_raw_output_order: deque[str] = deque()
_raw_output_limit = 20


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    if stripped.endswith("```") and "\n" not in stripped:
        inner = stripped[3:-3].strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].lstrip()
        return inner

    lines = stripped.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    if lines and lines[0].strip().lower() == "json":
        lines = lines[1:]
    return "\n".join(lines).strip()


def _coerce_int_grade(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match:
            return int(match.group(0))
        return None
    if isinstance(value, dict):
        for key in ("score", "grade", "value"):
            if key in value:
                return _coerce_int_grade(value[key])
    return None


def _extract_json_object(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _limit_comment_words(comment: str, max_words: int = 10) -> str:
    words = comment.strip().split()
    if not words:
        return "No feedback."
    return " ".join(words[:max_words])


def _coerce_comment(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("comment", "feedback", "note", "text"):
            if key in value:
                return str(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _coerce_grades_map(source: object) -> dict[str, int]:
    if not isinstance(source, dict):
        return {}

    grades: dict[str, int] = {}
    for key, value in source.items():
        grade = _coerce_int_grade(value)
        if grade is None:
            continue
        grades[str(key)] = max(0, min(10, grade))

    return grades


def _coerce_comments_map(source: object) -> dict[str, str]:
    if not isinstance(source, dict):
        return {}

    comments: dict[str, str] = {}
    for key, value in source.items():
        comments[str(key)] = _coerce_comment(value)

    return comments


def _extract_grades_and_comments(raw_output: str) -> tuple[dict[str, int], dict[str, str]]:
    data = _extract_json_object(raw_output)
    if not isinstance(data, dict):
        return {}, {}

    grades_source = None
    comments_source = None

    if "grades" in data or "comments" in data:
        grades_source = data.get("grades")
        comments_source = data.get("comments")
    elif "grades_for_each_question" in data or "comments_for_each_question" in data:
        grades_source = data.get("grades_for_each_question")
        comments_source = data.get("comments_for_each_question")
    else:
        grades_source = data
        comments_source = {}

    grades = _coerce_grades_map(grades_source)
    comments = _coerce_comments_map(comments_source)

    if not grades:
        return {}, {}

    normalized_comments: dict[str, str] = {}
    for key in sorted(grades.keys(), key=lambda k: int(k)):
        comment = comments.get(key, "")
        normalized_comments[key] = _limit_comment_words(comment)

    return grades, normalized_comments


def _extract_raw_output(run_result: AgentRunResult) -> str:
    try:
        response = run_result.response
    except Exception:
        return ""

    output_tool_name = getattr(run_result, "_output_tool_name", None)
    tool_calls = response.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            if output_tool_name is None or tool_call.tool_name == output_tool_name:
                try:
                    return tool_call.args_as_json_str()
                except Exception:
                    if isinstance(tool_call.args, str):
                        return tool_call.args
                    if tool_call.args is None:
                        return ""
                    return json.dumps(tool_call.args)

        fallback_call = tool_calls[-1]
        try:
            return fallback_call.args_as_json_str()
        except Exception:
            if isinstance(fallback_call.args, str):
                return fallback_call.args
            if fallback_call.args is None:
                return ""
            return json.dumps(fallback_call.args)

    return response.text or ""


def _attach_raw_output(raw_output: str) -> None:
    span = trace.get_current_span()
    if span is not None and span.is_recording():
        span.set_attribute("grady.raw_output", raw_output)
        span.set_attribute("grady.raw_output_len", len(raw_output))


def _attach_grades_map(grades: dict[str, int]) -> None:
    span = trace.get_current_span()
    if span is not None and span.is_recording():
        span.set_attribute("grady.grades", json.dumps(grades, ensure_ascii=False))


def _attach_comments_map(comments: dict[str, str]) -> None:
    span = trace.get_current_span()
    if span is not None and span.is_recording():
        span.set_attribute("grady.comments", json.dumps(comments, ensure_ascii=False))


def _store_raw_output(raw_output: str) -> str:
    run_id = uuid4().hex
    _raw_outputs[run_id] = raw_output
    _raw_output_order.append(run_id)
    if len(_raw_output_order) > _raw_output_limit:
        stale_id = _raw_output_order.popleft()
        _raw_outputs.pop(stale_id, None)
    return run_id


def get_agent() -> Agent:
    global _agent
    if _agent is not None:
        return _agent

    model = AnthropicModel("claude-sonnet-4-5", provider="gateway")
    _agent = Agent(
        model,
        output_type=GradesWithComments,
        system_prompt=SYSTEM_PROMPT,
        retries=1,
        model_settings=MODEL_SETTINGS,
    )
    return _agent


app = FastAPI(title="GRADY")
if hasattr(logfire, "instrument_fastapi"):
    try:
        logfire.instrument_fastapi(app)
    except Exception as exc:
        logfire.warning("FastAPI instrumentation disabled", error=str(exc))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=index_html)


@app.get("/grade/raw", response_model=RawOutputResponse)
def grade_raw(run_id: str) -> RawOutputResponse:
    raw_output = _raw_outputs.get(run_id)
    if raw_output is None:
        raise HTTPException(status_code=404, detail="Raw output not found")
    return RawOutputResponse(raw_output=raw_output)


@app.post("/grade", response_model=GradesWithComments)
def grade(payload: GradeRequest, response: Response) -> GradesWithComments | Response:
    prompt = USER_PROMPT_TEMPLATE.format(
        guide=payload.guide,
        student_answers=payload.student_answers,
    )

    try:
        agent = get_agent()
        logfire.info(
            "Grading request received",
            guide_chars=len(payload.guide),
            student_chars=len(payload.student_answers),
            bypass_validation=BYPASS_OUTPUT_VALIDATION,
        )
        if BYPASS_OUTPUT_VALIDATION:
            result = agent.run_sync(prompt, output_type=str)
            raw_output = result.output if isinstance(result.output, str) else _extract_raw_output(result)
            raw_output_clean = _strip_code_fences(raw_output)
            _attach_raw_output(raw_output_clean)
            grades_map, comments_map = _extract_grades_and_comments(raw_output_clean)
            if not grades_map:
                raise HTTPException(status_code=500, detail="Failed to extract grades from model output.")
            _attach_grades_map(grades_map)
            _attach_comments_map(comments_map)
            logfire.info("Raw model output (pre-validation)", raw_output=raw_output_clean)
            print("\nRaw model output (pre-validation, cleaned):", raw_output_clean)
            print("1:", result)
            run_id = _store_raw_output(raw_output_clean)
            response.headers["X-Grady-Run-Id"] = run_id
            return {
                "grades": grades_map,
                "comments": comments_map,
            }

        result = agent.run_sync(prompt)
        raw_output = _extract_raw_output(result)
        raw_output_clean = _strip_code_fences(raw_output)
        _attach_raw_output(raw_output_clean)
        _attach_grades_map(result.output.grades)
        _attach_comments_map(result.output.comments)
        logfire.info("Raw model output (pre-validation)", raw_output=raw_output_clean)
        print("\nRaw model output (pre-validation, cleaned):", raw_output_clean)
        print("1:", result)
        print("\nGrades result:", result.output)
        run_id = _store_raw_output(raw_output_clean)
        response.headers["X-Grady-Run-Id"] = run_id
        return result.output
    except Exception as exc:
        logfire.exception("Grading failed")
        raise HTTPException(status_code=500, detail=f"Grading failed: {exc}") from exc
