# GRADY

AI-powered full-stack grading assistant. Upload a guide (rubric/questions) and a student answer sheet — Grady returns per-question grades, comments, inline highlights (add/deduct), and rubric criteria scores. Supports `.txt`, PDF, and DOCX files. Built with FastAPI and pydantic-ai on top of Claude.

## Setup

1. Create a virtual environment (optional) and install deps:

```bash
pip install -r requirements.txt
```

2. If your gateway setup requires env vars, add them to `.env`.

## Logfire

This project is wired to Logfire for logging. Use one of the following:

- Set `LOGFIRE_KEY` in `.env` (the app maps it to `LOGFIRE_TOKEN`).
- Or authenticate with the CLI and set the project:

```bash
uv run logfire auth
uv run logfire projects use grady
```

## Run

```bash
uvicorn app.main:app --reload
```

Then open `http://localhost:8000` and upload `guide.txt` and `student_answers.txt`.

## API

### POST /grade

Request JSON:

```json
{
  "guide": "<raw guide.txt contents>",
  "student_answers": "<raw student_answers.txt contents>",
  "custom_prompt": "<optional extra instructions>"
}
```

Response JSON (exact shape):

```json
{
  "student_name": "<student name if present, otherwise empty string>",
  "questions": {
    "1": "<full question text>",
    "2": "<full question text>"
  },
  "answers": {
    "1": "<aligned student answer text>",
    "2": "<aligned student answer text>"
  },
  "grades": {
    "1": { "score": 7.5, "max_score": 10 },
    "2": { "score": 4, "max_score": 8 }
  },
  "comments": {
    "1": "Short grading explanation.",
    "2": "Short grading explanation."
  },
  "highlights": {
    "1": [
      { "start": 0, "end": 25, "effect": "add", "points": 2, "reason": "Correct definition" },
      { "start": 40, "end": 58, "effect": "deduct", "points": 1, "reason": "Missing example" }
    ],
    "2": []
  },
  "criteria": [
    {
      "title": "Overall quality",
      "score": 7,
      "max_score": 10,
      "comment": "Solid structure with minor language issues.",
      "items": [
        { "title": "Organization", "score": 2, "max_score": 3, "comment": "Clear sections." },
        { "title": "Language", "score": 2, "max_score": 3, "comment": "Some grammar issues." },
        { "title": "Bibliography", "score": 3, "max_score": 4, "comment": "Sources listed." }
      ]
    }
  ]
}
```

Notes:
- Question keys are sequential strings: `"1"`, `"2"`, ...
- `student_name` is optional; it is an empty string when no name is found.
- `start`/`end` are 0-based indices into `answers[k]` (end is exclusive).
- The UI lets you edit the final score and feedback locally (no persistence).
- Scores are computed as `max_score - sum(deduct highlight points)`; add highlights are informational only.

### POST /grade/stream

Streams NDJSON events while grading. Each line is a JSON object with a `type` field.

Event types:
- `status`: `{ "type": "status", "stage": "start" }`
- `text`: `{ "type": "text", "delta": "<partial text>" }`
- `final`: `{ "type": "final", "run_id": "...", "raw_output": "<full text>", "data": { ...grading view... } }`
- `raw`: `{ "type": "raw", "run_id": "...", "raw_output": "<full text>" }`
- `error`: `{ "type": "error", "detail": "..." }`

The frontend uses this endpoint to show a live, incremental response and then renders the final grading view.
