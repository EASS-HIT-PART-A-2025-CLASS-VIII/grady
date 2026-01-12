# GRADY

Minimal full-stack grader that accepts two raw .txt files and returns grades with short comments per question.

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
  "student_answers": "<raw student_answers.txt contents>"
}
```

Response JSON (exact shape):

```json
{
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
  }
}
```

Notes:
- Question keys are sequential strings: `"1"`, `"2"`, ...
- `start`/`end` are 0-based indices into `answers[k]` (end is exclusive).
- The UI lets you edit the final score and feedback locally (no persistence).
