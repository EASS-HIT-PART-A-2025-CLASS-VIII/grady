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
