# GDG On Campus Club FAQ Assistant

Slice 1: knowledge base & grounded Q&A. Answers club questions strictly from
a fixed KB, refusing when nothing relevant is retrieved and never fabricating
facts the retrieved section doesn't contain.

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` and set:

- `LLM_PROVIDER` — `anthropic` or `gemini` (defaults to `gemini` if unset).
- `GEMINI_API_KEY` — required when `LLM_PROVIDER=gemini`.
- `ANTHROPIC_API_KEY` — required when `LLM_PROVIDER=anthropic`.

Only one key is required, matching whichever provider is selected.

## Run

```bash
uv run python -m backend.cli
```

Refusals (queries with no relevant KB match) never call the LLM, so the CLI
works without any API key for those.

## Tests

```bash
uv run pytest
```

Every test mocks the LLM client, so the suite needs no API key and makes no
network calls.
