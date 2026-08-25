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
- `GEMINI_API_KEY` — required when `LLM_PROVIDER=gemini`. Get a free key at
  https://aistudio.google.com/apikey.
- `ANTHROPIC_API_KEY` — required when `LLM_PROVIDER=anthropic`.

Only one key is required, matching whichever provider is selected. `.env` is
loaded automatically (via `python-dotenv`) when the CLI starts.

**Free-tier quota:** a free Gemini key is limited to 5 requests/minute and
20 requests/day. Every grounded query costs 1 request, and every multi-turn
follow-up (a query with prior conversation history) costs up to 2 — one to
rewrite the follow-up into a standalone question, one to generate the
answer. A testing/demo session can exhaust the daily cap in roughly 10
turns. If the CLI starts responding with a "temporarily rate-limited or out
of quota" message instead of real answers, that's this limit, not a bug —
wait for the quota window to reset or switch to a key with more headroom.

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
