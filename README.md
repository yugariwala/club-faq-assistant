# GDG On Campus Club FAQ Assistant

Answers club questions strictly from a fixed KB (never fabricating facts the
retrieved section doesn't contain), carries context across turns, and tags
every message with an intent category.

- **Slice 1** — knowledge base & grounded Q&A.
- **Slice 2** — multi-turn memory (pronoun/ellipsis resolution across turns).
- **Slice 3** — hybrid intent classification (rules + LLM fallback).

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

## Intent classification (Slice 3)

Every user message is tagged with one of five categories -- `faq`,
`event_inquiry`, `action_request`, `out_of_scope`, `greeting` -- via a
hybrid classifier (`backend/intent.py`):

1. **Rule layer** — a fixed priority order of high-precision keyword/pattern
   rules (greeting → action_request → event_inquiry → faq). Each rule
   abstains (`None`) rather than guess; an abstained message falls through
   to the next rule and ultimately to the LLM. Rules never attempt
   `out_of_scope` -- that space is open-ended natural language, and a fixed
   keyword denylist would be wrong for anything not on it.
2. **LLM fallback** — a constrained single-label prompt (`llm_client.
   classify_intent`), validated against the 5-label enum with one retry on
   an unparseable response, falling back to `out_of_scope` if both attempts
   fail (never raises, never guesses an actionable label it isn't sure of).

Intent classifies the topic of the question, not whether the KB can answer
it -- "How many members are in Web Dev?" is `faq` intent (a team question)
even though that fact isn't in the KB; whether it's answerable is
`qa.answer_question`'s separate refusal/grounding concern.

Both the classified intent and which path resolved it (`rule` | `llm`) are
attached to every `AnswerResult` and logged per-turn.

### Evaluation

```bash
uv run python scripts/eval_intents.py            # full hybrid run (rules + batched LLM fallback)
uv run python scripts/eval_intents.py --rules-only # 0 API calls, iterate on rule patterns for free
```

Scores `data/intent_eval.jsonl` (56 hand-labeled queries, balanced across
all five categories, including deliberately ambiguous pairs like "Is
HackFest still open for registration?" vs. "Can I still sign up for
HackFest?"). Every rule-abstained item is batched into a single
`classify_intents_batch` call per run rather than one call each, to stay
within a limited daily Gemini quota. Results are written to
`data/intent_eval_results.md`.

**Latest results** (`LLM_PROVIDER=gemini`, `gemini-3.6-flash`, 1 LLM call for
the whole eval set):

| Path | Count | Fraction of scored traffic | Accuracy |
|---|---|---|---|
| rule | 33 | 58.9% | 100.0% |
| llm | 23 | 41.1% | 95.7% |
| **overall** | 56 | 100.0% | 98.2% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| action_request | 1.000 | 1.000 | 1.000 | 11 |
| event_inquiry | 1.000 | 1.000 | 1.000 | 12 |
| faq | 0.923 | 1.000 | 0.960 | 12 |
| greeting | 1.000 | 1.000 | 1.000 | 10 |
| out_of_scope | 1.000 | 0.909 | 0.952 | 11 |
| **macro avg** | 0.985 | 0.982 | 0.982 | 56 |
| **weighted avg** | 0.984 | 0.982 | 0.982 | 56 |

The one miss: "Does the club have a Discord server?" (gold `out_of_scope`)
was classified `faq` by the LLM fallback -- a defensible read given the
message is club-adjacent even though no KB section covers community
platforms. Rule-path accuracy is 100% by construction (rules only ever fire
on unambiguous patterns); see `data/intent_eval_results.md` for the full
confusion matrix and misclassification detail.
