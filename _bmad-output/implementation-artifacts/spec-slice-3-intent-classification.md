---
title: 'Slice 3 — Intent Classification'
type: 'feature'
created: '2026-08-25'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/requirements.md']
baseline_commit: 'a93fd0e'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** requirements.md §3.2 requires every user message to be tagged with an intent category (faq / event_inquiry / action_request / out_of_scope / greeting), shown in the UI and logged per-turn for the Slice 5 dashboard's intent breakdown. A pure LLM-prompt classifier would work but spends an API call (and a limited daily Gemini quota) on every single message, including trivially unambiguous ones ("hi", "register me for HackFest").

**Approach:** hybrid classification (requirements.md §5a, "Hybrid" row). A fixed-priority rule layer (`backend/intent.py`) resolves high-precision, unambiguous cases for free; any rule that isn't confident abstains (`None`) rather than guess, falling through to an LLM fallback (`llm_client.classify_intent`) constrained to return exactly one of the five labels. Confidence scoring, agentic actions, and the dashboard are explicitly out of scope for this slice.

## Boundaries & Constraints

**Always:**
- Rules are checked in a fixed priority order (greeting → action_request → event_inquiry → faq); the first non-abstaining rule wins.
- A rule abstains (`None`) rather than assert a label it isn't confident about -- precision over coverage.
- The LLM fallback's response is validated against the 5-label enum; an invalid/unparseable response is retried once, then falls back to `out_of_scope` (never raises, never guesses).
- Intent is classified from the original, raw user message -- never the multi-turn-rewritten form used for retrieval (Slice 2's `rewritten_query`).
- Every turn's classified intent and resolving path (`rule` | `llm`) are logged, alongside the existing original/rewritten-query log line.
- Rules never attempt to positively classify `out_of_scope` -- that space is open-ended and a fixed keyword list would be wrong for anything not on it; every `out_of_scope` case is LLM-resolved.

**Ask First:** None identified.

**Never:** Confidence scoring, agentic actions, the dashboard, or persisted structured logging (JSONL/DB) for the dashboard to read -- that store is a later slice's dependency, same precedent as the existing deferred-work.md entry for Slice 1's dashboard-log dependency. Changing `RETRIEVAL_THRESHOLD`/refusal semantics. Classifying on the rewritten query instead of the raw one.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Bare greeting | "Hi!" | Rule-resolved `greeting`, 0 LLM calls | N/A |
| Greeting with content | "hi, when's HackFest?" | Greeting rule abstains (not a whole-message match); falls through | N/A |
| Self-referential action | "Register me for HackFest 2025." | Rule-resolved `action_request`, wins over the event-name match (action_request checked first) | N/A |
| Informational event question | "Is HackFest still open for registration?" | Rule-resolved `event_inquiry` (no self-referential verb) | N/A |
| Genuinely ambiguous action/event pair | "Can I still sign up for HackFest?" | event_inquiry rule detects a competing weak-action-cue ("can i"/"sign up") and abstains; LLM fallback adjudicates | Falls to `out_of_scope` only if the LLM call itself fails |
| KB-adjacent but unanswerable detail | "How many members are in the Web Dev team?" | Rule-resolved `faq` (team-topic keyword) -- intent is topic-based, independent of whether the KB can answer it | N/A |
| Genuinely out-of-scope topic | "What's the club's budget?" | No rule fires; LLM fallback classifies `out_of_scope` | Falls back to `out_of_scope` (same label) if the call fails |
| LLM returns an invalid label | e.g. "I'm not sure, maybe faq?" | Retried once (`INTENT_CLASSIFY_MAX_ATTEMPTS`); falls back to `out_of_scope` if still invalid | Logged as a warning, never raised |
| Blank/whitespace query | "" or "   " | Resolves directly to `DEFAULT_INTENT_ON_LLM_FAILURE` ("out_of_scope"), no rule or LLM call | N/A |

</frozen-after-approval>

## Code Map

- `backend/config.py` -- add `INTENT_LABELS: frozenset[str]` (the 5-label enum), `DEFAULT_INTENT_ON_LLM_FAILURE: str`, `INTENT_CLASSIFY_MAX_ATTEMPTS: int`.
- `backend/intent.py` (NEW) -- rule layer (`_rule_greeting`, `_rule_action_request`, `_rule_event_inquiry`, `_rule_faq`, each `text -> str | None`) and the orchestrator `classify(query) -> IntentResult` (frozen dataclass: `label`, `path`).
- `backend/llm_client.py` -- `CLASSIFY_SYSTEM_PROMPT` (constrained, few-shot including the event_inquiry/action_request boundary), `_build_classify_message`, `_parse_intent_label`, `classify_intent(query) -> str` (single-item, production path, never raises), `_build_batch_classify_message`, `_parse_batch_intent_labels`, `classify_intents_batch(queries) -> list[str]` (bulk path for `scripts/eval_intents.py`, one call classifies many items).
- `backend/qa.py` -- `answer_question` calls `intent.classify(query)` once per turn (on the raw query, before the rewrite-based retrieval path), logs it alongside the existing query log line, and attaches it to every `AnswerResult` construction site. `AnswerResult` gains `intent: str = ""` and `intent_path: str = ""` (defaults preserve existing keyword-arg construction, same precedent as `rewritten_query` in Slice 2).
- `backend/cli.py` -- prints `[intent=... | path=...]` after every response.
- `data/intent_eval.jsonl` (NEW) -- 56 hand-labeled queries, balanced across all five categories, including deliberately ambiguous pairs.
- `scripts/eval_intents.py` (NEW) -- runs the eval set through the real rule layer plus one batched `classify_intents_batch` call for every rule-abstained item; reports per-class precision/recall/F1, macro/weighted averages, a confusion matrix, and the rule-path/LLM-path split with per-path accuracy. `--rules-only` skips the LLM entirely. Writes `data/intent_eval_results.md`.
- `tests/test_intent.py` (NEW) -- rule layer and `classify()` orchestrator, `llm_client.classify_intent` mocked.
- `tests/test_llm_client.py` -- `classify_intent`/`classify_intents_batch` coverage: valid response, punctuation/case normalization, retry-then-succeed, retry-exhausted fallback, provider failure, unknown provider, batch parsing/reordering/partial-retry.
- `tests/test_qa.py` -- intent attaches on both grounded and refused paths, classification runs on the raw (not rewritten) query, logged per-turn.
- `tests/test_cli.py` -- intent/path print line.

## Tasks & Acceptance

**Execution:**
- [x] `backend/config.py` -- intent label enum + fallback/retry constants
- [x] `backend/intent.py` -- rule layer + orchestrator
- [x] `backend/llm_client.py` -- constrained single-item + batch LLM classification
- [x] `backend/qa.py` -- per-turn classification, logging, `AnswerResult` fields
- [x] `backend/cli.py` -- intent/path display
- [x] `data/intent_eval.jsonl` -- labeled eval set
- [x] `scripts/eval_intents.py` -- metrics + path-split report
- [x] `tests/test_intent.py`, `tests/test_llm_client.py`, `tests/test_qa.py`, `tests/test_cli.py`
- [x] `README.md` -- Slice 3 section + evaluation results

**Acceptance Criteria:**
- Given a message an unambiguous rule can resolve, when classified, then the LLM is never called and the path is `rule`.
- Given a message where a rule would have to guess between two categories (e.g. the event/action boundary), when classified, then that rule abstains and the LLM fallback resolves it.
- Given any LLM fallback response outside the 5-label enum, when parsed, then it is retried once and, if still invalid, falls back to `out_of_scope` without raising.
- Given any turn (grounded, refused, or LLM-error), when it completes, then its `AnswerResult` carries a non-empty `intent` and `intent_path`, and both are logged.
- Given `scripts/eval_intents.py --rules-only`, when run, then zero LLM calls are made.
- Given `scripts/eval_intents.py` (default), when run, then every rule-abstained eval item is classified via a single batched call, not one call per item.

## Design Notes

**Why rules abstain instead of guessing `out_of_scope`:** a keyword denylist for "not part of the club's domain" is unbounded -- weather, homework, trivia, personal favors -- and any fixed list is simultaneously too narrow (misses most real out-of-scope traffic) and a precision risk (a listed word inside an otherwise in-scope message). Leaving `out_of_scope` entirely to the LLM keeps the rule layer's assertions high-precision by construction; the eval results confirm this (rule-path accuracy 100%, by definition -- rules only ever fire on patterns designed to be unambiguous).

**Topic-based intent, not answerability-based:** "How many members are in the Web Dev team?" and "What programming languages does AIML use?" are both rule-resolved `faq` even though the KB can't answer either -- intent tags the *kind* of question, not whether `qa.answer_question`'s grounding/refusal layer can satisfy it. Conflating the two would make intent classification depend on retrieval internals, and would misclassify plenty of legitimate FAQ-shaped questions as `out_of_scope` just because the KB happens to be thin on that specific detail.

**The event_inquiry/action_request boundary:** the pair in the brief -- "Is HackFest still open for registration?" vs. "Can I still sign up for HackFest?" -- is resolved by checking for a "weak action cue" (`sign up`, `register`, `can i`, `could i`, `join`) inside the event_inquiry rule itself. The first has none and resolves via rule; the second trips the cue and abstains to the LLM. This is the one place a rule's abstention condition is more than "no pattern matched" -- it's "a pattern matched, but so did a competing signal," which is exactly the discipline the brief asked for (precision over coverage, abstain don't guess).

**Batched eval, not batched production:** `classify_intents_batch` exists solely for `scripts/eval_intents.py`'s quota constraint -- one numbered-list prompt in, one numbered-list response out, parsed positionally and tolerant of reordered/missing lines (retried once for any position still unparsed). Production classification (`classify_intent`) stays one message at a time, since a live chat turn has no batch to join.

## Verification

**Commands:**
- `uv run pytest tests/ -v` -- 113 passed, 4 skipped (pre-existing live-provider tests requiring network), no new failures, no network access required for any new test (LLM mocked throughout).
- `uv run python scripts/eval_intents.py --rules-only` -- 33/33 rule-resolved items score 100% (by construction), 23 items reported as skipped, 0 API calls.
- `uv run python scripts/eval_intents.py` -- confirmed live against `GEMINI_API_KEY`: 1 LLM call total (`classify_intents_batch` on the 23 rule-abstained items), overall accuracy 98.2%, macro F1 0.982. See README.md "Intent classification (Slice 3)" for the full table and `data/intent_eval_results.md` for the confusion matrix.

## Suggested Review Order

**Rule layer (the slice's core mechanism)**

- Fixed priority order and abstain-not-guess discipline across all four rules.
  [`intent.py:196`](../../backend/intent.py#L196)

- The event_inquiry/action_request boundary: weak-action-cue check inside `_rule_event_inquiry`, the one place a rule reasons about a *competing* signal rather than just "no pattern matched."
  [`intent.py:140`](../../backend/intent.py#L140)

**LLM fallback**

- Constrained single-label parsing with retry-then-fallback, never raising.
  [`llm_client.py`](../../backend/llm_client.py) -- `classify_intent`

- Batch parsing tolerant of reordered/missing lines, retrying only unresolved positions.
  [`llm_client.py`](../../backend/llm_client.py) -- `classify_intents_batch`, `_parse_batch_intent_labels`

**Integration**

- Classification runs on the raw query (not the Slice 2 rewrite), attaches to every `AnswerResult` branch, logs per-turn.
  [`qa.py`](../../backend/qa.py) -- `answer_question`

**Evaluation**

- Rule pass reused directly from `backend.intent._RULES` (not reimplemented) so the eval script can never drift from production rule behavior; LLM-path items batched into one call.
  [`scripts/eval_intents.py`](../../scripts/eval_intents.py)

**Tests**

- Rule abstention on the deliberately ambiguous pair, and action_request winning over event_inquiry when both topics are present in one message.
  [`test_intent.py`](../../tests/test_intent.py)

- Retry-then-succeed and retry-exhausted-fallback for both single and batch classification.
  [`test_llm_client.py`](../../tests/test_llm_client.py)

- Intent attaches on the refusal path too, and is classified from the raw query even when Slice 2's rewrite changes it.
  [`test_qa.py`](../../tests/test_qa.py)
