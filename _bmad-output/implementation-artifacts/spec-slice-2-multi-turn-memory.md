---
title: 'Slice 2 — Multi-Turn Memory'
type: 'feature'
created: '2026-08-25'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/requirements.md']
baseline_commit: '5c2bff682f2031f855a13272bac0e09136786b09'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The Slice 1 bot answers each question in isolation via TF-IDF, so a follow-up like "Who leads it?" or "When is that?" carries almost no lexical signal and always fails retrieval, forcing the user to re-state full context every turn (requirements.md §3.2).

**Approach:** Track a bounded per-session conversation history (user message, bot answer, cited section), and before retrieval, ask the LLM to rewrite the incoming query into a standalone question using only that history — resolving pronouns/ellipsis without inventing facts. Retrieval and generation then run on the rewritten form; refusal mechanics are untouched.

## Boundaries & Constraints

**Always:**
- Sessions are independent, keyed by `session_id`; no session's history leaks into another's rewrite or retrieval.
- History window is bounded by one named constant (`config.MAX_HISTORY_TURNS`), read at call time (not baked into a fixed-size structure at session creation).
- Both the original and rewritten query are logged every turn, whether or not they differ.
- Rewriting never bypasses the Slice 1 refusal threshold: a rewritten query is retrieved exactly like any query, and refuses below `RETRIEVAL_THRESHOLD`.
- If history is empty for a session, `rewrite_query` is never called (nothing to resolve against) — original query passes straight to retrieval.
- Every turn (grounded, refused, or LLM-error) is recorded to that session's history, including its cited section (nullable).

**Ask First:** None identified.

**Never:** Intent classification, confidence scoring, agentic actions, dashboard/persisted logging store (later slices). Inventing an antecedent when history doesn't resolve a reference. Changing `RETRIEVAL_THRESHOLD` semantics or `REFUSAL_MESSAGE`.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Coreference resolves | History: turn about Cloud Study Jam (Events). Query: "When is that?" | Rewritten to a standalone Cloud Study Jam question; retrieval hits Events ≥ threshold; grounded answer | N/A |
| No resolvable antecedent | History present but doesn't resolve the reference (rewrite returns query unchanged) | Retrieval on the unchanged query scores below threshold; refuses exactly as Slice 1 | Never fabricate an antecedent |
| Session isolation | Two distinct `session_id`s; one has relevant history, the other has none | The empty-history session's follow-up is unaffected by the other session's history | N/A |
| Bounded window | More than `MAX_HISTORY_TURNS` turns added to one session | Only the most recent `MAX_HISTORY_TURNS` turns are retained/used for rewriting | N/A |

</frozen-after-approval>

## Code Map

- `backend/config.py` -- add `MAX_HISTORY_TURNS: int` named constant, read at call time (mirrors existing `RETRIEVAL_THRESHOLD` pattern at `config.py:12`).
- `backend/memory.py` (NEW) -- `Turn` frozen dataclass (`user_message`, `answer`, `source_section`); `SessionStore` with `get_history(session_id) -> list[Turn]` and `add_turn(session_id, turn)`, trimming to `config.MAX_HISTORY_TURNS` on each add.
- `backend/llm_client.py` -- refactor `_generate_anthropic`/`_generate_gemini` (currently hardcode `SYSTEM_PROMPT` at `llm_client.py:78,97`) to accept a `system_prompt` param; add `REWRITE_SYSTEM_PROMPT`, `_build_rewrite_message(query, history)`, and `rewrite_query(query, history) -> str` -- empty history short-circuits (no call); provider failure falls back to the original query unchanged (never raises into `qa.py`).
- `backend/qa.py` -- `answer_question` gains a required `session_id: str` param (plus optional `session_store` override mirroring the existing `retriever` override at `qa.py:33`); looks up history, rewrites, logs both queries via the module's existing `logger`, retrieves/generates on the rewritten form, and records the turn once at the end covering all three existing branches (refusal/grounded/LLM-error). `AnswerResult` gains `rewritten_query: str = ""` (default preserves existing keyword-arg construction in `tests/test_cli.py`).
- `backend/cli.py` -- generate one `session_id` (`uuid.uuid4().hex`) per REPL run; pass it to every `answer_question` call; print the rewritten query line when it differs from the original.
- `tests/test_memory.py` (NEW) -- `SessionStore` bounded window and session isolation.
- `tests/test_llm_client.py` -- add `rewrite_query` coverage: empty-history short-circuit, prompt construction from history, provider-failure fallback.
- `tests/test_qa.py` -- update existing calls to pass a unique `session_id` (each still needs zero prior history to stay LLM-mock-free); add coreference-across-turns, no-resolvable-antecedent, and session-isolation tests (all mocking `llm_client.rewrite_query`/`generate_answer`).
- `tests/test_cli.py` -- update `fake_answer_question` signatures for the new `session_id` argument.

## Tasks & Acceptance

**Execution:**
- [x] `backend/config.py` -- add `MAX_HISTORY_TURNS` -- named, tunable window size
- [x] `backend/memory.py` -- `Turn` + `SessionStore` -- per-session bounded history, independent sessions
- [x] `backend/llm_client.py` -- generic system-prompt param + `rewrite_query` -- reuses existing provider plumbing for reference resolution
- [x] `backend/qa.py` -- thread `session_id` through history lookup, rewrite, logging, retrieval/generation, and turn recording -- core integration point
- [x] `backend/cli.py` -- per-run `session_id` + rewritten-query display -- manual/demo verification
- [x] `tests/test_memory.py` -- bounded window + isolation
- [x] `tests/test_llm_client.py` -- `rewrite_query` coverage
- [x] `tests/test_qa.py` -- coreference, unresolvable antecedent, session isolation
- [x] `tests/test_cli.py` -- signature updates for `session_id`

**Acceptance Criteria:**
- Given a session where turn 1 is grounded in a KB section, when turn 2 asks a pronoun/ellipsis follow-up, then the rewritten standalone query is used for retrieval and the response grounds in the correct section.
- Given a follow-up whose reference `rewrite_query` cannot resolve from history, when `answer_question` runs, then it returns `refused=True` exactly as an ungrounded Slice-1 query would — no fabricated antecedent.
- Given two distinct `session_id`s, when one has history and the other has none, then the empty session's rewrite/retrieval behaves as if no conversation ever happened.
- Given more than `MAX_HISTORY_TURNS` turns recorded in one session, when history is read for rewriting, then only the most recent `MAX_HISTORY_TURNS` are present.
- Given any turn, when it completes, then both the original and rewritten query appear in the log output.

## Design Notes

**Why LLM-based rewriting over rule-based pronoun substitution:** resolving "it"/"that" requires identifying *which entity* a prior free-text turn introduced — semantic reference resolution, not pattern matching. The codebase already has a provider-agnostic, mockable LLM wrapper (`llm_client.generate_answer`) built for constrained NL tasks; `rewrite_query` reuses that same plumbing (same providers, same test-mocking approach) rather than adding a second NLP mechanism. Example: "When is that?" alone scores 0.0 against every KB section (no shared vocabulary); rewritten to "When is the Cloud Study Jam?" it scores 0.212 on Events (above `RETRIEVAL_THRESHOLD = 0.15`) — empirically confirmed against the live retriever.

**Anti-fabrication guardrail:** `REWRITE_SYSTEM_PROMPT` instructs the model to resolve references *only* from the given history and to return the follow-up **unchanged** if history doesn't resolve it — never invent an antecedent. An unresolved query then flows into the untouched threshold check and refuses exactly like Slice 1. Provider failures during rewriting fall back to the original query (never surfaced as an error) for the same reason: worst case is a Slice-1-style refusal, never a fabricated resolution.

**Why history is stored, not the rewritten form:** each session's history records what the user actually said and what was actually cited, so later rewrites resolve against real conversational ground truth rather than compounding paraphrases.

## Verification

**Commands:**
- `uv run pytest tests/ -v` -- expected: all tests pass, no network/API key required (rewrite/generation mocked at the `llm_client` boundary). Result: 66 passed (up from Slice 1's baseline), 0 failed, no network access.

**Manual checks (if no CLI):**
- `uv run python -m backend.cli`: ask "Tell me about the Cloud Study Jam", then "When is that?" → grounded answer citing Events, with the rewritten query printed and visibly different from the raw follow-up.

## Suggested Review Order

**Rewrite-then-retrieve orchestration**

- Entry point: history lookup, the rewrite-only-when-non-blank guard, and per-turn logging of both queries, all before retrieval runs.
  [`qa.py:74`](../../backend/qa.py#L74)

- Every branch (refused/grounded/LLM-error) builds a `result`, then one shared call records the turn -- the single place all three paths converge.
  [`qa.py:129`](../../backend/qa.py#L129)

- `rewritten_query` added to the uniform result shape with a safe default for pre-existing keyword-arg test construction.
  [`qa.py:36`](../../backend/qa.py#L36)

**Query rewriting (the slice's core mechanism)**

- Anti-fabrication guardrail: resolve only from history, return the follow-up unchanged if unresolvable -- never invent an antecedent.
  [`llm_client.py:32`](../../backend/llm_client.py#L32)

- Empty history short-circuits with zero provider calls; any failure (including unknown provider) falls back to the original query, now logged.
  [`llm_client.py:179`](../../backend/llm_client.py#L179)

- History renders as ordered user/assistant exchanges -- the only material the rewrite prompt is allowed to resolve references from.
  [`llm_client.py:97`](../../backend/llm_client.py#L97)

- Both provider functions now take an explicit `system_prompt` instead of a hardcoded constant, so one call site serves both grounded generation and rewriting.
  [`llm_client.py:113`](../../backend/llm_client.py#L113)

**Session state**

- `SessionStore.add_turn` reads `config.MAX_HISTORY_TURNS` at call time and trims -- tuning the constant changes every session's window immediately.
  [`memory.py:51`](../../backend/memory.py#L51)

- `get_history` returns a copy so callers can't mutate the store's internal list.
  [`memory.py:43`](../../backend/memory.py#L43)

- `Turn.source_section` is nullable -- refusals and LLM-errors are recorded too, not just grounded turns.
  [`memory.py:16`](../../backend/memory.py#L16)

**CLI (manual/demo verification harness)**

- One `session_id` per REPL run threads through every call; the rewritten query prints only when it actually differs.
  [`cli.py:18`](../../backend/cli.py#L18)

**Config**

- `MAX_HISTORY_TURNS` -- the one named, tunable window-size constant.
  [`config.py:41`](../../backend/config.py#L41)

**Tests**

- Coreference resolves via a mocked rewrite, retrieval/generation run on the rewritten form, and grounding lands on the correct section.
  [`test_qa.py:133`](../../tests/test_qa.py#L133)

- Unresolvable reference (rewrite returns the query unchanged) refuses exactly like Slice 1 -- no fabricated antecedent.
  [`test_qa.py:172`](../../tests/test_qa.py#L172)

- Session isolation: one session's history never affects another's rewrite/retrieval.
  [`test_qa.py:196`](../../tests/test_qa.py#L196)

- Bounded window: only the most recent `MAX_HISTORY_TURNS` turns reach the rewrite call.
  [`test_qa.py:218`](../../tests/test_qa.py#L218)

- Review patch: blank follow-up with existing history never triggers a real rewrite call.
  [`test_qa.py`](../../tests/test_qa.py)

- Review patch: `caplog`-based coverage that the log line actually contains both the original and rewritten query.
  [`test_qa.py`](../../tests/test_qa.py)

- `SessionStore` bounded window, read-at-call-time semantics, and session isolation in isolation from `qa.py`.
  [`test_memory.py:97`](../../tests/test_memory.py#L97)

- `rewrite_query`'s short-circuit, prompt construction/ordering, both providers, and every fallback path (blank result, `None` text, provider failure, unknown provider).
  [`test_llm_client.py:475`](../../tests/test_llm_client.py#L475) Result: confirmed live against `GEMINI_API_KEY` --
  ```
  > Tell me about the Cloud Study Jam
  Based on the provided context, the Cloud Study Jam is an upcoming event scheduled for September 20.
  [source=Events | score=0.212]

  > When is that?
  [rewritten: When is the Cloud Study Jam?]
  The Cloud Study Jam is on September 20.
  [source=Events | score=0.212]
  ```
  Matches the score empirically cited in Design Notes exactly (0.212), confirming the rewrite drove retrieval to the same section as a direct question would.

## Suggested Review Order

**Core integration point**

- History lookup, conditional rewrite (skipped entirely when history is empty), and the single end-of-function turn recording covering all three branches (refusal/grounded/LLM-error).
  [`qa.py:39`](../../backend/qa.py#L39), [`qa.py:127`](../../backend/qa.py#L127)

**Reference resolution**

- `rewrite_query`: empty-history short-circuit, provider-failure fallback to the original query, blank-result fallback -- every exit path short of a real rewrite returns the original query unchanged, never fabricates.
  [`llm_client.py:176`](../../backend/llm_client.py#L176)

- `_build_rewrite_message`: only prior turns' user/assistant text plus the follow-up are placed in context -- nothing else the model could resolve against.
  [`llm_client.py:97`](../../backend/llm_client.py#L97)

**Bounded, isolated history**

- `SessionStore.add_turn` reads `config.MAX_HISTORY_TURNS` at call time and trims on every add -- not fixed at session creation.
  [`memory.py:51`](../../backend/memory.py#L51)

- `MAX_HISTORY_TURNS` named constant.
  [`config.py:41`](../../backend/config.py#L41)

**Tests**

- Coreference resolution, unresolvable-antecedent refusal, session isolation, bounded-window trimming, and full turn-recording across all three branches.
  [`test_qa.py`](../../tests/test_qa.py)

- `rewrite_query` coverage: empty-history short-circuit, prompt construction from history (order preserved, correct system prompt), provider failure and blank-response fallback.
  [`test_llm_client.py`](../../tests/test_llm_client.py)

- `SessionStore` bounded window (including read-at-call-time semantics) and session isolation; copy-not-reference on `get_history`.
  [`test_memory.py`](../../tests/test_memory.py)

- CLI: shared `session_id` across a run, rewritten-query line only printed when it actually differs.
  [`test_cli.py`](../../tests/test_cli.py)
