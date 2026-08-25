- source_spec: `_bmad-output/implementation-artifacts/spec-slice-1-kb-grounded-qa.md`
  summary: Add a top-level README.md documenting setup, install, and how to run backend.cli, plus an .env.example for ANTHROPIC_API_KEY.
  evidence: Blind-hunter review of Slice 1 found no setup docs beyond a one-line mention in cli.py's docstring. README.md is an explicit Submission Requirement (requirements.md §4), scoped to submission-prep rather than Slice 1's own acceptance criteria.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-1-kb-grounded-qa.md`
  summary: Add structured logging/observability of each query -- question asked, matched section, retrieval score, refused vs grounded vs LLM-error outcome.
  evidence: Blind-hunter review of Slice 1 found no logging beyond interactive CLI output (a logger was added in the qa.py error path during patch triage, but only for exceptions, not routine query outcomes). requirements.md §3.4 (Dashboard) needs a persisted log of chat stats, intent breakdown, actions, and unanswered queries -- this is that slice's dependency, not Slice 1's.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-1-kb-grounded-qa.md`
  summary: Decide how a retrieval-successful-but-LLM-call-failed outcome should be surfaced/counted once intent classification and the dashboard exist (currently encoded only in AnswerResult.answer text with refused=False, source_section/score preserved).
  evidence: Edge-case-hunter review of Slice 1 flagged that an empty-text LLM response would silently look like a valid grounded answer. The Slice 1 patch fixed the crash risk (qa.py now catches LLM-call exceptions and returns a distinct LLM_ERROR_MESSAGE), but whether this LLM-error state needs its own field/bucket is a design decision for the confidence-scoring and dashboard "unanswered queries" work in later slices, not something Slice 1's spec covers.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-2-multi-turn-memory.md`
  summary: README.md is now further stale for Slice 2 -- it still describes Slice 1 only, with no mention of `session_id`, `MAX_HISTORY_TURNS`, the query-rewrite step, or that a grounded multi-turn follow-up now costs up to two LLM calls (rewrite + generate) instead of one.
  evidence: Blind-hunter review of Slice 2 found README.md unchanged from Slice 1. Compounds the existing README deferred-work entry above; per that entry's precedent, README maintenance is scoped to submission-prep rather than each slice's own acceptance criteria.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-2-multi-turn-memory.md`
  summary: `backend/memory.SessionStore` has no thread-safety/locking around `add_turn`/`get_history`, and no eviction/TTL across distinct `session_id`s -- `_sessions` grows unbounded for the life of the process.
  evidence: Blind-hunter and edge-case-hunter reviews of Slice 2 both flagged this. Not a defect for the current single-threaded CLI consumer, but will need attention once a concurrent multi-session frontend (e.g. a Streamlit dashboard, a later slice) is built.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-2-multi-turn-memory.md`
  summary: No sanitization/bounding of user input before interpolating it into LLM prompts; Slice 2 doubles the free-text surface fed to the model (the follow-up query and the accumulated conversation history, via `rewrite_query`) on top of Slice 1's already-unsanitized `generate_answer` prompt.
  evidence: Blind-hunter review of Slice 2 flagged this. Inherits Slice 1's existing trust model unchanged (no new precedent set), but is worth a dedicated security-review pass before submission given the larger prompt-injection surface.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-1-kb-grounded-qa.md`
  summary: `TfidfRetriever.retrieve()` is called from `qa.answer_question` with no try/except; if it ever raised, the exception would propagate uncaught out of `answer_question` and (as of Slice 2) skip recording that turn to session history.
  evidence: Blind-hunter review of Slice 2 surfaced this while reading `qa.py`, but the unwrapped call predates Slice 2 -- `retrieve()` was already called without error handling in Slice 1, and nothing in either spec asks for retrieval-layer error handling.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-3-intent-classification.md`
  summary: Intent and intent_path are logged per-turn via the existing `logger.info` text log only, not to a persisted structured store (JSONL/DB) the Slice 5 dashboard can query for its intent-breakdown counts. Same gap applies to Slice 2's original/rewritten query logging.
  evidence: requirements.md §3.2 says intent must be "logged per-turn, not just displayed and discarded (dashboard depends on this)"; requirements.md §3.4 says the dashboard "reads from persisted logs, not a hand-maintained/mocked list." Slice 3's own scope explicitly excludes building the dashboard, and per the existing Slice 1 deferred-work entry above, structured persisted logging is that later slice's dependency, not this one's.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-3-intent-classification.md`
  summary: Rule keyword lists in `backend/intent.py` (event names, team names, action verbs) are hardcoded and duplicate knowledge already present in `backend/kb_data.py` (event/team names) -- if the KB content ever changes, the rules must be updated by hand in a second place.
  evidence: Noted during implementation. Not a defect against Slice 3's own acceptance criteria (the KB is fixed per requirements.md §2, "Do not deviate from this document without updating it first"), but worth revisiting if the KB ever becomes dynamic.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-3-intent-classification.md`
  summary: `data/intent_eval.jsonl`'s gold labels were hand-authored by the same process that wrote the rules, and one boundary condition (`_rule_event_inquiry`'s weak-action-cue check) was tuned specifically so the eval set's ambiguous pair resolves the way the spec's own example described. This is normal for a first eval pass, but the set hasn't been independently reviewed by someone who didn't write the rules -- worth a second pass before citing the 98.2% figure as strong external validation.
  evidence: Noted during implementation while designing the eval set alongside the rules in the same session.
