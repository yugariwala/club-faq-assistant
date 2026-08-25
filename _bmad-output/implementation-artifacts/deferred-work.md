- source_spec: `_bmad-output/implementation-artifacts/spec-slice-1-kb-grounded-qa.md`
  summary: Add a top-level README.md documenting setup, install, and how to run backend.cli, plus an .env.example for ANTHROPIC_API_KEY.
  evidence: Blind-hunter review of Slice 1 found no setup docs beyond a one-line mention in cli.py's docstring. README.md is an explicit Submission Requirement (requirements.md §4), scoped to submission-prep rather than Slice 1's own acceptance criteria.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-1-kb-grounded-qa.md`
  summary: Add structured logging/observability of each query -- question asked, matched section, retrieval score, refused vs grounded vs LLM-error outcome.
  evidence: Blind-hunter review of Slice 1 found no logging beyond interactive CLI output (a logger was added in the qa.py error path during patch triage, but only for exceptions, not routine query outcomes). requirements.md §3.4 (Dashboard) needs a persisted log of chat stats, intent breakdown, actions, and unanswered queries -- this is that slice's dependency, not Slice 1's.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-1-kb-grounded-qa.md`
  summary: Decide how a retrieval-successful-but-LLM-call-failed outcome should be surfaced/counted once intent classification and the dashboard exist (currently encoded only in AnswerResult.answer text with refused=False, source_section/score preserved).
  evidence: Edge-case-hunter review of Slice 1 flagged that an empty-text LLM response would silently look like a valid grounded answer. The Slice 1 patch fixed the crash risk (qa.py now catches LLM-call exceptions and returns a distinct LLM_ERROR_MESSAGE), but whether this LLM-error state needs its own field/bucket is a design decision for the confidence-scoring and dashboard "unanswered queries" work in later slices, not something Slice 1's spec covers.
