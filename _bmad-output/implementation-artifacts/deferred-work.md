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

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-4-confidence-scoring.md`
  summary: The composite confidence evaluation (`uv run python scripts/eval_grounding.py`) has never completed a live run -- the Gemini free-tier daily cap (20 requests) was exhausted during this slice's implementation session, and no second provider is configured (`ANTHROPIC_API_KEY` is unset). The README quotes no composite numbers and commits no composite results file until it does.
  evidence: A first attempt hit repeated 503 UNAVAILABLE (transient model overload) and its retries consumed the remaining daily quota; a subsequent probe returned 429 RESOURCE_EXHAUSTED. The retrieval-only ablation (`--no-verify`, 0 calls) and the retrieval-signal probe both completed and are committed. The scoring pipeline itself is covered without a provider by `tests/test_eval_grounding.py`, but no test can measure how accurate the real LLM verifier is -- only the live run can.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-4-confidence-scoring.md`
  summary: `data/grounding_eval.jsonl`'s 24 labeled answers were hand-authored by the same session that wrote the verification prompt and the evidence-span check. Same independence caveat already recorded for `data/intent_eval.jsonl` in Slice 3 -- worth a second pass by someone who did not write the verifier before citing the composite results as strong external validation.
  evidence: Noted during implementation. The fabrications were deliberately made adversarial (a plausible VP email, an invented event built from a real team name, a 45-minute interview contradicting the KB's 15) rather than obvious, but adversarial-by-the-author is still not adversarial-by-an-independent-reviewer.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-4-confidence-scoring.md`
  summary: `backend/retrieval.py` fails to retrieve the Recruitment section for "Who is eligible to apply?" -- top-1 scores 0.0000 and returns Intro, so an answerable, in-scope question is refused. The KB says "Eligibility: 1st to 3rd year" and "Application Form"; the naive stemmer in `_stem` maps neither "eligible"->"eligibility" nor "apply"->"application".
  evidence: Surfaced by the retrieval-distribution probe written for Slice 4's confidence normalization. This is a Slice 1 retrieval-recall gap, not a confidence-scoring defect -- Slice 4's handling is correct (the turn refuses, and confidence reports `not_applicable` / `refused`, which is honest about a refusal). Fixing it means changing retrieval/refusal behavior, which both this spec and Slice 3's mark "Ask First" / "Never".

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-4-confidence-scoring.md`
  summary: `CONFIDENCE_BAND_HIGH` (0.85) and `CONFIDENCE_BAND_MEDIUM` (0.50) are justified analytically (0.85 is the point at which one unsupported claim in an answer of six or fewer claims cannot reach `high`) but have not yet been tuned against measured composite scores, because that run has not happened. Re-check them against the per-category distribution once the live eval completes.
  evidence: The disciplined order is build eval -> measure -> tune thresholds; only the first step and the retrieval-only ablation are done. The retrieval-only run shows the band boundaries land sensibly on the separation signal, but says nothing about where grounding scores cluster.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-4-confidence-scoring.md`
  summary: Confidence (band, composite score, both sub-scores, and reason code) is logged per-turn via `logger.info` text only, not to a persisted structured store the Slice 5 dashboard can query. Extends the identical gap already recorded for Slice 2's query logging and Slice 3's intent logging.
  evidence: requirements.md §3.2 requires confidence be "logged per-turn, not just displayed and discarded (dashboard depends on this)"; §3.4 requires the dashboard read "from persisted logs, not a hand-maintained/mocked list". Per the existing Slice 1 deferred-work precedent, the structured persisted store is the dashboard slice's dependency, not this one's. Note the reason codes (`refused`, `no_claims`, `verification_failed`, ...) were designed specifically so that store can bucket unscored turns rather than lumping them together.

- source_spec: `_bmad-output/implementation-artifacts/spec-slice-4-confidence-scoring.md`
  summary: Two further Slice 1 retrieval-recall gaps surfaced by the Slice 4 probe, alongside the "Who is eligible to apply?" one above. "What is the interview length?" scores 0.1450 -- just below `RETRIEVAL_THRESHOLD` (0.15) -- and is refused despite the Recruitment section stating "Interview (15 min)". "Can I switch teams?" is answered but retrieves Teams instead of Rules, whose content is the one that says "Team switching once per semester".
  evidence: `data/retrieval_signal_probe.md` keeps both in view rather than trimming them out of the accuracy figure. Neither is a confidence-scoring defect -- the refusal reports `not_applicable`/`refused`, which is honest -- but the 0.1450 near-miss in particular suggests `RETRIEVAL_THRESHOLD` is tuned tightly enough that small stemmer changes would flip real answers, and is worth revisiting together with the stemmer gaps. Changing the threshold or retrieval behavior is marked "Ask First" in this spec.
