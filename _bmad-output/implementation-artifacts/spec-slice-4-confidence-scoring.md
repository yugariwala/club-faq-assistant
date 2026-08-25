---
title: 'Slice 4 — Confidence Scoring'
type: 'feature'
created: '2026-08-25'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/requirements.md']
baseline_commit: '06fced1'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** requirements.md §3.2 requires every response to carry a confidence indicator, displayed in the UI and logged per-turn for the Slice 5 dashboard. §5b lays out four candidate signals and leaves the choice open. A number that doesn't discriminate between a grounded answer and a fabricated one is decoration, so whatever signal is chosen has to be measured against labeled data, not asserted.

**Approach:** the **Composite** row of requirements.md §5b — two independent signals combined with `min()`.

1. **Retrieval confidence** — how cleanly the query discriminated one KB section from the rest. Measures "did we find *the* right section."
2. **Grounding confidence** — post-hoc claim-by-claim verification of the generated answer against the retrieved KB text. Measures "is the answer actually faithful."

`final = min(retrieval, grounding)`: confidence is bounded by the weakest link, so a strong retrieval can never mask an ungrounded answer. Deliberately conservative — for a bot forbidden from fabricating, a false `high` costs far more than a false `medium`.

Agentic actions (§3.3) and the dashboard (§3.4) are explicitly out of scope for this slice.

## Boundaries & Constraints

**Always:**
- Retrieval confidence is the **separation ratio** `(top1 - top2) / top1`, clamped to [0, 1]; `0.0` when `top1 == 0`, `1.0` when there is no second candidate. Bounded by construction — no tuned rescaling constant.
- Raw top-1 magnitude keeps exactly its existing role: the binary refusal gate at `RETRIEVAL_THRESHOLD`. Magnitude decides *whether* to answer; separation decides *how confidently*.
- A claim may only be scored SUPPORTED if the verifier emits an evidence span that is **verbatim present in the retrieved section content** after normalization. A span that fails the substring check is downgraded to UNSUPPORTED in code, never trusted.
- Grounding confidence = supported claims / total claims, over claims extracted by the verifier.
- Verification runs as a **fresh LLM call** with an adversarial-auditor system prompt whose default verdict is UNSUPPORTED — it never sees that the same model authored the answer.
- Band thresholds are named constants in `backend/config.py`; no inline numbers at any call site.
- Both the band and the raw score are displayed, and both sub-scores are retained on the result for per-turn logging.
- Verification can be disabled (`VERIFY_GROUNDING=0`) for quota-constrained runs; the added per-turn cost is documented in README.md.

**Ask First:** Changing `RETRIEVAL_THRESHOLD` or any refusal semantics. Changing the `min()` combination rule.

**Never:** Agentic actions, the dashboard, or persisted structured logging (JSONL/DB) — that store remains a later slice's dependency, per the existing deferred-work.md precedent. LLM self-reported confidence (requirements.md §5b rates it "weakest under questioning"). Trusting a verifier's SUPPORTED verdict without the mechanical evidence-span check.

## Design Notes

**Why separation, not normalized magnitude.** requirements.md §5b's first row proposes the raw retrieval similarity score. Measured over 20 answerable queries spanning all 7 KB sections, plus 6 out-of-scope and 8 deliberately ambiguous ones:

| | raw top-1 cosine | separation ratio |
|---|---|---|
| answerable (n=20) | min 0.145, p25 0.219, median 0.345, max 0.712 | min 0.26, p25 0.57, median 0.89 |
| out-of-scope (n=6) | 0.0000 throughout | 0.0 |
| ambiguous ("cloud", "design", "2025", "workshops") | 0.107-0.321 | 0.03, 0.03, 0.11, 0.26 |

Top-1 *section accuracy was 19/20 across the entire non-zero magnitude range* — 0.1677 ("When is HackFest 2025?") is exactly as correct as 0.7121 ("Who is the Cloud team lead?"). Magnitude tracks query length and term-overlap density, not match quality; any ramp wide enough to act as a gradient bands correct short-query lookups `low`. Separation measures discrimination directly, which is what "did we find the right section" actually asks, and it is bounded [0,1] by construction with zero free parameters — a stronger normalization basis than any hand-tuned ramp.

*Limitation:* on a 7-entry KB, separation runs high whenever a query contains a section-unique term. On a larger corpus with overlapping documents this signal would compress and would need re-measuring.

**Why claims are extracted by the LLM, fused with verification into one call.** Structural (sentence-split) extraction is free but not atomic: "AIML is led by Rahul Sharma and Web Dev by Priya Patel" is two claims in one sentence, and scoring it as one supported claim hides a half-fabrication. Atomicity is the point of the metric. Decomposition and adjudication read the same two inputs (answer, context), so splitting them into two calls doubles the quota cost for no measurable gain — one call emits `CLAIM:`/`VERDICT:`/`EVIDENCE:` triples. What guards against self-confirmation is not call separation but the fresh call plus the adversarial prompt and the mechanical check below.

**What "supported" means.** Exact string match is too strict ("Rahul Sharma leads AIML" vs. KB "AIML (Lead: Rahul Sharma)"). Pure semantic judgment lets the verifier rubber-stamp its own hallucinated support. The resolution is **evidence anchoring with mechanical validation**: SUPPORTED requires an evidence span, and `confidence._evidence_supports` checks that span is a verbatim substring of the section content after NFKC/casefold/whitespace/dash normalization. A span that isn't found is downgraded to UNSUPPORTED. So "supported" = *the verifier asserted entailment **and** pointed at a real, machine-verified span*. The semantic step absorbs paraphrase; the substring check blocks invented support and makes the failure mode auditable in the report rather than silent. Validation lives in `confidence.py`, not `llm_client.py` — the client parses what the model said, policy decides what it means (same separation as the policy-free retriever).

**Aggregation.** "The club has 6 teams" is derived by counting, never literally stated. It is faithful, not fabricated, and stays SUPPORTED with the enumeration itself as the evidence span — which is verbatim, so the substring check still holds. Covered by a few-shot example and by a dedicated aggregate item in the eval set.

**The refusal case.** Scoring a refusal `0.0` is misleading — a correct refusal is the most trustworthy output this bot produces, and badging it `low` inverts that; it would also poison the eval by stacking the low band with correct behavior. Scoring it `1.0` is equally wrong, since nothing was verified. Confidence answers "how much should you trust this claim about the club"; a refusal makes no such claim, so the question does not apply: `score=None`, `band="not_applicable"`, with a reason code so the Slice 5 dashboard can bucket the states separately.

The asymmetry between `no_claims`/`refused` and `verification_failed` is deliberate: **`not_applicable` when there are no claims to check; `low` when there are claims we failed to check.** An answer that asserts facts we could not verify is precisely what this system exists to be suspicious of.

**Why `min()`.** Confidence is bounded by the weakest link. `min()` can never exceed either signal, so it under-reports when both are moderate for independent benign reasons — accepted deliberately. The alternative (a weighted sum) lets a strong retrieval average away a poorly grounded answer, which is the exact failure mode requirements.md §3.1 forbids.

**Band thresholds.** Grounding is `k/n` with n typically 1-6. `CONFIDENCE_BAND_HIGH = 0.85` is set so that **a single unsupported claim in any answer of six or fewer claims cannot reach `high`** (4/5 = 0.80, 5/6 = 0.83 — both `medium`). At 0.75, a 3-of-4 answer — one fabricated claim — would badge `high`, the precise failure the eval exists to catch. `CONFIDENCE_BAND_MEDIUM = 0.50` reads as "at least half the claims verified."

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Clean direct lookup | "Who is the Cloud team lead?" | separation 0.886, grounding 1.0 -> `min` 0.886, band `high`, reason `verified` | N/A |
| Correct but lexically thin | "When is HackFest 2025?" | raw magnitude 0.168 (weak) but separation 0.542 -> not falsely banded low | N/A |
| Ambiguous query | "cloud" | separation 0.028 -> caps composite at 0.028, band `low` even if fully grounded | N/A |
| Fabricated claim in answer | answer names a lead absent from context | that claim's evidence span fails the substring check -> UNSUPPORTED -> grounding < 1 | N/A |
| Verifier invents its own evidence | EVIDENCE span not in content | downgraded to UNSUPPORTED in code | Logged, counted in report |
| Below-threshold refusal | "What's the club's budget?" | `score=None`, band `not_applicable`, reason `refused` | N/A |
| LLM error / quota answer | `LLM_ERROR_MESSAGE` | `score=None`, band `not_applicable`, reason `llm_error` / `llm_quota` | N/A |
| In-context "I don't know" | generated text asserting nothing | verifier returns `NO_CLAIMS` -> `score=None`, `not_applicable`, reason `no_claims` — no 0/0 | N/A |
| Verification disabled | `VERIFY_GROUNDING=0` | `score = retrieval only`, banded, reason `verification_disabled` — never presented as a full composite | N/A |
| Verifier call fails | provider raises | `score=0.0`, band `low`, reason `verification_failed` | Logged, never raised |
| Single-candidate retrieval | only one KB entry | separation `1.0` (nothing competes) | N/A |
| Zero-score retrieval | `top1 == 0` | separation `0.0` | N/A |

</frozen-after-approval>

## Code Map

- `backend/config.py` — `CONFIDENCE_BAND_HIGH`, `CONFIDENCE_BAND_MEDIUM`, band-name constants, `CONFIDENCE_VERIFICATION_ENABLED`, `VERIFY_GROUNDING_ENV_VAR`, `VERIFY_BATCH_SIZE`, reason-code constants.
- `backend/llm_client.py` — `VERIFY_SYSTEM_PROMPT`, `ClaimVerdict` (raw parse product), `_build_verify_message`, `_parse_claim_verdicts`, `verify_grounding()`, `_build_batch_verify_message`, `_parse_batch_claim_verdicts`, `verify_groundings_batch()` (eval-only bulk path, mirrors `classify_intents_batch`).
- `backend/confidence.py` (NEW) — `VerifiedClaim`, `ConfidenceResult`, `retrieval_confidence()`, `band_for()`, `_normalize_for_match()`, `_evidence_supports()`, `verification_enabled()`, `not_applicable()`, `score_generated_answer()`.
- `backend/qa.py` — `AnswerResult.confidence: ConfidenceResult | None = None`; every branch constructs one; logged per-turn.
- `backend/cli.py` — prints band + raw score + reason.
- `data/grounding_eval.jsonl` (NEW) — labeled answers: `grounded` / `partially_grounded` / `fabricated`.
- `scripts/eval_grounding.py` (NEW) — per-category mean/min/max, band distribution, fabricated-scoring-high count, separation report; `--no-verify` for a 0-call run.
- `tests/test_confidence.py` (NEW), `tests/test_llm_client.py`, `tests/test_qa.py`, `tests/test_cli.py`.

## Tasks & Acceptance

**Execution:**
- [x] `backend/config.py` — band thresholds, reason codes, verification toggle
- [x] `backend/llm_client.py` — verification prompt, parsing, single + batch paths
- [x] `backend/confidence.py` — separation, evidence validation, orchestrator
- [x] `backend/qa.py` / `backend/cli.py` — wiring and display
- [x] `data/grounding_eval.jsonl` + `scripts/eval_grounding.py`
- [x] tests
- [x] `README.md` — Slice 4 section, quota cost, evaluation results

**Acceptance Criteria:**
- Given a verifier SUPPORTED verdict whose evidence span is absent from the content, when scored, then the claim counts as unsupported.
- Given any refusal or LLM-error answer, when scored, then `score is None` and `band == "not_applicable"` with the matching reason code.
- Given `VERIFY_GROUNDING=0`, when a grounded answer is scored, then zero verification calls are made and the reason is `verification_disabled`.
- Given a verifier call that raises, when scored, then `score == 0.0`, band `low`, reason `verification_failed`, and nothing propagates to the caller.
- Given the labeled eval set, when scored, then **no `fabricated` item lands in the `high` band**, and mean(`grounded`) > mean(`partially_grounded`) > mean(`fabricated`).
- Given band thresholds, when read, then they come from `config` — no inline numeric literal at any call site.

## Verification

**Commands:**
- `uv run pytest tests/ -q` -- **183 passed, 4 skipped** (the 4 pre-existing live-provider tests), no network access required. `tests/conftest.py` now enforces that structurally rather than by convention.
- `uv run python scripts/eval_grounding.py --probe` -- 0 API calls. Of 21 answerable queries, 2 fall below the refusal gate; among the 19 answered, top-1 section accuracy is **18/19** across a raw-magnitude range of 0.168-0.712 (>4x); separation 0.259-1.000 on those vs. 0.028-0.260 on ambiguous ones. Written to `data/retrieval_signal_probe.md`. This is the measurement the separation-ratio choice rests on.
- `uv run python scripts/eval_grounding.py --no-verify` -- 0 API calls, 24 items, complete. **3 fabricated answers reach the `high` band** on the retrieval signal alone, with a fabricated band distribution (3/1/4) barely distinguishable from grounded (4/2/2). Written to `data/grounding_eval_results_retrieval_only.md`. This is the ablation justifying the second LLM call.
- `uv run python scripts/eval_grounding.py` (composite) -- **NOT RUN.** Blocked on the Gemini free-tier daily cap (20 requests/day), exhausted during this session; no second provider configured. Recorded in deferred-work.md. No composite figures are quoted in README.md and no composite results file is committed until this completes.

**Acceptance criteria status:**
- Evidence-span downgrade -- verified (`test_supported_verdict_with_missing_evidence_is_downgraded`, `test_verifier_integrity_section_counts_downgraded_claims`).
- Refusal/LLM-error give `score is None` + `not_applicable` + matching reason -- verified (`test_refusal_carries_a_not_applicable_confidence_and_never_verifies`, `test_llm_error_answer_is_not_applicable_not_low_confidence`, `test_quota_error_answer_carries_its_own_confidence_reason`).
- `VERIFY_GROUNDING=0` makes zero verification calls -- verified (`test_verification_disabled_skips_the_extra_call_on_a_grounded_turn`).
- Verifier failure gives `0.0` / `low` / `verification_failed`, nothing propagates -- verified (`test_verification_failure_scores_low_not_not_applicable`, `test_unexpected_verifier_exception_does_not_propagate`). Also confirmed against a real 503 during the blocked eval attempt: every item degraded rather than crashing.
- Eval set separates the categories, no fabricated item reaches `high` -- **pipeline verified** with a stubbed correct verifier (`test_pipeline_separates_categories_given_a_correct_verifier`); **not yet measured live**, see above.
- Band thresholds read from `config` with no inline literal -- verified (`test_band_follows_config_when_thresholds_are_retuned`, `test_single_unsupported_claim_cannot_reach_high_in_a_short_answer`).

**Deviation from the brief:** the retrieval signal is the separation ratio `(top1 - top2) / top1`, not a normalized raw cosine. Raised with the user before implementation, with the measurement above; approved.

**Unplanned change:** `tests/conftest.py` (new). Wiring verification into `qa.answer_question` exposed that the suite was already making real network calls -- `classify_intent` (Slice 3) swallows provider failures, so an unpatched test passed while spending quota. Adding a verification call per grounded turn would have widened that leak, so the boundary is now blocked structurally.
