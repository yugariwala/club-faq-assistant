"""Composite confidence scoring for every answer (requirements.md §3.2, §5b).

Two independent signals, combined with `min()`:

1. **Retrieval confidence** -- how cleanly the query discriminated one KB
   section from the rest ("did we find *the* right section").
2. **Grounding confidence** -- the fraction of the generated answer's atomic
   claims that are verifiably supported by the retrieved section text ("is
   the answer actually faithful").

`min()` rather than a weighted sum: confidence is bounded by the weakest
link, so a strong retrieval can never mask an ungrounded answer. That makes
the number conservative by design -- it under-reports when both signals are
moderate for independent, benign reasons -- which is the right trade for a
bot forbidden from fabricating, where a false `high` costs far more than a
false `medium`.

This module owns *policy*: it decides what a verifier's verdict is worth,
what a score means, and when the question doesn't apply. `backend.llm_client`
only parses what the model said. Same split as `backend.retrieval`, which
scores candidates but never decides refusal.
"""

import logging
import os
import unicodedata
from dataclasses import dataclass

from backend import config, llm_client
from backend.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifiedClaim:
    """One claim after this module has re-checked the verifier's citation.

    `supported` is the verdict that counts, which is not necessarily the one
    the verifier reported: a SUPPORTED verdict whose evidence span isn't
    actually in the source is downgraded here. `evidence_found` records that
    downgrade explicitly so the eval report can surface how often the
    verifier tried to cite something it made up -- a silent correction would
    hide exactly the behavior worth measuring.
    """

    claim: str
    supported: bool
    evidence: str
    evidence_found: bool


@dataclass(frozen=True)
class ConfidenceResult:
    """The confidence indicator attached to every answer.

    `score` is None precisely when `band` is `not_applicable` -- the response
    makes no factual claim, so "how much should you trust this claim" has no
    answer (see `not_applicable`). Both sub-scores are kept alongside the
    composite so a turn's log records *why* confidence was what it was, and
    so the Slice 5 dashboard can break the number down rather than just
    counting bands.
    """

    score: float | None
    band: str
    reason: str
    retrieval_score: float | None = None
    grounding_score: float | None = None
    claims: tuple[VerifiedClaim, ...] = ()

    @property
    def supported_claims(self) -> int:
        return sum(1 for claim in self.claims if claim.supported)

    def display(self) -> str:
        """Human-readable band + raw score, for the CLI and any later UI.

        Both are shown, never just the band -- a band alone hides how close
        a `medium` was to either edge.
        """
        if self.score is None:
            return f"{self.band} ({self.reason})"
        return f"{self.band} ({self.score:.2f}, {self.reason})"


def retrieval_confidence(candidates: list[RetrievalResult]) -> float:
    """Score how cleanly retrieval discriminated one section, in [0, 1].

    The **separation ratio** `(top1 - top2) / top1`: the fraction of the top
    candidate's score that the runner-up does not account for.

    Not the raw top-1 cosine magnitude, which requirements.md §5b's first row
    suggests. Measured over 20 answerable queries spanning all 7 KB sections
    (see scripts/eval_grounding.py --probe, which reproduces this), top-1
    *section accuracy is 19/20 across the entire non-zero magnitude range* --
    "When is HackFest 2025?" scores 0.168 and is exactly as correct as "Who
    is the Cloud team lead?" at 0.712. Magnitude tracks query length and
    term-overlap density, not match quality, so any rescaling wide enough to
    act as a gradient bands correct short-query lookups `low`. Separation
    measures discrimination directly, which is what "did we find the right
    section" actually asks: it runs 0.26-1.00 for answerable queries but
    0.03-0.26 for genuinely ambiguous ones ("cloud", "design", "2025").

    It also needs no tuned constant -- being a ratio of two cosines with
    `0 <= top2 <= top1`, it is bounded [0, 1] by construction. Raw magnitude
    keeps the role it already has and is good at: the binary refusal gate at
    `config.RETRIEVAL_THRESHOLD`, applied by `qa.answer_question`. Magnitude
    decides *whether* to answer; separation decides *how confidently*.

    Known limitation: over a 7-entry KB, separation runs high whenever the
    query contains a term unique to one section. On a larger corpus with
    overlapping documents this signal would compress and would need
    re-measuring against fresh labeled data.
    """
    if not candidates:
        return 0.0

    top = candidates[0].score
    if top <= 0.0:
        # Nothing matched at all; there is no discrimination to measure.
        # (Such a query is below the refusal threshold anyway.)
        return 0.0

    runner_up = candidates[1].score if len(candidates) > 1 else 0.0
    return max(0.0, min(1.0, (top - runner_up) / top))


def band_for(score: float) -> str:
    """Map a score to its named band. Thresholds live in `config` and are
    read here at call time, so tuning them changes banding everywhere with
    no change at any call site."""
    if score >= config.CONFIDENCE_BAND_HIGH:
        return config.CONFIDENCE_BAND_HIGH_NAME
    if score >= config.CONFIDENCE_BAND_MEDIUM:
        return config.CONFIDENCE_BAND_MEDIUM_NAME
    return config.CONFIDENCE_BAND_LOW_NAME


_DASH_TRANSLATION = str.maketrans(
    {
        "–": "-",  # en dash, as used in the KB's "Sept 1-15, 2025"
        "—": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)


def _normalize_for_match(text: str) -> str:
    """Normalize text for the evidence substring check.

    Deliberately narrow: NFKC, casefold, unify the KB's typographic dashes
    and quotes, and collapse whitespace. Nothing here removes or reorders
    words, so the check stays a genuine verbatim-span test -- normalizing any
    harder (stripping punctuation, stemming) would start letting paraphrased
    "evidence" pass, which is the exact hole this check exists to close.
    """
    normalized = unicodedata.normalize("NFKC", text).translate(_DASH_TRANSLATION)
    return " ".join(normalized.casefold().split())


def _evidence_supports(evidence: str, content: str) -> bool:
    """Return whether `evidence` is a real verbatim span of `content`.

    This is what makes "supported" mean something. Exact string matching
    between a claim and the source is too strict -- "Rahul Sharma leads AIML"
    never literally appears in "AIML (Lead: Rahul Sharma)" -- so the verifier
    is allowed to judge entailment semantically. But a purely semantic
    verdict is only as trustworthy as the verifier, and a verifier can
    hallucinate support as readily as a generator can hallucinate a fact.

    Requiring a citation and then mechanically re-checking it splits the
    difference: the model does the semantic work, and this function confirms
    the text it pointed at actually exists. A verifier that wants to
    rubber-stamp a fabricated claim has to quote a span that is genuinely in
    the source -- which a human reading the eval report can check -- so the
    failure mode becomes visible rather than silent.
    """
    span = _normalize_for_match(evidence).strip("\"'")
    if not span:
        return False
    return span in _normalize_for_match(content)


def validate_claims(
    verdicts: list[llm_client.ClaimVerdict], content: str
) -> tuple[VerifiedClaim, ...]:
    """Re-check every reported verdict against `content` and return the
    claims that actually stand.

    A claim survives as supported only if the verifier said SUPPORTED *and*
    its evidence span is verbatim in the source. Both conditions, never
    either alone.
    """
    validated = []
    for verdict in verdicts:
        claimed_support = verdict.verdict == llm_client.SUPPORTED_VERDICT
        evidence_found = _evidence_supports(verdict.evidence, content)
        if claimed_support and not evidence_found:
            logger.warning(
                "confidence: downgrading SUPPORTED to UNSUPPORTED -- evidence "
                "%r is not a verbatim span of the source for claim %r",
                verdict.evidence,
                verdict.claim,
            )
        validated.append(
            VerifiedClaim(
                claim=verdict.claim,
                supported=claimed_support and evidence_found,
                evidence=verdict.evidence,
                evidence_found=evidence_found,
            )
        )
    return tuple(validated)


def verification_enabled() -> bool:
    """Whether grounding verification should run.

    Read from the environment at call time (not import time) so a
    quota-constrained demo can disable the extra per-turn LLM call without
    editing code or reloading the module -- same convention as LLM_PROVIDER.
    """
    raw = os.environ.get(config.VERIFY_GROUNDING_ENV_VAR, "").strip().lower()
    if not raw:
        return config.CONFIDENCE_VERIFICATION_ENABLED
    return raw not in {"0", "false", "no", "off"}


def not_applicable(reason: str, retrieval_score: float | None = None) -> ConfidenceResult:
    """Build the result for a response that makes no factual claim.

    Refusals, LLM-error/quota messages, and generated answers that assert
    nothing all land here. None of them can be scored, and both obvious
    defaults are wrong: `0.0` would badge a correct refusal -- the most
    trustworthy thing this bot does -- as untrustworthy, and would stack the
    low band with correct behavior in the eval; `1.0` would assert a
    verification that never ran.

    Confidence answers "how much should you trust this claim about the
    club". A response that makes no such claim doesn't have an answer, so it
    reports none, and the `reason` records which of these states it was so
    the Slice 5 dashboard can bucket them separately.
    """
    return ConfidenceResult(
        score=None,
        band=config.CONFIDENCE_BAND_NOT_APPLICABLE_NAME,
        reason=reason,
        retrieval_score=retrieval_score,
    )


def score_generated_answer(
    answer: str,
    section: str,
    content: str,
    candidates: list[RetrievalResult],
) -> ConfidenceResult:
    """Score a real, LLM-generated answer. Never raises.

    Costs one LLM call (the verification pass) when verification is enabled,
    and zero when it isn't -- see README.md "Quota cost".

    Three outcomes other than a normal score, each deliberately distinct:

    - **verification disabled** -- reports the retrieval signal alone,
      labeled so a degraded signal is never presented as a full composite.
    - **no claims** -- the answer asserts nothing (e.g. the model correctly
      said the context doesn't cover it); `not_applicable`, which also
      sidesteps a 0/0 grounding score.
    - **verification failed** -- every attempt errored or came back
      unparseable. This scores `0.0`, banded `low`, *not* `not_applicable`.
      The asymmetry is the point: `not_applicable` is for answers with no
      claims to check, `low` is for answers with claims we *failed* to
      check. An answer asserting facts we could not verify is precisely what
      this system exists to be suspicious of.
    """
    retrieval = retrieval_confidence(candidates)

    if not verification_enabled():
        return ConfidenceResult(
            score=retrieval,
            band=band_for(retrieval),
            reason=config.CONFIDENCE_REASON_VERIFICATION_DISABLED,
            retrieval_score=retrieval,
        )

    try:
        verdicts = llm_client.verify_grounding(answer, section, content)
    except Exception:
        # llm_client.verify_grounding already handles provider failures and
        # returns None; this guards anything unforeseen so a confidence
        # score can never take down an otherwise successful turn.
        logger.exception("confidence: verification raised for section=%r", section)
        verdicts = None

    if verdicts is None:
        return ConfidenceResult(
            score=0.0,
            band=config.CONFIDENCE_BAND_LOW_NAME,
            reason=config.CONFIDENCE_REASON_VERIFICATION_FAILED,
            retrieval_score=retrieval,
        )

    claims = validate_claims(verdicts, content)
    if not claims:
        return not_applicable(config.CONFIDENCE_REASON_NO_CLAIMS, retrieval_score=retrieval)

    grounding = sum(1 for claim in claims if claim.supported) / len(claims)
    score = min(retrieval, grounding)
    return ConfidenceResult(
        score=score,
        band=band_for(score),
        reason=config.CONFIDENCE_REASON_VERIFIED,
        retrieval_score=retrieval,
        grounding_score=grounding,
        claims=claims,
    )
