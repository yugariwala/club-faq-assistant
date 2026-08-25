"""Unit tests for `backend.confidence`, with `llm_client` mocked.

No network access or API key is required. Covers the spec's I/O &
Edge-Case Matrix rows for Slice 4.
"""

from unittest.mock import patch

import pytest

from backend import config, confidence
from backend.llm_client import SUPPORTED_VERDICT, UNSUPPORTED_VERDICT, ClaimVerdict
from backend.retrieval import RetrievalResult, TfidfRetriever

TEAMS_CONTENT = (
    "AIML (Lead: Rahul Sharma), Web Dev (Lead: Priya Patel), "
    "App Dev (Lead: Arjun Mehta), Cloud (Lead: Sneha Gupta), "
    "Cybersecurity (Lead: Vikram Singh), Design (Lead: Ananya Reddy)"
)


def _candidates(*scores: float) -> list[RetrievalResult]:
    return [
        RetrievalResult(section=f"S{i}", content="c", score=score)
        for i, score in enumerate(scores)
    ]


def _supported(claim: str, evidence: str) -> ClaimVerdict:
    return ClaimVerdict(claim=claim, verdict=SUPPORTED_VERDICT, evidence=evidence)


def _unsupported(claim: str) -> ClaimVerdict:
    return ClaimVerdict(claim=claim, verdict=UNSUPPORTED_VERDICT, evidence="")


# ---------------------------------------------------------------------------
# retrieval_confidence -- the separation ratio
# ---------------------------------------------------------------------------


def test_separation_is_one_when_runner_up_scores_nothing():
    assert confidence.retrieval_confidence(_candidates(0.4, 0.0)) == pytest.approx(1.0)


def test_separation_is_one_when_there_is_no_runner_up():
    """A single candidate means nothing competed for the match."""
    assert confidence.retrieval_confidence(_candidates(0.4)) == pytest.approx(1.0)


def test_separation_is_zero_when_top_score_is_zero():
    """Nothing matched, so there is no discrimination to measure."""
    assert confidence.retrieval_confidence(_candidates(0.0, 0.0)) == 0.0


def test_separation_is_zero_for_empty_candidates():
    assert confidence.retrieval_confidence([]) == 0.0


def test_separation_collapses_when_runner_up_is_nearly_as_good():
    """The ambiguous case the signal exists to catch."""
    assert confidence.retrieval_confidence(_candidates(0.30, 0.29)) < 0.05


def test_separation_is_bounded_to_unit_interval():
    for candidates in (_candidates(0.7, 0.08), _candidates(0.15, 0.149), _candidates(1.0, 0.0)):
        score = confidence.retrieval_confidence(candidates)
        assert 0.0 <= score <= 1.0


def test_ambiguous_real_query_scores_lower_than_specific_one():
    """Regression guard on the measurement the design rests on: a bare
    ambiguous term must not outscore a specific lookup, even though the two
    have comparable raw cosine magnitudes."""
    retriever = TfidfRetriever()
    ambiguous = confidence.retrieval_confidence(retriever.retrieve("cloud", top_k=3))
    specific = confidence.retrieval_confidence(
        retriever.retrieve("Who is the Cloud team lead?", top_k=3)
    )

    assert ambiguous < config.CONFIDENCE_BAND_MEDIUM
    assert specific >= config.CONFIDENCE_BAND_HIGH


# ---------------------------------------------------------------------------
# band_for -- thresholds come from config, never inline
# ---------------------------------------------------------------------------


def test_band_boundaries_are_inclusive_at_the_configured_thresholds():
    assert confidence.band_for(config.CONFIDENCE_BAND_HIGH) == config.CONFIDENCE_BAND_HIGH_NAME
    assert (
        confidence.band_for(config.CONFIDENCE_BAND_MEDIUM)
        == config.CONFIDENCE_BAND_MEDIUM_NAME
    )
    assert (
        confidence.band_for(config.CONFIDENCE_BAND_MEDIUM - 0.001)
        == config.CONFIDENCE_BAND_LOW_NAME
    )


def test_band_follows_config_when_thresholds_are_retuned(monkeypatch):
    """Thresholds are read at call time, so retuning config changes banding
    everywhere with no change at any call site."""
    monkeypatch.setattr(config, "CONFIDENCE_BAND_HIGH", 0.99)
    assert confidence.band_for(0.9) == config.CONFIDENCE_BAND_MEDIUM_NAME


def test_single_unsupported_claim_cannot_reach_high_in_a_short_answer():
    """The reason CONFIDENCE_BAND_HIGH is 0.85: an answer of six or fewer
    claims with one unsupported claim must never badge `high`."""
    for total in range(2, 7):
        grounding = (total - 1) / total
        assert confidence.band_for(grounding) != config.CONFIDENCE_BAND_HIGH_NAME


# ---------------------------------------------------------------------------
# Evidence-span validation -- what "supported" actually means
# ---------------------------------------------------------------------------


def test_verbatim_evidence_is_accepted():
    assert confidence._evidence_supports("AIML (Lead: Rahul Sharma)", TEAMS_CONTENT)


def test_evidence_match_ignores_case_and_whitespace_only():
    assert confidence._evidence_supports("  aiml   (lead:  rahul sharma) ", TEAMS_CONTENT)


def test_evidence_match_normalizes_typographic_dashes():
    """The KB uses en dashes ("Sept 1-15, 2025"); a verifier that emits an
    ASCII hyphen is quoting the same span, not inventing one."""
    assert confidence._evidence_supports("Sept 1-15, 2025", "Window: Sept 1–15, 2025.")


def test_paraphrased_evidence_is_rejected():
    """The verifier restating the fact in its own words is not a citation."""
    assert not confidence._evidence_supports("Rahul Sharma is the AIML lead", TEAMS_CONTENT)


def test_invented_evidence_is_rejected():
    assert not confidence._evidence_supports("AIML (Lead: Rohan Desai)", TEAMS_CONTENT)


def test_empty_evidence_is_rejected():
    assert not confidence._evidence_supports("", TEAMS_CONTENT)


def test_supported_verdict_with_missing_evidence_is_downgraded():
    """The core anti-rubber-stamp guarantee (spec Acceptance Criteria)."""
    verdicts = [ClaimVerdict("AIML has 25 members", SUPPORTED_VERDICT, "AIML has 25 members")]

    claims = confidence.validate_claims(verdicts, TEAMS_CONTENT)

    assert claims[0].supported is False
    assert claims[0].evidence_found is False


def test_supported_verdict_with_real_evidence_survives():
    verdicts = [_supported("AIML is led by Rahul Sharma", "AIML (Lead: Rahul Sharma)")]

    claims = confidence.validate_claims(verdicts, TEAMS_CONTENT)

    assert claims[0].supported is True
    assert claims[0].evidence_found is True


def test_unsupported_verdict_stays_unsupported_even_with_real_evidence():
    """Both conditions are required; a real span never overrides the
    verifier's own UNSUPPORTED judgment."""
    verdicts = [ClaimVerdict("AIML has 25 members", UNSUPPORTED_VERDICT, "AIML (Lead: Rahul Sharma)")]

    claims = confidence.validate_claims(verdicts, TEAMS_CONTENT)

    assert claims[0].supported is False
    assert claims[0].evidence_found is True


# ---------------------------------------------------------------------------
# score_generated_answer -- composition and the non-scored states
# ---------------------------------------------------------------------------


def test_composite_is_the_minimum_of_the_two_signals():
    """A strong retrieval must never mask a weakly grounded answer."""
    verdicts = [
        _supported("a", "AIML (Lead: Rahul Sharma)"),
        _unsupported("b"),
    ]
    with patch("backend.confidence.llm_client.verify_grounding", return_value=verdicts):
        result = confidence.score_generated_answer(
            "answer", "Teams", TEAMS_CONTENT, _candidates(0.9, 0.0)
        )

    assert result.retrieval_score == pytest.approx(1.0)
    assert result.grounding_score == pytest.approx(0.5)
    assert result.score == pytest.approx(0.5)
    assert result.reason == config.CONFIDENCE_REASON_VERIFIED


def test_weak_retrieval_caps_a_perfectly_grounded_answer():
    """The other direction: perfect grounding can't rescue an ambiguous match."""
    verdicts = [_supported("a", "AIML (Lead: Rahul Sharma)")]
    with patch("backend.confidence.llm_client.verify_grounding", return_value=verdicts):
        result = confidence.score_generated_answer(
            "answer", "Teams", TEAMS_CONTENT, _candidates(0.30, 0.29)
        )

    assert result.grounding_score == pytest.approx(1.0)
    assert result.score == result.retrieval_score
    assert result.band == config.CONFIDENCE_BAND_LOW_NAME


def test_no_claims_is_not_applicable_rather_than_zero():
    """An answer that asserts nothing has no confidence, not zero confidence
    -- and this is also what keeps grounding from dividing by zero."""
    with patch("backend.confidence.llm_client.verify_grounding", return_value=[]):
        result = confidence.score_generated_answer(
            "That isn't in the KB.", "Teams", TEAMS_CONTENT, _candidates(0.9, 0.0)
        )

    assert result.score is None
    assert result.band == config.CONFIDENCE_BAND_NOT_APPLICABLE_NAME
    assert result.reason == config.CONFIDENCE_REASON_NO_CLAIMS


def test_verification_failure_scores_low_not_not_applicable():
    """Claims we failed to check are suspicious; claims that don't exist are
    not. The asymmetry is deliberate (spec Design Notes)."""
    with patch("backend.confidence.llm_client.verify_grounding", return_value=None):
        result = confidence.score_generated_answer(
            "answer", "Teams", TEAMS_CONTENT, _candidates(0.9, 0.0)
        )

    assert result.score == 0.0
    assert result.band == config.CONFIDENCE_BAND_LOW_NAME
    assert result.reason == config.CONFIDENCE_REASON_VERIFICATION_FAILED


def test_unexpected_verifier_exception_does_not_propagate():
    with patch(
        "backend.confidence.llm_client.verify_grounding", side_effect=RuntimeError("boom")
    ):
        result = confidence.score_generated_answer(
            "answer", "Teams", TEAMS_CONTENT, _candidates(0.9, 0.0)
        )

    assert result.reason == config.CONFIDENCE_REASON_VERIFICATION_FAILED


def test_disabled_verification_makes_no_llm_call_and_reports_retrieval_only(monkeypatch):
    monkeypatch.setenv(config.VERIFY_GROUNDING_ENV_VAR, "0")

    with patch("backend.confidence.llm_client.verify_grounding") as mock_verify:
        result = confidence.score_generated_answer(
            "answer", "Teams", TEAMS_CONTENT, _candidates(0.9, 0.0)
        )

    mock_verify.assert_not_called()
    assert result.reason == config.CONFIDENCE_REASON_VERIFICATION_DISABLED
    assert result.score == result.retrieval_score
    assert result.grounding_score is None


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "Off"])
def test_verification_toggle_accepts_common_falsey_spellings(monkeypatch, value):
    monkeypatch.setenv(config.VERIFY_GROUNDING_ENV_VAR, value)
    assert confidence.verification_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", ""])
def test_verification_defaults_to_enabled(monkeypatch, value):
    monkeypatch.setenv(config.VERIFY_GROUNDING_ENV_VAR, value)
    assert confidence.verification_enabled() is True


# ---------------------------------------------------------------------------
# not_applicable / display
# ---------------------------------------------------------------------------


def test_not_applicable_carries_its_reason_and_no_score():
    result = confidence.not_applicable(config.CONFIDENCE_REASON_REFUSED)

    assert result.score is None
    assert result.band == config.CONFIDENCE_BAND_NOT_APPLICABLE_NAME
    assert result.reason == config.CONFIDENCE_REASON_REFUSED
    assert result.claims == ()


def test_display_shows_band_and_raw_score_together():
    verdicts = [_supported("a", "AIML (Lead: Rahul Sharma)")]
    with patch("backend.confidence.llm_client.verify_grounding", return_value=verdicts):
        result = confidence.score_generated_answer(
            "answer", "Teams", TEAMS_CONTENT, _candidates(0.9, 0.0)
        )

    rendered = result.display()
    assert result.band in rendered
    assert "1.00" in rendered


def test_display_of_a_not_applicable_result_omits_a_score():
    rendered = confidence.not_applicable(config.CONFIDENCE_REASON_REFUSED).display()

    assert config.CONFIDENCE_BAND_NOT_APPLICABLE_NAME in rendered
    assert config.CONFIDENCE_REASON_REFUSED in rendered
