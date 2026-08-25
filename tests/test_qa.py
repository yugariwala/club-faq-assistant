"""Unit tests for `backend.qa.answer_question`, with `llm_client` mocked.

No network access or ANTHROPIC_API_KEY is required to run these tests.
"""

from unittest.mock import patch

from backend import config
from backend.qa import AnswerResult, answer_question


def test_refusal_short_circuits_llm_call():
    """Below-threshold query never reaches the LLM (spec Acceptance Criteria)."""
    with patch("backend.qa.llm_client.generate_answer") as mock_generate:
        result = answer_question("What's the club's budget?")

    mock_generate.assert_not_called()
    assert isinstance(result, AnswerResult)
    assert result.refused is True
    assert result.source_section is None
    assert result.score < config.RETRIEVAL_THRESHOLD
    assert result.answer == config.REFUSAL_MESSAGE


def test_empty_query_refuses_without_crashing():
    with patch("backend.qa.llm_client.generate_answer") as mock_generate:
        result = answer_question("")

    mock_generate.assert_not_called()
    assert result.refused is True
    assert result.source_section is None


def test_grounded_path_returns_correct_shape_and_calls_llm_once():
    """A question clearly answered by the Teams section grounds through the LLM."""
    with patch(
        "backend.qa.llm_client.generate_answer", return_value="Rahul Sharma leads AIML."
    ) as mock_generate:
        result = answer_question("Who leads the AIML team?")

    mock_generate.assert_called_once()
    call_args = mock_generate.call_args.args
    assert call_args[0] == "Who leads the AIML team?"
    assert call_args[1] == "Teams"
    assert "Rahul Sharma" in call_args[2]  # retrieved content passed through verbatim

    assert isinstance(result, AnswerResult)
    assert result.refused is False
    assert result.source_section == "Teams"
    assert result.score >= config.RETRIEVAL_THRESHOLD
    assert result.answer == "Rahul Sharma leads AIML."


def test_aggregate_lookup_grounds_on_teams_section():
    with patch("backend.qa.llm_client.generate_answer", return_value="stub") as mock_generate:
        result = answer_question("List all the teams")

    mock_generate.assert_called_once()
    assert result.refused is False
    assert result.source_section == "Teams"


def test_threshold_change_flips_refusal_without_touching_qa_logic():
    """Editing RETRIEVAL_THRESHOLD in config.py changes refusal behavior
    (spec Acceptance Criteria) -- verified here by patching the config
    value qa.py reads at call time, with no change to qa.py itself."""
    query = "Who leads the AIML team?"

    with patch("backend.qa.llm_client.generate_answer", return_value="stub") as mock_generate:
        with patch.object(config, "RETRIEVAL_THRESHOLD", 0.99):
            result = answer_question(query)

    mock_generate.assert_not_called()
    assert result.refused is True

    with patch("backend.qa.llm_client.generate_answer", return_value="stub") as mock_generate:
        with patch.object(config, "RETRIEVAL_THRESHOLD", 0.0):
            result = answer_question("gibberish zxcv asdkjqwe")

    mock_generate.assert_called_once()
    assert result.refused is False


def test_score_is_raw_float_not_rounded():
    with patch("backend.qa.llm_client.generate_answer", return_value="stub"):
        result = answer_question("Who leads the AIML team?")

    assert isinstance(result.score, float)


def test_llm_call_failure_degrades_gracefully_without_crashing():
    """A retrieval hit (in-scope query) whose LLM call raises must not
    propagate the exception, and must NOT be reported as refused=True --
    that flag is reserved for the below-threshold "not in the KB" path,
    not an LLM-call failure on an in-scope query."""
    with patch(
        "backend.qa.llm_client.generate_answer", side_effect=RuntimeError("boom")
    ) as mock_generate:
        result = answer_question("Who leads the AIML team?")

    mock_generate.assert_called_once()
    assert isinstance(result, AnswerResult)
    assert result.refused is False
    assert result.source_section == "Teams"
    assert result.score >= config.RETRIEVAL_THRESHOLD
    assert result.answer == config.LLM_ERROR_MESSAGE
    assert result.answer != config.REFUSAL_MESSAGE
