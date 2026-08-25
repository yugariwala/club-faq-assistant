"""Unit tests for `backend.llm_client.generate_answer`.

Mocks each provider's client so no network access or API key is required.
Covers provider dispatch (LLM_PROVIDER env var, default), request
construction per provider (model/system/message shape, context isolation),
response-parsing parity between providers, and error handling on API
failures.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend import llm_client
from backend.config import ANTHROPIC_MODEL, GEMINI_MODEL
from backend.memory import Turn


def _anthropic_block(block_type: str, text: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(type=block_type, text=text)


def _anthropic_response(blocks: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(content=blocks)


def _mock_anthropic_client(response: SimpleNamespace) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = response
    return client


def _mock_gemini_client(text: str) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text=text)
    return client


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


def test_defaults_to_gemini_when_llm_provider_unset(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = _mock_gemini_client("stub answer")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.generate_answer("q", "Teams", "content")

    client.models.generate_content.assert_called_once()
    assert result == "stub answer"


def test_selects_anthropic_when_llm_provider_set(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = _mock_anthropic_client(_anthropic_response([_anthropic_block("text", "stub")]))

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        result = llm_client.generate_answer("q", "Teams", "content")

    client.messages.create.assert_called_once()
    assert result == "stub"


def test_selects_gemini_when_llm_provider_set(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("stub")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.generate_answer("q", "Teams", "content")

    client.models.generate_content.assert_called_once()
    assert result == "stub"


def test_provider_env_var_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Anthropic")
    client = _mock_anthropic_client(_anthropic_response([_anthropic_block("text", "stub")]))

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        llm_client.generate_answer("q", "Teams", "content")

    client.messages.create.assert_called_once()


def test_unknown_provider_raises_llm_provider_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(llm_client.LLMProviderError):
        llm_client.generate_answer("q", "Teams", "content")


# ---------------------------------------------------------------------------
# Anthropic request construction
# ---------------------------------------------------------------------------


def test_anthropic_calls_api_with_expected_model_and_system(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = _mock_anthropic_client(_anthropic_response([_anthropic_block("text", "stub answer")]))

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        llm_client.generate_answer("Who leads AIML?", "Teams", "AIML (Lead: Rahul Sharma)")

    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == ANTHROPIC_MODEL
    assert kwargs["system"] == llm_client.SYSTEM_PROMPT


def test_anthropic_message_contains_only_passed_section_content_and_question(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = _mock_anthropic_client(_anthropic_response([_anthropic_block("text", "stub answer")]))

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        llm_client.generate_answer(
            "Who leads AIML?", "Teams", "AIML (Lead: Rahul Sharma), Web Dev (Lead: Priya Patel)"
        )

    _, kwargs = client.messages.create.call_args
    messages = kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    content = messages[0]["content"]
    assert "Who leads AIML?" in content
    assert "AIML (Lead: Rahul Sharma), Web Dev (Lead: Priya Patel)" in content
    assert content == (
        "Context (Teams section):\n"
        "AIML (Lead: Rahul Sharma), Web Dev (Lead: Priya Patel)\n\n"
        "Question: Who leads AIML?"
    )


def test_anthropic_extracts_single_text_block(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = _mock_anthropic_client(
        _anthropic_response([_anthropic_block("text", "Rahul Sharma leads AIML.")])
    )

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        result = llm_client.generate_answer("Who leads AIML?", "Teams", "content")

    assert result == "Rahul Sharma leads AIML."


def test_anthropic_joins_multiple_text_blocks(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = _mock_anthropic_client(
        _anthropic_response([_anthropic_block("text", "Hello "), _anthropic_block("text", "World")])
    )

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        result = llm_client.generate_answer("q", "Teams", "content")

    assert result == "Hello World"


def test_anthropic_returns_empty_string_when_no_text_block(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = _mock_anthropic_client(_anthropic_response([_anthropic_block("tool_use")]))

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        result = llm_client.generate_answer("q", "Teams", "content")

    assert result == ""


def test_anthropic_skips_non_text_blocks_when_joining(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = _mock_anthropic_client(
        _anthropic_response(
            [_anthropic_block("thinking", "internal reasoning"), _anthropic_block("text", "final answer")]
        )
    )

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        result = llm_client.generate_answer("q", "Teams", "content")

    assert result == "final answer"


def test_anthropic_api_error_raises_llm_provider_error(monkeypatch):
    import anthropic

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = MagicMock()
    client.messages.create.side_effect = anthropic.RateLimitError(
        "rate limited", response=MagicMock(), body=None
    )

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        with pytest.raises(llm_client.LLMProviderError):
            llm_client.generate_answer("q", "Teams", "content")


# ---------------------------------------------------------------------------
# Gemini request construction
# ---------------------------------------------------------------------------


def test_gemini_calls_api_with_expected_model_and_system(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("stub answer")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        llm_client.generate_answer("Who leads AIML?", "Teams", "AIML (Lead: Rahul Sharma)")

    _, kwargs = client.models.generate_content.call_args
    assert kwargs["model"] == GEMINI_MODEL
    assert kwargs["config"].system_instruction == llm_client.SYSTEM_PROMPT
    assert 0 <= kwargs["config"].temperature <= 0.2


def test_gemini_contents_contains_only_passed_section_content_and_question(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("stub answer")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        llm_client.generate_answer(
            "Who leads AIML?", "Teams", "AIML (Lead: Rahul Sharma), Web Dev (Lead: Priya Patel)"
        )

    _, kwargs = client.models.generate_content.call_args
    assert kwargs["contents"] == (
        "Context (Teams section):\n"
        "AIML (Lead: Rahul Sharma), Web Dev (Lead: Priya Patel)\n\n"
        "Question: Who leads AIML?"
    )


def test_gemini_returns_response_text(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("Rahul Sharma leads AIML.")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.generate_answer("Who leads AIML?", "Teams", "content")

    assert result == "Rahul Sharma leads AIML."


def test_gemini_returns_empty_string_when_response_text_is_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text=None)

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.generate_answer("q", "Teams", "content")

    assert result == ""


def test_gemini_api_error_raises_llm_provider_error(monkeypatch):
    from google.genai import errors

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = MagicMock()
    client.models.generate_content.side_effect = errors.ClientError(
        429, {"error": {"message": "rate limited"}}
    )

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        with pytest.raises(llm_client.LLMProviderError):
            llm_client.generate_answer("q", "Teams", "content")


# ---------------------------------------------------------------------------
# Quota/rate-limit error classification (distinct from generic failures, so
# qa.py can surface a distinct user-facing message for each)
# ---------------------------------------------------------------------------


def test_gemini_429_raises_llm_quota_error(monkeypatch):
    from google.genai import errors

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = MagicMock()
    client.models.generate_content.side_effect = errors.ClientError(
        429, {"error": {"message": "RESOURCE_EXHAUSTED"}}
    )

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        with pytest.raises(llm_client.LLMQuotaError):
            llm_client.generate_answer("q", "Teams", "content")


def test_gemini_non_429_error_raises_plain_llm_provider_error_not_quota(monkeypatch):
    from google.genai import errors

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = MagicMock()
    client.models.generate_content.side_effect = errors.ClientError(
        400, {"error": {"message": "bad request"}}
    )

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        with pytest.raises(llm_client.LLMProviderError) as exc_info:
            llm_client.generate_answer("q", "Teams", "content")

    assert not isinstance(exc_info.value, llm_client.LLMQuotaError)


def test_anthropic_rate_limit_raises_llm_quota_error(monkeypatch):
    import anthropic

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = MagicMock()
    client.messages.create.side_effect = anthropic.RateLimitError(
        "rate limited", response=MagicMock(), body=None
    )

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        with pytest.raises(llm_client.LLMQuotaError):
            llm_client.generate_answer("q", "Teams", "content")


def test_anthropic_non_rate_limit_error_raises_plain_llm_provider_error_not_quota(monkeypatch):
    import anthropic

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = MagicMock()
    client.messages.create.side_effect = anthropic.BadRequestError(
        "bad request", response=MagicMock(), body=None
    )

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        with pytest.raises(llm_client.LLMProviderError) as exc_info:
            llm_client.generate_answer("q", "Teams", "content")

    assert not isinstance(exc_info.value, llm_client.LLMQuotaError)


# ---------------------------------------------------------------------------
# missing_api_key_var (startup key-presence check)
# ---------------------------------------------------------------------------


def test_missing_api_key_var_returns_gemini_var_when_unset(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert llm_client.missing_api_key_var() == "GEMINI_API_KEY"


def test_missing_api_key_var_returns_none_when_gemini_key_set(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "some-key")

    assert llm_client.missing_api_key_var() is None


def test_missing_api_key_var_treats_blank_key_as_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "   ")

    assert llm_client.missing_api_key_var() == "GEMINI_API_KEY"


def test_missing_api_key_var_returns_anthropic_var_when_selected_and_unset(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert llm_client.missing_api_key_var() == "ANTHROPIC_API_KEY"


def test_missing_api_key_var_defaults_to_gemini_when_provider_unset(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert llm_client.missing_api_key_var() == "GEMINI_API_KEY"


def test_missing_api_key_var_returns_none_for_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    assert llm_client.missing_api_key_var() is None


# ---------------------------------------------------------------------------
# Cross-provider parity
# ---------------------------------------------------------------------------


def test_both_providers_return_identical_shape_for_identical_answer_text(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    anthropic_client = _mock_anthropic_client(
        _anthropic_response([_anthropic_block("text", "Rahul Sharma leads AIML.")])
    )
    with patch("backend.llm_client._get_anthropic_client", return_value=anthropic_client):
        anthropic_result = llm_client.generate_answer("Who leads AIML?", "Teams", "content")

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    gemini_client = _mock_gemini_client("Rahul Sharma leads AIML.")
    with patch("backend.llm_client._get_gemini_client", return_value=gemini_client):
        gemini_result = llm_client.generate_answer("Who leads AIML?", "Teams", "content")

    assert type(anthropic_result) is type(gemini_result) is str
    assert anthropic_result == gemini_result


# ---------------------------------------------------------------------------
# rewrite_query
# ---------------------------------------------------------------------------


def test_rewrite_query_short_circuits_on_empty_history(monkeypatch):
    """Nothing to resolve against -> no provider call at all (spec Code Map:
    "empty history short-circuits (no call)")."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("should never be used")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.rewrite_query("When is that?", [])

    client.models.generate_content.assert_not_called()
    assert result == "When is that?"


def test_rewrite_query_builds_prompt_from_history_and_uses_rewrite_system_prompt(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("When is the Cloud Study Jam?")
    history = [
        Turn(
            user_message="Tell me about the Cloud Study Jam",
            answer="The Cloud Study Jam is on Sept 20.",
            source_section="Events",
        )
    ]

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.rewrite_query("When is that?", history)

    _, kwargs = client.models.generate_content.call_args
    assert kwargs["config"].system_instruction == llm_client.REWRITE_SYSTEM_PROMPT
    assert kwargs["config"].system_instruction != llm_client.SYSTEM_PROMPT
    assert "Tell me about the Cloud Study Jam" in kwargs["contents"]
    assert "The Cloud Study Jam is on Sept 20." in kwargs["contents"]
    assert "When is that?" in kwargs["contents"]
    assert result == "When is the Cloud Study Jam?"


def test_rewrite_query_includes_multiple_history_turns_in_order(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("stub")
    history = [
        Turn(user_message="Tell me about Cloud team", answer="Sneha Gupta leads Cloud.", source_section="Teams"),
        Turn(user_message="What about AIML?", answer="Rahul Sharma leads AIML.", source_section="Teams"),
    ]

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        llm_client.rewrite_query("Who leads it?", history)

    _, kwargs = client.models.generate_content.call_args
    contents = kwargs["contents"]
    assert contents.index("Tell me about Cloud team") < contents.index("What about AIML?")
    assert contents.index("What about AIML?") < contents.index("Who leads it?")


def test_rewrite_query_uses_anthropic_when_selected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = _mock_anthropic_client(
        _anthropic_response([_anthropic_block("text", "When is the Cloud Study Jam?")])
    )
    history = [Turn(user_message="q", answer="a", source_section="Events")]

    with patch("backend.llm_client._get_anthropic_client", return_value=client):
        result = llm_client.rewrite_query("When is that?", history)

    _, kwargs = client.messages.create.call_args
    assert kwargs["system"] == llm_client.REWRITE_SYSTEM_PROMPT
    assert result == "When is the Cloud Study Jam?"


def test_rewrite_query_strips_whitespace_from_result(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("  When is the Cloud Study Jam?  \n")
    history = [Turn(user_message="q", answer="a", source_section="Events")]

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.rewrite_query("When is that?", history)

    assert result == "When is the Cloud Study Jam?"


def test_rewrite_query_falls_back_to_original_on_provider_failure(monkeypatch):
    """Provider failure falls back to the original query, never raises
    (spec Code Map: "provider failure falls back to the original query
    unchanged")."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("boom")
    history = [Turn(user_message="q", answer="a", source_section="Events")]

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.rewrite_query("When is that?", history)

    assert result == "When is that?"


def test_rewrite_query_falls_back_to_original_when_result_is_blank(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("   ")
    history = [Turn(user_message="q", answer="a", source_section="Events")]

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.rewrite_query("When is that?", history)

    assert result == "When is that?"


def test_rewrite_query_falls_back_to_original_when_response_text_is_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text=None)
    history = [Turn(user_message="q", answer="a", source_section="Events")]

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.rewrite_query("When is that?", history)

    assert result == "When is that?"


def test_rewrite_query_falls_back_to_original_on_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    history = [Turn(user_message="q", answer="a", source_section="Events")]

    result = llm_client.rewrite_query("When is that?", history)

    assert result == "When is that?"


# ---------------------------------------------------------------------------
# classify_intent (Slice 3)
# ---------------------------------------------------------------------------


def test_classify_intent_returns_valid_label_from_first_attempt(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("faq")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.classify_intent("What teams does the club have?")

    assert result == "faq"
    client.models.generate_content.assert_called_once()


def test_classify_intent_strips_punctuation_and_case_from_response(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client(" Event_Inquiry.\n")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.classify_intent("When is HackFest?")

    assert result == "event_inquiry"


def test_classify_intent_retries_once_on_unparseable_response_then_succeeds(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = MagicMock()
    client.models.generate_content.side_effect = [
        SimpleNamespace(text="I'm not sure, maybe faq?"),
        SimpleNamespace(text="faq"),
    ]

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.classify_intent("What teams does the club have?")

    assert result == "faq"
    assert client.models.generate_content.call_count == 2


def test_classify_intent_falls_back_to_default_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("not a real label")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.classify_intent("gibberish")

    assert result == llm_client.DEFAULT_INTENT_ON_LLM_FAILURE
    assert client.models.generate_content.call_count == llm_client.INTENT_CLASSIFY_MAX_ATTEMPTS


def test_classify_intent_falls_back_to_default_on_provider_failure(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("boom")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.classify_intent("gibberish")

    assert result == llm_client.DEFAULT_INTENT_ON_LLM_FAILURE


def test_classify_intent_falls_back_to_default_on_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    result = llm_client.classify_intent("What teams does the club have?")

    assert result == llm_client.DEFAULT_INTENT_ON_LLM_FAILURE


def test_classify_intent_never_returns_a_label_outside_the_enum(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("definitely_not_a_label")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.classify_intent("anything")

    assert result in llm_client.INTENT_LABELS


# ---------------------------------------------------------------------------
# classify_intents_batch (Slice 3 -- eval script quota efficiency)
# ---------------------------------------------------------------------------


def test_classify_intents_batch_empty_list_returns_empty_without_a_call(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    with patch("backend.llm_client._get_gemini_client") as mock_get_client:
        result = llm_client.classify_intents_batch([])

    assert result == []
    mock_get_client.assert_not_called()


def test_classify_intents_batch_parses_numbered_response_in_order(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("1. faq\n2. event_inquiry\n3. greeting")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.classify_intents_batch(
            ["What teams?", "When is HackFest?", "Hi!"]
        )

    assert result == ["faq", "event_inquiry", "greeting"]
    client.models.generate_content.assert_called_once()


def test_classify_intents_batch_tolerates_out_of_order_lines(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("2. greeting\n1. faq")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.classify_intents_batch(["What teams?", "Hi!"])

    assert result == ["faq", "greeting"]


def test_classify_intents_batch_retries_only_the_missing_positions(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = MagicMock()
    client.models.generate_content.side_effect = [
        SimpleNamespace(text="1. faq\n2. not_a_label"),
        SimpleNamespace(text="1. faq\n2. greeting"),
    ]

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.classify_intents_batch(["What teams?", "Hi!"])

    assert result == ["faq", "greeting"]
    assert client.models.generate_content.call_count == 2


def test_classify_intents_batch_falls_back_to_default_for_positions_still_unparsed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client = _mock_gemini_client("1. faq\n2. still_broken")

    with patch("backend.llm_client._get_gemini_client", return_value=client):
        result = llm_client.classify_intents_batch(["What teams?", "gibberish"])

    assert result == ["faq", llm_client.DEFAULT_INTENT_ON_LLM_FAILURE]
    assert client.models.generate_content.call_count == llm_client.INTENT_CLASSIFY_MAX_ATTEMPTS


def test_classify_intents_batch_falls_back_to_default_on_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    result = llm_client.classify_intents_batch(["a", "b"])

    assert result == [llm_client.DEFAULT_INTENT_ON_LLM_FAILURE] * 2
