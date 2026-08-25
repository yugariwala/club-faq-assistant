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
