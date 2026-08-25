"""Shared test fixtures.

The suite is meant to run with no API key and no network access. Until now
that held by convention -- each test remembered to patch whichever
`llm_client` entry point it exercised. That convention is easy to breach
silently: `llm_client.classify_intent` and `llm_client.verify_grounding`
both swallow provider failures by design, so a test that forgets to patch
one still *passes* while quietly opening a real connection and burning
quota.

This fixture makes the property structural instead. It blocks the two lazy
client constructors -- the single boundary every provider call crosses on
its way to the network -- so an unpatched call fails locally and loudly.
Tests that need a specific response still patch what they always patched
(`_get_gemini_client`, `_get_anthropic_client`, or a higher-level entry
point), and their patch wins inside its own scope; tests that don't simply
exercise the already-covered provider-failure path.
"""

import pytest

from backend import llm_client


class UnmockedProviderCall(AssertionError):
    """Raised when a test reaches a real provider client. Always a test bug."""


@pytest.fixture(autouse=True)
def block_provider_network_calls(monkeypatch):
    def _blocked(*_args, **_kwargs):
        raise UnmockedProviderCall(
            "A test tried to construct a real LLM provider client. Patch the "
            "llm_client entry point this test exercises (generate_answer / "
            "rewrite_query / classify_intent / verify_grounding), or patch "
            "_get_gemini_client / _get_anthropic_client directly."
        )

    monkeypatch.setattr(llm_client, "_get_gemini_client", _blocked)
    monkeypatch.setattr(llm_client, "_get_anthropic_client", _blocked)
