"""Live-provider tests for `llm_client.rewrite_query` quality.

`tests/test_llm_client.py` mocks every provider call, so passing tests
there prove the orchestration (prompt construction, fallback behavior) but
give zero signal on whether the real model actually resolves references
well. This suite calls the real, configured provider -- no mocking -- and
checks that the *rewritten* query actually retrieves the right KB section,
rather than asserting exact wording (LLM output isn't stable enough for
that).

Opt-in and skipped by default so it never burns API quota on a normal
`uv run pytest` run: set RUN_LIVE_LLM_TESTS=1 and a valid API key for the
configured LLM_PROVIDER (see .env.example) to run it.
"""

import os

import pytest

from backend import llm_client
from backend.config import RETRIEVAL_THRESHOLD
from backend.memory import Turn
from backend.retrieval import TfidfRetriever

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TESTS", "").strip() != "1",
    reason=(
        "live rewrite-quality tests are opt-in (real API calls, burns quota) -- "
        "set RUN_LIVE_LLM_TESTS=1 and a valid provider API key to run"
    ),
)

_retriever = TfidfRetriever()


def _grounds_in_section(query: str, expected_section: str) -> bool:
    """True if `query` alone retrieves `expected_section` above the same
    threshold `qa.answer_question` uses to decide grounded vs. refused."""
    top = _retriever.retrieve(query, top_k=1)[0]
    return top.section == expected_section and top.score >= RETRIEVAL_THRESHOLD


def test_live_pronoun_coreference_resolves_to_teams():
    history = [
        Turn(
            user_message="Tell me about the AIML team",
            answer="AIML is led by Rahul Sharma.",
            source_section="Teams",
        )
    ]

    rewritten = llm_client.rewrite_query("Who leads it?", history)

    assert _grounds_in_section(rewritten, "Teams"), (
        f"rewritten query {rewritten!r} did not ground in Teams "
        f"(score={_retriever.retrieve(rewritten, top_k=1)[0].score:.3f})"
    )


def test_live_ellipsis_topic_switch_resolves_to_teams():
    """The scenario a health-check run found broken: an elliptical topic
    switch that names a new entity without a pronoun ("what about Cloud?")
    following an unrelated team's leadership question."""
    history = [
        Turn(
            user_message="Tell me about the AIML team",
            answer="AIML is led by Rahul Sharma.",
            source_section="Teams",
        )
    ]

    rewritten = llm_client.rewrite_query("What about Cloud?", history)

    assert _grounds_in_section(rewritten, "Teams"), (
        f"rewritten query {rewritten!r} did not ground in Teams "
        f"(score={_retriever.retrieve(rewritten, top_k=1)[0].score:.3f})"
    )


def test_live_chained_ellipsis_then_pronoun_resolves_to_teams():
    """A pronoun follow-up ("that one") after an already-resolved topic
    switch must resolve to the most recently discussed entity (Cloud), not
    the original one (AIML)."""
    history = [
        Turn(
            user_message="Tell me about the AIML team",
            answer="AIML is led by Rahul Sharma.",
            source_section="Teams",
        ),
        Turn(
            user_message="What about Cloud?",
            answer="Sneha Gupta leads the Cloud team.",
            source_section="Teams",
        ),
    ]

    rewritten = llm_client.rewrite_query("And who leads that one?", history)

    assert _grounds_in_section(rewritten, "Teams"), (
        f"rewritten query {rewritten!r} did not ground in Teams "
        f"(score={_retriever.retrieve(rewritten, top_k=1)[0].score:.3f})"
    )


def test_live_unresolvable_reference_stays_below_threshold():
    """A follow-up unrelated to history must never be forced into a
    resolution -- it should either come back unchanged or, if rewritten,
    still fail to ground (never invents an antecedent)."""
    history = [
        Turn(
            user_message="Tell me about the AIML team",
            answer="AIML is led by Rahul Sharma.",
            source_section="Teams",
        )
    ]

    rewritten = llm_client.rewrite_query("What's the club's budget?", history)

    top = _retriever.retrieve(rewritten, top_k=1)[0]
    assert top.score < RETRIEVAL_THRESHOLD, (
        f"rewritten query {rewritten!r} unexpectedly grounded in "
        f"{top.section!r} (score={top.score:.3f}) for an unrelated follow-up"
    )
