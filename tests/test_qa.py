"""Unit tests for `backend.qa.answer_question`, with `llm_client` mocked.

No network access or ANTHROPIC_API_KEY is required to run these tests.

Every call below passes a unique `session_id`, so each test still starts
with zero prior history (mirroring Slice 1's LLM-mock-free calls) unless a
test's whole point is to exercise history across turns -- those build up
their own session_id's history turn by turn within the test itself.
"""

import logging
from unittest.mock import patch

from backend import config
from backend.memory import SessionStore, Turn
from backend.qa import AnswerResult, answer_question


def test_refusal_short_circuits_llm_call():
    """Below-threshold query never reaches the LLM (spec Acceptance Criteria)."""
    with patch("backend.qa.llm_client.generate_answer") as mock_generate:
        result = answer_question("What's the club's budget?", session_id="test-refusal")

    mock_generate.assert_not_called()
    assert isinstance(result, AnswerResult)
    assert result.refused is True
    assert result.source_section is None
    assert result.score < config.RETRIEVAL_THRESHOLD
    assert result.answer == config.REFUSAL_MESSAGE


def test_empty_query_refuses_without_crashing():
    with patch("backend.qa.llm_client.generate_answer") as mock_generate:
        result = answer_question("", session_id="test-empty-query")

    mock_generate.assert_not_called()
    assert result.refused is True
    assert result.source_section is None


def test_grounded_path_returns_correct_shape_and_calls_llm_once():
    """A question clearly answered by the Teams section grounds through the LLM."""
    with patch(
        "backend.qa.llm_client.generate_answer", return_value="Rahul Sharma leads AIML."
    ) as mock_generate:
        result = answer_question("Who leads the AIML team?", session_id="test-grounded")

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
    assert result.rewritten_query == "Who leads the AIML team?"


def test_aggregate_lookup_grounds_on_teams_section():
    with patch("backend.qa.llm_client.generate_answer", return_value="stub") as mock_generate:
        result = answer_question("List all the teams", session_id="test-aggregate")

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
            result = answer_question(query, session_id="test-threshold-high")

    mock_generate.assert_not_called()
    assert result.refused is True

    with patch("backend.qa.llm_client.generate_answer", return_value="stub") as mock_generate:
        with patch.object(config, "RETRIEVAL_THRESHOLD", 0.0):
            result = answer_question("gibberish zxcv asdkjqwe", session_id="test-threshold-low")

    mock_generate.assert_called_once()
    assert result.refused is False


def test_score_is_raw_float_not_rounded():
    with patch("backend.qa.llm_client.generate_answer", return_value="stub"):
        result = answer_question("Who leads the AIML team?", session_id="test-score-type")

    assert isinstance(result.score, float)


def test_llm_call_failure_degrades_gracefully_without_crashing():
    """A retrieval hit (in-scope query) whose LLM call raises must not
    propagate the exception, and must NOT be reported as refused=True --
    that flag is reserved for the below-threshold "not in the KB" path,
    not an LLM-call failure on an in-scope query."""
    with patch(
        "backend.qa.llm_client.generate_answer", side_effect=RuntimeError("boom")
    ) as mock_generate:
        result = answer_question("Who leads the AIML team?", session_id="test-llm-error")

    mock_generate.assert_called_once()
    assert isinstance(result, AnswerResult)
    assert result.refused is False
    assert result.source_section == "Teams"
    assert result.score >= config.RETRIEVAL_THRESHOLD
    assert result.answer == config.LLM_ERROR_MESSAGE
    assert result.answer != config.REFUSAL_MESSAGE


# ---------------------------------------------------------------------------
# Multi-turn memory (Slice 2)
# ---------------------------------------------------------------------------


def test_first_turn_in_a_session_never_calls_rewrite_query():
    """No history yet -> rewrite_query is never called (spec: "If history
    is empty for a session, rewrite_query is never called")."""
    with patch("backend.qa.llm_client.rewrite_query") as mock_rewrite:
        with patch("backend.qa.llm_client.generate_answer", return_value="stub"):
            result = answer_question("Who leads the AIML team?", session_id="test-first-turn")

    mock_rewrite.assert_not_called()
    assert result.rewritten_query == "Who leads the AIML team?"


def test_coreference_follow_up_resolves_via_rewrite_and_grounds_correct_section():
    """Turn 1 grounds in Events; turn 2's pronoun follow-up is rewritten to a
    standalone Events question and retrieval/generation run on that
    rewritten form, grounding in the same section (spec Acceptance Criteria
    / I/O matrix -> "Coreference resolves")."""
    session_id = "test-coreference"

    with patch(
        "backend.qa.llm_client.generate_answer",
        return_value="The Cloud Study Jam is on Sept 20.",
    ):
        first = answer_question("Tell me about the Cloud Study Jam", session_id=session_id)

    assert first.refused is False
    assert first.source_section == "Events"

    with patch(
        "backend.qa.llm_client.rewrite_query",
        return_value="When is the Cloud Study Jam?",
    ) as mock_rewrite:
        with patch(
            "backend.qa.llm_client.generate_answer", return_value="Sept 20."
        ) as mock_generate:
            second = answer_question("When is that?", session_id=session_id)

    mock_rewrite.assert_called_once()
    rewrite_args = mock_rewrite.call_args.args
    assert rewrite_args[0] == "When is that?"
    assert len(rewrite_args[1]) == 1  # turn 1's history, passed through unmodified

    mock_generate.assert_called_once()
    generate_args = mock_generate.call_args.args
    assert generate_args[0] == "When is the Cloud Study Jam?"  # retrieval ran on rewritten form

    assert second.refused is False
    assert second.source_section == "Events"
    assert second.rewritten_query == "When is the Cloud Study Jam?"


def test_unresolvable_reference_refuses_exactly_like_slice_one():
    """rewrite_query can't resolve the follow-up and returns it unchanged ->
    retrieval on the unchanged query scores below threshold -> refuses
    exactly as Slice 1 would, no fabricated antecedent (spec I/O matrix ->
    "No resolvable antecedent")."""
    session_id = "test-unresolvable"

    with patch("backend.qa.llm_client.generate_answer", return_value="stub"):
        first = answer_question("Who leads the AIML team?", session_id=session_id)
    assert first.refused is False

    with patch(
        "backend.qa.llm_client.rewrite_query", return_value="What's the club's budget?"
    ) as mock_rewrite:
        with patch("backend.qa.llm_client.generate_answer") as mock_generate:
            second = answer_question("What's the club's budget?", session_id=session_id)

    mock_rewrite.assert_called_once()
    mock_generate.assert_not_called()
    assert second.refused is True
    assert second.source_section is None
    assert second.answer == config.REFUSAL_MESSAGE


def test_session_isolation_empty_session_unaffected_by_other_sessions_history():
    """Two distinct session_ids; one has relevant history, the other has
    none -- the empty session's follow-up behaves as if no conversation
    ever happened (spec I/O matrix -> "Session isolation")."""
    session_with_history = "test-isolation-a"
    session_without_history = "test-isolation-b"

    with patch("backend.qa.llm_client.generate_answer", return_value="stub"):
        answer_question("Tell me about the Cloud Study Jam", session_id=session_with_history)

    with patch("backend.qa.llm_client.rewrite_query") as mock_rewrite:
        with patch("backend.qa.llm_client.generate_answer") as mock_generate:
            result = answer_question("When is that?", session_id=session_without_history)

    # No history for THIS session -> rewrite_query is never called; the
    # original, unresolved query is retrieved as-is and refuses below
    # threshold, unaffected by the other session's Cloud Study Jam history.
    mock_rewrite.assert_not_called()
    mock_generate.assert_not_called()
    assert result.refused is True


def test_bounded_window_only_most_recent_max_history_turns_used_for_rewriting():
    """More than MAX_HISTORY_TURNS turns added to one session -> only the
    most recent MAX_HISTORY_TURNS are retained/used for rewriting (spec I/O
    matrix -> "Bounded window")."""
    session_id = "test-bounded-window"

    with patch.object(config, "MAX_HISTORY_TURNS", 2):
        with patch(
            "backend.qa.llm_client.rewrite_query", side_effect=lambda q, h: q
        ) as mock_rewrite:
            with patch("backend.qa.llm_client.generate_answer", return_value="stub"):
                answer_question("Who leads the AIML team?", session_id=session_id)
                answer_question("Who leads the Web Dev team?", session_id=session_id)
                answer_question("Who leads the App Dev team?", session_id=session_id)
                answer_question("Who leads it?", session_id=session_id)

        # The last call's history should reflect only the most recent
        # MAX_HISTORY_TURNS=2 turns, not all three prior turns.
        history_passed = mock_rewrite.call_args.args[1]

    assert len(history_passed) == 2
    assert history_passed[0].user_message == "Who leads the Web Dev team?"
    assert history_passed[1].user_message == "Who leads the App Dev team?"


def test_every_turn_is_recorded_including_refusals_and_llm_errors():
    """Every turn -- grounded, refused, or LLM-error -- is recorded to the
    session's history once, including its cited section, which is nullable
    for refusals (spec: Boundaries & Constraints)."""
    session_id = "test-history-recording"
    store = SessionStore()

    with patch("backend.qa.llm_client.rewrite_query", side_effect=lambda q, h: q):
        with patch(
            "backend.qa.llm_client.generate_answer", return_value="Rahul Sharma leads AIML."
        ):
            answer_question(
                "Who leads the AIML team?", session_id=session_id, session_store=store
            )
        with patch("backend.qa.llm_client.generate_answer"):
            answer_question(
                "What's the club's budget?", session_id=session_id, session_store=store
            )
        with patch("backend.qa.llm_client.generate_answer", side_effect=RuntimeError("boom")):
            answer_question(
                "Who leads the AIML team?", session_id=session_id, session_store=store
            )

    history = store.get_history(session_id)
    assert len(history) == 3

    assert history[0].answer == "Rahul Sharma leads AIML."
    assert history[0].source_section == "Teams"

    assert history[1].answer == config.REFUSAL_MESSAGE
    assert history[1].source_section is None

    assert history[2].answer == config.LLM_ERROR_MESSAGE
    assert history[2].source_section == "Teams"


def test_blank_follow_up_with_history_never_calls_rewrite_query():
    """A session with existing history followed by an empty/whitespace-only
    query must NOT trigger a real `rewrite_query` call -- retrieval already
    scores empty input below threshold regardless, so calling out to the LLM
    here would be pure wasted cost/latency."""
    session_id = "test-blank-follow-up"

    with patch("backend.qa.llm_client.generate_answer", return_value="stub"):
        first = answer_question("Who leads the AIML team?", session_id=session_id)
    assert first.refused is False

    with patch("backend.qa.llm_client.rewrite_query") as mock_rewrite:
        with patch("backend.qa.llm_client.generate_answer") as mock_generate:
            second = answer_question("   ", session_id=session_id)

    mock_rewrite.assert_not_called()
    mock_generate.assert_not_called()
    assert second.refused is True
    assert second.answer == config.REFUSAL_MESSAGE


def test_log_output_contains_both_original_and_rewritten_query(caplog):
    """Spec Acceptance Criteria: "Given any turn, when it completes, then
    both the original and rewritten query appear in the log output."
    Exercises a real two-turn conversation where turn 2 is rewritten via a
    mocked `llm_client.rewrite_query`, then asserts the logged record for
    that second call contains both query strings."""
    session_id = "test-log-output"
    original_query = "When is that?"
    rewritten_query = "When is the Cloud Study Jam?"

    with patch(
        "backend.qa.llm_client.generate_answer",
        return_value="The Cloud Study Jam is on Sept 20.",
    ):
        answer_question("Tell me about the Cloud Study Jam", session_id=session_id)

    with caplog.at_level(logging.INFO, logger="backend.qa"):
        with patch(
            "backend.qa.llm_client.rewrite_query", return_value=rewritten_query
        ):
            with patch("backend.qa.llm_client.generate_answer", return_value="Sept 20."):
                answer_question(original_query, session_id=session_id)

    second_turn_records = [r for r in caplog.records if original_query in r.getMessage()]
    assert second_turn_records, "expected a log record mentioning the original query"
    assert any(rewritten_query in r.getMessage() for r in second_turn_records)


def test_session_store_override_is_used_instead_of_module_singleton():
    """The optional `session_store` override behaves like the existing
    `retriever` override -- history lands in the passed-in store, not the
    module-level singleton (spec Code Map: "optional session_store override
    mirroring the existing retriever override")."""
    session_id = "test-store-override"
    store = SessionStore()

    with patch("backend.qa.llm_client.generate_answer", return_value="stub"):
        answer_question(
            "Who leads the AIML team?", session_id=session_id, session_store=store
        )

    assert len(store.get_history(session_id)) == 1
    assert store.get_history(session_id)[0].user_message == "Who leads the AIML team?"

    from backend.qa import _session_store

    assert _session_store.get_history(session_id) == []
