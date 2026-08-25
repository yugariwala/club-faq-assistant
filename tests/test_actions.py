"""Unit tests for `backend.actions` -- slot-filling, the state machine, KB
validation, and persistence for the two Slice 5 actions (event registration,
feedback submission).

No API key or network access is required: the happy paths below never reach
an LLM at all (slot extraction and state transitions are pure regex --
that's the point, see README.md "Quota cost"), and `tests/conftest.py`'s
autouse fixture makes any accidental real call fail loudly rather than
silently succeed. The one path that *does* call the LLM layer (an
interruption answered via `qa.answer_question`) mocks `llm_client.generate_answer`
exactly like `tests/test_qa.py` does.
"""

from unittest.mock import patch

import pytest

from backend import config, intent
from backend.actions import (
    ActionRecordStore,
    ActiveAction,
    ActiveActionStore,
    handle_turn,
)
from backend.qa import AnswerResult


@pytest.fixture
def stores(tmp_path):
    """A fresh, isolated pair of stores per test -- mirrors the
    retriever/session_store override pattern already used throughout
    tests/test_qa.py, so no test's action state or persisted records leak
    into another's."""
    return ActiveActionStore(), ActionRecordStore(tmp_path / "actions_log.jsonl")


def _turn(query, session_id, action_store, record_store):
    return handle_turn(
        query, session_id, active_action_store=action_store, record_store=record_store
    )


# ---------------------------------------------------------------------------
# Slot-filling: partial info in the opening message
# ---------------------------------------------------------------------------


def test_opening_message_with_event_only_asks_for_name_not_the_event_again(stores):
    """The example from the brief: "Register me for HackFest" already gives
    the event, so the very next question must be about name (or email),
    never re-asking which event."""
    action_store, record_store = stores
    result = _turn("Register me for HackFest", "s1", action_store, record_store)

    assert result.refused is False
    assert result.source_section is None
    assert result.intent == "action_request"
    assert "name" in result.answer.lower()
    assert "event" not in result.answer.lower()

    active = action_store.get("s1")
    assert active.action_type == "register"
    assert active.slots == {"event": "HackFest 2025"}


def test_opening_message_never_swallows_the_whole_sentence_as_the_name(stores):
    """Without an explicit lead-in cue, the opening message's leftover text
    must never be mistaken for the name slot -- "Register me for HackFest"
    is the request, not the caller's name."""
    action_store, record_store = stores
    _turn("Register me for HackFest", "s1", action_store, record_store)

    active = action_store.get("s1")
    assert "name" not in active.slots


def test_opening_message_with_explicit_leadin_fills_name_immediately(stores):
    action_store, record_store = stores
    result = _turn(
        "Register me for HackFest, my name is Yug Ariwala", "s1", action_store, record_store
    )

    active = action_store.get("s1")
    assert active.slots["name"] == "Yug Ariwala"
    assert "email" in result.answer.lower()


def test_feedback_opening_message_asks_only_for_missing_feedback_text(stores):
    action_store, record_store = stores
    result = _turn(
        "I'd like to submit feedback about Flutter Forward", "s1", action_store, record_store
    )

    active = action_store.get("s1")
    assert active.action_type == "feedback"
    assert active.slots == {"event": "Flutter Forward"}
    assert "feedback" in result.answer.lower()


def test_action_request_with_no_recognized_action_type_asks_which_one(stores):
    """"Remind me about HackFest" matches Slice 3's action_request rule but
    names an action outside this slice's scope -- must degrade gracefully,
    not crash or silently start the wrong flow, and must not create state."""
    action_store, record_store = stores
    result = _turn("Remind me about HackFest.", "s1", action_store, record_store)

    assert result.refused is False
    assert "register" in result.answer.lower() or "feedback" in result.answer.lower()
    assert action_store.get("s1") is None


# ---------------------------------------------------------------------------
# Full slot-by-slot flow, end to end, with a confirmation
# ---------------------------------------------------------------------------


def test_full_registration_flow_completes_and_confirms_every_value(stores):
    action_store, record_store = stores
    session_id = "s-full-register"

    r1 = _turn("Register me for the Cloud Study Jam", session_id, action_store, record_store)
    assert "name" in r1.answer.lower()

    r2 = _turn("Yug Ariwala", session_id, action_store, record_store)
    assert "email" in r2.answer.lower()

    r3 = _turn("yug@example.com", session_id, action_store, record_store)
    assert "confirm" in r3.answer.lower()
    assert "Cloud Study Jam" in r3.answer
    assert "Yug Ariwala" in r3.answer
    assert "yug@example.com" in r3.answer

    r4 = _turn("yes", session_id, action_store, record_store)
    assert "registered" in r4.answer.lower()
    assert "Cloud Study Jam" in r4.answer
    assert "Yug Ariwala" in r4.answer
    assert "yug@example.com" in r4.answer
    assert action_store.get(session_id) is None

    records = record_store.read_all()
    assert len(records) == 1
    assert records[0]["action_type"] == "register"
    assert records[0]["status"] == config.ACTION_STATUS_COMPLETED
    assert records[0]["slots"] == {
        "event": "Cloud Study Jam",
        "name": "Yug Ariwala",
        "email": "yug@example.com",
    }
    assert "timestamp" in records[0]


def test_full_registration_flow_makes_zero_llm_calls(tmp_path):
    """Every turn in a well-formed flow resolves via regex/rule matching --
    the opening action_request rule fires deterministically, and no reply
    ever falls through to an LLM classify or generate call. Relies on
    tests/conftest.py's autouse fixture to fail loudly if that's wrong."""
    action_store = ActiveActionStore()
    record_store = ActionRecordStore(tmp_path / "actions_log.jsonl")
    session_id = "s-zero-calls"

    _turn("Register me for HackFest", session_id, action_store, record_store)
    _turn("Yug Ariwala", session_id, action_store, record_store)
    _turn("yug@example.com", session_id, action_store, record_store)
    result = _turn("yes", session_id, action_store, record_store)

    assert "registered" in result.answer.lower()


def test_full_feedback_flow_completes_and_persists(stores):
    action_store, record_store = stores
    session_id = "s-full-feedback"

    r1 = _turn("I want to submit feedback", session_id, action_store, record_store)
    assert "event" in r1.answer.lower()

    r2 = _turn("It's about HackFest", session_id, action_store, record_store)
    assert "feedback" in r2.answer.lower()

    r3 = _turn("The workshop was really well organized.", session_id, action_store, record_store)
    assert "confirm" in r3.answer.lower()

    r4 = _turn("yes", session_id, action_store, record_store)
    assert "thanks" in r4.answer.lower()

    records = record_store.read_all()
    assert records[0]["action_type"] == "feedback"
    assert records[0]["slots"]["event"] == "HackFest 2025"
    assert "well organized" in records[0]["slots"]["feedback_text"]


# ---------------------------------------------------------------------------
# Confirmation rejection
# ---------------------------------------------------------------------------


def test_rejecting_the_confirmation_discards_and_persists_abandoned(stores):
    action_store, record_store = stores
    session_id = "s-reject"

    _turn("Register me for HackFest", session_id, action_store, record_store)
    _turn("Yug Ariwala", session_id, action_store, record_store)
    _turn("wrong@example.com", session_id, action_store, record_store)
    result = _turn("no", session_id, action_store, record_store)

    assert "discarded" in result.answer.lower() or "start again" in result.answer.lower()
    assert action_store.get(session_id) is None

    records = record_store.read_all()
    assert records[0]["status"] == config.ACTION_STATUS_ABANDONED
    assert records[0]["reason"] == "rejected_at_confirmation"
    assert records[0]["slots"]["email"] == "wrong@example.com"


# ---------------------------------------------------------------------------
# KB validation: unknown / completed events
# ---------------------------------------------------------------------------


def test_registering_for_an_unknown_event_is_flagged_not_accepted(stores):
    action_store, record_store = stores
    session_id = "s-unknown-event"

    result = _turn(
        "Register me for the Rocket Launch Party", session_id, action_store, record_store
    )

    assert "don't see" in result.answer.lower()
    active = action_store.get(session_id)
    assert active is not None
    assert "event" not in active.slots


def test_registering_for_a_completed_event_is_flagged(stores):
    """Flutter Forward is Completed in the KB -- registration must be
    refused for it even though it's a real event."""
    action_store, record_store = stores
    session_id = "s-completed-event"

    result = _turn("Register me for Flutter Forward", session_id, action_store, record_store)

    assert "already happened" in result.answer.lower()
    active = action_store.get(session_id)
    assert "event" not in active.slots


def test_feedback_is_allowed_for_a_completed_event(stores):
    """Unlike registration, feedback naturally applies to an event that
    already happened -- Completed status must not block it."""
    action_store, record_store = stores
    session_id = "s-feedback-completed"

    result = _turn(
        "I'd like to submit feedback about Flutter Forward", session_id, action_store, record_store
    )

    active = action_store.get(session_id)
    assert active.slots.get("event") == "Flutter Forward"
    assert "already happened" not in result.answer.lower()


def test_invalid_event_named_mid_flow_is_flagged_and_reprompted(stores):
    action_store, record_store = stores
    session_id = "s-invalid-mid-flow"

    _turn("I would like to register", session_id, action_store, record_store)
    result = _turn("Rocket Launch Party", session_id, action_store, record_store)

    assert "don't see" in result.answer.lower()
    active = action_store.get(session_id)
    assert "event" not in active.slots
    assert active.idle_turns == 1


# ---------------------------------------------------------------------------
# Unrelated interruption mid-action
# ---------------------------------------------------------------------------


def test_unrelated_question_mid_action_is_answered_and_reminds_of_pending_slot(stores):
    action_store, record_store = stores
    session_id = "s-interrupt"

    _turn("Register me for HackFest", session_id, action_store, record_store)

    with patch(
        "backend.actions.qa.answer_question",
        return_value=AnswerResult(
            answer="HackFest 2025 is on Oct 10.",
            source_section="Events",
            score=0.6,
            refused=False,
            intent="event_inquiry",
            intent_path="rule",
        ),
    ) as mock_qa:
        result = _turn("When is HackFest again?", session_id, action_store, record_store)

    mock_qa.assert_called_once()
    assert "HackFest 2025 is on Oct 10." in result.answer
    assert "picking back up" in result.answer.lower()
    assert result.intent == "event_inquiry"  # the aside's own intent, not the action's

    # The action survives the interruption -- slots aren't lost, and the
    # user can resume answering the pending slot on the next turn.
    active = action_store.get(session_id)
    assert active is not None
    assert active.slots == {"event": "HackFest 2025"}

    result2 = _turn("Yug Ariwala", session_id, action_store, record_store)
    assert "email" in result2.answer.lower()


def test_greeting_mid_action_is_treated_as_interruption_without_a_question_mark(stores):
    """Greeting is exempt from the "?" requirement -- "hi" is never
    plausible slot content and virtually never carries a question mark."""
    action_store, record_store = stores
    session_id = "s-greeting-interrupt"

    _turn("Register me for HackFest", session_id, action_store, record_store)

    with patch(
        "backend.actions.qa.answer_question",
        return_value=AnswerResult(
            answer="Hi there!", source_section=None, score=0.0, refused=False,
            intent="greeting", intent_path="rule",
        ),
    ) as mock_qa:
        _turn("hi!", session_id, action_store, record_store)

    mock_qa.assert_called_once()


def test_faq_keyword_without_question_mark_is_not_misread_as_interruption(stores):
    """A slot reply that happens to mention club vocabulary but isn't
    phrased as a question must be treated as slot content, not an
    interruption -- e.g. feedback naming a team by name."""
    action_store, record_store = stores
    session_id = "s-no-false-interrupt"

    _turn("I want to submit feedback", session_id, action_store, record_store)
    _turn("It's about HackFest", session_id, action_store, record_store)

    with patch("backend.actions.qa.answer_question") as mock_qa:
        result = _turn(
            "The Web Dev team ran a great session at this one.",
            session_id, action_store, record_store,
        )

    mock_qa.assert_not_called()
    assert "confirm" in result.answer.lower()


# ---------------------------------------------------------------------------
# Switching actions mid-flow
# ---------------------------------------------------------------------------


def test_action_request_mid_flow_is_blocked_not_silently_switched(stores):
    action_store, record_store = stores
    session_id = "s-switch-attempt"

    _turn("Register me for HackFest", session_id, action_store, record_store)
    result = _turn(
        "Actually, sign me up for the Cloud Study Jam instead", session_id, action_store, record_store
    )

    assert "finish this" in result.answer.lower()
    active = action_store.get(session_id)
    assert active.action_type == "register"
    assert active.slots == {"event": "HackFest 2025"}


# ---------------------------------------------------------------------------
# Abandonment: explicit cancel
# ---------------------------------------------------------------------------


def test_explicit_cancel_abandons_and_persists(stores):
    action_store, record_store = stores
    session_id = "s-cancel"

    _turn("Register me for HackFest", session_id, action_store, record_store)
    _turn("Yug Ariwala", session_id, action_store, record_store)
    result = _turn("actually never mind, cancel that", session_id, action_store, record_store)

    assert "cancelled" in result.answer.lower()
    assert action_store.get(session_id) is None

    records = record_store.read_all()
    assert records[0]["status"] == config.ACTION_STATUS_ABANDONED
    assert records[0]["reason"] == "user_cancelled"
    assert records[0]["slots"] == {"event": "HackFest 2025", "name": "Yug Ariwala"}


def test_cancel_works_from_the_confirming_stage_too(stores):
    action_store, record_store = stores
    session_id = "s-cancel-confirming"

    _turn("Register me for HackFest", session_id, action_store, record_store)
    _turn("Yug Ariwala", session_id, action_store, record_store)
    _turn("yug@example.com", session_id, action_store, record_store)
    result = _turn("cancel", session_id, action_store, record_store)

    assert "cancelled" in result.answer.lower()
    records = record_store.read_all()
    assert records[0]["reason"] == "user_cancelled"
    assert records[0]["status"] == config.ACTION_STATUS_ABANDONED


# ---------------------------------------------------------------------------
# Abandonment: idle-turn safety net
# ---------------------------------------------------------------------------


def test_action_auto_abandons_after_the_idle_turn_limit(stores):
    action_store, record_store = stores
    session_id = "s-idle-timeout"

    _turn("Register me for HackFest", session_id, action_store, record_store)

    qa_stub = AnswerResult(
        answer="Rahul Sharma leads AIML.",
        source_section="Teams", score=0.5, refused=False,
        intent="faq", intent_path="rule",
    )
    with patch("backend.actions.qa.answer_question", return_value=qa_stub):
        for _ in range(config.ACTION_IDLE_TURN_LIMIT):
            result = _turn("Who leads the AIML team?", session_id, action_store, record_store)
            assert action_store.get(session_id) is not None

        final = _turn("Who leads the AIML team?", session_id, action_store, record_store)

    assert action_store.get(session_id) is None
    assert "cancelled" in final.answer.lower()
    records = record_store.read_all()
    assert records[0]["reason"] == "idle_timeout"
    assert records[0]["status"] == config.ACTION_STATUS_ABANDONED


# ---------------------------------------------------------------------------
# Persistence across "restart"
# ---------------------------------------------------------------------------


def test_records_survive_a_fresh_store_instance_pointed_at_the_same_file(tmp_path):
    """Simulates a process restart: a brand-new ActionRecordStore instance,
    backed by the same file, must see records an earlier instance wrote."""
    path = tmp_path / "actions_log.jsonl"
    action_store = ActiveActionStore()
    first_process_store = ActionRecordStore(path)

    _turn("Register me for HackFest", "s-restart", action_store, first_process_store)
    _turn("Yug Ariwala", "s-restart", action_store, first_process_store)
    _turn("yug@example.com", "s-restart", action_store, first_process_store)
    _turn("yes", "s-restart", action_store, first_process_store)

    second_process_store = ActionRecordStore(path)
    records = second_process_store.read_all()

    assert len(records) == 1
    assert records[0]["status"] == config.ACTION_STATUS_COMPLETED
    assert records[0]["slots"]["email"] == "yug@example.com"


def test_record_store_read_all_on_nonexistent_file_returns_empty_list(tmp_path):
    store = ActionRecordStore(tmp_path / "never_written.jsonl")
    assert store.read_all() == []


def test_multiple_records_append_across_separate_actions(tmp_path):
    path = tmp_path / "actions_log.jsonl"
    action_store = ActiveActionStore()
    record_store = ActionRecordStore(path)

    _turn("Register me for HackFest", "s-a", action_store, record_store)
    _turn("Yug Ariwala", "s-a", action_store, record_store)
    _turn("yug@example.com", "s-a", action_store, record_store)
    _turn("yes", "s-a", action_store, record_store)

    _turn("I want to submit feedback", "s-b", action_store, record_store)
    _turn("It's about HackFest", "s-b", action_store, record_store)
    _turn("Great event overall.", "s-b", action_store, record_store)
    _turn("cancel", "s-b", action_store, record_store)

    records = ActionRecordStore(path).read_all()
    assert len(records) == 2
    assert records[0]["action_type"] == "register"
    assert records[0]["status"] == config.ACTION_STATUS_COMPLETED
    assert records[1]["action_type"] == "feedback"
    assert records[1]["status"] == config.ACTION_STATUS_ABANDONED


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


def test_two_sessions_have_independent_active_actions(stores):
    action_store, record_store = stores

    _turn("Register me for HackFest", "s-x", action_store, record_store)
    with patch("backend.qa.llm_client.generate_answer", return_value="Rahul Sharma leads AIML."):
        result_y = _turn("Who leads the AIML team?", "s-y", action_store, record_store)

    # session s-y was never routed into the action state machine at all.
    assert action_store.get("s-y") is None
    assert action_store.get("s-x") is not None
    assert result_y.intent in ("faq",)


# ---------------------------------------------------------------------------
# Router: non-action turns are unaffected and never double-classified
# ---------------------------------------------------------------------------


def test_plain_faq_turn_routes_straight_through_to_qa_unchanged(stores):
    action_store, record_store = stores

    with patch("backend.actions.qa.answer_question") as mock_qa:
        mock_qa.return_value = AnswerResult(
            answer="Rahul Sharma leads AIML.", source_section="Teams", score=0.7,
            refused=False, intent="faq", intent_path="rule",
        )
        result = _turn("Who leads the AIML team?", "s-plain", action_store, record_store)

    mock_qa.assert_called_once()
    assert result.answer == "Rahul Sharma leads AIML."


def test_router_classifies_at_most_once_for_a_non_action_turn(stores, monkeypatch):
    """The router's own classify() call must be the only one for the whole
    turn -- `backend.actions` and `backend.qa` both do `from backend import
    intent`, so it is the *same* module-level `classify` function either
    way; if qa.answer_question ignored `precomputed_intent` and classified
    again, this count would be 2, not 1."""
    action_store, record_store = stores
    monkeypatch.setenv(config.VERIFY_GROUNDING_ENV_VAR, "0")

    with patch("backend.intent.classify", wraps=intent.classify) as mock_classify:
        with patch("backend.qa.llm_client.generate_answer", return_value="Rahul Sharma."):
            _turn("Who leads the AIML team?", "s-single-classify", action_store, record_store)

    mock_classify.assert_called_once()


# ---------------------------------------------------------------------------
# ActiveAction / ActiveActionStore basics
# ---------------------------------------------------------------------------


def test_active_action_store_get_returns_none_for_unknown_session():
    store = ActiveActionStore()
    assert store.get("nope") is None


def test_active_action_store_clear_is_idempotent():
    store = ActiveActionStore()
    store.set("s1", ActiveAction(action_type="register"))
    store.clear("s1")
    store.clear("s1")  # must not raise
    assert store.get("s1") is None
