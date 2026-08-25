"""Unit tests for `backend.memory.SessionStore`.

Covers the bounded per-session history window and session isolation (spec:
I/O & Edge-Case Matrix -> "Bounded window", "Session isolation").
"""

from unittest.mock import patch

from backend import config
from backend.memory import SessionStore, Turn


def _turn(n: int) -> Turn:
    return Turn(user_message=f"question {n}", answer=f"answer {n}", source_section="Teams")


def test_empty_session_has_no_history():
    store = SessionStore()
    assert store.get_history("session-a") == []


def test_add_turn_appends_to_history_in_order():
    store = SessionStore()
    store.add_turn("session-a", _turn(1))
    store.add_turn("session-a", _turn(2))

    assert store.get_history("session-a") == [_turn(1), _turn(2)]


def test_history_retains_only_most_recent_max_history_turns():
    """More than MAX_HISTORY_TURNS turns added -> only the most recent
    MAX_HISTORY_TURNS are retained (spec Acceptance Criteria)."""
    store = SessionStore()

    with patch.object(config, "MAX_HISTORY_TURNS", 3):
        for n in range(5):
            store.add_turn("session-a", _turn(n))
        history = store.get_history("session-a")

    assert history == [_turn(2), _turn(3), _turn(4)]


def test_max_history_turns_is_read_at_call_time_not_baked_in_at_creation():
    """Changing config.MAX_HISTORY_TURNS between calls changes trimming
    immediately -- the window isn't baked into a fixed-size structure when
    the session is first created (spec: Boundaries & Constraints ->
    "History window is bounded by one named constant, read at call time")."""
    store = SessionStore()

    with patch.object(config, "MAX_HISTORY_TURNS", 10):
        for n in range(5):
            store.add_turn("session-a", _turn(n))
    assert len(store.get_history("session-a")) == 5

    with patch.object(config, "MAX_HISTORY_TURNS", 2):
        store.add_turn("session-a", _turn(5))

    assert store.get_history("session-a") == [_turn(4), _turn(5)]


def test_sessions_are_isolated():
    """Two distinct session_ids; one has history, the other doesn't --
    neither call affects the other's history (spec I/O matrix -> "Session
    isolation")."""
    store = SessionStore()
    store.add_turn("session-a", _turn(1))

    assert store.get_history("session-a") == [_turn(1)]
    assert store.get_history("session-b") == []


def test_get_history_returns_a_copy_not_the_internal_list():
    store = SessionStore()
    store.add_turn("session-a", _turn(1))

    history = store.get_history("session-a")
    history.append(_turn(99))

    assert store.get_history("session-a") == [_turn(1)]


def test_turn_source_section_is_nullable():
    """Refusals/LLM-error turns record with no cited section (spec: "Every
    turn ... is recorded ... including its cited section (nullable)")."""
    store = SessionStore()
    turn = Turn(user_message="What's the club's budget?", answer="refused", source_section=None)

    store.add_turn("session-a", turn)

    assert store.get_history("session-a")[0].source_section is None
