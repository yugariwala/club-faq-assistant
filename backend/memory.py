"""Per-session, bounded conversational history for multi-turn memory.

Sessions are independent and keyed by `session_id` -- no session's history
ever leaks into another's rewrite or retrieval (spec: Boundaries &
Constraints -> "Sessions are independent"). The window size is read from
`config.MAX_HISTORY_TURNS` at call time (inside `add_turn`), not captured
once at construction, so tuning that constant changes trimming behavior for
every session immediately (spec: "read at call time").
"""

from dataclasses import dataclass

from backend import config


@dataclass(frozen=True)
class Turn:
    """One completed conversational turn: what the user asked, what the bot
    answered, and which KB section (if any) grounded that answer.

    `source_section` is nullable -- refusals and LLM-error turns are still
    recorded, with no cited section (spec: "Every turn (grounded, refused,
    or LLM-error) is recorded to that session's history, including its
    cited section (nullable).").
    """

    user_message: str
    answer: str
    source_section: str | None


class SessionStore:
    """In-memory store of each session's turn history.

    Not persisted -- multi-turn memory within a running process only
    (persisted logging is a later slice, per spec Boundaries & Constraints
    -> "Never").
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[Turn]] = {}

    def get_history(self, session_id: str) -> list[Turn]:
        """Return this session's turns, oldest first.

        A copy is returned so callers can't mutate the store's internal list
        directly. Unknown/empty sessions return an empty list.
        """
        return list(self._sessions.get(session_id, []))

    def add_turn(self, session_id: str, turn: Turn) -> None:
        """Append `turn` to `session_id`'s history, then trim to the most
        recent `config.MAX_HISTORY_TURNS` turns.

        `config.MAX_HISTORY_TURNS` is read here, at call time, so it can be
        tuned (or patched in tests) without touching this method or
        pre-sizing anything at session creation.
        """
        history = self._sessions.setdefault(session_id, [])
        history.append(turn)

        max_turns = config.MAX_HISTORY_TURNS
        if len(history) > max_turns:
            del history[: len(history) - max_turns]
