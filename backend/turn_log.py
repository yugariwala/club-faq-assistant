"""Persisted per-turn log: one JSON-Lines record per chat turn, for the
Slice 6 dashboard (requirements.md §3.2, §3.4).

Intent and confidence were already logged per-turn from Slice 3/4 onward, but
only to the free-text `logger.info` log (`club_faq_assistant.log`) -- fine
for debugging, useless for a dashboard to query. This module is the
persisted, structured counterpart: append-only JSON Lines, same shape and
durability rationale as `backend.actions.ActionRecordStore` (a record is
written exactly once, at the end of the turn, so there is never a
half-written line to recover from after a restart).

`record_turn` is called from exactly one place -- `backend.actions.
handle_turn`, the single per-turn entrypoint every real chat turn (QA or
action) passes through (see that module's docstring). Evaluation scripts
(`scripts/eval_intents.py`, `scripts/eval_grounding.py`) call `backend.qa.
answer_question` / `backend.intent.classify` directly and never touch this
log -- an offline eval run is not real chat usage, and the dashboard's "chat
stats" would be misleading if it counted one.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend import config
from backend.qa import AnswerResult

logger = logging.getLogger(__name__)


class TurnLogStore:
    """Append-only JSON-Lines log of every chat turn.

    Each line: timestamp, session_id, the raw and rewritten query, the
    classified intent and which path resolved it, whether the turn refused,
    the cited KB section (nullable), and the confidence band/score/reason.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else _default_log_path()

    def append(self, record: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        """Return every persisted record, oldest first. Empty list if the
        log doesn't exist yet -- a fresh checkout/restart before any turn has
        ever been logged, not an error."""
        if not self._path.exists():
            return []
        records = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def _default_log_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / config.TURN_LOG_DIR / config.TURN_LOG_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_turn(store: TurnLogStore, session_id: str, query: str, result: AnswerResult) -> None:
    """Append one turn's record to `store`, derived from the `AnswerResult`
    `backend.actions.handle_turn` is about to return.

    `query` is the raw user message, not `result.rewritten_query` -- the
    dashboard's "unanswered queries" panel needs what the user actually
    typed, and the rewrite (when it happened at all) is kept alongside it,
    not in its place.
    """
    conf = result.confidence
    record = {
        "timestamp": _now_iso(),
        "session_id": session_id,
        "query": query,
        "rewritten_query": result.rewritten_query,
        "intent": result.intent,
        "intent_path": result.intent_path,
        "refused": result.refused,
        "source_section": result.source_section,
        "confidence_band": conf.band if conf is not None else None,
        "confidence_score": conf.score if conf is not None else None,
        "confidence_reason": conf.reason if conf is not None else None,
    }
    store.append(record)
    logger.info("turn record persisted: session_id=%r intent=%r refused=%r", session_id, result.intent, result.refused)
