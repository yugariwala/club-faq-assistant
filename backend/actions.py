"""Agentic actions: slot-filling, state machine, KB validation, and
persistence for the two actions in scope (requirements.md §3.3):
event registration and feedback submission.

`handle_turn` is the top-level per-turn entrypoint (backend/cli.py calls
this instead of `qa.answer_question` directly). It routes a session's
message to one of three places:

1. An in-progress action for this session -> `_continue_action`, which owns
   the whole state machine (cancel / switch-attempt / confirm / slot-fill /
   interruption).
2. No in-progress action, message classifies as `action_request` ->
   `_start_action`, which picks register vs. feedback and best-effort
   extracts whatever slots the opening message already gave.
3. Anything else -> unchanged, straight through to `qa.answer_question`.

Every path returns a plain `backend.qa.AnswerResult` -- action turns reuse
that shape rather than inventing a parallel one, so `answer` carries the
conversational text (prompt, confirmation, or completion message) exactly
the way it already carries a factual answer, and the CLI's existing print
logic needs no new branches. Confidence is `not_applicable` with reason
`action_turn` (config.CONFIDENCE_REASON_ACTION_TURN) for every action turn:
none of them assert a claim about the club, so "how much should you trust
this" has no answer, same reasoning as a refusal
(see `backend.confidence.not_applicable`).

State machine and interruption/abandonment policy are explained in the
module-level docstrings of `_continue_action` and `_looks_like_interruption`
below; see also README.md "Agentic actions (Slice 5)".
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backend import config, confidence, intent, qa
from backend.intent import IntentResult, _rule_action_request, _rule_event_inquiry, _rule_faq, _rule_greeting
from backend.kb_data import EVENTS
from backend.memory import SessionStore
from backend.qa import AnswerResult
from backend.retrieval import TfidfRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionDefinition:
    """What one action needs, and how to ask for each piece.

    `required_slots` is ordered -- it's both the fill priority (the first
    missing slot is always what gets asked next) and the confirmation/record
    field order, so tests and logs see a stable order.
    """

    action_type: str
    required_slots: tuple[str, ...]
    prompts: dict[str, str]


ACTION_DEFINITIONS: dict[str, ActionDefinition] = {
    "register": ActionDefinition(
        action_type="register",
        required_slots=("event", "name", "email"),
        prompts={
            "event": "Which event would you like to register for?",
            "name": "What name should I register you under?",
            "email": "What's your email address?",
        },
    ),
    "feedback": ActionDefinition(
        action_type="feedback",
        required_slots=("event", "feedback_text"),
        prompts={
            "event": "Which event is this feedback about?",
            "feedback_text": "What feedback would you like to share?",
        },
    ),
}


@dataclass
class ActiveAction:
    """One session's in-progress action. Mutated turn over turn (unlike the
    frozen dataclasses elsewhere in this codebase) because that's what it
    is -- scratch conversational state, never itself persisted; only a
    snapshot of `.slots` at a terminal transition becomes a permanent
    record (see `_complete`/`_abandon`)."""

    action_type: str
    slots: dict[str, str] = field(default_factory=dict)
    stage: str = "collecting"  # "collecting" | "confirming"
    idle_turns: int = 0


class ActiveActionStore:
    """In-memory, per-session active-action state. Same shape as
    `backend.memory.SessionStore`: not persisted -- an in-progress action
    does not survive a process restart, only a *completed or abandoned* one
    does (spec: "Persists the resulting record", not the in-flight
    conversation)."""

    def __init__(self) -> None:
        self._active: dict[str, ActiveAction] = {}

    def get(self, session_id: str) -> ActiveAction | None:
        return self._active.get(session_id)

    def set(self, session_id: str, active: ActiveAction) -> None:
        self._active[session_id] = active

    def clear(self, session_id: str) -> None:
        self._active.pop(session_id, None)


class ActionRecordStore:
    """Append-only JSON-Lines log of every completed/abandoned action.

    Each line is one record: timestamp, action_type, the captured slots, and
    status (spec: "Include timestamp, action type, captured slots, and
    status -- Slice 6's dashboard reads this."). Appending (never
    read-modify-write) is what makes this safe across a restart: a record is
    written exactly once, at the terminal transition, so there is never a
    half-written line to recover from.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else _default_log_path()

    def append(self, record: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        """Return every persisted record, oldest first. Empty list if the
        log doesn't exist yet -- a fresh checkout/restart before any action
        has ever completed, not an error."""
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
    return project_root / config.ACTIONS_LOG_DIR / config.ACTIONS_LOG_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Module-level singletons, mirroring backend.qa's `_retriever`/`_session_store`
# pattern -- every override below has the same optional-param escape hatch
# tests already use for those.
_active_action_store = ActiveActionStore()
_action_record_store = ActionRecordStore()


# ---------------------------------------------------------------------------
# Slot extraction -- regex/keyword only, zero LLM calls (spec: "keep slot
# extraction as cheap as possible").
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _match_event(text: str) -> dict | None:
    """Return the KB event `text` names, or None if it names none of them.

    Matches on the longest alias found anywhere in the normalized text, so a
    short, forgiving alias ("hackfest") doesn't block a more specific one
    from winning when both happen to be present. This is the sole gate for
    "is this a real KB event" (requirements.md §3.3 "Actions may only
    reference real KB entities") -- callers never accept an event slot
    value that didn't come through here.
    """
    normalized = _normalize(text)
    best: tuple[dict, str] | None = None
    for event in EVENTS:
        for alias in event["aliases"]:
            if alias in normalized and (best is None or len(alias) > len(best[1])):
                best = (event, alias)
    return best[0] if best else None


_EVENT_CUE_RE = re.compile(r"\b(?:for|about)\s+(.+?)[.!?]*$", re.IGNORECASE)


def _candidate_event_phrase(text: str) -> str | None:
    """Best-effort guess at what event an opening message *tried* to name,
    when `_match_event` found no real KB event at all.

    Only used to decide whether an unrecognized event is worth flagging
    immediately on the opening message (spec: "must be caught and reported,
    not accepted") versus just not asking about it yet -- e.g. "I'd like to
    submit feedback" alone names nothing and should just prompt for the
    event, not accuse the user of naming an invalid one. Requires at least
    two words after the cue so "for it"/"for me" doesn't false-fire.
    """
    match = _EVENT_CUE_RE.search(text)
    if not match:
        return None
    candidate = match.group(1).strip()
    return candidate if len(candidate.split()) >= 2 else None


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _extract_email(text: str) -> str | None:
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else None


_NAME_LEADIN_RE = re.compile(
    r"(?:my name is|my name'?s|name'?s|it'?s|i'?m|this is|call me)\s+", re.IGNORECASE
)
_FEEDBACK_LEADIN_RE = re.compile(
    r"(?:my feedback is|feedback\s*[:\-]|here'?s my feedback[:,]?|i think|i feel)\s*",
    re.IGNORECASE,
)


def _extract_freetext_slot(text: str, leadin_re: re.Pattern, strict: bool) -> str | None:
    """Shared extractor for the two free-text slots (name, feedback_text).

    An explicit lead-in phrase ("my name is X"), searched anywhere in the
    message (not just at the start -- "Register me for HackFest, my name is
    Yug" has it mid-sentence), always wins: everything after the match is
    the value. Without one, `strict=False` (a direct reply to "what's your
    name?") falls back to using the whole trimmed message -- appropriate
    when we just asked exactly this question. `strict=True` (parsing the
    *opening* action request itself) never takes that fallback: the opening
    message is the action request ("Register me for HackFest"), not the
    slot's content, and grabbing whatever's left over after a shallow strip
    would silently mis-fill the slot with the request sentence itself.
    """
    stripped = text.strip()
    match = leadin_re.search(stripped)
    if match:
        value = stripped[match.end() :].strip().strip(".!")
        return value or None
    if strict:
        return None
    value = stripped.strip(".!")
    return value or None


def _extract_name(text: str, strict: bool) -> str | None:
    return _extract_freetext_slot(text, _NAME_LEADIN_RE, strict)


def _extract_feedback_text(text: str, strict: bool) -> str | None:
    return _extract_freetext_slot(text, _FEEDBACK_LEADIN_RE, strict)


def _valid_events_hint(action_type: str) -> str:
    if action_type == "register":
        names = [e["name"] for e in EVENTS if e["status"] == "Upcoming"]
        return "You can register for: " + ", ".join(names) + "."
    names = [e["name"] for e in EVENTS]
    return "Events on record: " + ", ".join(names) + "."


def _validate_event_for_action(event: dict, action_type: str) -> tuple[dict | None, str | None]:
    """Apply the per-action status rule to an already-matched KB event.

    Registration only makes sense for an `Upcoming` event -- you cannot sign
    up for something that already happened (spec: "Registering for a
    Completed event should also be flagged"). Feedback is the opposite case:
    it is *most* useful for an event that already happened, so no status
    restriction applies there, only "is this a real event at all"
    (`_match_event` already guarantees that by the time this runs).
    """
    if action_type == "register" and event["status"] != "Upcoming":
        return None, (
            f"{event['name']} already happened (status: {event['status']}), so "
            f"registration isn't open for it. {_valid_events_hint(action_type)}"
        )
    return event, None


def _try_fill_slot(slot_name: str, text: str, action_type: str) -> tuple[str, str | IntentResult]:
    """Attempt to fill one pending slot from a continuation-turn reply.

    Returns one of:
    - ("filled", value) -- accepted.
    - ("interrupt", IntentResult) -- this reply is recognizably about
      something else; caller should answer it via `qa.answer_question`
      rather than force-fitting it into the slot.
    - ("invalid", message) -- looked like an attempt to answer, but didn't
      validate; caller reprompts with `message`.
    """
    if slot_name == "event":
        event = _match_event(text)
        if event is None:
            interruption = _looks_like_interruption(text)
            if interruption is not None:
                return ("interrupt", interruption)
            return (
                "invalid",
                f"I don't see that in our events list. {_valid_events_hint(action_type)}",
            )
        validated, error = _validate_event_for_action(event, action_type)
        if error is not None:
            return ("invalid", error)
        return ("filled", validated["name"])

    if slot_name == "email":
        email = _extract_email(text)
        if email is not None:
            return ("filled", email)
        interruption = _looks_like_interruption(text)
        if interruption is not None:
            return ("interrupt", interruption)
        return ("invalid", "That doesn't look like a valid email address -- could you share it again?")

    if slot_name in ("name", "feedback_text"):
        # Checked before extraction, not after (unlike event/email): the
        # free-text fallback below accepts almost anything, so there is no
        # "extraction failed" moment to hang the interruption check off of.
        # `_looks_like_interruption` already applies its own "?"-gating for
        # faq/event_inquiry (greeting is unconditional), so no extra gate
        # is needed here.
        interruption = _looks_like_interruption(text)
        if interruption is not None:
            return ("interrupt", interruption)
        extractor = _extract_name if slot_name == "name" else _extract_feedback_text
        value = extractor(text, strict=False)
        if value:
            return ("filled", value)
        return ("invalid", "Sorry, I didn't catch that -- could you say that again?")

    raise ValueError(f"unknown slot {slot_name!r}")  # unreachable: every ActionDefinition slot is handled above


def _extract_opening_slots(action_type: str, query: str) -> tuple[dict[str, str], str | None]:
    """Best-effort slot extraction from the message that *started* the
    action (spec: "Extract whatever the opening message already provides").

    Structured slots (event, email) are extracted opportunistically from the
    full message -- their extractors are pattern-anchored (an alias match, an
    email regex) so there's no real risk of grabbing the wrong thing.
    Free-text slots (name, feedback_text) only fill on an explicit lead-in
    cue (`strict=True`) -- see `_extract_freetext_slot` for why the greedy
    fallback would be wrong here: "Register me for HackFest" must never be
    read as the caller's *name*.

    Returns (filled_slots, validation_error_or_None). The error is set only
    when the message named something that looks like an attempted event and
    it didn't validate -- naming *nothing* is not an error, it's just a slot
    to ask for next.
    """
    required = ACTION_DEFINITIONS[action_type].required_slots
    filled: dict[str, str] = {}
    error: str | None = None

    if "event" in required:
        event = _match_event(query)
        if event is not None:
            validated, err = _validate_event_for_action(event, action_type)
            if validated is not None:
                filled["event"] = validated["name"]
            else:
                error = err
        else:
            candidate = _candidate_event_phrase(query)
            if candidate is not None:
                error = f'I don\'t see "{candidate}" in our events list. {_valid_events_hint(action_type)}'

    if "email" in required:
        email = _extract_email(query)
        if email is not None:
            filled["email"] = email

    if "name" in required:
        name = _extract_name(query, strict=True)
        if name is not None:
            filled["name"] = name

    if "feedback_text" in required:
        feedback_text = _extract_feedback_text(query, strict=True)
        if feedback_text is not None:
            filled["feedback_text"] = feedback_text

    return filled, error


# ---------------------------------------------------------------------------
# Interruption detection -- rule layer only, never the LLM fallback.
# ---------------------------------------------------------------------------


def _looks_like_interruption(text: str) -> IntentResult | None:
    """Decide whether `text` is recognizably about something else entirely,
    rather than an (even if invalid) attempt to answer the pending slot.

    Deliberately reuses only Slice 3's high-precision *rule* layer
    (`backend.intent._rule_greeting/_rule_faq/_rule_event_inquiry`), never
    `intent.classify`'s LLM fallback -- two reasons, not one:

    1. **Cost**: this runs on every failed slot-fill attempt; routing it
       through the LLM would silently add a call to what's supposed to be
       the cheap path (spec: "keep slot extraction as cheap as possible").
    2. **Precision**: slot *content* in this domain routinely mentions club
       vocabulary on purpose -- feedback text naming the team or event it's
       about, an event slot reply that's just a wrong/misremembered event
       name ("the AIML workshop"). A keyword hit alone is a weak signal here
       and would false-positive constantly. Requiring a literal `?` alongside
       the faq/event_inquiry rules (greeting is exempted -- "hi"/"thanks"
       are never plausible slot content and rarely carry a `?`) keeps this
       to genuine questions, at the cost of occasionally missing an aside
       that isn't phrased as one -- which just falls through to a normal
       reprompt, never a crash or a silently wrong slot value.

    `_rule_action_request` is deliberately excluded: an action_request-shaped
    message mid-action is a *switch attempt*, handled separately in
    `_continue_action` (finish-or-cancel-first), not treated as an
    interruption to answer and return from.
    """
    greeting = _rule_greeting(text)
    if greeting is not None:
        return IntentResult(label=greeting, path="rule")
    if "?" not in text:
        return None
    for rule in (_rule_faq, _rule_event_inquiry):
        label = rule(text)
        if label is not None:
            return IntentResult(label=label, path="rule")
    return None


# ---------------------------------------------------------------------------
# Confirmation / cancel keyword matching
# ---------------------------------------------------------------------------

_CANCEL_RE = re.compile(r"\b(cancel|never\s*mind|stop|abort|forget (?:it|this))\b", re.IGNORECASE)
_AFFIRM_RE = re.compile(
    r"^(yes|yeah|yep|yup|correct|confirm(?:ed)?|right|that'?s right|sounds good|ok(?:ay)?)[.!]*$",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(r"^(no|nope|nah|incorrect|wrong|that'?s wrong)[.!]*$", re.IGNORECASE)


def _first_missing_slot(action_def: ActionDefinition, active: ActiveAction) -> str | None:
    for slot in action_def.required_slots:
        if slot not in active.slots:
            return slot
    return None


def _confirmation_summary(action_def: ActionDefinition, active: ActiveAction) -> str:
    s = active.slots
    if active.action_type == "register":
        body = f"event: {s['event']}, name: {s['name']}, email: {s['email']}"
    else:
        body = f'event: {s["event"]}, feedback: "{s["feedback_text"]}"'
    return f"Please confirm -- {body}. Reply 'yes' to confirm or 'no' to cancel."


def _completion_message(active: ActiveAction) -> str:
    s = active.slots
    if active.action_type == "register":
        return f"You're registered! Recorded: event={s['event']}, name={s['name']}, email={s['email']}."
    return f'Thanks! Recorded your feedback for {s["event"]}: "{s["feedback_text"]}"'


_ABANDON_MESSAGES = {
    "user_cancelled": "OK, I've cancelled that -- nothing was saved.",
    "rejected_at_confirmation": "No problem -- I've discarded that. Start again if you'd like to redo it with the corrected details.",
    "idle_timeout": "I've cancelled the in-progress action since we moved on -- just start again if you still want to do that.",
}


def _pending_reminder(action_def: ActionDefinition, active: ActiveAction) -> str:
    if active.stage == "confirming":
        return f"(Picking back up -- {_confirmation_summary(action_def, active)})"
    missing = _first_missing_slot(action_def, active)
    return f"(Picking back up on your {active.action_type} -- {action_def.prompts[missing]})"


def _make_result(
    answer: str, intent_label: str = "action_request", intent_path: str = "action_state"
) -> AnswerResult:
    """Build the AnswerResult shape shared by every action-turn response.

    `intent_path="action_state"` marks a turn resolved purely by the state
    machine (a slot fill, a yes/no, a cancel) -- distinct from "rule"/"llm",
    which mean a fresh `intent.classify` call actually ran. Only
    `_start_action`'s own turn (and any interruption, whose intent is
    genuinely `qa.answer_question`'s) uses a real rule/llm path.
    """
    return AnswerResult(
        answer=answer,
        source_section=None,
        score=0.0,
        refused=False,
        intent=intent_label,
        intent_path=intent_path,
        confidence=confidence.not_applicable(config.CONFIDENCE_REASON_ACTION_TURN),
    )


def _persist(
    active: ActiveAction, record_store: ActionRecordStore, status: str, reason: str | None
) -> None:
    record = {
        "timestamp": _now_iso(),
        "action_type": active.action_type,
        "slots": dict(active.slots),
        "status": status,
    }
    if reason is not None:
        record["reason"] = reason
    record_store.append(record)
    logger.info(
        "action record persisted: action_type=%r status=%r reason=%r slots=%r",
        active.action_type,
        status,
        reason,
        record["slots"],
    )


def _abandon(
    active: ActiveAction, session_id: str, action_store: ActiveActionStore,
    record_store: ActionRecordStore, reason: str,
) -> AnswerResult:
    _persist(active, record_store, config.ACTION_STATUS_ABANDONED, reason)
    action_store.clear(session_id)
    return _make_result(_ABANDON_MESSAGES[reason])


def _complete(
    active: ActiveAction, session_id: str, action_store: ActiveActionStore,
    record_store: ActionRecordStore,
) -> AnswerResult:
    _persist(active, record_store, config.ACTION_STATUS_COMPLETED, None)
    action_store.clear(session_id)
    return _make_result(_completion_message(active))


def _reprompt_no_progress(
    active: ActiveAction, session_id: str, action_store: ActiveActionStore,
    record_store: ActionRecordStore, message: str,
) -> AnswerResult:
    """A turn that didn't advance the action: a failed slot-fill attempt, or
    an unrecognized reply during confirmation. Counts against the idle-turn
    cap exactly like an interruption does -- see `_handle_interruption` for
    why that cap exists."""
    active.idle_turns += 1
    if active.idle_turns > config.ACTION_IDLE_TURN_LIMIT:
        return _abandon(active, session_id, action_store, record_store, "idle_timeout")
    action_store.set(session_id, active)
    return _make_result(message)


def _handle_interruption(
    query: str, session_id: str, active: ActiveAction, action_def: ActionDefinition,
    action_store: ActiveActionStore, record_store: ActionRecordStore,
    retriever: TfidfRetriever | None, session_store: SessionStore | None,
    intent_result: IntentResult,
) -> AnswerResult:
    """Answer the unrelated question for real, then remind the user what's
    still pending -- see module docstring and `_looks_like_interruption` for
    the reasoning. The pending action's slots are untouched; only the
    idle-turn counter advances, same bound as any other non-progressing turn.
    """
    qa_result = qa.answer_question(
        query, session_id, retriever=retriever, session_store=session_store,
        precomputed_intent=intent_result,
    )
    active.idle_turns += 1
    if active.idle_turns > config.ACTION_IDLE_TURN_LIMIT:
        abandon_result = _abandon(active, session_id, action_store, record_store, "idle_timeout")
        combined = f"{qa_result.answer}\n\n{abandon_result.answer}"
    else:
        action_store.set(session_id, active)
        combined = f"{qa_result.answer}\n\n{_pending_reminder(action_def, active)}"
    return AnswerResult(
        answer=combined,
        source_section=qa_result.source_section,
        score=qa_result.score,
        refused=qa_result.refused,
        rewritten_query=qa_result.rewritten_query,
        intent=qa_result.intent,
        intent_path=qa_result.intent_path,
        confidence=qa_result.confidence,
    )


def _continue_action(
    query: str, session_id: str, active: ActiveAction, action_store: ActiveActionStore,
    record_store: ActionRecordStore, retriever: TfidfRetriever | None,
    session_store: SessionStore | None,
) -> AnswerResult:
    """Advance an in-progress action by exactly one turn.

    Checked in a fixed priority order every turn: explicit cancel, then a
    switch-to-a-different-action attempt (blocked, not silently accepted or
    silently dropped), then the stage-specific logic (confirming: yes/no;
    collecting: fill the one pending slot). See the module docstring for the
    interruption/abandonment policy this implements.
    """
    action_def = ACTION_DEFINITIONS[active.action_type]

    if _CANCEL_RE.search(query):
        return _abandon(active, session_id, action_store, record_store, "user_cancelled")

    if _rule_action_request(query) is not None:
        # A second action_request mid-flow reads as "switch to something
        # else", not as this action's slot content -- finishing or
        # cancelling the current one is a decision only the user can make
        # explicitly (via the cancel keyword above), so this never switches
        # on its own.
        reminder = (
            _confirmation_summary(action_def, active)
            if active.stage == "confirming"
            else action_def.prompts[_first_missing_slot(action_def, active)]
        )
        return _reprompt_no_progress(
            active, session_id, action_store, record_store,
            f"Let's finish this {active.action_type} first (or say 'cancel' to stop it). {reminder}",
        )

    if active.stage == "confirming":
        stripped = query.strip()
        if _AFFIRM_RE.match(stripped):
            return _complete(active, session_id, action_store, record_store)
        if _NEGATIVE_RE.match(stripped):
            return _abandon(active, session_id, action_store, record_store, "rejected_at_confirmation")
        interruption = _looks_like_interruption(query)
        if interruption is not None:
            return _handle_interruption(
                query, session_id, active, action_def, action_store, record_store,
                retriever, session_store, interruption,
            )
        return _reprompt_no_progress(
            active, session_id, action_store, record_store,
            "Sorry, I didn't catch that -- reply 'yes' to confirm or 'no' to cancel.\n\n"
            + _confirmation_summary(action_def, active),
        )

    # stage == "collecting"
    pending_slot = _first_missing_slot(action_def, active)
    outcome, payload = _try_fill_slot(pending_slot, query, active.action_type)

    if outcome == "filled":
        active.slots[pending_slot] = payload
        active.idle_turns = 0
        next_missing = _first_missing_slot(action_def, active)
        if next_missing is None:
            active.stage = "confirming"
            action_store.set(session_id, active)
            return _make_result(_confirmation_summary(action_def, active))
        action_store.set(session_id, active)
        return _make_result(action_def.prompts[next_missing])

    if outcome == "interrupt":
        return _handle_interruption(
            query, session_id, active, action_def, action_store, record_store,
            retriever, session_store, payload,
        )

    return _reprompt_no_progress(active, session_id, action_store, record_store, payload)


_FEEDBACK_CUE_RE = re.compile(r"\bfeedback\b", re.IGNORECASE)
_REGISTER_CUE_RE = re.compile(r"\b(register|sign\s*(?:me\s*)?up)\b", re.IGNORECASE)


def _route_action_type(text: str) -> str | None:
    """Which of the two implemented actions this opening message means.
    Feedback checked first since it's the more specific cue -- a message
    could in principle mention both, and "feedback about registration" is
    feedback, not a registration attempt."""
    if _FEEDBACK_CUE_RE.search(text):
        return "feedback"
    if _REGISTER_CUE_RE.search(text):
        return "register"
    return None


def _start_action(
    query: str, session_id: str, intent_result: IntentResult, action_store: ActiveActionStore
) -> AnswerResult:
    action_type = _route_action_type(query)
    if action_type is None:
        # action_request fired (spec: e.g. "remind me", "check my status")
        # but names an action outside this slice's scope (minimum 2:
        # register, feedback). No state is created -- there is nothing to
        # hold open.
        return AnswerResult(
            answer=(
                "I can help you register for an event or submit feedback about "
                "one -- which would you like to do?"
            ),
            source_section=None,
            score=0.0,
            refused=False,
            intent=intent_result.label,
            intent_path=intent_result.path,
            confidence=confidence.not_applicable(config.CONFIDENCE_REASON_ACTION_TURN),
        )

    action_def = ACTION_DEFINITIONS[action_type]
    filled, error = _extract_opening_slots(action_type, query)
    active = ActiveAction(action_type=action_type, slots=filled)

    missing = _first_missing_slot(action_def, active)
    if missing is None:
        active.stage = "confirming"
        body = _confirmation_summary(action_def, active)
    else:
        body = action_def.prompts[missing]
    if error is not None:
        body = f"{error}\n\n{body}"

    action_store.set(session_id, active)
    return AnswerResult(
        answer=body,
        source_section=None,
        score=0.0,
        refused=False,
        intent=intent_result.label,
        intent_path=intent_result.path,
        confidence=confidence.not_applicable(config.CONFIDENCE_REASON_ACTION_TURN),
    )


def handle_turn(
    query: str,
    session_id: str,
    retriever: TfidfRetriever | None = None,
    session_store: SessionStore | None = None,
    active_action_store: ActiveActionStore | None = None,
    record_store: ActionRecordStore | None = None,
) -> AnswerResult:
    """Single per-turn entrypoint: route to the action state machine or to
    plain KB Q&A, and return a uniform `AnswerResult` either way.

    Classifies intent at most once per turn. A session with no in-progress
    action gets exactly the one `intent.classify` call it would have paid
    for anyway (previously made inside `qa.answer_question` itself); when
    that call resolves to anything other than `action_request`, the same
    result is threaded through to `qa.answer_question` via
    `precomputed_intent` so it is never classified twice (spec: "keep slot
    extraction as cheap as possible" -- the router itself must not double
    the cost of every ordinary QA turn to get there).
    """
    action_store = active_action_store if active_action_store is not None else _active_action_store
    records = record_store if record_store is not None else _action_record_store

    active = action_store.get(session_id)
    if active is not None:
        return _continue_action(query, session_id, active, action_store, records, retriever, session_store)

    intent_result = intent.classify(query)
    if intent_result.label == "action_request":
        return _start_action(query, session_id, intent_result, action_store)

    return qa.answer_question(
        query, session_id, retriever=retriever, session_store=session_store,
        precomputed_intent=intent_result,
    )
