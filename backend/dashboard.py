"""Streamlit app: a Chat tab over the live assistant, and a read-only
Dashboard tab over the persisted chat/action logs (requirements.md §3.4,
Slice 6).

Two tabs, one browser window:

* **Chat** -- a thin UI over `backend.actions.handle_turn`, the same single
  per-turn entrypoint `backend.cli` drives. One Streamlit session holds one
  conversation (one `session_id`), and each turn's metadata is surfaced the
  way the CLI prints it: intent + path, confidence band + score, cited
  source section, and the `[action]` marker on agentic turns.
* **Dashboard** -- reads the two append-only JSON-Lines logs `handle_turn`
  writes on every real chat turn: `backend.turn_log.TurnLogStore` (one
  record per turn: intent, path, confidence, refused) and
  `backend.actions.ActionRecordStore` (one record per completed/abandoned
  action). Never a hand-maintained or mocked list (spec: "reads from
  persisted logs, not a hand-maintained/mocked list").

Because the Chat tab writes to exactly those logs, the Dashboard tab
refreshes off real traffic as you chat -- the logs stay the single source of
truth, and the dashboard still reads them rather than the in-memory history.
The Chat tab renders before the Dashboard tab in each script run, so a turn
answered in this run is already on disk by the time the dashboard reads it.

Chatting makes LLM calls and so needs an API key (see README Setup); the
Dashboard tab alone still needs neither. The CLI (`backend.cli`) remains a
fully supported entrypoint -- this app doesn't replace it, and both can run
against the same logs at once.

Run: `uv run streamlit run backend/dashboard.py`
"""

import logging
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# This is an application entry point, so it loads `.env` itself -- library
# modules only ever read from `os.environ` (see backend/cli.py's docstring).
load_dotenv()

from backend import llm_client  # noqa: E402 -- after load_dotenv(), see module docstring
from backend.actions import ActionRecordStore, handle_turn  # noqa: E402
from backend.qa import AnswerResult  # noqa: E402
from backend.turn_log import TurnLogStore  # noqa: E402

# Send this project's logs (including the full tracebacks `qa.py` logs on an
# LLM-call failure) to the same file the CLI uses, and stop them propagating
# to the root logger -- otherwise every failure would also print a stack
# trace into the terminal running Streamlit, mid-demo. Scoped to the
# `backend` logger rather than `logging.basicConfig`, which would either
# no-op (Streamlit has already configured the root logger) or, with
# force=True, silence Streamlit's own console output too.
_backend_logger = logging.getLogger("backend")
if not _backend_logger.handlers:
    _handler = logging.FileHandler("club_faq_assistant.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _backend_logger.addHandler(_handler)
    _backend_logger.setLevel(logging.INFO)
    _backend_logger.propagate = False

st.set_page_config(page_title="Club FAQ Assistant")
st.title("GDG On Campus Club FAQ Assistant")

# One session_id per browser session, so this tab's whole conversation shares
# one bounded history -- the same contract backend/cli.py holds for a REPL
# run. `st.session_state` survives reruns; the script itself re-executes on
# every interaction, so the guard matters.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []


# --- Chat tab ------------------------------------------------------------

# Explicit avatars rather than Streamlit's near-identical defaults, so a long
# exchange stays scannable as question / answer.
_AVATARS = {"user": "🙋", "assistant": "🤖"}


def _metadata_lines(result: AnswerResult) -> list[str]:
    """The per-response metadata the CLI prints, as display lines.

    Mirrors the printing block in `backend.cli.main` -- same fields, same
    three-way split between a refusal, an agentic-action turn (no KB section
    grounds a prompt/confirmation message, and "source=None" would read as a
    bug), and a grounded answer.

    The CLI's surrounding "[...]" markers are dropped: `_muted` wraps each
    line in a `:gray[...]` span, which a nested "]" would close early, and
    the muted styling already separates these from the answer text.
    """
    if result.refused:
        lines = ["refused | no source | score={:.3f}".format(result.score)]
    elif result.source_section is None:
        lines = ["action turn | no source"]
    else:
        lines = ["source={} | score={:.3f}".format(result.source_section, result.score)]

    lines.append("intent={} | path={}".format(result.intent, result.intent_path))
    if result.confidence:
        # Band and raw score together -- the band alone hides how close a
        # `medium` sat to either edge.
        lines.append("confidence={}".format(result.confidence.display()))
        if result.confidence.claims:
            lines.append(
                "grounding={}/{} claims verified".format(
                    result.confidence.supported_claims,
                    len(result.confidence.claims),
                )
            )
    return lines


def _muted(line: str) -> str:
    """One metadata line as a small, muted annotation.

    `st.caption` supplies the smaller type; the `:gray[...]` span supplies a
    colour that reads as an annotation on the answer rather than as more body
    text. Callers must pass bracket-free text -- a "]" inside closes the span.
    """
    return ":gray[{}]".format(line)


def _answer_prompt(prompt: str) -> None:
    """Run one turn through `handle_turn` and append both messages to the
    session's history. `handle_turn` persists the turn record itself."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        with st.spinner("Thinking..."):
            result = handle_turn(prompt, st.session_state.session_id)
    except Exception:
        # A grounded-answer LLM failure is already handled inside qa.py and
        # comes back as a degraded answer; anything reaching here is
        # unexpected, so keep the traceback in the log file and leave the
        # conversation usable instead of blanking the app with a stack trace.
        _backend_logger.exception("chat turn failed: query=%r", prompt)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Something went wrong on that turn -- see club_faq_assistant.log.",
                "meta": ["error"],
            }
        )
        return

    message = {
        "role": "assistant",
        "content": result.answer,
        "meta": _metadata_lines(result),
    }
    if result.rewritten_query and result.rewritten_query != prompt:
        message["rewritten"] = result.rewritten_query
    st.session_state.messages.append(message)


def _reset_conversation() -> None:
    """Start a fresh conversation: a new session_id and an empty history.

    Both `backend.memory.SessionStore` (multi-turn memory) and
    `ActiveActionStore` are keyed by session_id, so minting a new one is a
    clean break -- no history and no in-progress action carry over, and no
    old session's state has to be reached into and cleared.

    An action left mid-slot-fill is dropped without an `abandoned` record,
    exactly as quitting the CLI mid-action drops it: no terminal transition
    happened, so there is nothing to log.
    """
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []


def render_chat() -> None:
    st.subheader("Ask about the club")
    st.markdown(
        "Answers questions from the club handbook -- **intro, teams, events, "
        "recruitment, rules, contacts and achievements** -- and cites the "
        "section each answer came from. It can also **register you for an "
        "event** or **record event feedback**, asking for whatever details it "
        "still needs. When nothing in the handbook matches, it says so rather "
        "than guessing."
    )
    st.caption(
        "Same engine as the CLI (`backend.actions.handle_turn`). Every turn is "
        "written to the log the Dashboard tab reads."
    )

    # Handled before the history container is filled below, so the cleared
    # conversation shows up in this same run. Deliberately not disabled on an
    # empty history: the button renders before this run's turn is processed,
    # so a `disabled=not messages` flag would always be one run stale (still
    # greyed out right after the first question). Clicking it with nothing to
    # clear is a harmless no-op.
    if st.button("New conversation"):
        _reset_conversation()

    missing_var = llm_client.missing_api_key_var()
    if missing_var:
        st.warning(
            f"{missing_var} is not set, so grounded answers will fail with a "
            "quota/error message until it's set in .env (see README.md Setup). "
            "Refusals (no KB match) still work without it."
        )

    # The history is rendered into a container declared *before* the input
    # box, but filled *after* the turn is answered -- so a new exchange shows
    # up in the same run, above the input, with no st.rerun() round trip.
    history = st.container()
    prompt = st.chat_input("Ask a question about the club")
    if prompt and prompt.strip():
        _answer_prompt(prompt.strip())

    with history:
        if not st.session_state.messages:
            st.info("No messages yet. Ask something like *When does the AIML team meet?*")
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=_AVATARS[message["role"]]):
                if message.get("rewritten"):
                    st.caption(_muted("rewritten: {}".format(message["rewritten"])))
                st.markdown(message["content"])
                if message.get("meta"):
                    # Two trailing spaces = a markdown hard break, so the
                    # metadata stacks one field per line as it does in the CLI.
                    st.caption("  \n".join(_muted(line) for line in message["meta"]))


# --- Dashboard tab -------------------------------------------------------


def render_dashboard() -> None:
    # Read on every run, after the Chat tab has had its turn: a question
    # answered in this run is already in the log by now.
    turns = TurnLogStore().read_all()
    actions = ActionRecordStore().read_all()
    turns_df = pd.DataFrame(turns) if turns else None

    # --- 1. Chat stats -----------------------------------------------------

    st.header("1. Chat stats")
    total_messages = len(turns)
    total_sessions = len({t["session_id"] for t in turns})
    col1, col2 = st.columns(2)
    col1.metric("Total sessions", total_sessions)
    col2.metric("Total messages", total_messages)

    # --- 2. Intent breakdown -------------------------------------------------

    st.header("2. Intent breakdown")
    if turns_df is not None:
        intent_counts = (
            turns_df["intent"].value_counts().rename_axis("intent").reset_index(name="count")
        )
        st.subheader("Counts per intent")
        st.dataframe(intent_counts, width="stretch", hide_index=True)

        st.subheader("Rule-path vs. LLM-path split")
        st.caption(
            "`action_state` turns are continuations of an in-progress action "
            "(slot fill, confirm, cancel) -- resolved by the state machine, not "
            "a fresh classify() call."
        )
        path_counts = (
            turns_df["intent_path"].value_counts().rename_axis("path").reset_index(name="count")
        )
        st.dataframe(path_counts, width="stretch", hide_index=True)
    else:
        st.write("No chat turns logged yet.")

    # --- Confidence band distribution (Slice 4) -------------------------------

    st.header("Confidence band distribution")
    if turns_df is not None:
        band_counts = (
            turns_df["confidence_band"]
            .value_counts(dropna=False)
            .rename_axis("band")
            .reset_index(name="count")
        )
        st.dataframe(band_counts, width="stretch", hide_index=True)
    else:
        st.write("No chat turns logged yet.")

    # --- 3. Actions log --------------------------------------------------------

    st.header("3. Actions log")
    if actions:
        actions_df = pd.DataFrame(actions)
        if "reason" not in actions_df.columns:
            actions_df["reason"] = None
        actions_df["reason"] = actions_df["reason"].fillna("")
        actions_df["slots"] = actions_df["slots"].apply(
            lambda slots: ", ".join(f"{key}={value}" for key, value in slots.items())
        )
        actions_df = actions_df[["timestamp", "action_type", "slots", "status", "reason"]]
        st.dataframe(actions_df.iloc[::-1], width="stretch", hide_index=True)
    else:
        st.write("No actions recorded yet.")

    # --- 4. Unanswered queries -------------------------------------------------

    st.header("4. Unanswered queries")
    st.caption(
        "Questions that hit the refusal path -- nothing in the KB matched -- for gap analysis."
    )
    if turns_df is not None:
        unanswered = turns_df[turns_df["refused"]][["timestamp", "session_id", "query"]]
        if len(unanswered):
            st.dataframe(unanswered.iloc[::-1], width="stretch", hide_index=True)
        else:
            st.write("No unanswered queries yet.")
    else:
        st.write("No chat turns logged yet.")


chat_tab, dashboard_tab = st.tabs(["Chat", "Dashboard"])
with chat_tab:
    render_chat()
with dashboard_tab:
    render_dashboard()
