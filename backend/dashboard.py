"""Read-only Streamlit dashboard over the persisted chat/action logs
(requirements.md §3.4, Slice 6).

Reads the two append-only JSON-Lines logs `backend.actions.handle_turn`
already writes on every real chat turn -- `backend.turn_log.TurnLogStore`
(one record per turn: intent, path, confidence, refused) and
`backend.actions.ActionRecordStore` (one record per completed/abandoned
action). Never a hand-maintained or mocked list (spec: "reads from
persisted logs, not a hand-maintained/mocked list").

Runs alongside the CLI (`backend.cli`), not instead of it: this process only
reads the log files, it never calls `handle_turn` itself, so it needs no
API key and makes no LLM calls.

Run: `uv run streamlit run backend/dashboard.py`
"""

import pandas as pd
import streamlit as st

from backend.actions import ActionRecordStore
from backend.turn_log import TurnLogStore

st.set_page_config(page_title="Club FAQ Assistant Dashboard")
st.title("GDG On Campus Club FAQ Assistant — Dashboard")

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
    st.dataframe(intent_counts, use_container_width=True, hide_index=True)

    st.subheader("Rule-path vs. LLM-path split")
    st.caption(
        "`action_state` turns are continuations of an in-progress action "
        "(slot fill, confirm, cancel) -- resolved by the state machine, not "
        "a fresh classify() call."
    )
    path_counts = (
        turns_df["intent_path"].value_counts().rename_axis("path").reset_index(name="count")
    )
    st.dataframe(path_counts, use_container_width=True, hide_index=True)
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
    st.dataframe(band_counts, use_container_width=True, hide_index=True)
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
    st.dataframe(actions_df.iloc[::-1], use_container_width=True, hide_index=True)
else:
    st.write("No actions recorded yet.")

# --- 4. Unanswered queries -------------------------------------------------

st.header("4. Unanswered queries")
st.caption("Questions that hit the refusal path -- nothing in the KB matched -- for gap analysis.")
if turns_df is not None:
    unanswered = turns_df[turns_df["refused"]][["timestamp", "session_id", "query"]]
    if len(unanswered):
        st.dataframe(unanswered.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.write("No unanswered queries yet.")
else:
    st.write("No chat turns logged yet.")
