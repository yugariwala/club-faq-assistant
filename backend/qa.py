"""Single entrypoint for KB-grounded question answering.

Later slices (multi-turn memory, intent classification, agentic actions,
dashboard) build on top of `answer_question` without restructuring this
layer (spec: Code Map).
"""

import logging
from dataclasses import dataclass

from backend import config, llm_client
from backend.memory import SessionStore, Turn
from backend.retrieval import RetrievalResult, TfidfRetriever

logger = logging.getLogger(__name__)

# Module-level singletons: the KB is fixed and fitting TF-IDF is cheap, but
# there is no reason to refit it on every call. Likewise one process-wide
# SessionStore backs every session unless a caller overrides it (e.g. tests).
_retriever = TfidfRetriever()
_session_store = SessionStore()


@dataclass(frozen=True)
class AnswerResult:
    """Uniform result shape for every answer path, grounded or refused
    (spec: "Every answer path ... returns a uniform result shape: answer,
    source_section, score, refused flag"). `rewritten_query` defaults to
    ""  so existing keyword-arg construction (e.g. in tests/test_cli.py)
    that doesn't pass it keeps working unchanged."""

    answer: str
    source_section: str | None
    score: float
    refused: bool
    rewritten_query: str = ""


def answer_question(
    query: str,
    session_id: str,
    retriever: TfidfRetriever | None = None,
    session_store: SessionStore | None = None,
) -> AnswerResult:
    """Answer `query` strictly from the KB, or refuse if nothing is relevant enough.

    `session_id` scopes this call to one conversation's bounded history
    (spec: Boundaries & Constraints -> "Sessions are independent, keyed by
    session_id; no session's history leaks into another's rewrite or
    retrieval"). When that session already has history, the incoming query
    is first rewritten into a standalone form via `llm_client.rewrite_query`
    so pronoun/ellipsis follow-ups ("When is that?") carry enough lexical
    signal for retrieval; with no history yet, `rewrite_query` is never
    called and the original query passes straight through (spec: "If
    history is empty for a session, rewrite_query is never called").

    Retrieval and generation then run on the rewritten form exactly like
    Slice 1 ran on the raw query -- the refusal threshold check below is
    untouched, so a rewrite that can't resolve anything (returned
    unchanged) refuses exactly as an ungrounded Slice-1 query would (spec:
    "Rewriting never bypasses the Slice 1 refusal threshold"). The
    threshold itself is still read from `config` at call time, not imported
    by name, so editing `RETRIEVAL_THRESHOLD` changes refusal behavior with
    no change here.

    Every turn -- refused, grounded, or LLM-error -- is recorded to the
    session's history once, at the end, covering all three branches (spec:
    "Every turn ... is recorded to that session's history, including its
    cited section (nullable).").
    """
    active_retriever = retriever if retriever is not None else _retriever
    active_store = session_store if session_store is not None else _session_store

    history = active_store.get_history(session_id)
    rewritten_query = (
        llm_client.rewrite_query(query, history) if history and query.strip() else query
    )

    # Both the original and rewritten query are logged every turn, whether
    # or not they differ (spec: Boundaries & Constraints).
    logger.info(
        "session_id=%r original_query=%r rewritten_query=%r",
        session_id,
        query,
        rewritten_query,
    )

    candidates: list[RetrievalResult] = active_retriever.retrieve(rewritten_query, top_k=3)
    top = candidates[0] if candidates else None

    if top is None or top.score < config.RETRIEVAL_THRESHOLD:
        result = AnswerResult(
            answer=config.REFUSAL_MESSAGE,
            source_section=None,
            score=top.score if top is not None else 0.0,
            refused=True,
            rewritten_query=rewritten_query,
        )
    else:
        try:
            answer_text = llm_client.generate_answer(rewritten_query, top.section, top.content)
        except Exception:
            # Any failure reaching or parsing the LLM (auth, network, rate
            # limit, malformed response, ...) degrades gracefully instead of
            # crashing the caller (e.g. the CLI REPL). This is NOT the
            # below-threshold refusal path -- retrieval found a relevant KB
            # section (refused stays False, source_section/score are
            # preserved) and the query genuinely is in scope; the LLM call
            # itself failed.
            logger.exception(
                "llm_client.generate_answer failed for rewritten_query=%r", rewritten_query
            )
            result = AnswerResult(
                answer=config.LLM_ERROR_MESSAGE,
                source_section=top.section,
                score=top.score,
                refused=False,
                rewritten_query=rewritten_query,
            )
        else:
            result = AnswerResult(
                answer=answer_text,
                source_section=top.section,
                score=top.score,
                refused=False,
                rewritten_query=rewritten_query,
            )

    active_store.add_turn(
        session_id,
        Turn(user_message=query, answer=result.answer, source_section=result.source_section),
    )
    return result
