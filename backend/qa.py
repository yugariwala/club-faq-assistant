"""Single entrypoint for KB-grounded question answering.

Later slices (multi-turn memory, intent classification, agentic actions,
dashboard) build on top of `answer_question` without restructuring this
layer (spec: Code Map).
"""

import logging
from dataclasses import dataclass

from backend import config, llm_client
from backend.retrieval import RetrievalResult, TfidfRetriever

logger = logging.getLogger(__name__)

# Module-level singleton: the KB is fixed and fitting TF-IDF is cheap, but
# there is no reason to refit it on every call.
_retriever = TfidfRetriever()


@dataclass(frozen=True)
class AnswerResult:
    """Uniform result shape for every answer path, grounded or refused
    (spec: "Every answer path ... returns a uniform result shape: answer,
    source_section, score, refused flag")."""

    answer: str
    source_section: str | None
    score: float
    refused: bool


def answer_question(query: str, retriever: TfidfRetriever | None = None) -> AnswerResult:
    """Answer `query` strictly from the KB, or refuse if nothing is relevant enough.

    Retrieval always runs and its top score is compared against
    `config.RETRIEVAL_THRESHOLD` -- the only place that decision is made
    (spec: Design Notes -> "Refusal mechanics"). Below threshold,
    `llm_client` is never invoked. The threshold is read from the `config`
    module at call time (not imported by name) so editing
    `RETRIEVAL_THRESHOLD` in config.py changes refusal behavior without any
    change to this function (spec Acceptance Criteria).
    """
    active_retriever = retriever if retriever is not None else _retriever
    candidates: list[RetrievalResult] = active_retriever.retrieve(query, top_k=3)

    top = candidates[0] if candidates else None

    if top is None or top.score < config.RETRIEVAL_THRESHOLD:
        return AnswerResult(
            answer=config.REFUSAL_MESSAGE,
            source_section=None,
            score=top.score if top is not None else 0.0,
            refused=True,
        )

    try:
        answer_text = llm_client.generate_answer(query, top.section, top.content)
    except Exception:
        # Any failure reaching or parsing the LLM (auth, network, rate
        # limit, malformed response, ...) degrades gracefully instead of
        # crashing the caller (e.g. the CLI REPL). This is NOT the
        # below-threshold refusal path -- retrieval found a relevant KB
        # section (refused stays False, source_section/score are preserved)
        # and the query genuinely is in scope; the LLM call itself failed.
        logger.exception("llm_client.generate_answer failed for query=%r", query)
        return AnswerResult(
            answer=config.LLM_ERROR_MESSAGE,
            source_section=top.section,
            score=top.score,
            refused=False,
        )

    return AnswerResult(
        answer=answer_text,
        source_section=top.section,
        score=top.score,
        refused=False,
    )
