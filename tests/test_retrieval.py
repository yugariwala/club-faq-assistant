"""Unit tests for `backend.retrieval.TfidfRetriever`.

Covers the spec's I/O & Edge-Case Matrix rows: direct lookup, aggregate
lookup, out-of-scope, and empty/gibberish input.
"""

from backend.config import RETRIEVAL_THRESHOLD
from backend.kb_data import KB_ENTRIES
from backend.retrieval import RetrievalResult, TfidfRetriever


def test_direct_lookup_matches_teams_section_above_threshold():
    """"Who leads the AIML team?" should retrieve Teams with score >= threshold."""
    retriever = TfidfRetriever()

    results = retriever.retrieve("Who leads the AIML team?", top_k=3)
    top = results[0]

    assert top.section == "Teams"
    assert top.score >= RETRIEVAL_THRESHOLD


def test_aggregate_lookup_matches_teams_section():
    """"List all the teams" should retrieve Teams, whose content lists all 6 teams."""
    retriever = TfidfRetriever()

    results = retriever.retrieve("List all the teams", top_k=3)
    top = results[0]

    assert top.section == "Teams"
    assert top.score >= RETRIEVAL_THRESHOLD
    # Content is verbatim KB text, so all 6 team leads are present for the
    # LLM to enumerate.
    assert "Rahul Sharma" in top.content
    assert "Ananya Reddy" in top.content


def test_out_of_scope_query_scores_below_threshold():
    """"What's the club's budget?" is not covered by the KB and must score low."""
    retriever = TfidfRetriever()

    results = retriever.retrieve("What's the club's budget?", top_k=3)
    top = results[0]

    assert top.score < RETRIEVAL_THRESHOLD


def test_empty_input_does_not_crash_and_scores_below_threshold():
    retriever = TfidfRetriever()

    results = retriever.retrieve("", top_k=3)

    assert len(results) > 0
    assert all(r.score < RETRIEVAL_THRESHOLD for r in results)


def test_whitespace_only_input_does_not_crash():
    retriever = TfidfRetriever()

    results = retriever.retrieve("   ", top_k=3)

    assert len(results) > 0
    assert all(r.score < RETRIEVAL_THRESHOLD for r in results)


def test_gibberish_input_does_not_crash_and_scores_below_threshold():
    retriever = TfidfRetriever()

    results = retriever.retrieve("asdkjqwe zxcv random gibberish", top_k=3)

    assert len(results) > 0
    assert results[0].score < RETRIEVAL_THRESHOLD


def test_retrieve_returns_raw_unrounded_scores_for_every_candidate():
    """Scores are never discarded or rounded before reaching the caller."""
    retriever = TfidfRetriever()

    results = retriever.retrieve("Who leads the AIML team?", top_k=len(KB_ENTRIES))

    assert len(results) == len(KB_ENTRIES)
    for result in results:
        assert isinstance(result, RetrievalResult)
        assert isinstance(result.score, float)


def test_results_sorted_descending_by_score():
    retriever = TfidfRetriever()

    results = retriever.retrieve("Who is the president?", top_k=len(KB_ENTRIES))

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_result_count():
    retriever = TfidfRetriever()

    results = retriever.retrieve("Who leads the AIML team?", top_k=2)

    assert len(results) == 2
