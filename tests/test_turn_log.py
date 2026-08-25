"""Unit tests for `backend.turn_log` -- the persisted per-turn log that
backs the Slice 6 dashboard.

No API key or network access required: `record_turn` only ever reads fields
off an already-built `AnswerResult`, never calls into `llm_client`.
"""

from backend.confidence import ConfidenceResult
from backend.qa import AnswerResult
from backend.turn_log import TurnLogStore, record_turn


def test_read_all_on_nonexistent_file_returns_empty_list(tmp_path):
    store = TurnLogStore(tmp_path / "never_written.jsonl")
    assert store.read_all() == []


def test_append_and_read_all_round_trips(tmp_path):
    store = TurnLogStore(tmp_path / "turns_log.jsonl")
    store.append({"query": "Who leads AIML?", "intent": "faq"})
    store.append({"query": "When is HackFest?", "intent": "event_inquiry"})

    records = store.read_all()
    assert len(records) == 2
    assert records[0]["query"] == "Who leads AIML?"
    assert records[1]["intent"] == "event_inquiry"


def test_records_survive_a_fresh_store_instance_pointed_at_the_same_file(tmp_path):
    """Simulates a process restart, mirroring
    tests/test_actions.py::test_records_survive_a_fresh_store_instance_pointed_at_the_same_file."""
    path = tmp_path / "turns_log.jsonl"
    TurnLogStore(path).append({"query": "Who leads AIML?"})

    second_process_store = TurnLogStore(path)
    assert second_process_store.read_all() == [{"query": "Who leads AIML?"}]


def test_record_turn_maps_a_grounded_answer_result(tmp_path):
    store = TurnLogStore(tmp_path / "turns_log.jsonl")
    result = AnswerResult(
        answer="Rahul Sharma leads AIML.",
        source_section="Teams",
        score=0.693,
        refused=False,
        rewritten_query="Who leads the AIML team?",
        intent="faq",
        intent_path="rule",
        confidence=ConfidenceResult(
            score=0.9, band="high", reason="verified", retrieval_score=1.0, grounding_score=0.9,
        ),
    )

    record_turn(store, "s1", "Who leads AIML?", result)

    records = store.read_all()
    assert len(records) == 1
    record = records[0]
    assert record["session_id"] == "s1"
    assert record["query"] == "Who leads AIML?"
    assert record["rewritten_query"] == "Who leads the AIML team?"
    assert record["intent"] == "faq"
    assert record["intent_path"] == "rule"
    assert record["refused"] is False
    assert record["source_section"] == "Teams"
    assert record["confidence_band"] == "high"
    assert record["confidence_score"] == 0.9
    assert record["confidence_reason"] == "verified"
    assert "timestamp" in record


def test_record_turn_handles_a_refusal_with_not_applicable_confidence(tmp_path):
    store = TurnLogStore(tmp_path / "turns_log.jsonl")
    result = AnswerResult(
        answer="I don't have that information in the club's knowledge base.",
        source_section=None,
        score=0.05,
        refused=True,
        intent="out_of_scope",
        intent_path="llm",
        confidence=ConfidenceResult(score=None, band="not_applicable", reason="refused"),
    )

    record_turn(store, "s2", "What's the club's budget?", result)

    record = store.read_all()[0]
    assert record["refused"] is True
    assert record["source_section"] is None
    assert record["confidence_band"] == "not_applicable"
    assert record["confidence_score"] is None
    assert record["confidence_reason"] == "refused"


def test_record_turn_handles_a_result_with_no_confidence(tmp_path):
    """`AnswerResult.confidence` defaults to None (pre-Slice-4 construction,
    still used by some tests/callers) -- must not raise."""
    store = TurnLogStore(tmp_path / "turns_log.jsonl")
    result = AnswerResult(answer="stub", source_section="Teams", score=0.5, refused=False)

    record_turn(store, "s3", "anything", result)

    record = store.read_all()[0]
    assert record["confidence_band"] is None
    assert record["confidence_score"] is None
    assert record["confidence_reason"] is None
