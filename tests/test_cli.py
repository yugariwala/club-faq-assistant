"""Unit tests for `backend.cli.main` (the interactive REPL).

`input()` and `backend.cli.answer_question` are monkeypatched so these tests
run with no terminal interaction, no network access, and no ANTHROPIC_API_KEY.
"""

from backend import config
from backend.cli import main
from backend.confidence import ConfidenceResult, VerifiedClaim
from backend.qa import AnswerResult


def _run_cli(inputs, monkeypatch, capsys, fake_answer_question):
    """Drive `main()` with a scripted sequence of `input()` responses."""
    it = iter(inputs)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("backend.cli.answer_question", fake_answer_question)

    main()

    return capsys.readouterr()


def test_quit_terminates_loop_without_calling_answer_question(monkeypatch, capsys):
    calls = []

    def fake_answer_question(query, session_id):
        calls.append(query)
        raise AssertionError("answer_question should not be called for 'quit'")

    result = _run_cli(["quit"], monkeypatch, capsys, fake_answer_question)

    assert calls == []
    assert "GDG On Campus Club FAQ Assistant" in result.out


def test_exit_terminates_loop_without_calling_answer_question(monkeypatch, capsys):
    calls = []

    def fake_answer_question(query, session_id):
        calls.append(query)
        raise AssertionError("answer_question should not be called for 'exit'")

    result = _run_cli(["exit"], monkeypatch, capsys, fake_answer_question)

    assert calls == []
    assert "GDG On Campus Club FAQ Assistant" in result.out


def test_quit_uppercase_terminates_loop(monkeypatch, capsys):
    calls = []

    def fake_answer_question(query, session_id):
        calls.append(query)
        raise AssertionError("answer_question should not be called for 'QUIT'")

    _run_cli(["QUIT"], monkeypatch, capsys, fake_answer_question)

    assert calls == []


def test_empty_and_whitespace_input_is_skipped_without_crashing(monkeypatch, capsys):
    calls = []

    def fake_answer_question(query, session_id):
        calls.append(query)
        return AnswerResult(answer="stub", source_section="Teams", score=0.5, refused=False)

    result = _run_cli(["", "   ", "quit"], monkeypatch, capsys, fake_answer_question)

    # Both blank and whitespace-only input are stripped and skipped before
    # ever reaching answer_question -- the loop just continues.
    assert calls == []
    assert "GDG On Campus Club FAQ Assistant" in result.out


def test_refused_result_prints_refused_format(monkeypatch, capsys):
    def fake_answer_question(query, session_id):
        return AnswerResult(
            answer="I don't have that information in the club's knowledge base.",
            source_section=None,
            score=0.05,
            refused=True,
        )

    result = _run_cli(["What's the club's budget?", "quit"], monkeypatch, capsys, fake_answer_question)

    assert "I don't have that information in the club's knowledge base." in result.out
    assert "[refused | no source | score=0.050]" in result.out
    # The grounded-path branch must not have also fired.
    assert "[source=" not in result.out


def test_grounded_result_prints_source_format(monkeypatch, capsys):
    def fake_answer_question(query, session_id):
        return AnswerResult(
            answer="Rahul Sharma leads AIML.",
            source_section="Teams",
            score=0.693,
            refused=False,
        )

    result = _run_cli(["Who leads the AIML team?", "quit"], monkeypatch, capsys, fake_answer_question)

    assert "Rahul Sharma leads AIML." in result.out
    assert "[source=Teams | score=0.693]" in result.out
    # The refused-path branch must not have also fired.
    assert "[refused" not in result.out


def test_all_calls_in_one_run_share_the_same_session_id(monkeypatch, capsys):
    """One session_id is generated per REPL run and passed to every
    answer_question call in that run (spec Code Map: "generate one
    session_id per REPL run; pass it to every answer_question call")."""
    session_ids = []

    def fake_answer_question(query, session_id):
        session_ids.append(session_id)
        return AnswerResult(answer="stub", source_section="Teams", score=0.5, refused=False)

    _run_cli(["first question", "second question", "quit"], monkeypatch, capsys, fake_answer_question)

    assert len(session_ids) == 2
    assert session_ids[0] == session_ids[1]
    assert session_ids[0] != ""


def test_rewritten_query_is_printed_when_it_differs_from_original(monkeypatch, capsys):
    def fake_answer_question(query, session_id):
        return AnswerResult(
            answer="Sept 20.",
            source_section="Events",
            score=0.5,
            refused=False,
            rewritten_query="When is the Cloud Study Jam?",
        )

    result = _run_cli(["When is that?", "quit"], monkeypatch, capsys, fake_answer_question)

    assert "[rewritten: When is the Cloud Study Jam?]" in result.out


def test_rewritten_query_is_not_printed_when_unchanged(monkeypatch, capsys):
    def fake_answer_question(query, session_id):
        return AnswerResult(
            answer="Rahul Sharma leads AIML.",
            source_section="Teams",
            score=0.5,
            refused=False,
            rewritten_query="Who leads the AIML team?",
        )

    result = _run_cli(["Who leads the AIML team?", "quit"], monkeypatch, capsys, fake_answer_question)

    assert "[rewritten:" not in result.out


def test_intent_and_path_are_printed_for_every_response(monkeypatch, capsys):
    def fake_answer_question(query, session_id):
        return AnswerResult(
            answer="Rahul Sharma leads AIML.",
            source_section="Teams",
            score=0.693,
            refused=False,
            intent="faq",
            intent_path="rule",
        )

    result = _run_cli(
        ["Who leads the AIML team?", "quit"], monkeypatch, capsys, fake_answer_question
    )

    assert "[intent=faq | path=rule]" in result.out


def test_startup_warns_when_api_key_missing(monkeypatch, capsys):
    monkeypatch.setattr("backend.cli.llm_client.missing_api_key_var", lambda: "GEMINI_API_KEY")

    def fake_answer_question(query, session_id):
        return AnswerResult(answer="stub", source_section="Teams", score=0.5, refused=False)

    result = _run_cli(["quit"], monkeypatch, capsys, fake_answer_question)

    assert "WARNING" in result.out
    assert "GEMINI_API_KEY" in result.out


def test_startup_prints_no_warning_when_api_key_present(monkeypatch, capsys):
    monkeypatch.setattr("backend.cli.llm_client.missing_api_key_var", lambda: None)

    def fake_answer_question(query, session_id):
        return AnswerResult(answer="stub", source_section="Teams", score=0.5, refused=False)

    result = _run_cli(["quit"], monkeypatch, capsys, fake_answer_question)

    assert "WARNING" not in result.out


# ---------------------------------------------------------------------------
# Confidence display (Slice 4)
# ---------------------------------------------------------------------------


def test_confidence_band_and_raw_score_are_both_printed(monkeypatch, capsys):
    """The band alone hides how close a `medium` sat to either edge, so both
    are always shown (spec: "Display the band and the raw score")."""
    claims = (
        VerifiedClaim("Rahul Sharma leads AIML", True, "AIML (Lead: Rahul Sharma)", True),
        VerifiedClaim("AIML has 25 members", False, "", False),
    )

    def fake_answer_question(query, session_id):
        return AnswerResult(
            answer="Rahul Sharma leads AIML.",
            source_section="Teams",
            score=0.693,
            refused=False,
            intent="faq",
            intent_path="rule",
            confidence=ConfidenceResult(
                score=0.5,
                band=config.CONFIDENCE_BAND_MEDIUM_NAME,
                reason=config.CONFIDENCE_REASON_VERIFIED,
                retrieval_score=0.89,
                grounding_score=0.5,
                claims=claims,
            ),
        )

    result = _run_cli(["Who leads AIML?", "quit"], monkeypatch, capsys, fake_answer_question)

    assert config.CONFIDENCE_BAND_MEDIUM_NAME in result.out
    assert "0.50" in result.out
    assert "1/2 claims verified" in result.out


def test_not_applicable_confidence_prints_its_reason_and_no_score(monkeypatch, capsys):
    def fake_answer_question(query, session_id):
        return AnswerResult(
            answer=config.REFUSAL_MESSAGE,
            source_section=None,
            score=0.0,
            refused=True,
            intent="out_of_scope",
            intent_path="llm",
            confidence=ConfidenceResult(
                score=None,
                band=config.CONFIDENCE_BAND_NOT_APPLICABLE_NAME,
                reason=config.CONFIDENCE_REASON_REFUSED,
            ),
        )

    result = _run_cli(["What's the budget?", "quit"], monkeypatch, capsys, fake_answer_question)

    assert config.CONFIDENCE_BAND_NOT_APPLICABLE_NAME in result.out
    assert config.CONFIDENCE_REASON_REFUSED in result.out
    assert "claims verified" not in result.out


def test_cli_tolerates_a_result_without_confidence(monkeypatch, capsys):
    """`confidence` defaults to None so pre-Slice-4 construction still works."""

    def fake_answer_question(query, session_id):
        return AnswerResult(answer="stub", source_section="Teams", score=0.5, refused=False)

    result = _run_cli(["Who leads AIML?", "quit"], monkeypatch, capsys, fake_answer_question)

    assert "stub" in result.out
