"""Unit tests for `backend.cli.main` (the interactive REPL).

`input()` and `backend.cli.answer_question` are monkeypatched so these tests
run with no terminal interaction, no network access, and no ANTHROPIC_API_KEY.
"""

from backend.cli import main
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

    def fake_answer_question(query):
        calls.append(query)
        raise AssertionError("answer_question should not be called for 'quit'")

    result = _run_cli(["quit"], monkeypatch, capsys, fake_answer_question)

    assert calls == []
    assert "GDG On Campus Club FAQ Assistant" in result.out


def test_exit_terminates_loop_without_calling_answer_question(monkeypatch, capsys):
    calls = []

    def fake_answer_question(query):
        calls.append(query)
        raise AssertionError("answer_question should not be called for 'exit'")

    result = _run_cli(["exit"], monkeypatch, capsys, fake_answer_question)

    assert calls == []
    assert "GDG On Campus Club FAQ Assistant" in result.out


def test_quit_uppercase_terminates_loop(monkeypatch, capsys):
    calls = []

    def fake_answer_question(query):
        calls.append(query)
        raise AssertionError("answer_question should not be called for 'QUIT'")

    _run_cli(["QUIT"], monkeypatch, capsys, fake_answer_question)

    assert calls == []


def test_empty_and_whitespace_input_is_skipped_without_crashing(monkeypatch, capsys):
    calls = []

    def fake_answer_question(query):
        calls.append(query)
        return AnswerResult(answer="stub", source_section="Teams", score=0.5, refused=False)

    result = _run_cli(["", "   ", "quit"], monkeypatch, capsys, fake_answer_question)

    # Both blank and whitespace-only input are stripped and skipped before
    # ever reaching answer_question -- the loop just continues.
    assert calls == []
    assert "GDG On Campus Club FAQ Assistant" in result.out


def test_refused_result_prints_refused_format(monkeypatch, capsys):
    def fake_answer_question(query):
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
    def fake_answer_question(query):
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
