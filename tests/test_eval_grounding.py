"""Unit tests for `scripts/eval_grounding.py`.

The eval script is the artifact that makes the confidence number defensible,
so its own scoring and reporting logic needs to be correct independently of
whether the live provider is reachable. These tests drive the full pipeline
with a stubbed verifier -- they prove the machinery (scoring, banding,
aggregation, the run-validity guard) works; they say nothing about how good
the real LLM verifier is, which only a live run against
`data/grounding_eval.jsonl` can measure.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from backend import config
from backend.llm_client import SUPPORTED_VERDICT, UNSUPPORTED_VERDICT, ClaimVerdict

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_grounding.py"


@pytest.fixture(scope="module")
def eval_grounding():
    spec = importlib.util.spec_from_file_location("eval_grounding", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_grounding"] = module
    spec.loader.exec_module(module)
    return module


def test_eval_set_is_balanced_and_content_is_verbatim_kb(eval_grounding):
    """A skewed set would make the category means incomparable, and content
    that has drifted from the KB would silently break every evidence check."""
    from backend.kb_data import KB_ENTRIES

    kb = {e["section"]: e["content"] for e in KB_ENTRIES}
    items = eval_grounding.load_eval_set(eval_grounding.DEFAULT_EVAL_SET)

    counts = {cat: sum(1 for i in items if i.label == cat) for cat in eval_grounding.CATEGORIES}
    assert len(set(counts.values())) == 1, f"unbalanced eval set: {counts}"
    assert all(i.content == kb[i.section] for i in items)


def test_no_verify_mode_makes_no_llm_call(eval_grounding):
    items = eval_grounding.load_eval_set(eval_grounding.DEFAULT_EVAL_SET)

    with patch("backend.llm_client.verify_groundings_batch") as mock_verify:
        scored = eval_grounding.run_eval(items, verify=False)

    mock_verify.assert_not_called()
    assert all(
        s.result.reason == config.CONFIDENCE_REASON_VERIFICATION_DISABLED for s in scored
    )


def test_pipeline_separates_categories_given_a_correct_verifier(eval_grounding):
    """End-to-end proof that the scoring path turns claim verdicts into a
    discriminating score. The stub plays a verifier that is always right, so
    this isolates the machinery from the model's accuracy."""
    items = eval_grounding.load_eval_set(eval_grounding.DEFAULT_EVAL_SET)

    def perfect_verifier(batch):
        verdicts = []
        for answer, _section, content in batch:
            item = next(i for i in items if i.answer == answer)
            if item.label == "grounded":
                claims = [ClaimVerdict("c1", SUPPORTED_VERDICT, content)]
            elif item.label == "partially_grounded":
                claims = [
                    ClaimVerdict("c1", SUPPORTED_VERDICT, content),
                    ClaimVerdict("c2", UNSUPPORTED_VERDICT, ""),
                ]
            else:
                claims = [ClaimVerdict("c1", UNSUPPORTED_VERDICT, "")]
            verdicts.append(claims)
        return verdicts

    with patch(
        "backend.llm_client.verify_groundings_batch", side_effect=perfect_verifier
    ):
        scored = eval_grounding.run_eval(items, verify=True)

    means = {
        cat: sum(s.result.score for s in scored if s.item.label == cat)
        / sum(1 for s in scored if s.item.label == cat)
        for cat in eval_grounding.CATEGORIES
    }

    assert means["fabricated"] == 0.0
    assert means["fabricated"] < means["partially_grounded"] < means["grounded"]
    assert not [
        s
        for s in scored
        if s.item.label == "fabricated" and s.result.band == config.CONFIDENCE_BAND_HIGH_NAME
    ]


def test_report_flags_a_run_where_verification_never_completed(eval_grounding):
    """A run in which every call failed would otherwise report "0 fabricated
    answers reached high" -- trivially true, and exactly the decoration this
    eval exists to rule out."""
    items = eval_grounding.load_eval_set(eval_grounding.DEFAULT_EVAL_SET)

    with patch(
        "backend.llm_client.verify_groundings_batch",
        side_effect=lambda batch: [None] * len(batch),
    ):
        scored = eval_grounding.run_eval(items, verify=True)

    report = eval_grounding.build_report(scored, verify=True)

    assert "INCOMPLETE RUN" in report
    assert all(
        s.result.reason == config.CONFIDENCE_REASON_VERIFICATION_FAILED for s in scored
    )


def test_report_does_not_flag_a_complete_run(eval_grounding):
    items = eval_grounding.load_eval_set(eval_grounding.DEFAULT_EVAL_SET)

    with patch(
        "backend.llm_client.verify_groundings_batch",
        side_effect=lambda batch: [
            [ClaimVerdict("c", SUPPORTED_VERDICT, content)] for _a, _s, content in batch
        ],
    ):
        scored = eval_grounding.run_eval(items, verify=True)

    assert "INCOMPLETE RUN" not in eval_grounding.build_report(scored, verify=True)


def test_verifier_integrity_section_counts_downgraded_claims(eval_grounding):
    """A verifier citing evidence that isn't in the source must show up in
    the report, not be silently corrected."""
    items = eval_grounding.load_eval_set(eval_grounding.DEFAULT_EVAL_SET)[:2]

    with patch(
        "backend.llm_client.verify_groundings_batch",
        side_effect=lambda batch: [
            [ClaimVerdict("c", SUPPORTED_VERDICT, "text that is not in the source")]
            for _ in batch
        ],
    ):
        scored = eval_grounding.run_eval(items, verify=True)

    report = eval_grounding.build_report(scored, verify=True)

    assert "Claims downgraded: **2**" in report
    assert all(s.result.grounding_score == 0.0 for s in scored)


def test_probe_reports_section_accuracy_without_llm_calls(eval_grounding):
    with patch("backend.llm_client.verify_groundings_batch") as mock_verify:
        report = eval_grounding.run_probe()

    mock_verify.assert_not_called()
    assert "top-1 section accuracy" in report.lower()
    assert "separation" in report.lower()
    # The probe must keep its own failures visible; trimming them would make
    # the accuracy figure that justifies the separation signal unfalsifiable.
    assert "Answerable but refused" in report
