"""Evaluate `backend.intent.classify` against `data/intent_eval.jsonl`.

Usage:
    uv run python scripts/eval_intents.py [--rules-only] [--out PATH]

Requires GEMINI_API_KEY (or ANTHROPIC_API_KEY with LLM_PROVIDER=anthropic)
unless --rules-only is passed. `.env` is loaded here, once, at this entry
point -- library modules never load it themselves (see backend/cli.py's
module docstring for why).

Quota efficiency: every eval item a rule can resolve costs zero LLM calls.
Every item the rules abstain on is batched into a single
`llm_client.classify_intents_batch` call per eval run (not one call per
item) -- for the default 56-item set with ~23 rule-abstained items, that's
1 LLM call total instead of 23. --rules-only skips the LLM path entirely
(0 calls) so rule patterns can be iterated on for free.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, llm_client  # noqa: E402
from backend.intent import _RULES  # noqa: E402

DEFAULT_EVAL_SET = Path(__file__).resolve().parent.parent / "data" / "intent_eval.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "intent_eval_results.md"

LABELS = sorted(config.INTENT_LABELS)


@dataclass(frozen=True)
class EvalItem:
    query: str
    gold: str


@dataclass(frozen=True)
class Prediction:
    query: str
    gold: str
    predicted: str
    path: str  # "rule" | "llm" | "skipped" (--rules-only, rule abstained)


def load_eval_set(path: Path) -> list[EvalItem]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(EvalItem(query=row["query"], gold=row["label"]))
    return items


def _classify_by_rules_only(query: str) -> str | None:
    """Same priority order as backend.intent.classify, but never falls
    through to the LLM -- used for both the real hybrid run's rule pass and
    the --rules-only mode."""
    for rule in _RULES:
        label = rule(query)
        if label is not None:
            return label
    return None


def run_eval(items: list[EvalItem], rules_only: bool) -> list[Prediction]:
    predictions: list[Prediction] = []
    llm_pending: list[tuple[int, EvalItem]] = []

    for item in items:
        if not item.query.strip():
            predictions.append(
                Prediction(item.query, item.gold, config.DEFAULT_INTENT_ON_LLM_FAILURE, "rule")
            )
            continue

        rule_label = _classify_by_rules_only(item.query)
        if rule_label is not None:
            predictions.append(Prediction(item.query, item.gold, rule_label, "rule"))
        elif rules_only:
            predictions.append(Prediction(item.query, item.gold, "", "skipped"))
        else:
            llm_pending.append((len(predictions), item))
            predictions.append(Prediction(item.query, item.gold, "", "llm"))  # placeholder

    if llm_pending:
        queries = [item.query for _, item in llm_pending]
        labels = llm_client.classify_intents_batch(queries)
        for (index, item), label in zip(llm_pending, labels):
            predictions[index] = Prediction(item.query, item.gold, label, "llm")

    return predictions


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def build_report(predictions: list[Prediction], rules_only: bool) -> str:
    scored = [p for p in predictions if p.path != "skipped"]
    skipped = [p for p in predictions if p.path == "skipped"]

    lines: list[str] = []
    lines.append("# Intent Classification Evaluation")
    lines.append("")
    lines.append(f"Mode: {'rules-only' if rules_only else 'hybrid (rules + LLM fallback)'}")
    lines.append(f"Eval set: {len(predictions)} items")
    if skipped:
        lines.append(
            f"Skipped (rule abstained, --rules-only): {len(skipped)} "
            f"({len(skipped) / len(predictions):.1%})"
        )
    lines.append("")

    # -- Path split -----------------------------------------------------
    rule_preds = [p for p in scored if p.path == "rule"]
    llm_preds = [p for p in scored if p.path == "llm"]

    def _accuracy(preds: list[Prediction]) -> float:
        if not preds:
            return 0.0
        return sum(1 for p in preds if p.predicted == p.gold) / len(preds)

    lines.append("## Rule-path vs. LLM-path split")
    lines.append("")
    lines.append("| Path | Count | Fraction of scored traffic | Accuracy |")
    lines.append("|---|---|---|---|")
    total_scored = len(scored) or 1
    lines.append(
        f"| rule | {len(rule_preds)} | {len(rule_preds) / total_scored:.1%} "
        f"| {_accuracy(rule_preds):.1%} |"
    )
    if not rules_only:
        lines.append(
            f"| llm | {len(llm_preds)} | {len(llm_preds) / total_scored:.1%} "
            f"| {_accuracy(llm_preds):.1%} |"
        )
    lines.append(f"| **overall** | {len(scored)} | 100.0% | {_accuracy(scored):.1%} |")
    lines.append("")

    if not scored:
        lines.append("No scored predictions (all items skipped) -- nothing further to report.")
        return "\n".join(lines)

    # -- Per-class precision/recall/F1 -----------------------------------
    confusion: dict[str, Counter] = {gold: Counter() for gold in LABELS}
    for p in scored:
        confusion.setdefault(p.gold, Counter())[p.predicted] += 1

    per_class = {}
    support = {}
    for label in LABELS:
        tp = confusion.get(label, Counter()).get(label, 0)
        fn = sum(c for pred, c in confusion.get(label, Counter()).items() if pred != label)
        fp = sum(
            confusion.get(other, Counter()).get(label, 0) for other in LABELS if other != label
        )
        precision, recall, f1 = _prf1(tp, fp, fn)
        per_class[label] = (precision, recall, f1)
        support[label] = sum(confusion.get(label, Counter()).values())

    lines.append("## Per-class precision / recall / F1")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for label in LABELS:
        precision, recall, f1 = per_class[label]
        lines.append(
            f"| {label} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {support[label]} |"
        )

    macro_p = sum(v[0] for v in per_class.values()) / len(LABELS)
    macro_r = sum(v[1] for v in per_class.values()) / len(LABELS)
    macro_f1 = sum(v[2] for v in per_class.values()) / len(LABELS)
    total_support = sum(support.values()) or 1
    weighted_p = sum(per_class[label][0] * support[label] for label in LABELS) / total_support
    weighted_r = sum(per_class[label][1] * support[label] for label in LABELS) / total_support
    weighted_f1 = sum(per_class[label][2] * support[label] for label in LABELS) / total_support

    lines.append(f"| **macro avg** | {macro_p:.3f} | {macro_r:.3f} | {macro_f1:.3f} | {total_support} |")
    lines.append(
        f"| **weighted avg** | {weighted_p:.3f} | {weighted_r:.3f} | {weighted_f1:.3f} | {total_support} |"
    )
    lines.append("")

    # -- Confusion matrix -------------------------------------------------
    lines.append("## Confusion matrix (rows = gold, columns = predicted)")
    lines.append("")
    header = "| gold \\ predicted | " + " | ".join(LABELS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(LABELS) + 1))
    for gold in LABELS:
        row = confusion.get(gold, Counter())
        cells = " | ".join(str(row.get(pred, 0)) for pred in LABELS)
        lines.append(f"| {gold} | {cells} |")
    lines.append("")

    # -- Misclassifications, for qualitative error inspection -------------
    errors = [p for p in scored if p.predicted != p.gold]
    if errors:
        lines.append("## Misclassifications")
        lines.append("")
        lines.append("| Query | Gold | Predicted | Path |")
        lines.append("|---|---|---|---|")
        for e in errors:
            lines.append(f"| {e.query} | {e.gold} | {e.predicted} | {e.path} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set", type=Path, default=DEFAULT_EVAL_SET, help="Path to the labeled JSONL eval set."
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT, help="Path to write the Markdown results report."
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Skip the LLM fallback entirely (0 API calls) -- items no rule "
        "resolves are reported as skipped, not scored. For iterating on "
        "rule patterns without spending quota.",
    )
    args = parser.parse_args()

    if not args.rules_only:
        missing_var = llm_client.missing_api_key_var()
        if missing_var:
            print(
                f"WARNING: {missing_var} is not set. Rule-resolved items will still "
                "score correctly; items requiring the LLM fallback will fail to "
                "classify. Pass --rules-only to skip the LLM path entirely, or set "
                f"{missing_var} in .env.",
                file=sys.stderr,
            )

    items = load_eval_set(args.eval_set)
    predictions = run_eval(items, rules_only=args.rules_only)
    report = build_report(predictions, rules_only=args.rules_only)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
