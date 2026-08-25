"""Evaluate composite confidence scoring against `data/grounding_eval.jsonl`.

Usage:
    uv run python scripts/eval_grounding.py [--no-verify] [--probe] [--out PATH]

The question this answers is the only one that makes a confidence number
worth displaying: **does it separate grounded answers from fabricated ones?**
A score that doesn't discriminate is decoration. The report therefore leads
with the safety metric -- how many `fabricated` answers reached the `high`
band, which must be zero -- and with the mean score per category.

Answers in the eval set are fixed strings, not generated at eval time. That
is deliberate: it makes the measurement reproducible (no generation variance
confounding the result), it costs no generation quota, and it is the only way
to include deliberate fabrications, since a correctly-working generator will
not produce them on demand.

Quota: one verification call per `config.VERIFY_BATCH_SIZE` items (the eval
uses `llm_client.verify_groundings_batch`, not the per-turn path), so the
24-item set costs 4 calls rather than 24. `--no-verify` scores the retrieval
signal alone for 0 calls.

`--probe` reproduces the retrieval measurement behind
`confidence.retrieval_confidence`'s choice of the separation ratio over raw
cosine magnitude, and makes no LLM calls.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, confidence, llm_client  # noqa: E402
from backend.retrieval import TfidfRetriever  # noqa: E402

DEFAULT_EVAL_SET = Path(__file__).resolve().parent.parent / "data" / "grounding_eval.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "grounding_eval_results.md"

# Ordered worst-to-best so every table reads in the same direction.
CATEGORIES = ["fabricated", "partially_grounded", "grounded"]
BANDS = [
    config.CONFIDENCE_BAND_HIGH_NAME,
    config.CONFIDENCE_BAND_MEDIUM_NAME,
    config.CONFIDENCE_BAND_LOW_NAME,
    config.CONFIDENCE_BAND_NOT_APPLICABLE_NAME,
]


@dataclass(frozen=True)
class EvalItem:
    query: str
    section: str
    content: str
    answer: str
    label: str
    note: str


@dataclass(frozen=True)
class Scored:
    item: EvalItem
    result: confidence.ConfidenceResult
    retrieved_section: str | None


def load_eval_set(path: Path) -> list[EvalItem]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                EvalItem(
                    query=row["query"],
                    section=row["section"],
                    content=row["content"],
                    answer=row["answer"],
                    label=row["label"],
                    note=row.get("note", ""),
                )
            )
    return items


def run_eval(items: list[EvalItem], verify: bool) -> list[Scored]:
    """Score every item exactly the way production would, with two
    deliberate differences.

    Verification runs against the item's *labeled* section, not whatever the
    retriever happens to return, so a retrieval miss can never be mistaken
    for a grounding failure -- the two signals stay independently
    measurable. Retrieval confidence still comes from the real retriever on
    the real query, so the composite is realistic; where the retriever's top
    section differs from the labeled one, the report says so.

    Verification is batched into one call per `config.VERIFY_BATCH_SIZE`
    items; production verifies one answer at a time.
    """
    retriever = TfidfRetriever()
    retrievals = [retriever.retrieve(item.query, top_k=3) for item in items]
    retrieval_scores = [confidence.retrieval_confidence(c) for c in retrievals]
    top_sections = [c[0].section if c else None for c in retrievals]

    if not verify:
        return [
            Scored(
                item=item,
                result=confidence.ConfidenceResult(
                    score=retrieval,
                    band=confidence.band_for(retrieval),
                    reason=config.CONFIDENCE_REASON_VERIFICATION_DISABLED,
                    retrieval_score=retrieval,
                ),
                retrieved_section=section,
            )
            for item, retrieval, section in zip(items, retrieval_scores, top_sections)
        ]

    batch = [(item.answer, item.section, item.content) for item in items]
    verdict_lists = llm_client.verify_groundings_batch(batch)

    scored = []
    for item, verdicts, retrieval, section in zip(
        items, verdict_lists, retrieval_scores, top_sections
    ):
        if verdicts is None:
            result = confidence.ConfidenceResult(
                score=0.0,
                band=config.CONFIDENCE_BAND_LOW_NAME,
                reason=config.CONFIDENCE_REASON_VERIFICATION_FAILED,
                retrieval_score=retrieval,
            )
        else:
            claims = confidence.validate_claims(verdicts, item.content)
            if not claims:
                result = confidence.not_applicable(
                    config.CONFIDENCE_REASON_NO_CLAIMS, retrieval_score=retrieval
                )
            else:
                grounding = sum(1 for c in claims if c.supported) / len(claims)
                score = min(retrieval, grounding)
                result = confidence.ConfidenceResult(
                    score=score,
                    band=confidence.band_for(score),
                    reason=config.CONFIDENCE_REASON_VERIFIED,
                    retrieval_score=retrieval,
                    grounding_score=grounding,
                    claims=claims,
                )
        scored.append(Scored(item=item, result=result, retrieved_section=section))
    return scored


def _stats(values: list[float]) -> str:
    if not values:
        return "| n/a | n/a | n/a | n/a |"
    return (
        f"| {statistics.mean(values):.3f} | {min(values):.3f} "
        f"| {max(values):.3f} | {len(values)} |"
    )


def build_report(scored: list[Scored], verify: bool) -> str:
    lines: list[str] = []
    lines.append("# Confidence Scoring Evaluation")
    lines.append("")
    lines.append(
        f"Mode: {'composite (retrieval + grounding verification)' if verify else 'retrieval-only (--no-verify, 0 LLM calls)'}"
    )
    lines.append(f"Eval set: {len(scored)} labeled answers")
    lines.append(
        f"Bands: high >= {config.CONFIDENCE_BAND_HIGH}, "
        f"medium >= {config.CONFIDENCE_BAND_MEDIUM}, low below that"
    )
    if verify:
        calls = -(-len(scored) // config.VERIFY_BATCH_SIZE)
        lines.append(
            f"Verification cost: {calls} LLM call(s) "
            f"(batched at {config.VERIFY_BATCH_SIZE}/call; the per-turn path would cost {len(scored)})"
        )
    lines.append("")

    # -- Run validity ------------------------------------------------------
    # Without this, a run where every verification call failed reports
    # "0 fabricated answers reached the high band" -- trivially true, because
    # nothing was scored at all. That is precisely the decoration this eval
    # exists to rule out, so an incomplete run has to say so before any
    # number below it is read.
    unverified = [
        s
        for s in scored
        if s.result.reason
        in (
            config.CONFIDENCE_REASON_VERIFICATION_FAILED,
            config.CONFIDENCE_REASON_VERIFICATION_DISABLED,
        )
    ]
    if verify and unverified:
        lines.append(
            f"> **INCOMPLETE RUN — {len(unverified)} of {len(scored)} items were never "
            "verified** (provider error or unparseable response). Grounding was not "
            "measured for those items, so every figure below understates them and the "
            "safety metric is not meaningful. Re-run before citing these results."
        )
        lines.append("")

    # -- The headline safety metric --------------------------------------
    lines.append("## Does confidence separate grounded from fabricated?")
    lines.append("")
    fabricated_high = [
        s
        for s in scored
        if s.item.label == "fabricated"
        and s.result.band == config.CONFIDENCE_BAND_HIGH_NAME
    ]
    means = {
        cat: [s.result.score for s in scored if s.item.label == cat and s.result.score is not None]
        for cat in CATEGORIES
    }
    lines.append(
        f"- **Fabricated answers reaching the `high` band: {len(fabricated_high)}** "
        "(target: 0 -- this is the failure that would make the number unsafe to display)"
    )
    if means["grounded"] and means["fabricated"]:
        gap = statistics.mean(means["grounded"]) - statistics.mean(means["fabricated"])
        lines.append(f"- Separation, mean(grounded) - mean(fabricated): **{gap:+.3f}**")
    ordered = [
        statistics.mean(means[cat]) if means[cat] else float("nan") for cat in CATEGORIES
    ]
    monotonic = all(a <= b for a, b in zip(ordered, ordered[1:]))
    lines.append(
        f"- Means ordered fabricated <= partially_grounded <= grounded: "
        f"**{'yes' if monotonic else 'NO'}**"
    )
    lines.append("")

    # -- Composite score by category --------------------------------------
    lines.append("## Composite confidence by category")
    lines.append("")
    lines.append("| Label | Mean | Min | Max | Scored |")
    lines.append("|---|---|---|---|---|")
    for cat in CATEGORIES:
        lines.append(f"| {cat} " + _stats(means[cat]))
    lines.append("")

    # -- Sub-scores, so it's visible which signal did the work -------------
    lines.append("## Sub-scores by category")
    lines.append("")
    lines.append("Which of the two signals is actually discriminating.")
    lines.append("")
    lines.append("| Label | Mean retrieval | Mean grounding |")
    lines.append("|---|---|---|")
    for cat in CATEGORIES:
        rs = [
            s.result.retrieval_score
            for s in scored
            if s.item.label == cat and s.result.retrieval_score is not None
        ]
        gs = [
            s.result.grounding_score
            for s in scored
            if s.item.label == cat and s.result.grounding_score is not None
        ]
        rtxt = f"{statistics.mean(rs):.3f}" if rs else "n/a"
        gtxt = f"{statistics.mean(gs):.3f}" if gs else "n/a"
        lines.append(f"| {cat} | {rtxt} | {gtxt} |")
    lines.append("")

    # -- Band distribution -------------------------------------------------
    lines.append("## Band distribution")
    lines.append("")
    lines.append("| Label | " + " | ".join(BANDS) + " |")
    lines.append("|" + "---|" * (len(BANDS) + 1))
    for cat in CATEGORIES:
        counts = Counter(s.result.band for s in scored if s.item.label == cat)
        lines.append(f"| {cat} | " + " | ".join(str(counts.get(b, 0)) for b in BANDS) + " |")
    lines.append("")

    # -- Verifier integrity ------------------------------------------------
    if verify:
        downgraded = [
            (s, c)
            for s in scored
            for c in s.result.claims
            if c.evidence and not c.evidence_found
        ]
        lines.append("## Verifier integrity (evidence-span check)")
        lines.append("")
        lines.append(
            "Claims where the verifier cited evidence that is **not** a verbatim span "
            "of the source. Each was downgraded to unsupported in code. A non-zero "
            "count here is the check earning its keep -- it is the verifier "
            "manufacturing its own support, caught mechanically."
        )
        lines.append("")
        lines.append(f"- Claims downgraded: **{len(downgraded)}**")
        total_claims = sum(len(s.result.claims) for s in scored)
        lines.append(f"- Total claims extracted: {total_claims}")
        lines.append("")
        if downgraded:
            lines.append("| Claim | Cited evidence (not found in source) |")
            lines.append("|---|---|")
            for _, c in downgraded[:15]:
                lines.append(f"| {c.claim} | {c.evidence} |")
            lines.append("")

    # -- Per-item detail ---------------------------------------------------
    lines.append("## Per-item detail")
    lines.append("")
    lines.append("| Label | Query | Confidence | Band | Retrieval | Grounding | Claims | Reason |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cat in CATEGORIES:
        for s in [x for x in scored if x.item.label == cat]:
            r = s.result
            score = f"{r.score:.3f}" if r.score is not None else "n/a"
            rs = f"{r.retrieval_score:.3f}" if r.retrieval_score is not None else "n/a"
            gs = f"{r.grounding_score:.3f}" if r.grounding_score is not None else "n/a"
            claims = f"{r.supported_claims}/{len(r.claims)}" if r.claims else "-"
            lines.append(
                f"| {s.item.label} | {s.item.query} | {score} | {r.band} "
                f"| {rs} | {gs} | {claims} | {r.reason} |"
            )
    lines.append("")

    # -- Retrieval/label mismatches, so the confound is visible ------------
    mismatched = [s for s in scored if s.retrieved_section != s.item.section]
    if mismatched:
        lines.append("## Retrieval / labeled-section mismatches")
        lines.append("")
        lines.append(
            "Items where the live retriever's top section differs from the section "
            "the answer was verified against. Verification always uses the labeled "
            "section, so grounding is unaffected; this only explains an unexpectedly "
            "low retrieval sub-score."
        )
        lines.append("")
        lines.append("| Query | Labeled | Retrieved |")
        lines.append("|---|---|---|")
        for s in mismatched:
            lines.append(f"| {s.item.query} | {s.item.section} | {s.retrieved_section} |")
        lines.append("")

    return "\n".join(lines)


# Reproduces the measurement behind the separation-ratio choice. Queries span
# all 7 KB sections (answerable), plus clearly out-of-scope and genuinely
# ambiguous ones.
#
# The two queries the retriever gets WRONG are deliberately kept in this list
# rather than trimmed to make the accuracy figure look better: "Who is
# eligible to apply?" and "Can I switch teams?" are both answerable from the
# KB and both fail. They are what the accuracy line below is honest about,
# and the first is recorded in deferred-work.md as a Slice 1 retrieval-recall
# gap (the stemmer maps neither "eligible" to "eligibility" nor "apply" to
# "application"). Neither weakens the separation argument -- the point being
# measured is that magnitude does not predict correctness, and a probe that
# hid its own failures could not support that claim.
_PROBE_ANSWERABLE = [
    ("Who leads the AIML team?", "Teams"),
    ("List all the teams", "Teams"),
    ("Who is the Cloud team lead?", "Teams"),
    ("Tell me about the Design team", "Teams"),
    ("When is HackFest 2025?", "Events"),
    ("What events are upcoming?", "Events"),
    ("Is Flutter Forward completed?", "Events"),
    ("What is the recruitment process?", "Recruitment"),
    ("When does the recruitment window open?", "Recruitment"),
    ("What is the interview length?", "Recruitment"),
    ("Who is eligible to apply?", "Recruitment"),
    ("What are the club rules?", "Rules"),
    ("How many events per month must I attend?", "Rules"),
    ("What happens if I am inactive for two months?", "Rules"),
    ("Can I switch teams?", "Rules"),
    ("Who is the president?", "Contacts"),
    ("What is the general contact email?", "Contacts"),
    ("What awards has the club won?", "Achievements"),
    ("How many open-source projects does the club have?", "Achievements"),
    ("When was GDG On Campus founded?", "Intro"),
    ("How many members are in the community?", "Intro"),
]
_PROBE_OUT_OF_SCOPE = [
    "What's the club's budget?",
    "Can you help me with my calculus homework?",
    "What's the weather today?",
    "Who won the cricket match yesterday?",
]
_PROBE_AMBIGUOUS = ["cloud", "design", "2025", "workshops"]


def run_probe() -> str:
    """Print the raw-magnitude vs. separation comparison. No LLM calls."""
    retriever = TfidfRetriever()
    lines = ["# Retrieval signal probe", ""]
    lines.append(
        "Why `confidence.retrieval_confidence` uses the separation ratio "
        "`(top1 - top2) / top1` rather than the raw top-1 cosine magnitude "
        "suggested by requirements.md §5b."
    )
    lines.append("")
    lines.append("| Group | Query | Top-1 section | Correct | Raw magnitude | Separation |")
    lines.append("|---|---|---|---|---|---|")

    seps = []
    for query, expected in _PROBE_ANSWERABLE:
        candidates = retriever.retrieve(query, top_k=3)
        top = candidates[0]
        sep = confidence.retrieval_confidence(candidates)
        # Only queries that clear the refusal gate get scored for confidence
        # at all, so only those belong in the separation range quoted below.
        if top.score >= config.RETRIEVAL_THRESHOLD:
            seps.append(sep)
        ok = "yes" if top.section == expected else "**no**"
        lines.append(
            f"| answerable | {query} | {top.section} | {ok} | {top.score:.4f} | {sep:.4f} |"
        )
    for query in _PROBE_OUT_OF_SCOPE:
        candidates = retriever.retrieve(query, top_k=3)
        lines.append(
            f"| out-of-scope | {query} | {candidates[0].section} | - "
            f"| {candidates[0].score:.4f} | {confidence.retrieval_confidence(candidates):.4f} |"
        )
    for query in _PROBE_AMBIGUOUS:
        candidates = retriever.retrieve(query, top_k=3)
        lines.append(
            f"| ambiguous | {query} | {candidates[0].section} | - "
            f"| {candidates[0].score:.4f} | {confidence.retrieval_confidence(candidates):.4f} |"
        )

    # Score only the queries that clear the refusal gate: one below-threshold
    # query is refused rather than answered wrongly, so counting it as a
    # retrieval error would misstate what the gate does.
    answered = [
        (query, expected, retriever.retrieve(query, top_k=1)[0])
        for query, expected in _PROBE_ANSWERABLE
    ]
    above = [(q, e, t) for q, e, t in answered if t.score >= config.RETRIEVAL_THRESHOLD]
    refused = [(q, e, t) for q, e, t in answered if t.score < config.RETRIEVAL_THRESHOLD]
    correct = [(q, e, t) for q, e, t in above if t.section == e]
    above_mags = [t.score for _, _, t in above]

    lines.append("")
    lines.append(
        f"Of the {len(_PROBE_ANSWERABLE)} answerable queries, {len(refused)} score below "
        f"`RETRIEVAL_THRESHOLD` ({config.RETRIEVAL_THRESHOLD}) and are refused rather than "
        f"answered. Among the {len(above)} that are answered, top-1 section accuracy is "
        f"**{len(correct)}/{len(above)}** across a raw-magnitude range of "
        f"{min(above_mags):.3f}-{max(above_mags):.3f}."
    )
    lines.append("")
    lines.append(
        "That range is the point: magnitude varies more than fourfold among answered "
        "queries without tracking correctness, so it cannot carry a graded confidence "
        "and is used only as the binary refusal gate. Separation spans "
        f"{min(seps):.3f}-{max(seps):.3f} on answerable queries while collapsing toward 0 "
        "on the ambiguous ones, which is the discrimination the confidence score needs."
    )
    if len(correct) != len(above):
        lines.append("")
        lines.append(
            "Answered-but-wrong (kept in this probe rather than trimmed; see "
            "deferred-work.md):"
        )
        for query, expected, top in above:
            if top.section != expected:
                lines.append(
                    f"- \"{query}\" -> {top.section}, expected {expected} "
                    f"(magnitude {top.score:.4f})"
                )
    if refused:
        lines.append("")
        lines.append("Answerable but refused (Slice 1 retrieval-recall gap):")
        for query, expected, top in refused:
            lines.append(
                f"- \"{query}\" -> magnitude {top.score:.4f}, expected {expected}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip grounding verification entirely (0 LLM calls). Scores the "
        "retrieval signal alone -- useful for iterating without spending quota, "
        "but note this cannot measure grounding, which is the point of the eval.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Print the retrieval-signal measurement (raw magnitude vs. "
        "separation ratio) and exit. Makes no LLM calls.",
    )
    args = parser.parse_args()

    if args.probe:
        print(run_probe())
        return

    verify = not args.no_verify
    if verify:
        missing_var = llm_client.missing_api_key_var()
        if missing_var:
            print(
                f"WARNING: {missing_var} is not set, so every item will report "
                "verification_failed. Pass --no-verify to score the retrieval signal "
                f"alone, or set {missing_var} in .env.",
                file=sys.stderr,
            )

    items = load_eval_set(args.eval_set)
    scored = run_eval(items, verify=verify)
    report = build_report(scored, verify=verify)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
