# Club FAQ Assistant

An AI chatbot for GDG On Campus that answers questions strictly from a fixed club knowledge base, holds context across turns, classifies what the user is asking for, scores its own confidence, and completes simple actions like event registration. A dashboard reads the per-turn logs so the whole thing can be verified end to end.

Built for the GDG AI/ML team recruitment task (Round 2).

---

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Add a Gemini API key to `.env` (get one at [aistudio.google.com](https://aistudio.google.com)):

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

Run the chatbot:

```bash
uv run python -m backend.cli
```

Run the dashboard (separate terminal, works alongside the CLI):

```bash
uv run streamlit run backend/dashboard.py
```

Run the tests (no API key needed — the LLM is mocked throughout):

```bash
uv run pytest -q
```

**Quota note.** The Gemini free tier allows 5 requests/minute and 20 requests/day per project. A single multi-turn question can cost up to 3 calls (rewrite + generate + grounding verification), so a testing session can exhaust the daily cap in roughly 7 turns. Set `VERIFY_GROUNDING=0` to disable grounding verification and cut one call per turn. Rate-limit errors are reported distinctly from real failures, so "out of quota" never reads as "the app is broken."

---

## How a turn works

```
user message
  │
  ├─ 1. Intent classification    rules first, LLM only if rules abstain
  │
  ├─ 2. Action routing           if action_request or mid-action → state machine
  │
  ├─ 3. Query rewriting          resolve pronouns/ellipsis against history
  │
  ├─ 4. Retrieval                TF-IDF cosine over 7 KB sections
  │
  ├─ 5. Refusal gate             below threshold → refuse, no LLM call
  │
  ├─ 6. Generation               answer from retrieved section only
  │
  ├─ 7. Confidence               min(separation ratio, grounding verification)
  │
  └─ 8. Logging                  append to JSONL for the dashboard
```

| Module | Responsibility |
|---|---|
| `backend/kb_data.py` | The 7 knowledge base sections; the only source of truth |
| `backend/retrieval.py` | TF-IDF vectorization, cosine ranking. Scores only — makes no refusal decision |
| `backend/intent.py` | Rule layer + orchestration for the 5 intent labels |
| `backend/confidence.py` | Separation ratio, evidence-span validation, composite scoring |
| `backend/actions.py` | Action state machine, slot-filling, KB validation, persistence |
| `backend/qa.py` | Orchestrates a turn end to end |
| `backend/turn_log.py` | Append-only per-turn JSONL log |
| `backend/dashboard.py` | Streamlit dashboard, read-only |

---

## Key design decisions

### Query rewriting happens before retrieval, not after

TF-IDF matches words, and "it" doesn't carry any. "Who leads it?" scores 0.107 against the Teams section — below the 0.15 refusal threshold — so the bot would refuse a question it could easily answer. Rewrite it to "Who leads the AIML team?" first and the score jumps to 0.693.

I considered just handing the LLM the raw conversation history at generation time instead, but that doesn't fix the actual problem: by the time the LLM ever sees the question, retrieval has already fetched the wrong section, or nothing at all. The rewrite has to happen upstream of retrieval, not downstream of it. That's one extra API call on every follow-up, and I think it's worth it.

### Intent classification is hybrid, not pure LLM

Five labels: `faq`, `event_inquiry`, `action_request`, `out_of_scope`, `greeting`.

Rules run first, checked in a fixed priority order, and the important part is that a rule can abstain instead of guessing. If a pattern matches but a competing signal also fires, the rule backs off and lets the LLM decide. "Is HackFest still open for registration?" resolves cleanly by rule as `event_inquiry`, but "Can I still sign up for HackFest?" trips a weak action cue too, so the rule punts.

`out_of_scope` is never assigned by rule. That category is unbounded — weather, homework, random trivia — and no fixed keyword list was ever going to cover it without also being a precision risk. It's always LLM-resolved.

End result: 58.9% of traffic gets classified by rule at zero API cost, and overall accuracy across the eval set is 98.2%.

### Retrieval confidence uses separation ratio, not raw similarity

My first instinct was to just normalize the raw TF-IDF cosine score into a confidence value. Measuring it before committing to that plan is what saved me from shipping it.

Of 21 answerable queries across all 7 sections, 19 clear the refusal threshold and get answered — and among those, top-1 section accuracy is 18/19 across a magnitude range from 0.168 to 0.712. "When is HackFest 2025?" scores 0.168 and is exactly as correct as "Who is the Cloud team lead?" at 0.712. The magnitude tracks query length and term-overlap density, not whether the match is actually good. Any ramp wide enough to act as a real gradient would have banded plenty of correct short queries as low-confidence.

So instead I use the **separation ratio**: `(top1 - top2) / top1` — how far ahead the best-matching section is from its runner-up. That's a much closer proxy for "did we actually find *the* right section," it's bounded [0, 1] by construction, and it needs no tuned constant.

Raw magnitude keeps doing the one job it's actually good at: the binary refusal gate. Magnitude decides *whether* to answer; separation decides *how confidently*.

*Limitation:* on a 7-section KB, separation runs high the moment a query hits a section-unique term. On a bigger, more overlapping corpus this signal would compress and need re-measuring — I wouldn't trust it blindly at scale.

### Confidence is the minimum of two signals, not their average

Final confidence is `min(retrieval_confidence, grounding_confidence)`.

The grounding signal comes from a second LLM call — an adversarial auditor that defaults to UNSUPPORTED, breaks the answer into atomic claims, and checks each one against the retrieved text. A SUPPORTED verdict only counts if the verifier also quotes an evidence span, and that span has to be verbatim present in the source after normalization — checked mechanically in code, not just trusted because the model said so. If the span can't be found, the claim gets downgraded. The LLM absorbs paraphrase; the substring check stops it from inventing its own support.

I used `min()` instead of a weighted average on purpose: a strong retrieval score should never be able to paper over a badly grounded answer, since that's exactly the failure this whole system is meant to prevent. Yes, this under-reports confidence when both signals are moderate for unrelated, benign reasons — I'm fine with that trade. For a bot that isn't allowed to fabricate, a false *low* is cheap and a false *high* is expensive.

A refusal is scored `not_applicable`, not zero. A correct refusal is the single most trustworthy thing this bot does — banding it `low` would get that exactly backwards.

---

## Evaluation

### Intent classification

56 hand-labeled queries across all five categories, including deliberately ambiguous cases.

| Path | Count | Share | Accuracy |
|---|---|---|---|
| rule | 33 | 58.9% | 100.0% |
| llm | 23 | 41.1% | 95.7% |
| **overall** | **56** | — | **98.2%** |

Macro F1: **0.982**. One miss: "Does the club have a Discord server?" (labeled `out_of_scope`, classified `faq`) — a defensible call on a genuinely borderline item.

Reproduce: `uv run python scripts/eval_intents.py` (1 API call, batched) or `--rules-only` for a zero-call run.

### Confidence scoring

24 labeled answers across three categories: grounded, partially grounded, fabricated.

| Label | Mean confidence | Mean retrieval | Mean grounding |
|---|---|---|---|
| fabricated | 0.000 | 0.444 | 0.000 |
| partially_grounded | 0.430 | 0.625 | 0.625 |
| grounded | 0.661 | 0.661 | 1.000 |

| Label | high | medium | low |
|---|---|---|---|
| fabricated | **0** | 0 | 8 |
| partially_grounded | 0 | 6 | 2 |
| grounded | 4 | 2 | 2 |

- **Fabricated answers reaching the high band: 0**
- Separation, mean(grounded) − mean(fabricated): **+0.661**
- Category means correctly ordered

The sub-score columns show which signal is doing the work. Fabricated answers averaged **0.444 on retrieval but 0.000 on grounding** — retrieval often found a plausible section, and grounding verification is what caught the fabrication.

### Ablation: is the second LLM call worth it?

Running the same eval on the retrieval signal alone (`--no-verify`, 0 API calls): **3 fabricated answers reach the high band**, with a fabricated band distribution (3/1/4) barely distinguishable from grounded (4/2/2).

That's the justification for the grounding verification call. Without it, confidence doesn't discriminate.

Reproduce: `uv run python scripts/eval_grounding.py`.

### Test suite

223 tests passing. All LLM calls are mocked — the suite makes no network calls and requires no API key, enforced structurally in `conftest.py` rather than by convention.

---

## Known limitations

**Conservative confidence under-reports on correct answers.** Four grounded answers were banded low or medium despite 1.000 grounding — for example "What has the club achieved?" scored 0.000 retrieval confidence. This traces to retrieval returning the Intro section instead of the correct one on six eval items: Intro shares vocabulary with everything on a 7-section KB, and TF-IDF has no way to prefer specificity. Since `min()` is bounded by the weaker signal, a retrieval miss caps confidence even when grounding is perfect. This is the accepted cost of the conservative design, but on a larger KB the retrieval signal itself would need improving — semantic embeddings rather than pure lexical matching.

**Rule-path accuracy is 100% by construction, not by luck.** Rules only fire on patterns designed to be unambiguous, so a perfect rule-path score reflects that design choice rather than an independent measurement. The meaningful number is overall accuracy.

**The eval sets were reviewed but not independently authored.** Labels were checked by hand, but the same process that designed the rules also drafted the eval items. A genuinely independent labeler would be a stronger validation.

**Quota constrains live testing.** The free tier's 20 requests/day makes repeated live evaluation impractical. Both eval scripts support zero-call modes, and batching keeps full runs cheap (the 56-item intent eval costs 1 call; the 24-item grounding eval costs 4).

**Ragas and similar RAG evaluation frameworks** would be an obvious next step for standardized faithfulness and answer-relevancy metrics. The custom verifier here was chosen for transparency — every step of the scoring is inspectable — but a standard framework would make results comparable across projects.

---

## Repository layout

```
backend/          application code
data/             KB eval sets, persisted logs, eval results
scripts/          evaluation scripts
tests/            223 tests, no network access
requirements.md   the authoritative spec this was built against
_bmad-output/     per-slice implementation specs
```

The project was built in six incremental slices — knowledge base and grounded Q&A, multi-turn memory, intent classification, confidence scoring, agentic actions, and the dashboard — with each slice committed separately. The commit history reflects that progression.
