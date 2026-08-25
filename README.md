# GDG On Campus Club FAQ Assistant

Answers club questions strictly from a fixed KB (never fabricating facts the
retrieved section doesn't contain), carries context across turns, and tags
every message with an intent category.

- **Slice 1** — knowledge base & grounded Q&A.
- **Slice 2** — multi-turn memory (pronoun/ellipsis resolution across turns).
- **Slice 3** — hybrid intent classification (rules + LLM fallback).
- **Slice 4** — composite confidence scoring (retrieval separation + grounding verification).

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` and set:

- `LLM_PROVIDER` — `anthropic` or `gemini` (defaults to `gemini` if unset).
- `GEMINI_API_KEY` — required when `LLM_PROVIDER=gemini`. Get a free key at
  https://aistudio.google.com/apikey.
- `ANTHROPIC_API_KEY` — required when `LLM_PROVIDER=anthropic`.

Only one key is required, matching whichever provider is selected. `.env` is
loaded automatically (via `python-dotenv`) when the CLI starts.

**Free-tier quota:** a free Gemini key is limited to 5 requests/minute and
20 requests/day. If the CLI starts responding with a "temporarily
rate-limited or out of quota" message instead of real answers, that's this
limit, not a bug — wait for the quota window to reset or switch to a key
with more headroom. Per-turn cost:

| Turn | Calls | Made up of |
|---|---|---|
| Refusal (nothing in the KB matched) | **0** | never reaches the LLM at all |
| Grounded, first turn of a session | **2** | generate + verify |
| Grounded follow-up (has history) | **3** | rewrite + generate + verify |
| \+ ambiguous intent (rules abstain) | **+1** | LLM intent fallback |

Grounding verification (Slice 4) is the `verify` call and is the one you can
turn off: set `VERIFY_GROUNDING=0` to drop it, roughly doubling how many
grounded turns fit in a day. The cost is that confidence then falls back to
the retrieval signal alone — see [Confidence scoring](#confidence-scoring-slice-4).

## Run

```bash
uv run python -m backend.cli
```

Refusals (queries with no relevant KB match) never call the LLM, so the CLI
works without any API key for those.

## Tests

```bash
uv run pytest
```

The suite needs no API key and makes no network calls. That is enforced
structurally by `tests/conftest.py`, which blocks the provider client
constructors so any unmocked call fails locally and loudly, rather than
relying on every test remembering to patch the right entry point. The
convention alone was not sufficient: `classify_intent` and
`verify_grounding` both swallow provider failures by design, so a test that
forgot to patch one still passed while quietly opening a real connection and
spending quota.

## Intent classification (Slice 3)

Every user message is tagged with one of five categories -- `faq`,
`event_inquiry`, `action_request`, `out_of_scope`, `greeting` -- via a
hybrid classifier (`backend/intent.py`):

1. **Rule layer** — a fixed priority order of high-precision keyword/pattern
   rules (greeting → action_request → event_inquiry → faq). Each rule
   abstains (`None`) rather than guess; an abstained message falls through
   to the next rule and ultimately to the LLM. Rules never attempt
   `out_of_scope` -- that space is open-ended natural language, and a fixed
   keyword denylist would be wrong for anything not on it.
2. **LLM fallback** — a constrained single-label prompt (`llm_client.
   classify_intent`), validated against the 5-label enum with one retry on
   an unparseable response, falling back to `out_of_scope` if both attempts
   fail (never raises, never guesses an actionable label it isn't sure of).

Intent classifies the topic of the question, not whether the KB can answer
it -- "How many members are in Web Dev?" is `faq` intent (a team question)
even though that fact isn't in the KB; whether it's answerable is
`qa.answer_question`'s separate refusal/grounding concern.

Both the classified intent and which path resolved it (`rule` | `llm`) are
attached to every `AnswerResult` and logged per-turn.

### Evaluation

```bash
uv run python scripts/eval_intents.py            # full hybrid run (rules + batched LLM fallback)
uv run python scripts/eval_intents.py --rules-only # 0 API calls, iterate on rule patterns for free
```

Scores `data/intent_eval.jsonl` (56 hand-labeled queries, balanced across
all five categories, including deliberately ambiguous pairs like "Is
HackFest still open for registration?" vs. "Can I still sign up for
HackFest?"). Every rule-abstained item is batched into a single
`classify_intents_batch` call per run rather than one call each, to stay
within a limited daily Gemini quota. Results are written to
`data/intent_eval_results.md`.

**Latest results** (`LLM_PROVIDER=gemini`, `gemini-3.6-flash`, 1 LLM call for
the whole eval set):

| Path | Count | Fraction of scored traffic | Accuracy |
|---|---|---|---|
| rule | 33 | 58.9% | 100.0% |
| llm | 23 | 41.1% | 95.7% |
| **overall** | 56 | 100.0% | 98.2% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| action_request | 1.000 | 1.000 | 1.000 | 11 |
| event_inquiry | 1.000 | 1.000 | 1.000 | 12 |
| faq | 0.923 | 1.000 | 0.960 | 12 |
| greeting | 1.000 | 1.000 | 1.000 | 10 |
| out_of_scope | 1.000 | 0.909 | 0.952 | 11 |
| **macro avg** | 0.985 | 0.982 | 0.982 | 56 |
| **weighted avg** | 0.984 | 0.982 | 0.982 | 56 |

The one miss: "Does the club have a Discord server?" (gold `out_of_scope`)
was classified `faq` by the LLM fallback -- a defensible read given the
message is club-adjacent even though no KB section covers community
platforms. Rule-path accuracy is 100% by construction (rules only ever fire
on unambiguous patterns); see `data/intent_eval_results.md` for the full
confusion matrix and misclassification detail.

## Confidence scoring (Slice 4)

Every response carries a confidence indicator: a band (`high` / `medium` /
`low` / `not_applicable`) plus the raw score, both displayed and both logged
per-turn. It is the **Composite** option from `requirements.md` §5b — two
independent signals combined with `min()`:

```
final_confidence = min(retrieval_score, grounding_score)
```

`min()` rather than a weighted average because confidence is bounded by the
weakest link: a strong retrieval must never be able to average away an
ungrounded answer. That makes the number deliberately conservative — it
under-reports when both signals are moderate for unrelated benign reasons —
which is the right trade for a bot forbidden from fabricating, where a false
`high` costs far more than a false `medium`.

### Signal 1 — retrieval separation

`retrieval_score = (top1 − top2) / top1`, the fraction of the top KB
candidate's cosine score that the runner-up does not account for.

This is **not** the raw top-1 cosine that §5b's first row suggests, and the
reason is measured rather than asserted. Run
`uv run python scripts/eval_grounding.py --probe` (0 API calls; output in
`data/retrieval_signal_probe.md`):

| | raw top-1 cosine | separation |
|---|---|---|
| answerable & answered (n=19, all 7 sections) | 0.168 – 0.712 | 0.259 – 1.000 |
| out-of-scope (n=4) | 0.000 | 0.000 |
| ambiguous — "cloud", "design", "2025", "workshops" | 0.107 – 0.177 | 0.028, 0.028, 0.110, 0.260 |

Among the 19 answerable queries that clear the refusal gate, **top-1 section
accuracy is 18/19 across a magnitude range that varies more than fourfold**:
"When is HackFest 2025?" scores 0.168 and is exactly as correct as "Who is
the Cloud team lead?" at 0.712. Magnitude tracks query length and term
overlap, not match quality, so any rescaling wide enough to act as a
gradient would band correct short-query lookups `low`. Separation measures
discrimination directly — which is what "did we find *the* right section"
actually asks — and it needs no tuned constant: as a ratio of two cosines
with `0 ≤ top2 ≤ top1`, it is bounded to [0, 1] by construction.

The probe keeps its own failures in view rather than trimming to a nicer
number: one answered-but-wrong query ("Can I switch teams?" → Teams, should
be Rules) and two answerable queries refused outright ("What is the
interview length?" at 0.145, just under the 0.15 threshold, and "Who is
eligible to apply?" at 0.000). All three are Slice 1 retrieval-recall gaps,
recorded in `deferred-work.md`; none is a confidence-scoring defect, and the
refusals are reported honestly as `not_applicable` rather than as low
confidence.

Raw magnitude keeps the job it is already good at: the binary refusal gate
at `RETRIEVAL_THRESHOLD`. **Magnitude decides *whether* to answer;
separation decides *how confidently*.**

*Limitation:* on a 7-entry KB, separation runs high whenever a query
contains a term unique to one section. On a larger corpus with overlapping
documents it would compress and would need re-measuring.

### Signal 2 — grounding verification

After generation, a **fresh LLM call** re-reads the answer against the
retrieved section, decomposes it into atomic factual claims, and adjudicates
each one. `grounding_score = supported / total`.

**Claims are extracted by the LLM, fused with adjudication into one call.**
Structural sentence-splitting would be free but isn't atomic — "AIML is led
by Rahul Sharma and Web Dev by Priya Patel" is two claims in one sentence,
and scoring it as one supported claim hides a half-fabrication. Splitting
extraction and adjudication into two calls would double the quota cost for
no gain, since both steps read the same two inputs. What guards against the
model rubber-stamping its own output is not call separation but the fresh
call, the adversarial prompt (default verdict UNSUPPORTED), and the check
below.

**"Supported" means evidence-anchored, then mechanically validated.** Exact
string matching is too strict — "Rahul Sharma leads AIML" never literally
appears in "AIML (Lead: Rahul Sharma)". But a purely semantic verdict is
only as trustworthy as the verifier, which can hallucinate support as easily
as a generator hallucinates a fact. So the verifier may return SUPPORTED
only if it also emits an evidence span, and `confidence._evidence_supports`
then re-checks in code that the span is **verbatim present in the source**
(after NFKC / casefold / whitespace / dash normalization). A span that isn't
there is downgraded to UNSUPPORTED.

The model does the semantic work, so paraphrase isn't punished; the
substring check blocks invented support and — because the report prints
every downgraded claim — makes that failure mode auditable instead of
silent.

Aggregation is handled explicitly: "the club has 6 teams" is derived by
counting and appears nowhere in the source, but it is faithful, so it stays
SUPPORTED with the enumeration itself as the (verbatim) evidence span.

### The refusal case

A refusal is scored `not_applicable`, **not** `0.0`. Scoring it zero would
badge the most trustworthy thing this bot does as untrustworthy, and would
stack the low band with correct behavior in the eval. Scoring it `1.0` would
assert a verification that never ran. Confidence answers "how much should
you trust this claim about the club" — a response that makes no such claim
has no answer to give.

| State | Confidence | Band | Reason code |
|---|---|---|---|
| Below-threshold refusal | `None` | `not_applicable` | `refused` |
| LLM error / quota message | `None` | `not_applicable` | `llm_error` / `llm_quota` |
| Generated, 0 claims ("that isn't in the KB") | `None` | `not_applicable` | `no_claims` |
| Generated, verification disabled | retrieval only | banded | `verification_disabled` |
| Generated, **verifier call failed** | **`0.0`** | **`low`** | `verification_failed` |
| Generated and verified | `min(r, g)` | banded | `verified` |

The last two rows differ deliberately: **`not_applicable` when there are no
claims to check; `low` when there are claims we *failed* to check.** An
answer asserting facts we could not verify is precisely what this system
exists to be suspicious of. The `no_claims` row also removes the 0/0 case
rather than special-casing it.

### Bands

Named constants in `backend/config.py`, never inline numbers:
`CONFIDENCE_BAND_HIGH = 0.85`, `CONFIDENCE_BAND_MEDIUM = 0.50`.

0.85 is not arbitrary. Grounding is `k/n` with n typically 1–6 claims, so
0.85 is the threshold at which **a single unsupported claim in any answer of
six or fewer claims can no longer reach `high`** (4/5 = 0.80, 5/6 = 0.83 —
both `medium`). At 0.75, a 3-of-4 answer — one fabricated claim — would
badge `high`, which is exactly the failure the eval set exists to catch.

### Evaluation

```bash
uv run python scripts/eval_grounding.py              # composite; 4 LLM calls
uv run python scripts/eval_grounding.py --no-verify  # retrieval-only ablation; 0 calls
uv run python scripts/eval_grounding.py --probe      # retrieval signal measurement; 0 calls
```

`data/grounding_eval.jsonl` holds 24 hand-labeled answers, balanced 8/8/8
across `grounded` / `partially_grounded` / `fabricated`. The answers are
**fixed strings, not generated at eval time** — that makes the measurement
reproducible (no generation variance confounding it), costs no generation
quota, and is the only way to include deliberate fabrications, since a
correctly-working generator will not produce them on demand. Fabrications
are adversarial rather than obvious: a plausible-looking VP email, an
invented event built from a real team name, a 45-minute interview
contradicting the KB's 15.

Verification is batched at `VERIFY_BATCH_SIZE` (6) items per call, so the
24-item set costs 4 calls instead of 24.

#### Result: retrieval alone is not enough (ablation, complete)

`--no-verify`, 24 items, 0 API calls — full report in
`data/grounding_eval_results_retrieval_only.md`:

| Label | Mean | high | medium | low |
|---|---|---|---|---|
| fabricated | 0.444 | **3** | 1 | 4 |
| partially_grounded | 0.625 | 4 | 2 | 2 |
| grounded | 0.661 | 4 | 2 | 2 |

**Three fabricated answers reach the `high` band, and the fabricated band
distribution (3/1/4) is barely distinguishable from the grounded one
(4/2/2).** This is the ablation that justifies spending a second LLM call:
the retrieval signal knows only that a question matched a section cleanly,
and every one of these fabrications was written in answer to a question that
did. It cannot see that the answer then made things up. A confidence number
built on retrieval alone would be decoration here — precisely the outcome
this evaluation was built to detect.

#### Result: composite — not yet measured

The composite run is **blocked on the free-tier daily quota** (20
requests/day), which this session exhausted; only a Gemini key is configured
(`ANTHROPIC_API_KEY` is unset). Re-run after the quota window resets:

```bash
uv run python scripts/eval_grounding.py   # writes data/grounding_eval_results.md
```

No composite numbers are quoted here, and no results file is committed,
until that run completes. If a run's verification calls fail, the report
prints an **INCOMPLETE RUN** banner above every figure — without it, a
fully-failed run would report "0 fabricated answers reached `high`", which
is trivially true when nothing was scored at all, and would be the exact
decoration this evaluation exists to rule out.

What *is* already verified without the provider
(`tests/test_eval_grounding.py`): the scoring pipeline separates the three
categories correctly given a correct verifier, fabricated items never reach
`high`, cited evidence that isn't in the source is downgraded and counted,
and the INCOMPLETE RUN guard fires when it should. Those tests prove the
machinery; only the live run can measure how good the LLM verifier itself
is.
