# Confidence Scoring Evaluation

Mode: composite (retrieval + grounding verification)
Eval set: 24 labeled answers
Bands: high >= 0.85, medium >= 0.5, low below that
Verification cost: 4 LLM call(s) (batched at 6/call; the per-turn path would cost 24)

> **INCOMPLETE RUN — 24 of 24 items were never verified** (provider error or unparseable response). Grounding was not measured for those items, so every figure below understates them and the safety metric is not meaningful. Re-run before citing these results.

## Does confidence separate grounded from fabricated?

- **Fabricated answers reaching the `high` band: 0** (target: 0 -- this is the failure that would make the number unsafe to display)
- Separation, mean(grounded) - mean(fabricated): **+0.000**
- Means ordered fabricated <= partially_grounded <= grounded: **yes**

## Composite confidence by category

| Label | Mean | Min | Max | Scored |
|---|---|---|---|---|
| fabricated | 0.000 | 0.000 | 0.000 | 8 |
| partially_grounded | 0.000 | 0.000 | 0.000 | 8 |
| grounded | 0.000 | 0.000 | 0.000 | 8 |

## Sub-scores by category

Which of the two signals is actually discriminating.

| Label | Mean retrieval | Mean grounding |
|---|---|---|
| fabricated | 0.444 | n/a |
| partially_grounded | 0.625 | n/a |
| grounded | 0.661 | n/a |

## Band distribution

| Label | high | medium | low | not_applicable |
|---|---|---|---|---|
| fabricated | 0 | 0 | 8 | 0 |
| partially_grounded | 0 | 0 | 8 | 0 |
| grounded | 0 | 0 | 8 | 0 |

## Verifier integrity (evidence-span check)

Claims where the verifier cited evidence that is **not** a verbatim span of the source. Each was downgraded to unsupported in code. A non-zero count here is the check earning its keep -- it is the verifier manufacturing its own support, caught mechanically.

- Claims downgraded: **0**
- Total claims extracted: 0

## Per-item detail

| Label | Query | Confidence | Band | Retrieval | Grounding | Claims | Reason |
|---|---|---|---|---|---|---|---|
| fabricated | Who leads the Robotics team? | 0.000 | low | 0.877 | n/a | - | verification_failed |
| fabricated | What is the club's phone number? | 0.000 | low | 0.000 | n/a | - | verification_failed |
| fabricated | When is the AI Summit? | 0.000 | low | 0.000 | n/a | - | verification_failed |
| fabricated | What is the attendance policy during exams? | 0.000 | low | 0.000 | n/a | - | verification_failed |
| fabricated | How many members does the club have? | 0.000 | low | 0.000 | n/a | - | verification_failed |
| fabricated | What is the interview format? | 0.000 | low | 1.000 | n/a | - | verification_failed |
| fabricated | How many hackathons has the club won? | 0.000 | low | 1.000 | n/a | - | verification_failed |
| fabricated | What does the AIML team work on? | 0.000 | low | 0.676 | n/a | - | verification_failed |
| partially_grounded | Who leads AIML and how big is the team? | 0.000 | low | 0.890 | n/a | - | verification_failed |
| partially_grounded | Tell me about HackFest. | 0.000 | low | 1.000 | n/a | - | verification_failed |
| partially_grounded | Who are the club officers? | 0.000 | low | 0.000 | n/a | - | verification_failed |
| partially_grounded | When does recruitment open? | 0.000 | low | 0.576 | n/a | - | verification_failed |
| partially_grounded | What are the membership rules? | 0.000 | low | 1.000 | n/a | - | verification_failed |
| partially_grounded | Tell me about the club. | 0.000 | low | 0.000 | n/a | - | verification_failed |
| partially_grounded | What awards has the club won? | 0.000 | low | 1.000 | n/a | - | verification_failed |
| partially_grounded | What events are coming up? | 0.000 | low | 0.532 | n/a | - | verification_failed |
| grounded | Who leads the AIML team? | 0.000 | low | 0.890 | n/a | - | verification_failed |
| grounded | Who leads the Cloud team? | 0.000 | low | 0.886 | n/a | - | verification_failed |
| grounded | How many teams does the club have? | 0.000 | low | 0.519 | n/a | - | verification_failed |
| grounded | When is HackFest 2025? | 0.000 | low | 0.542 | n/a | - | verification_failed |
| grounded | How many events must I attend per month? | 0.000 | low | 0.452 | n/a | - | verification_failed |
| grounded | Who is the president and what is their email? | 0.000 | low | 1.000 | n/a | - | verification_failed |
| grounded | What is the recruitment process? | 0.000 | low | 1.000 | n/a | - | verification_failed |
| grounded | What has the club achieved? | 0.000 | low | 0.000 | n/a | - | verification_failed |

## Retrieval / labeled-section mismatches

Items where the live retriever's top section differs from the section the answer was verified against. Verification always uses the labeled section, so grounding is unaffected; this only explains an unexpectedly low retrieval sub-score.

| Query | Labeled | Retrieved |
|---|---|---|
| What has the club achieved? | Achievements | Intro |
| Who are the club officers? | Contacts | Intro |
| What is the club's phone number? | Contacts | Intro |
| When is the AI Summit? | Events | Intro |
| What is the attendance policy during exams? | Rules | Intro |
| How many hackathons has the club won? | Achievements | Intro |
