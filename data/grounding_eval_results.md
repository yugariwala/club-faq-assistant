# Confidence Scoring Evaluation

Mode: composite (retrieval + grounding verification)
Eval set: 24 labeled answers
Bands: high >= 0.85, medium >= 0.5, low below that
Verification cost: 4 LLM call(s) (batched at 6/call; the per-turn path would cost 24)

## Does confidence separate grounded from fabricated?

- **Fabricated answers reaching the `high` band: 0** (target: 0 -- this is the failure that would make the number unsafe to display)
- Separation, mean(grounded) - mean(fabricated): **+0.661**
- Means ordered fabricated <= partially_grounded <= grounded: **yes**

## Composite confidence by category

| Label | Mean | Min | Max | Scored |
|---|---|---|---|---|
| fabricated | 0.000 | 0.000 | 0.000 | 8 |
| partially_grounded | 0.430 | 0.000 | 0.667 | 8 |
| grounded | 0.661 | 0.000 | 1.000 | 8 |

## Sub-scores by category

Which of the two signals is actually discriminating.

| Label | Mean retrieval | Mean grounding |
|---|---|---|
| fabricated | 0.444 | 0.000 |
| partially_grounded | 0.625 | 0.625 |
| grounded | 0.661 | 1.000 |

## Band distribution

| Label | high | medium | low | not_applicable |
|---|---|---|---|---|
| fabricated | 0 | 0 | 8 | 0 |
| partially_grounded | 0 | 6 | 2 | 0 |
| grounded | 4 | 2 | 2 | 0 |

## Verifier integrity (evidence-span check)

Claims where the verifier cited evidence that is **not** a verbatim span of the source. Each was downgraded to unsupported in code. A non-zero count here is the check earning its keep -- it is the verifier manufacturing its own support, caught mechanically.

- Claims downgraded: **0**
- Total claims extracted: 54

## Per-item detail

| Label | Query | Confidence | Band | Retrieval | Grounding | Claims | Reason |
|---|---|---|---|---|---|---|---|
| fabricated | Who leads the Robotics team? | 0.000 | low | 0.877 | 0.000 | 0/1 | verified |
| fabricated | What is the club's phone number? | 0.000 | low | 0.000 | 0.000 | 0/1 | verified |
| fabricated | When is the AI Summit? | 0.000 | low | 0.000 | 0.000 | 0/2 | verified |
| fabricated | What is the attendance policy during exams? | 0.000 | low | 0.000 | 0.000 | 0/2 | verified |
| fabricated | How many members does the club have? | 0.000 | low | 0.000 | 0.000 | 0/2 | verified |
| fabricated | What is the interview format? | 0.000 | low | 1.000 | 0.000 | 0/3 | verified |
| fabricated | How many hackathons has the club won? | 0.000 | low | 1.000 | 0.000 | 0/2 | verified |
| fabricated | What does the AIML team work on? | 0.000 | low | 0.676 | 0.000 | 0/3 | verified |
| partially_grounded | Who leads AIML and how big is the team? | 0.500 | medium | 0.890 | 0.500 | 1/2 | verified |
| partially_grounded | Tell me about HackFest. | 0.667 | medium | 1.000 | 0.667 | 2/3 | verified |
| partially_grounded | Who are the club officers? | 0.000 | low | 0.000 | 0.667 | 2/3 | verified |
| partially_grounded | When does recruitment open? | 0.576 | medium | 0.576 | 0.667 | 2/3 | verified |
| partially_grounded | What are the membership rules? | 0.667 | medium | 1.000 | 0.667 | 2/3 | verified |
| partially_grounded | Tell me about the club. | 0.000 | low | 0.000 | 0.667 | 2/3 | verified |
| partially_grounded | What awards has the club won? | 0.500 | medium | 1.000 | 0.500 | 1/2 | verified |
| partially_grounded | What events are coming up? | 0.532 | medium | 0.532 | 0.667 | 2/3 | verified |
| grounded | Who leads the AIML team? | 0.890 | high | 0.890 | 1.000 | 1/1 | verified |
| grounded | Who leads the Cloud team? | 0.886 | high | 0.886 | 1.000 | 1/1 | verified |
| grounded | How many teams does the club have? | 0.519 | medium | 0.519 | 1.000 | 1/1 | verified |
| grounded | When is HackFest 2025? | 0.542 | medium | 0.542 | 1.000 | 2/2 | verified |
| grounded | How many events must I attend per month? | 0.452 | low | 0.452 | 1.000 | 1/1 | verified |
| grounded | Who is the president and what is their email? | 1.000 | high | 1.000 | 1.000 | 2/2 | verified |
| grounded | What is the recruitment process? | 1.000 | high | 1.000 | 1.000 | 5/5 | verified |
| grounded | What has the club achieved? | 0.000 | low | 0.000 | 1.000 | 3/3 | verified |

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
