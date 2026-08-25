# Confidence Scoring Evaluation

Mode: retrieval-only (--no-verify, 0 LLM calls)
Eval set: 24 labeled answers
Bands: high >= 0.85, medium >= 0.5, low below that

## Does confidence separate grounded from fabricated?

- **Fabricated answers reaching the `high` band: 3** (target: 0 -- this is the failure that would make the number unsafe to display)
- Separation, mean(grounded) - mean(fabricated): **+0.217**
- Means ordered fabricated <= partially_grounded <= grounded: **yes**

## Composite confidence by category

| Label | Mean | Min | Max | Scored |
|---|---|---|---|---|
| fabricated | 0.444 | 0.000 | 1.000 | 8 |
| partially_grounded | 0.625 | 0.000 | 1.000 | 8 |
| grounded | 0.661 | 0.000 | 1.000 | 8 |

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
| fabricated | 3 | 1 | 4 | 0 |
| partially_grounded | 4 | 2 | 2 | 0 |
| grounded | 4 | 2 | 2 | 0 |

## Per-item detail

| Label | Query | Confidence | Band | Retrieval | Grounding | Claims | Reason |
|---|---|---|---|---|---|---|---|
| fabricated | Who leads the Robotics team? | 0.877 | high | 0.877 | n/a | - | verification_disabled |
| fabricated | What is the club's phone number? | 0.000 | low | 0.000 | n/a | - | verification_disabled |
| fabricated | When is the AI Summit? | 0.000 | low | 0.000 | n/a | - | verification_disabled |
| fabricated | What is the attendance policy during exams? | 0.000 | low | 0.000 | n/a | - | verification_disabled |
| fabricated | How many members does the club have? | 0.000 | low | 0.000 | n/a | - | verification_disabled |
| fabricated | What is the interview format? | 1.000 | high | 1.000 | n/a | - | verification_disabled |
| fabricated | How many hackathons has the club won? | 1.000 | high | 1.000 | n/a | - | verification_disabled |
| fabricated | What does the AIML team work on? | 0.676 | medium | 0.676 | n/a | - | verification_disabled |
| partially_grounded | Who leads AIML and how big is the team? | 0.890 | high | 0.890 | n/a | - | verification_disabled |
| partially_grounded | Tell me about HackFest. | 1.000 | high | 1.000 | n/a | - | verification_disabled |
| partially_grounded | Who are the club officers? | 0.000 | low | 0.000 | n/a | - | verification_disabled |
| partially_grounded | When does recruitment open? | 0.576 | medium | 0.576 | n/a | - | verification_disabled |
| partially_grounded | What are the membership rules? | 1.000 | high | 1.000 | n/a | - | verification_disabled |
| partially_grounded | Tell me about the club. | 0.000 | low | 0.000 | n/a | - | verification_disabled |
| partially_grounded | What awards has the club won? | 1.000 | high | 1.000 | n/a | - | verification_disabled |
| partially_grounded | What events are coming up? | 0.532 | medium | 0.532 | n/a | - | verification_disabled |
| grounded | Who leads the AIML team? | 0.890 | high | 0.890 | n/a | - | verification_disabled |
| grounded | Who leads the Cloud team? | 0.886 | high | 0.886 | n/a | - | verification_disabled |
| grounded | How many teams does the club have? | 0.519 | medium | 0.519 | n/a | - | verification_disabled |
| grounded | When is HackFest 2025? | 0.542 | medium | 0.542 | n/a | - | verification_disabled |
| grounded | How many events must I attend per month? | 0.452 | low | 0.452 | n/a | - | verification_disabled |
| grounded | Who is the president and what is their email? | 1.000 | high | 1.000 | n/a | - | verification_disabled |
| grounded | What is the recruitment process? | 1.000 | high | 1.000 | n/a | - | verification_disabled |
| grounded | What has the club achieved? | 0.000 | low | 0.000 | n/a | - | verification_disabled |

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
