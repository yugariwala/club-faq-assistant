# Club FAQ Assistant — Requirements

Authoritative spec for this build. Sourced from `Task 2_ Club FAQ Assistant.md`. Do not deviate from this document without updating it first.

---

## 1. Project Overview

Build an AI-powered chatbot for GDG On Campus that answers member questions strictly from a fixed club knowledge base, holds conversational context across turns, and can execute simple agentic actions (e.g. event registration) by collecting missing details, persisting them, and confirming completion. The bot must never fabricate information not present in the knowledge base. A companion dashboard exposes chat volume, intent breakdown, an actions log, and any queries the bot could not answer — so correctness and coverage can be verified end-to-end. Audience: club members asking about teams, events, recruitment, rules, contacts, and achievements; and whoever evaluates this submission for grounding, agentic behavior, and observability.

---

## 2. Knowledge Base

This is the bot's only source of truth. No external facts, no invented details. Reproduced verbatim from the task doc.

**Club Introduction** — GDG On Campus is a community of 150+ tech enthusiasts. Founded in 2022. Organizes workshops, hackathons, and speaker sessions.

**Teams** — AIML (Lead: Rahul Sharma), Web Dev (Lead: Priya Patel), App Dev (Lead: Arjun Mehta), Cloud (Lead: Sneha Gupta), Cybersecurity (Lead: Vikram Singh), Design (Lead: Ananya Reddy)

**Events** — Intro to GenAI Workshop (Sept 15, Upcoming), HackFest 2025 (Oct 10, Upcoming), Cloud Study Jam (Sept 20, Upcoming), Flutter Forward (Aug 30, Completed), CyberCTF Challenge (Nov 5, Upcoming), Design Thinking Bootcamp (Sept 25, Upcoming)

**Recruitment** — Application Form → Technical Assessment (1 week) → Interview (15 min) → Results (1 week) → Onboarding (2 weeks). Window: Sept 1–15, 2025. Eligibility: 1st to 3rd year.

**Rules** — Minimum 2 events/month to stay active. Inactive for 2 months = alumni status. Team switching once per semester. At least 1 project contribution per semester.

**Contacts** — President: Aditya Kumar (president@gdgoncampus.com), VP: Meera Joshi, Tech Head: Rohan Desai, General: info@gdgoncampus.com

**Achievements** — Best Community Award at DevFest 2024, 12 open-source projects (500+ GitHub stars), 25+ workshops in 2024–25, partnerships with 3 college clubs.

---

## 3. Functional Requirements

### 3.1 Core Chatbot

Answers user questions using only the knowledge base above (any LLM API or open-source model).

**Done when:**
- Every factual claim in a response traces to a KB field; no invented facts (names, dates, emails, numbers).
- If the answer isn't in the KB, the bot explicitly says it doesn't know / isn't in scope — never guesses or extrapolates.
- Handles direct lookups (single field), aggregate lookups (e.g. "list all teams"), and out-of-scope questions (e.g. "what's the club's budget?").

### 3.2 Smart Features

- **Multi-turn memory** — resolves references to prior turns within a session (e.g. "Who leads it?" after asking about a team resolves the pronoun/ellipsis correctly).
- **Source citation** — every grounded answer names the KB section it came from (e.g. "Teams", "Events").
- **Confidence scoring** — every response carries a confidence indicator.
- **Intent classification** — every user message is tagged with a category (e.g. FAQ, event inquiry, action request, out-of-scope) and the tag is shown in the UI.

**Done when:**
- A scripted multi-turn conversation (follow-up questions with pronouns/ellipsis) resolves correctly without the user re-stating context.
- Every bot response in the UI displays: cited source section (or "not found in KB"), a confidence value, and the classified intent.
- Confidence and intent are logged per-turn, not just displayed and discarded (dashboard depends on this).

### 3.3 Agentic Actions

Minimum 2 actions (e.g. event registration, feedback submission, status check, reminder setup). Each action:
- Gathers missing required fields conversationally (slot-filling) — asks only for what's missing, not a rigid form dump.
- Persists the resulting record (survives restart — file/db, not in-memory only).
- Confirms completion back to the user with a summary of what was recorded.

**Done when:**
- Two distinct actions are implemented end-to-end.
- Starting an action with partial info (e.g. "register me for HackFest") triggers targeted follow-up questions for exactly the missing slots, not all slots.
- Completed action records are visible in persisted storage after the process restarts.
- User receives an explicit confirmation message naming the values recorded.
- Actions only reference real KB entities (e.g. can't "register" for an event not in the Events list — bot should flag this).

### 3.4 Dashboard

Simple dashboard showing: chat stats, intent breakdown, actions log, unanswered queries.

**Done when:**
- Chat stats: at minimum total conversations/messages count, visible and updating as new chats happen.
- Intent breakdown: counts/proportions per intent category, sourced from the same classification logged in 3.2.
- Actions log: every completed (and, ideally, abandoned) agentic action with timestamp and captured slot values.
- Unanswered queries: a list of user questions the bot couldn't ground in the KB, for gap analysis.
- Dashboard reads from persisted logs, not a hand-maintained/mocked list — reflects actual chatbot usage.

---

## 4. Submission Requirements

- **GitHub repo** — organized (clear module separation: KB, chatbot logic, actions, dashboard), runnable from a clean clone with documented setup steps.
- **README.md** — explains the approach taken (architecture, model/framework choices) and includes evaluation results (how grounding/intent/actions were tested and what the results were).
- **Streamlit demo** — optional.
- **Recorded demo video** — walks through the approach and the results (i.e. shows the working chatbot, agentic actions, and dashboard, not just a talk-through of code).

---

## 5. Open Technical Decisions

Not decided here — options and trade-offs only. Choice to be made separately.

### 5a. Intent Classification: Trained Classifier vs. LLM Prompt

| Option | Implementation effort | Explainability | Evaluation metrics enabled |
|---|---|---|---|
| **Trained classifier** (e.g. logistic regression / small transformer fine-tune on labeled intent examples) | Higher upfront — needs labeled training data, training pipeline, versioned model artifact | High — inspectable weights/features, deterministic given input, easy to unit test on a fixed set | Standard supervised metrics: precision/recall/F1 per class, confusion matrix, accuracy on a held-out test set — clean numbers for the README's evaluation section |
| **LLM prompt classification** (ask the same or a cheaper LLM to output an intent label, few-shot or zero-shot) | Lower upfront — no training data or pipeline, just prompt design | Lower — behavior depends on prompt wording and model version, harder to guarantee consistency; can mitigate with structured output (enum constraint) | Can still compute precision/recall/F1 against a hand-labeled eval set, but results can drift across model versions/temperature; also enables qualitative error inspection via the LLM's reasoning if asked to explain |
| **Hybrid** (rule/keyword pre-filter for obvious cases, LLM fallback for ambiguous ones) | Medium — some rule authoring plus prompt design | Medium — rules are fully explainable, LLM portion inherits its caveats | Metrics split by path (rule-covered vs. LLM-covered), useful for showing what fraction of traffic is deterministically explainable |

**Choice: _(blank — to be decided)_**

### 5b. Confidence Scoring: What Signal Produces the Number

| Option | Signal source | Defensibility |
|---|---|---|
| **Retrieval similarity score** (e.g. cosine similarity between query and matched KB section/embedding) | Vector search score at retrieval time | Defensible as "how well the question matched a KB entry" — directly traceable to a number you can show; doesn't reflect whether the LLM's phrasing of the answer is accurate, only that the right section was found |
| **LLM self-reported confidence** (ask the model to output a confidence value alongside its answer) | Model's own token-level or verbalized estimate | Weakest under questioning — LLMs are known to be poorly calibrated when self-reporting; only defensible if paired with a calibration check against a labeled eval set |
| **Grounding/verification score** (post-hoc check: does every claim in the generated answer appear in the retrieved KB text? score = fraction of claims supported) | Comparison of generated output against source KB text | Strong defensibility — directly measures the property that matters most here (no fabrication), and the method (claim-by-claim match) can be shown/audited on demand |
| **Composite** (combine retrieval similarity + grounding verification, e.g. weighted or min of the two) | Both of the above | Most defensible overall but adds implementation complexity and requires justifying the combination method itself |

**Choice: _(blank — to be decided)_**

---
