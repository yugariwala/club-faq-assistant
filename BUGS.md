# Health Check — Slice 1 (grounded Q&A) + Slice 2 (multi-turn memory)

Run 2026-08-25. Environment: Windows 11, `uv 0.12.5`, Python 3.12, `LLM_PROVIDER=gemini`,
`GEMINI_MODEL=gemini-3.6-flash` (backend/config.py:22). No `ANTHROPIC_API_KEY` was available in
`.env`, so all live LLM calls in this report used the Gemini path.

Findings are ranked most-severe first. No code was changed as part of this check.

---

## CRITICAL

### C1. Documented run command does not work from a clean shell — `.env` is never loaded

- **Input:** Follow README.md:9-26 literally: `uv sync`, `cp .env.example .env`, fill in
  `GEMINI_API_KEY`, then `uv run python -m backend.cli`.
- **Expected:** Grounded (non-refusal) queries answer using the KB, per README.md:28-29
  ("Refusals ... never call the LLM, so the CLI works without any API key **for those**" —
  implying everything else needs the key from `.env`).
- **Actual:** Confirmed two ways:
  1. `uv run python -c "import os; print('GEMINI_API_KEY' in os.environ)"` → `False` with a
     populated `.env` sitting next to it. `uv run` does not auto-source `.env`.
  2. Nothing in the codebase loads it either — no `python-dotenv` in `pyproject.toml:6-10`, no
     `load_dotenv()` call anywhere (`grep -r dotenv` → no matches).
  3. Ran the CLI exactly as documented (no manual `export`/`$env:` step) and asked a direct
     lookup ("who leads the AIML team?"). Retrieval succeeded (`score=0.693`, section=Teams),
     but generation raised `ValueError: No API key was provided...` inside
     `google.genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))`
     (backend/llm_client.py:84). `qa.answer_question`'s bare `except Exception` (qa.py:100-119)
     catches it and degrades gracefully to `LLM_ERROR_MESSAGE`, so it doesn't crash — but every
     single grounded answer silently becomes *"I couldn't reach the model right now."*
- **PASS/FAIL:** FAIL.
- **File:line:** README.md:9-26 (documented steps), backend/llm_client.py:84 (where the env var
  is actually read), pyproject.toml:6-10 (no dotenv dependency).
- **Why CRITICAL:** This is the only documented run path. A grader or new contributor following
  the README exactly gets a bot that appears to work (no crash, no visible error unless they
  read stderr) but never produces a single real grounded answer — it looks broken end-to-end.
  Not a correctness bug in the grounding logic itself, but it defeats the whole feature via
  setup, silently.

### C2. `.env.example` currently has a live-looking API key pasted into it (uncommitted)

- **Input:** `git status` / `git diff .env.example` at session start.
- **Expected:** `.env.example` is a template — every value blank, matching `git show
  HEAD:.env.example` (`GEMINI_API_KEY=`, empty).
- **Actual:** The working tree has an **unstaged** edit to `.env.example` that sets
  `GEMINI_API_KEY=` to what looks like a real Gemini key. `.env.example` is tracked (unlike
  `.env`, which is correctly in `.gitignore:1`), so if this gets `git add`ed and committed, the
  key leaks into git history permanently.
- **PASS/FAIL:** FAIL (flagged live to the user at the start of this session; not written to
  any other file; not committed by this check).
- **File:line:** `.env.example` (working tree, uncommitted).
- **Recommendation:** Rotate the key and revert `.env.example` to blank before any commit. Not
  fixed here per your instruction not to fix anything yet.

---

## HIGH

### H1. Slice 2 multi-turn memory: elliptical topic-switch ("what about X?") is not resolved — causes a false refusal on an answerable, in-scope question

- **Input (session, in order):**
  1. `"tell me about the AIML team"`
  2. `"who leads it?"`
  3. `"what about Cloud?"`
- **Expected:** Per requirements.md:46 ("Multi-turn memory — resolves references to prior turns
  ... e.g. 'Who leads it?'") and requirements.md:52 ("A scripted multi-turn conversation
  (follow-up questions with pronouns/ellipsis) resolves correctly without the user re-stating
  context"), turn 3 should be rewritten into something like "what about the Cloud team?" /
  "tell me about the Cloud team", retrieve the Teams section, and answer with Sneha Gupta as
  Cloud's lead (that fact **is** in the KB — kb_data.py:23).
- **Actual (live run, real outputs):**
  ```
  turn 1: query='tell me about the AIML team'
          rewritten_query='tell me about the AIML team'  (no history yet, unchanged — correct)
          refused=False source_section='Teams' score=0.2960
          answer='Based on the provided context, the AIML team is led by Rahul Sharma.'

  turn 2: query='who leads it?'
          rewritten_query='Who leads the AIML team?'      (coreference resolved correctly)
          refused=False source_section='Teams' score=0.6934
          answer='Rahul Sharma leads the AIML team.'

  turn 3: query='what about Cloud?'
          rewritten_query='what about Cloud?'              (rewrite returned it UNCHANGED)
          refused=True source_section=None score=0.1071
          answer="I don't have that information in the club's knowledge base. I can only
                   answer questions about GDG On Campus's intro, teams, events, recruitment,
                   rules, contacts, or achievements."
  ```
  Raw retrieval on the unrewritten "what about Cloud?" scores 0.107, below
  `RETRIEVAL_THRESHOLD = 0.15` (config.py:12), so it refuses before the LLM is even consulted
  a second time.
- **PASS/FAIL:** FAIL. This is exactly the "Topic switch" scenario this health check was asked
  to verify.
- **File:line:** backend/llm_client.py:35-45 (`REWRITE_SYSTEM_PROMPT`), backend/qa.py:75-77
  (rewrite gate), backend/qa.py:91 (threshold check that then refuses the unresolved query).
- **Not a contract violation, but a real capability gap:** The Slice 2 spec
  (`_bmad-output/implementation-artifacts/spec-slice-2-multi-turn-memory.md:38`) explicitly
  documents "rewrite returns query unchanged → refuses like Slice 1" as *intended* fallback
  behavior for a genuinely unresolvable reference. The bug is that this reference **was**
  resolvable from history (a Cloud team question, right after an AIML team question) and the
  live Gemini rewrite call failed to do it anyway — it only reliably resolves pronouns
  ("it"/"that"), not "what about X" ellipsis, in this run. Distinct from fabrication (no wrong
  fact was stated), but it directly fails the "ellipsis" half of requirements.md §3.2's
  acceptance criterion, so I'm ranking it HIGH rather than a minor edge case.
- **Related test-coverage gap:** tests/test_qa.py has exactly two rewrite-behavior tests —
  `test_coreference_follow_up_resolves_via_rewrite_and_grounds_correct_section` (test_qa.py:134)
  and `test_unresolvable_reference_refuses_exactly_like_slice_one` (test_qa.py:173) — and both
  **mock** `rewrite_query`, so the suite never actually exercises the live LLM's rewrite quality.
  68/68 tests passing gives no signal on this failure mode; it can only be caught by a live run
  against the real provider, which is what this health check did.

---

## MEDIUM

### M1. Ambiguous pronoun in a brand-new session (no history) reaches the LLM instead of refusing, relying entirely on the model's good behavior

- **Input:** Fresh session, zero prior turns, query = `"who leads it?"`.
- **Expected (per the health-check brief's own framing of this as the "Unresolvable" case):**
  the bot should not resolve "it" to any specific team, since nothing established what "it"
  refers to.
- **Actual:**
  ```
  query='who leads it?' (fresh session)
  rewritten_query='who leads it?'   (rewrite_query never called — history is empty, correct
                                      per spec: qa.py:76)
  refused=False source_section='Teams' score=0.7525
  answer='The provided context does not specify which team "it" refers to, but lists the
           following domain leads:
           * AIML: Rahul Sharma
           * Web Dev: Priya Patel
           * App Dev: Arjun Mehta
           * Cloud: Sneha Gupta
           * Cybersecurity: Vikram Singh
           * Design: Ananya Reddy'
  ```
  Retrieval alone (no rewrite involved) scores 0.7525 against the Teams section, purely because
  the stemmed token "lead" (from "leads") heavily overlaps the repeated "Lead:" labels in
  kb_data.py:22-24 (retrieval.py:58-72 repeats the section label 3x specifically to boost this
  kind of match — retrieval.py:55,65-71). That pushes it well above
  `RETRIEVAL_THRESHOLD = 0.15`, so the LLM is invoked on a genuinely ambiguous question instead
  of the bot refusing.
- **This run did not fabricate** — Gemini correctly declined to guess and enumerated all leads
  instead. But this is not a code-level guarantee: nothing in `qa.answer_question` or
  `retrieval.py` distinguishes "high lexical overlap" from "actually answerable," so a future
  call (different phrasing, different model, different temperature) could just as easily pick
  one team and assert it as the answer to "it" — which would be exactly the kind of fabrication
  requirements.md:41 prohibits ("no invented facts... never guesses or extrapolates"), except
  the fabrication would be *which entity* the pronoun resolves to rather than an invented fact
  from outside the KB.
- **PASS/FAIL:** PASS this run (no fabrication observed) / flagged as a risk, not a currently
  reproduced failure.
- **File:line:** backend/qa.py:74-77 (empty-history short-circuit), backend/retrieval.py:55-72
  (label-repetition weighting that inflates this match).

### M2. Gemini free-tier quota (5 req/min, 20 req/day on this key) makes live testing/demoing unreliable

- **Observed directly during this check:** two distinct 429s from `google.genai`:
  - `GenerateRequestsPerMinutePerProjectPerModel-FreeTier`, `quotaValue: '5'`
  - `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quotaValue: '20'`
  The daily cap was hit partway through this session's testing; several later calls
  (`"and who leads that one?"`, the session-isolation probe) returned
  `LLM_ERROR_MESSAGE` ("I couldn't reach the model right now") purely due to quota exhaustion,
  not application logic — those two results are **inconclusive**, not a pass or fail.
- **Why this matters beyond "just an API limit":** a live grading run, a recorded demo, or even
  a second developer testing locally can silently exhaust this in a few dozen turns (each
  multi-turn follow-up costs up to 2 calls — rewrite + generate, per
  `_bmad-output/implementation-artifacts/spec-slice-2-multi-turn-memory.md:14`), and the failure
  mode is indistinguishable in the UI from a real outage (`LLM_ERROR_MESSAGE` in both cases).
  Every previously-passing grounded-answer test in this report could flip to "looks broken" on a
  quota-exhausted key with no code change.
- **PASS/FAIL:** N/A — environment/operational risk, not a code defect.
- **File:line:** backend/config.py:22 (`GEMINI_MODEL = "gemini-3.6-flash"`), backend/config.py:9
  (comment already notes `gemini-2.5-flash` 404s for new keys — this model's rate limits are the
  next issue down that same path).

---

## LOW / INFO

### L1. Uncaught-exception tracebacks print straight to the console on every LLM failure

Every `LLMProviderError` path (`llm_client.py:203`, `qa.py:110`) logs via
`logger.exception(...)`, and with no logging config set up anywhere (no `logging.basicConfig`
call in `cli.py` or elsewhere), Python's root-logger lastResort handler dumps the full
traceback to stderr. Functionally harmless (the graceful-degradation message still prints
correctly after it — see the C1 and M2 transcripts above), but it's noisy in an interactive demo
and could read as "the app crashed" to someone not watching closely. Not fixed here.

### L2. "CLI args" input handling doesn't apply as scoped — `backend/cli.py` takes no command-line arguments

Checked for `argparse`/`sys.argv` usage in `backend/`: none found. `backend/cli.py` is a pure
REPL (`input("> ")` loop, cli.py:25); there is no argument-parsing surface to test for
quote/whitespace pollution. Tested the equivalent interactive-input surface instead
(pure-Python, no LLM cost, via `backend.retrieval._analyzer` and `TfidfRetriever.retrieve`):

| Input | Analyzer tokens | Section | Score |
|---|---|---|---|
| `who leads the AIML team?` | `['lead','aiml','team']` | Teams | 0.693 |
| `  who leads the AIML team?  ` (padded) | `['lead','aiml','team']` | Teams | 0.693 |
| `"who leads the AIML team?"` (literal double quotes) | `['lead','aiml','team']` | Teams | 0.693 |
| `'who leads the AIML team?'` (literal single quotes) | `['lead','aiml','team']` | Teams | 0.693 |
| `   ` (whitespace only) | `[]` | Intro (default, score 0) | 0.000 |
| `` (empty) | `[]` | Intro (default, score 0) | 0.000 |

**PASS.** `retrieval.py:18`'s tokenizer (`[A-Za-z]+|\d+`) only ever extracts letters/digits, so
stray quote characters are silently dropped rather than polluting terms, and both whitespace-only
and empty queries are guarded explicitly (retrieval.py:101-108) before hitting scikit-learn.
`cli.py:25` also `.strip()`s every line and skips empty input before calling `answer_question` at
all (cli.py:32-33), so an all-whitespace REPL line never even reaches retrieval in practice.

---

## PASS — confirmed working as specified

| # | Category | Input | Result |
|---|---|---|---|
| P1 | Slice 1 direct lookup | "who leads the AIML team?" | `Rahul Sharma leads the AIML team.` — source=Teams, score=0.693 |
| P2 | Slice 1 direct lookup | "when is HackFest 2025?" | `HackFest 2025 is on October 10.` — source=Events, score=0.168 |
| P3 | Slice 1 direct lookup | "how do I contact the president?" | `You can contact the President, Aditya Kumar, via email at president@gdgoncampus.com.` — source=Contacts, score=0.628 |
| P4 | Slice 1 aggregate | "list all teams" | Correctly lists all 6 teams — source=Teams, score=0.312 |
| P5 | Slice 1 aggregate | "what events are upcoming?" | Correctly lists only the 5 events tagged "Upcoming" in the KB, **excluding** Flutter Forward (tagged "Completed") — source=Events, score=0.702. No fabrication; correctly filtered on KB-stated status. |
| P6 | Slice 1 refusal | "what's the club's budget?" | Refused, `REFUSAL_MESSAGE`, score=0.000. No LLM call (retrieval-only refusal). |
| P7 | Slice 1 refusal | "who is the treasurer?" | Refused, `REFUSAL_MESSAGE`, score=0.000. |
| P8 | Slice 1 in-scope-unanswerable | "how many members are in Web Dev?" | Retrieval hit Teams (score=0.266, above threshold) but the LLM correctly declined to invent a count: `"The provided context does not state how many members are in Web Dev; it only mentions that the Lead is Priya Patel."` No fabricated number. |
| P9 | Slice 1 in-scope-unanswerable (borderline) | "what time does HackFest start?" | Retrieval score=0.129, just below `RETRIEVAL_THRESHOLD=0.15` → refused via the threshold path, never reaches the LLM. Correct outcome (KB genuinely has no time-of-day), though it's close enough to the threshold that it's worth knowing this one refuses "by luck of the score" rather than the LLM ever seeing it. |
| P10 | Slice 2 coreference | "tell me about the AIML team" → "who leads it?" | Turn 2 rewritten to `"Who leads the AIML team?"`, correctly grounds in Teams, answers `Rahul Sharma leads the AIML team.` |
| P11 | Slice 2 session isolation | Fresh `session_id`, no prior turns, asked "who leads it?" | `rewritten_query == query` (unchanged) confirms `rewrite_query` was never invoked for this session — `qa.py:74-77`'s `history and query.strip()` guard correctly saw an empty history list. No cross-session leakage possible through the rewrite path by construction (mechanism verified directly; the LLM-generated answer text for this particular call was not obtainable — see M2, daily quota exhausted mid-run). |
| P12 | Test suite | `uv run pytest -q` | **68 passed, 0 failed** (2.89s). Every test mocks the LLM client per README.md:37-38 — no network calls, no API key needed. See H1's note: this coverage validates orchestration/plumbing, not live rewrite quality. |
| P13 | Environment: imports | `import backend.qa, backend.retrieval, backend.memory, backend.llm_client, backend.config, backend.kb_data` | Clean import, no errors, under `uv run`. |

---

## Inconclusive (quota-blocked, not a code finding)

- **"and who leads that one?"** (turn 4, same session as H1) — hit the same-day 429 at both the
  rewrite and generate stage; fell back to `LLM_ERROR_MESSAGE`. Retrieval on the *unrewritten*
  text still scored 0.7525 against Teams purely by lexical luck on "leads" (same mechanism as
  M1), so this particular phrasing can't cleanly distinguish "rewrite worked" from "rewrite
  wasn't needed." Would need a re-run on a fresh-quota key to get a real signal.
- **Session-isolation LLM answer text** (P11) — mechanism confirmed (no history read into the
  rewrite), but the actual generated answer for that call also hit the exhausted daily quota.

## Summary

| Severity | Count |
|---|---|
| CRITICAL | 2 |
| HIGH | 1 |
| MEDIUM | 2 |
| LOW/INFO | 2 |
| PASS | 13 |
| Inconclusive | 2 |

No fabrication was observed in any run today. The two CRITICAL items are both about getting the
system into a state where it can be evaluated at all (setup correctness, secret hygiene), not
about the grounding logic itself, which held up well under adversarial-ish testing (P8, M1). The
one HIGH item is a real, reproducible gap in the specific "ellipsis topic-switch" phrasing this
health check was asked to check for.
