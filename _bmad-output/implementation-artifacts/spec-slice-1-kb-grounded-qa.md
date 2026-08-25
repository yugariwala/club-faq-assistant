---
title: 'Slice 1 — Knowledge Base & Grounded Q&A'
type: 'feature'
created: '2026-08-24'
status: 'done'
review_loop_iteration: 0
context: ['{project-root}/requirements.md']
baseline_commit: 'NO_VCS'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Club members need an assistant that answers questions using only GDG On Campus's fixed knowledge base (Intro, Teams, Events, Recruitment, Rules, Contacts, Achievements), citing its source and refusing when the answer isn't covered — no chatbot code exists yet.

**Approach:** Build a Python retrieval-augmented pipeline: TF-IDF vectorized KB sections retrieved via cosine similarity, an LLM generation step constrained to answer only from the retrieved section text, and a named refusal threshold constant that short-circuits generation when nothing is relevant enough.

## Boundaries & Constraints

**Always:**
- KB content matches requirements.md §2 verbatim, each entry tagged with its section label (Intro, Teams, Events, Recruitment, Rules, Contacts, Achievements).
- Retrieval returns section + raw similarity score for every candidate; score is never discarded before reaching the caller.
- Generation prompt includes ONLY the retrieved section text as context; the LLM is instructed to answer solely from that context.
- Refusal threshold is one named constant in a config module (e.g. `RETRIEVAL_THRESHOLD`), never inlined at call sites.
- Below-threshold queries never reach the LLM; the bot returns a fixed "not in the club's information" response with no fabricated answer.
- Every answer path (grounded or refusal) returns a uniform result shape: answer, source_section (nullable on refusal), score, refused flag.

**Ask First:** Changing KB wording/values vs. requirements.md §2. Swapping the LLM provider/model.

**Never:** Multi-turn memory, intent classification, agentic actions, dashboard (later slices). Calling the LLM when retrieval is below threshold. Any KB fact absent from requirements.md §2.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Direct lookup | "Who leads the AIML team?" | Answer "Rahul Sharma", source_section=Teams, score ≥ threshold | N/A |
| Aggregate lookup | "List all the teams" | Answer lists all 6 teams + leads, source_section=Teams | N/A |
| Out-of-scope | "What's the club's budget?" | Refusal message, source_section=None, score < threshold, LLM not called | N/A |
| Empty/gibberish input | `""` or random chars | Refusal path, no crash | Guard empty string before vectorizing |

</frozen-after-approval>

## Code Map

Greenfield — no existing project files. This spec establishes the initial module layout under `backend/`; later slices (memory, intent classification, actions, dashboard) build on top without restructuring this layer.

## Tasks & Acceptance

**Execution:**
- [x] `backend/kb_data.py` -- define `KB_ENTRIES: list[dict]` with `section`/`content` for all 7 sections, verbatim from requirements.md §2 -- single source of truth for storage
- [x] `backend/config.py` -- define `RETRIEVAL_THRESHOLD` and LLM model name as named constants -- tunable, not magic numbers
- [x] `backend/retrieval.py` -- `TfidfRetriever` fit over `KB_ENTRIES`; `retrieve(query, top_k=3) -> list[RetrievalResult(section, content, score)]` sorted desc -- exposes raw score for confidence scoring later
- [x] `backend/llm_client.py` -- thin wrapper calling the Anthropic API with a grounded system prompt -- isolates the provider so tests can mock it
- [x] `backend/qa.py` -- `answer_question(query) -> AnswerResult(answer, source_section, score, refused)`; checks `RETRIEVAL_THRESHOLD` before calling `llm_client` -- single entrypoint later slices build on
- [x] `backend/cli.py` -- interactive REPL calling `answer_question` for manual verification
- [x] `tests/test_retrieval.py` -- unit tests covering the I/O matrix's lookup/aggregate/out-of-scope/empty rows
- [x] `tests/test_qa.py` -- unit tests with `llm_client` mocked: refusal short-circuits the LLM call; grounded path returns the correct shape
- [x] `pyproject.toml` -- declare `anthropic`, `scikit-learn` deps; configure `uv run pytest`

**Acceptance Criteria:**
- Given a question answered by the Teams section, when `answer_question` runs, then the response has `source_section="Teams"` and `score >= RETRIEVAL_THRESHOLD`.
- Given a question with no relevant KB match, when `answer_question` runs, then `llm_client` is never invoked and `refused=True` with a message stating the information isn't available.
- Given any retrieval call, when scores are computed, then the returned `score` is the raw, unrounded similarity float.
- Given `RETRIEVAL_THRESHOLD` is edited in `config.py`, when `answer_question` runs, then refusal behavior changes accordingly without touching `qa.py` logic.

## Spec Change Log

- 2026-08-24 (implementation): Empirically calibrated `TfidfRetriever` beyond the plain "TF-IDF + cosine similarity" description in Design Notes. Plain TF-IDF over raw KB content misrouted both required matrix cases -- "Who leads the AIML team?" scored higher on Rules than Teams (the word "team" only occurs in Rules' "Team switching..." line), and "List all the teams" scored 0.0 on every section (the Teams entry's content lists team names/leads but never literally contains the word "team"/"teams"). Fixed by: (1) indexing each entry's section label alongside its content, repeated `_LABEL_REPEATS = 3` times so the label carries enough TF-IDF weight against the entry's own longer content; (2) a lightweight custom analyzer (regex tokenizer + naive suffix-stripping stemmer, no new dependency) so "teams"/"leads" match "Team"/"Lead" in the corpus; (3) an extra stopword set (`gdg`, `campus`, `club`, `clubs`) on top of scikit-learn's English list, since "club"/"clubs" appears only incidentally in the Achievements entry ("partnerships with 3 college clubs") and was otherwise a false-positive magnet for any generic "the club" question. `RETRIEVAL_THRESHOLD = 0.15` was chosen against the resulting score distribution (grounded queries tested at 0.31-0.69; out-of-scope/empty/gibberish at 0.0). Retrieval interface (`retrieve(query, top_k) -> list[RetrievalResult]`) and refusal mechanics are unchanged from Design Notes. See `backend/retrieval.py` for the implementation.

## Design Notes

**Storage/retrieval trade-off:** TF-IDF + cosine similarity (scikit-learn) over dense embeddings or LLM-only routing. With only 7 KB entries, TF-IDF is deterministic, has zero network dependency, needs no model download, and yields an explainable score (term-overlap based) that plugs directly into later confidence scoring. Cost: weaker on pure paraphrase with no shared vocabulary — acceptable here because KB terms (team names, "lead", event names, dates) tend to appear in natural questions about them. Embeddings would handle paraphrase better but add a model-download dependency and a less legible score for defensibility; LLM-only classification of "which section" would remove the numeric score entirely, which the spec requires. If retrieval misses on real queries during testing, swapping `TfidfRetriever` for an embedding-based retriever is a localized change behind the same `retrieve()` interface.

**Refusal mechanics:** `retrieve()` always returns scored candidates (never filters). `qa.answer_question` alone decides refusal by comparing the top score to `RETRIEVAL_THRESHOLD`, keeping the policy in one place and the retriever policy-free.

## Verification

**Commands:**
- `uv run pytest tests/ -v` -- expected: all tests pass, including the refusal-path test with `llm_client` mocked (no network/API key required)

**Manual checks (if no CLI):**
- `uv run python -m backend.cli` (requires `ANTHROPIC_API_KEY`): ask "What teams are in the club?" → grounded answer citing Teams; ask "What's the ticket price for HackFest?" → refusal, no fabricated price.

## Suggested Review Order

**Refusal & grounding policy**

- Entry point: threshold check decides refusal vs. grounding, in the one place that decision is made.
  [`qa.py:49`](../../backend/qa.py#L49)

- LLM call now degrades gracefully on failure instead of crashing the caller; distinct from the below-threshold refusal path.
  [`qa.py:57`](../../backend/qa.py#L57)

- Named, tunable refusal threshold plus the two distinct outbound messages (not-in-KB vs. LLM-unreachable).
  [`config.py:12`](../../backend/config.py#L12)

**Retrieval**

- TF-IDF corpus construction: section label repeated to carry retrieval weight, without altering stored/returned content.
  [`retrieval.py:58`](../../backend/retrieval.py#L58)

- Scored, sorted candidates returned with raw, unrounded similarity — retriever never filters or decides refusal itself.
  [`retrieval.py:93`](../../backend/retrieval.py#L93)

**Generation**

- Prompt is constrained to only the retrieved section's content — nothing else in context.
  [`llm_client.py:33`](../../backend/llm_client.py#L33)

- Text extraction joins all text blocks rather than taking only the first, so nothing is silently dropped.
  [`llm_client.py:59`](../../backend/llm_client.py#L59)

**Knowledge base**

- All 7 sections, verbatim from requirements.md §2, each tagged with the section label used as the citation.
  [`kb_data.py:11`](../../backend/kb_data.py#L11)

**CLI (manual verification harness)**

- Refused vs. grounded output formats — the branch a review found untested and could have silently inverted.
  [`cli.py:31`](../../backend/cli.py#L31)

**Tests**

- Covers the I/O matrix: direct lookup, aggregate lookup, out-of-scope, empty/whitespace/gibberish input.
  [`test_retrieval.py:12`](../../tests/test_retrieval.py#L12)

- Refusal short-circuits the LLM call; threshold edits in config.py flip behavior with zero qa.py changes.
  [`test_qa.py:12`](../../tests/test_qa.py#L12)

- New: the LLM-call-failure path added during review no longer has zero coverage.
  [`test_qa.py`](../../tests/test_qa.py)

- New: request construction and response-parsing of the only network-calling function, previously fully mocked out of coverage.
  [`test_llm_client.py`](../../tests/test_llm_client.py)

- New: quit/exit/empty-input handling and the refused-vs-grounded print branch.
  [`test_cli.py`](../../tests/test_cli.py)

- Dependencies and `uv run pytest` configuration.
  [`pyproject.toml`](../../pyproject.toml)
