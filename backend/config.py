"""Tunable configuration constants for the KB-grounded Q&A pipeline.

Named constants only -- no magic numbers at call sites (spec: Boundaries &
Constraints -> "Refusal threshold is one named constant in a config module").
"""

# Minimum cosine similarity (raw TF-IDF score, range [0, 1]) a retrieval
# candidate's top result must meet for `qa.answer_question` to call the LLM.
# Below this, the bot refuses without ever contacting the LLM (see
# Design Notes -> "Refusal mechanics" in the spec). Tuned empirically against
# the I/O matrix's direct-lookup vs. out-of-scope examples.
RETRIEVAL_THRESHOLD: float = 0.15

# Model used for grounded answer generation on each provider. Single named
# constants so the provider/model can be swapped in one place (spec:
# "Ask First: Swapping the LLM provider/model").
ANTHROPIC_MODEL: str = "claude-sonnet-5"
# gemini-2.5-flash 404s at generate_content time for newer API keys ("no
# longer available to new users") despite still appearing in models.list();
# gemini-3.6-flash is the model Google's own error message points to, and is
# confirmed live via models.list() (see .env-configured live probe run).
GEMINI_MODEL: str = "gemini-3.6-flash"

# Provider selected when the LLM_PROVIDER env var is unset or blank.
DEFAULT_LLM_PROVIDER: str = "gemini"

# Fixed refusal message returned when retrieval doesn't clear
# RETRIEVAL_THRESHOLD. Never fabricated, never LLM-generated.
REFUSAL_MESSAGE: str = (
    "I don't have that information in the club's knowledge base. "
    "I can only answer questions about GDG On Campus's intro, teams, "
    "events, recruitment, rules, contacts, or achievements."
)

# Maximum number of prior turns retained per session and available to
# `llm_client.rewrite_query` for reference resolution. Read at call time by
# `memory.SessionStore.add_turn` (not baked into a fixed-size structure at
# session creation) so tuning this value changes trimming behavior for every
# existing session immediately (spec: "History window is bounded by one
# named constant, read at call time").
MAX_HISTORY_TURNS: int = 5

# Fixed message returned when retrieval found a relevant KB section (so this
# is NOT a below-threshold refusal) but the call into `llm_client` itself
# failed -- auth failure, network error, rate limit, malformed response,
# etc. Distinct from REFUSAL_MESSAGE: that one means "not in the KB", this
# one means "couldn't reach the model to answer a question that IS in scope".
LLM_ERROR_MESSAGE: str = (
    "I couldn't reach the model right now. Please try again in a moment."
)

# Fixed message returned when the LLM call itself failed specifically
# because the provider is rate-limited or out of quota (HTTP 429), as
# distinct from any other failure. Kept separate from LLM_ERROR_MESSAGE so
# "out of quota, try again shortly" doesn't read the same as "something is
# broken" to the user.
LLM_QUOTA_MESSAGE: str = (
    "The model is temporarily rate-limited or out of quota, not broken -- "
    "please try again in a minute."
)

# The exact five intent categories every user message is tagged with
# (requirements.md §3.2). The LLM fallback (backend/llm_client.classify_intent)
# is constrained to return one of these labels, nothing else.
INTENT_LABELS: frozenset[str] = frozenset(
    {"faq", "event_inquiry", "action_request", "out_of_scope", "greeting"}
)

# Fallback label used only when the LLM classification path exhausts
# INTENT_CLASSIFY_MAX_ATTEMPTS with no valid label (unparseable response, or
# the call itself fails every attempt). Never a positive claim about what
# the user wants -- "couldn't determine" is closest in spirit to "not
# answerable", not to any specific actionable/informational category.
DEFAULT_INTENT_ON_LLM_FAILURE: str = "out_of_scope"

# Number of attempts (including the first) classify_intent/classify_intents_batch
# make against the LLM before giving up and returning
# DEFAULT_INTENT_ON_LLM_FAILURE for whatever couldn't be parsed.
INTENT_CLASSIFY_MAX_ATTEMPTS: int = 2

# --- Confidence scoring (requirements.md §3.2, §5b "Composite") -------------

# Band names for the displayed confidence indicator. NOT_APPLICABLE is used
# when the response makes no factual claim about the club at all (a refusal,
# an LLM-error message, or a generated "the context doesn't say") -- see
# CONFIDENCE_REASON_* below. Scoring such a response 0.0 would badge the
# bot's most trustworthy behavior as untrustworthy; scoring it 1.0 would
# assert a verification that never ran.
CONFIDENCE_BAND_HIGH_NAME: str = "high"
CONFIDENCE_BAND_MEDIUM_NAME: str = "medium"
CONFIDENCE_BAND_LOW_NAME: str = "low"
CONFIDENCE_BAND_NOT_APPLICABLE_NAME: str = "not_applicable"

# Score at or above which a response is banded `high`. Grounding score is
# supported/total claims, so with the 1-6 claims a concise FAQ answer
# typically yields, 0.85 is the threshold at which a single unsupported
# claim can no longer reach `high` (4/5 = 0.80, 5/6 = 0.83, both `medium`).
# A lower bar such as 0.75 would badge a 3-of-4 answer -- one fabricated
# claim -- as high confidence, which is the exact failure the eval set in
# data/grounding_eval.jsonl exists to catch.
CONFIDENCE_BAND_HIGH: float = 0.85

# Score at or above which a response is banded `medium`; below it, `low`.
# Reads as "at least half the answer's claims were verified against source".
CONFIDENCE_BAND_MEDIUM: float = 0.50

# Why a confidence score is what it is. Attached to every scored response so
# a `not_applicable` band can be told apart from a genuinely low-confidence
# one, and so the Slice 5 dashboard can bucket these states separately
# rather than lumping every unscored turn together.
CONFIDENCE_REASON_VERIFIED: str = "verified"
CONFIDENCE_REASON_REFUSED: str = "refused"
CONFIDENCE_REASON_LLM_ERROR: str = "llm_error"
CONFIDENCE_REASON_LLM_QUOTA: str = "llm_quota"
CONFIDENCE_REASON_NO_CLAIMS: str = "no_claims"
CONFIDENCE_REASON_VERIFICATION_DISABLED: str = "verification_disabled"
CONFIDENCE_REASON_VERIFICATION_FAILED: str = "verification_failed"

# Whether post-hoc grounding verification runs by default. Verification adds
# exactly one LLM call per grounded turn (see README.md "Quota cost"), which
# matters on a free Gemini key capped at 20 requests/day. Overridable at call
# time via the VERIFY_GROUNDING env var so a quota-constrained demo can turn
# it off without editing code; with it off, the reported confidence is the
# retrieval signal alone and is labeled `verification_disabled` so a
# degraded signal is never presented as a full composite.
CONFIDENCE_VERIFICATION_ENABLED: bool = True
VERIFY_GROUNDING_ENV_VAR: str = "VERIFY_GROUNDING"

# How many answers `llm_client.verify_groundings_batch` packs into a single
# call. Bulk path for scripts/eval_grounding.py only (the production path
# verifies one answer at a time -- a live turn has no batch to join), same
# precedent as classify_intents_batch. Bounded well below the eval set size
# so one oversized response can't exceed the provider's output token limit
# and lose the whole run.
VERIFY_BATCH_SIZE: int = 6

# Attempts (including the first) verify_grounding/verify_groundings_batch
# make before giving up. A retry costs another LLM call, so this is
# deliberately low: the strict CLAIM/VERDICT/EVIDENCE output format makes an
# unparseable response rare, and the give-up path is already safe
# (CONFIDENCE_REASON_VERIFICATION_FAILED bands the answer `low` rather than
# letting an unverified answer look confident).
VERIFY_MAX_ATTEMPTS: int = 2
