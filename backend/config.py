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
