"""Thin wrapper around the LLM provider (Anthropic or Gemini) used for
grounded answer generation.

Isolates the provider behind `generate_answer()` so `qa.py` (and its tests)
can mock this module instead of talking to the network (spec: "isolates the
provider so tests can mock it"). The active provider is chosen at call time
via the LLM_PROVIDER env var ("anthropic" | "gemini", default "gemini") so it
can be swapped without touching qa.py or any retrieval/grounding/refusal
logic there.
"""

import logging
import os

from backend.config import ANTHROPIC_MODEL, DEFAULT_LLM_PROVIDER, GEMINI_MODEL
from backend.memory import Turn

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the GDG On Campus club FAQ assistant. Answer the user's question "
    "using ONLY the KB section text provided below as context. Do not use any "
    "outside knowledge, do not invent or infer facts not present in the context, "
    "and do not guess. If the provided context does not actually contain the "
    "answer, say so plainly instead of fabricating one. Keep answers concise "
    "and factual."
)

# System prompt for `rewrite_query`. Resolution is scoped strictly to the
# given history -- never invents an antecedent (spec: Design Notes ->
# "Anti-fabrication guardrail"). Returning the query unchanged when it can't
# be resolved lets the untouched Slice-1 threshold check refuse it exactly
# like any other ungrounded query, rather than rewrite_query ever having to
# decide "in scope or not" itself.
REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's follow-up question into a standalone question, "
    "using ONLY the conversation history provided below to resolve pronouns "
    "and ellipsis (e.g. \"it\", \"that\", \"who leads it\"). Do not use any "
    "outside knowledge, and do not invent, assume, or introduce any fact "
    "that is not explicitly present in the history. If the history does "
    "not contain enough information to resolve the follow-up, return the "
    "follow-up question EXACTLY as given, unchanged -- never guess at an "
    "antecedent. Respond with ONLY the rewritten (or unchanged) question, "
    "nothing else -- no explanation, no quotation marks, no extra text."
)

# Fixed low temperature for providers that expose the knob, so grounding
# behavior stays as deterministic as the provider allows (spec: "Set
# temperature low (0-0.2) for deterministic grounding behavior"). Anthropic's
# SDK here has no temperature parameter; its determinism lever is the
# `output_config={"effort": "low"}` passed below instead.
GEMINI_TEMPERATURE = 0.1

_anthropic_client = None
_gemini_client = None


class LLMProviderError(RuntimeError):
    """Raised when a provider call fails -- auth failure, network error, rate
    limit, or an unparseable response. Always raised, never swallowed into a
    fabricated answer, so a failed call can never look like a successful one
    to the caller (qa.py treats this as the in-scope "couldn't reach the
    model" path, distinct from the below-threshold refusal path)."""


def _get_anthropic_client():
    """Lazily construct the Anthropic client so importing this module never
    requires network access or credentials (only calling it does)."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _get_gemini_client():
    """Lazily construct the Gemini client so importing this module never
    requires network access or credentials (only calling it does)."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _gemini_client


def _build_user_message(query: str, section: str, content: str) -> str:
    """Build the single user-turn string shared by every provider.

    Only the retrieved section's raw content text is placed in the prompt
    context -- no other KB entries, and no content beyond that one section
    (spec: "Generation prompt includes ONLY the retrieved section text as
    context"). `section` is used solely to label the context block for the
    model, not as additional factual content.
    """
    return f"Context ({section} section):\n{content}\n\nQuestion: {query}"


def _build_rewrite_message(query: str, history: list[Turn]) -> str:
    """Build the single user-turn string given to the rewrite LLM call.

    Renders each prior turn as a user/assistant exchange, in order, followed
    by the incoming follow-up -- the only material `rewrite_query`'s system
    prompt permits it to resolve references from (spec: Design Notes ->
    "Anti-fabrication guardrail"). `history` is expected non-empty; callers
    (`rewrite_query`, and `qa.answer_question` upstream of it) are
    responsible for skipping the call entirely when there's no history.
    """
    turns = "\n\n".join(
        f"User: {turn.user_message}\nAssistant: {turn.answer}" for turn in history
    )
    return f"Conversation history:\n{turns}\n\nFollow-up question: {query}"


def _generate_anthropic(user_message: str, system_prompt: str) -> str:
    import anthropic

    try:
        response = _get_anthropic_client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            system=system_prompt,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as exc:
        raise LLMProviderError(f"Anthropic API call failed: {exc}") from exc

    # Join every text block rather than taking only the first -- a response
    # with more than one text block would otherwise silently drop content.
    return "".join(block.text for block in response.content if block.type == "text")


def _generate_gemini(user_message: str, system_prompt: str) -> str:
    from google.genai import errors, types

    try:
        response = _get_gemini_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=GEMINI_TEMPERATURE,
            ),
        )
    except errors.APIError as exc:
        raise LLMProviderError(f"Gemini API call failed: {exc}") from exc

    return response.text or ""


_PROVIDERS = {
    "anthropic": _generate_anthropic,
    "gemini": _generate_gemini,
}


def generate_answer(query: str, section: str, content: str) -> str:
    """Generate a grounded answer for `query` using only `content` as context.

    Reads LLM_PROVIDER at call time (not import time) so tests and callers
    can switch providers without reloading this module. Both providers are
    given the identical SYSTEM_PROMPT and user message and both return a
    plain string, so this function's behavior and return shape do not vary
    with the provider (spec: provider parity).
    """
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    try:
        generate = _PROVIDERS[provider]
    except KeyError:
        raise LLMProviderError(
            f"Unknown LLM_PROVIDER {provider!r}; expected 'anthropic' or 'gemini'"
        ) from None

    return generate(_build_user_message(query, section, content), SYSTEM_PROMPT)


def rewrite_query(query: str, history: list[Turn]) -> str:
    """Rewrite `query` into a standalone question using `history` to resolve
    pronouns/ellipsis (e.g. "When is that?").

    Empty history short-circuits with no provider call at all -- there's
    nothing to resolve against (spec: Code Map -> "empty history
    short-circuits (no call)"). Any failure reaching or parsing the
    provider (including an unknown LLM_PROVIDER) falls back to the original
    `query`, unchanged, rather than raising into `qa.answer_question` --
    worst case is then a normal Slice-1-style refusal on the unrewritten
    query, never a crash and never a fabricated resolution (spec: Design
    Notes -> "Anti-fabrication guardrail").
    """
    if not history:
        return query

    provider = os.environ.get("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    generate = _PROVIDERS.get(provider)
    if generate is None:
        return query

    try:
        rewritten = generate(_build_rewrite_message(query, history), REWRITE_SYSTEM_PROMPT)
    except Exception:
        logger.exception("llm_client.rewrite_query failed for query=%r", query)
        return query

    rewritten = rewritten.strip() if rewritten else ""
    return rewritten if rewritten else query
