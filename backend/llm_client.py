"""Thin wrapper around the LLM provider (Anthropic or Gemini) used for
grounded answer generation.

Isolates the provider behind `generate_answer()` so `qa.py` (and its tests)
can mock this module instead of talking to the network (spec: "isolates the
provider so tests can mock it"). The active provider is chosen at call time
via the LLM_PROVIDER env var ("anthropic" | "gemini", default "gemini") so it
can be swapped without touching qa.py or any retrieval/grounding/refusal
logic there.
"""

import os

from backend.config import ANTHROPIC_MODEL, DEFAULT_LLM_PROVIDER, GEMINI_MODEL

SYSTEM_PROMPT = (
    "You are the GDG On Campus club FAQ assistant. Answer the user's question "
    "using ONLY the KB section text provided below as context. Do not use any "
    "outside knowledge, do not invent or infer facts not present in the context, "
    "and do not guess. If the provided context does not actually contain the "
    "answer, say so plainly instead of fabricating one. Keep answers concise "
    "and factual."
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


def _generate_anthropic(user_message: str) -> str:
    import anthropic

    try:
        response = _get_anthropic_client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as exc:
        raise LLMProviderError(f"Anthropic API call failed: {exc}") from exc

    # Join every text block rather than taking only the first -- a response
    # with more than one text block would otherwise silently drop content.
    return "".join(block.text for block in response.content if block.type == "text")


def _generate_gemini(user_message: str) -> str:
    from google.genai import errors, types

    try:
        response = _get_gemini_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
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

    return generate(_build_user_message(query, section, content))
