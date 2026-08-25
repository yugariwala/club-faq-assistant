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
import re

from backend.config import (
    ANTHROPIC_MODEL,
    DEFAULT_INTENT_ON_LLM_FAILURE,
    DEFAULT_LLM_PROVIDER,
    GEMINI_MODEL,
    INTENT_CLASSIFY_MAX_ATTEMPTS,
    INTENT_LABELS,
)
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
#
# Few-shot examples cover two distinct resolution patterns, not just one:
# pronoun coreference ("who leads it?") AND elliptical topic switches
# ("what about Cloud?", "and who leads that one?") that name a new entity
# without a pronoun at all. A health-check run found the plain instruction
# above resolved pronouns reliably but left "what about X?" unresolved,
# causing a false refusal on an answerable, in-scope follow-up -- these
# examples close that gap. The last example is a negative case (nothing in
# history resolves the follow-up), reinforcing the anti-fabrication
# guardrail rather than making the model over-eager to resolve everything.
REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's follow-up question into a standalone question, "
    "using ONLY the conversation history provided below to resolve pronouns "
    "and ellipsis (e.g. \"it\", \"that\", \"who leads it\", \"what about "
    "X?\", \"and that one?\"). An elliptical topic switch that names a new "
    "entity but omits the rest of the question (e.g. \"what about Cloud?\" "
    "right after a question about a team) still needs to be resolved into a "
    "standalone question about that entity, using the same topic as the "
    "prior turn -- do not treat naming a new entity as automatically "
    "unresolvable just because it isn't a pronoun. Do not use any outside "
    "knowledge, and do not invent, assume, or introduce any fact that is "
    "not explicitly present in the history. If the history does not "
    "contain enough information to resolve the follow-up, return the "
    "follow-up question EXACTLY as given, unchanged -- never guess at an "
    "antecedent. Respond with ONLY the rewritten (or unchanged) question, "
    "nothing else -- no explanation, no quotation marks, no extra text.\n\n"
    "Examples:\n\n"
    "Conversation history:\n"
    "User: Tell me about the AIML team\n"
    "Assistant: AIML is led by Rahul Sharma.\n\n"
    "Follow-up question: Who leads it?\n"
    "Rewritten: Who leads the AIML team?\n\n"
    "Conversation history:\n"
    "User: Tell me about the AIML team\n"
    "Assistant: AIML is led by Rahul Sharma.\n\n"
    "Follow-up question: What about Cloud?\n"
    "Rewritten: Who leads the Cloud team?\n\n"
    "Conversation history:\n"
    "User: Tell me about the AIML team\n"
    "Assistant: AIML is led by Rahul Sharma.\n"
    "User: What about Cloud?\n"
    "Assistant: Sneha Gupta leads the Cloud team.\n\n"
    "Follow-up question: And who leads that one?\n"
    "Rewritten: Who leads the Cloud team?\n\n"
    "Conversation history:\n"
    "User: Tell me about the AIML team\n"
    "Assistant: AIML is led by Rahul Sharma.\n\n"
    "Follow-up question: What's the club's budget?\n"
    "Rewritten: What's the club's budget?"
)

# System prompt shared by `classify_intent` and `classify_intents_batch`.
# Constrained to return exactly one of the five labels (spec: "Constrained
# output -- the model returns one of the five labels, nothing else").
# Few-shot examples deliberately include the event_inquiry/action_request
# boundary case, since that's where the rule layer in backend/intent.py
# also abstains and defers here.
CLASSIFY_SYSTEM_PROMPT = (
    "You classify a single user message from the GDG On Campus club "
    "chatbot into exactly ONE of these five intent categories:\n\n"
    "faq - general club info: rules, achievements, the club intro, teams, "
    "contacts, recruitment process (not about one specific event).\n"
    "event_inquiry - a question about a specific event or events in "
    "general (what/when/where/status), asked informationally.\n"
    "action_request - the user wants something done on their behalf right "
    "now: register/sign up for something, submit feedback, set a "
    "reminder, check a status -- phrased as a request to act, not just a "
    "question.\n"
    "out_of_scope - the topic itself isn't part of the club's domain at "
    "all (unrelated to teams/events/rules/recruitment/contacts/"
    "achievements, or unrelated to the club entirely).\n"
    "greeting - a greeting, thanks, or goodbye with no other content.\n\n"
    "Classify by the TOPIC of the question, not by whether you personally "
    "know the answer -- a question about a team or event is faq/"
    "event_inquiry even if the specific detail asked for isn't something "
    "you'd know.\n\n"
    "Respond with ONLY the single matching category name, exactly as "
    "spelled above, in lowercase, with no punctuation, no explanation, "
    "and nothing else.\n\n"
    "Examples:\n"
    "Message: What teams does the club have?\nCategory: faq\n\n"
    "Message: When is HackFest 2025?\nCategory: event_inquiry\n\n"
    "Message: Is HackFest still open for registration?\n"
    "Category: event_inquiry\n\n"
    "Message: Register me for HackFest\nCategory: action_request\n\n"
    "Message: Can I still sign up for HackFest?\n"
    "Category: action_request\n\n"
    "Message: I'd like to submit feedback about the last workshop\n"
    "Category: action_request\n\n"
    "Message: What's the club's budget?\nCategory: out_of_scope\n\n"
    "Message: Can you help me with my calculus homework?\n"
    "Category: out_of_scope\n\n"
    "Message: Hey, thanks so much!\nCategory: greeting"
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


class LLMQuotaError(LLMProviderError):
    """Raised specifically when a provider rejects a call for being
    rate-limited or out of quota (HTTP 429), as opposed to any other
    failure. qa.py surfaces this as a distinct user-facing message
    (`config.LLM_QUOTA_MESSAGE`) so "out of quota" doesn't read the same as
    "the app is broken" (`config.LLM_ERROR_MESSAGE`)."""


# Maps the LLM_PROVIDER value to the env var that must be set for that
# provider. Single source of truth for `missing_api_key_var` below.
_PROVIDER_API_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def missing_api_key_var() -> str | None:
    """Return the name of the env var required by the currently configured
    LLM_PROVIDER if it's unset or blank, or None if the provider is
    correctly configured.

    An unrecognized LLM_PROVIDER value returns None here -- that failure is
    already surfaced clearly at call time by `generate_answer` raising
    `LLMProviderError`, so it isn't this function's job to re-detect it.
    """
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    var_name = _PROVIDER_API_KEY_VARS.get(provider)
    if var_name is None:
        return None
    return None if os.environ.get(var_name, "").strip() else var_name


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


def _build_classify_message(query: str) -> str:
    return f"Message: {query}\nCategory:"


def _parse_intent_label(raw: str) -> str | None:
    """Normalize a raw model response into one of config.INTENT_LABELS, or
    None if it isn't one after stripping whitespace/punctuation -- the
    caller decides whether to retry or fall back (spec: "Reject and retry
    on anything outside the enum")."""
    cleaned = raw.strip().lower().strip(" .!\"'")
    return cleaned if cleaned in INTENT_LABELS else None


def _build_batch_classify_message(queries: list[str]) -> str:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(queries, start=1))
    return (
        f"Classify each of the following {len(queries)} messages. Respond "
        f"with exactly {len(queries)} lines, one category per line, in the "
        'same order, formatted as "<number>. <category>" -- nothing else, '
        "no blank lines, no commentary.\n\n"
        f"{numbered}"
    )


def _parse_batch_intent_labels(raw: str, expected_count: int) -> list[str | None]:
    """Parse a numbered "<n>. <category>" batch response into a
    positional list of labels (None where a line is missing/unparseable),
    tolerant of the model reordering or dropping lines."""
    labels: list[str | None] = [None] * expected_count
    for line in (raw or "").splitlines():
        match = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", line)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if 0 <= index < expected_count:
            labels[index] = _parse_intent_label(match.group(2))
    return labels


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
    except anthropic.RateLimitError as exc:
        raise LLMQuotaError(f"Anthropic API rate limit/quota exceeded: {exc}") from exc
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
        if getattr(exc, "code", None) == 429:
            raise LLMQuotaError(f"Gemini API rate limit/quota exceeded: {exc}") from exc
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


def classify_intent(query: str) -> str:
    """Classify `query` into one of config.INTENT_LABELS via the LLM.

    Never raises -- called only after backend/intent.py's rule layer has
    already abstained, so this is the last word on an already-uncertain
    message; any provider failure, or a response that isn't one of the five
    labels after config.INTENT_CLASSIFY_MAX_ATTEMPTS attempts, falls back to
    config.DEFAULT_INTENT_ON_LLM_FAILURE rather than propagate into the
    caller or guess a label it isn't confident about (mirrors
    rewrite_query's never-raise, anti-fabrication contract).
    """
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    generate = _PROVIDERS.get(provider)
    if generate is None:
        return DEFAULT_INTENT_ON_LLM_FAILURE

    message = _build_classify_message(query)
    for attempt in range(INTENT_CLASSIFY_MAX_ATTEMPTS):
        try:
            raw = generate(message, CLASSIFY_SYSTEM_PROMPT)
        except Exception:
            logger.exception(
                "llm_client.classify_intent attempt %d failed for query=%r",
                attempt + 1,
                query,
            )
            continue

        label = _parse_intent_label(raw or "")
        if label is not None:
            return label
        logger.warning(
            "llm_client.classify_intent attempt %d returned unparseable "
            "label %r for query=%r",
            attempt + 1,
            raw,
            query,
        )

    return DEFAULT_INTENT_ON_LLM_FAILURE


def classify_intents_batch(queries: list[str]) -> list[str]:
    """Classify many queries in as few LLM calls as possible.

    Built for scripts/eval_intents.py's quota-constrained evaluation runs
    (the production per-turn path is `classify_intent`, one message at a
    time) -- one call classifies the whole batch via a numbered-list
    prompt/response. Any position that never parses to a valid label after
    config.INTENT_CLASSIFY_MAX_ATTEMPTS whole-batch attempts falls back to
    config.DEFAULT_INTENT_ON_LLM_FAILURE for that item only; this never
    raises and always returns exactly len(queries) labels.
    """
    if not queries:
        return []

    provider = os.environ.get("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    generate = _PROVIDERS.get(provider)
    if generate is None:
        return [DEFAULT_INTENT_ON_LLM_FAILURE] * len(queries)

    labels: list[str | None] = [None] * len(queries)
    message = _build_batch_classify_message(queries)

    for attempt in range(INTENT_CLASSIFY_MAX_ATTEMPTS):
        try:
            raw = generate(message, CLASSIFY_SYSTEM_PROMPT)
        except Exception:
            logger.exception(
                "llm_client.classify_intents_batch attempt %d failed for %d queries",
                attempt + 1,
                len(queries),
            )
            continue

        parsed = _parse_batch_intent_labels(raw or "", len(queries))
        for i, label in enumerate(parsed):
            if labels[i] is None and label is not None:
                labels[i] = label

        if all(label is not None for label in labels):
            break

    return [label if label is not None else DEFAULT_INTENT_ON_LLM_FAILURE for label in labels]
