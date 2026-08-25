"""Hybrid intent classification: deterministic rules first, LLM fallback for
everything the rules aren't confident about.

Every rule is high-precision and abstains (returns None) rather than guess --
an uncertain rule falls through to the next rule, and ultimately to
`llm_client.classify_intent`, rather than risk a wrong label. Rules are
checked in a fixed priority order so a message matching more than one
rule's *topic* (e.g. "remind me about HackFest" mentions both a reminder
action and a named event) resolves to the more specific/actionable
category, not whichever rule happens to be scanned last.

Intent here means the topical category of what the user is asking, not
whether the knowledge base can actually answer it -- "How many members are
in Web Dev?" is faq intent (a team question) even though that fact isn't in
the KB; whether it's answerable is `qa.answer_question`'s separate refusal
concern.
"""

import re
from dataclasses import dataclass

from backend import config, llm_client


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace so every rule matches consistently
    regardless of the caller's casing/spacing."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Rule 1: greeting -- whole-message match only (never a substring match), so
# a greeting attached to real content ("hi, when's HackFest?") falls through
# instead of being misread as a bare greeting.
# ---------------------------------------------------------------------------

_GREETING_PHRASES = (
    "hi",
    "hello",
    "hey",
    "hiya",
    "yo",
    "hi there",
    "hello there",
    "good morning",
    "good afternoon",
    "good evening",
    "what's up",
    "whats up",
    "thanks",
    "thank you",
    "thanks a lot",
    "thank you so much",
    "thanks so much",
    "cheers",
    "much appreciated",
    "ok thanks",
    "okay thanks",
    "great thanks",
    "bye",
    "goodbye",
    "see you",
    "see ya",
    "take care",
    "later",
    "have a good day",
    "have a nice day",
)
# Longest-first so the regex engine doesn't stop at a shorter alternative
# that's a prefix of a longer valid phrase.
_GREETING_RE = re.compile(
    "|".join(re.escape(p) for p in sorted(_GREETING_PHRASES, key=len, reverse=True))
)


def _rule_greeting(text: str) -> str | None:
    normalized = _normalize(text).rstrip("!.? ")
    return "greeting" if _GREETING_RE.fullmatch(normalized) else None


# ---------------------------------------------------------------------------
# Rule 2: action_request -- self-referential imperative patterns only.
# Topic overlap with an action word alone never qualifies (e.g. "how do I
# register" is informational, not a command) -- these all require explicit
# first-person "do this for/to me" framing.
# ---------------------------------------------------------------------------

_ACTION_REQUEST_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\bregister me\b",
        r"\bsign me up\b",
        r"\bplease register me\b",
        r"\bcan you register me\b",
        r"\bi(?:'d|\s+would|\s+want to)\s+(?:like to\s+)?register\b",
        r"\bsubmit (?:my )?feedback\b",
        r"\bi(?:'d|\s+would|\s+want to)\s+(?:like to\s+)?(?:submit|give|leave)\s+feedback\b",
        r"\bleave (?:some )?feedback\b",
        r"\bhere'?s my feedback\b",
        r"\bremind me\b",
        r"\bset (?:a |up a )?reminder\b",
        r"\bcheck my (?:registration|status)\b",
    )
)


def _rule_action_request(text: str) -> str | None:
    normalized = _normalize(text)
    matched = any(pattern.search(normalized) for pattern in _ACTION_REQUEST_PATTERNS)
    return "action_request" if matched else None


# ---------------------------------------------------------------------------
# Rule 3: event_inquiry -- a named KB event, with no competing "weak action
# cue" in the same message. When both are present ("Can I still sign up for
# HackFest?"), the rule abstains rather than guess between event_inquiry and
# action_request -- that's the LLM fallback's job.
# ---------------------------------------------------------------------------

_EVENT_NAME_MARKERS = (
    "hackfest",
    "genai",
    "cloud study jam",
    "flutter forward",
    "cyberctf",
    "cyber ctf",
    "design thinking bootcamp",
)
_EVENT_NAME_RE = re.compile("|".join(re.escape(m) for m in _EVENT_NAME_MARKERS))
_WEAK_ACTION_CUE_RE = re.compile(r"\b(sign up|signup|register|can i|could i|join)\b")


def _rule_event_inquiry(text: str) -> str | None:
    normalized = _normalize(text)
    if not _EVENT_NAME_RE.search(normalized):
        return None
    if _WEAK_ACTION_CUE_RE.search(normalized):
        return None
    return "event_inquiry"


# ---------------------------------------------------------------------------
# Rule 4: faq -- non-event KB topic vocabulary (teams, rules, achievements,
# contacts, recruitment, founding). Checked last among the assertive rules
# so a team name that's also part of an event name (e.g. "Cloud") never gets
# misrouted here -- event_inquiry's full-phrase check already had first
# refusal on any message naming an actual event.
# ---------------------------------------------------------------------------

_FAQ_KEYWORD_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\baiml\b",
        r"\bweb ?dev\b",
        r"\bapp ?dev\b",
        r"\bcybersecurity\b",
        r"\bdesign team\b",
        r"\bcloud team\b",
        r"\brules?\b",
        r"\bachievements?\b",
        r"\bcontacts?\b",
        r"\bpresident\b",
        r"\bvice president\b",
        r"\bvp\b",
        r"\btech head\b",
        r"\bfounded\b",
        r"\bfounder\b",
        r"\brecruitment\b",
        r"\beligib\w*\b",
        r"\bwho leads\b",
        r"\bteam lead\b",
        r"\bhow many members\b",
        r"\babout (?:the )?club\b",
        r"\bwhen was gdg\b",
    )
)


def _rule_faq(text: str) -> str | None:
    normalized = _normalize(text)
    matched = any(pattern.search(normalized) for pattern in _FAQ_KEYWORD_PATTERNS)
    return "faq" if matched else None


_RULES = (_rule_greeting, _rule_action_request, _rule_event_inquiry, _rule_faq)


@dataclass(frozen=True)
class IntentResult:
    """One classified message: its label and which path resolved it, so
    callers (and the eval script) can report the rule-path/LLM-path split
    (spec: "Report what fraction of traffic each path resolves")."""

    label: str
    path: str  # "rule" | "llm"


def classify(query: str) -> IntentResult:
    """Classify `query` into one of config.INTENT_LABELS.

    Runs every rule in priority order; the first non-abstaining rule wins.
    A blank/whitespace-only query never reaches a rule or the LLM -- there's
    nothing to classify -- and resolves directly to
    config.DEFAULT_INTENT_ON_LLM_FAILURE (mirrors qa.answer_question's
    existing "blank input never triggers a real LLM call" convention).
    """
    if not query.strip():
        return IntentResult(label=config.DEFAULT_INTENT_ON_LLM_FAILURE, path="rule")

    for rule in _RULES:
        label = rule(query)
        if label is not None:
            return IntentResult(label=label, path="rule")

    return IntentResult(label=llm_client.classify_intent(query), path="llm")
