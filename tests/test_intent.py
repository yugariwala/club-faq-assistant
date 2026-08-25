"""Unit tests for `backend.intent` -- the rule layer and the hybrid
classify() orchestrator. `llm_client.classify_intent` is mocked throughout,
so these tests need no network access or API key.
"""

from unittest.mock import patch

from backend import config
from backend.intent import (
    IntentResult,
    _rule_action_request,
    _rule_event_inquiry,
    _rule_faq,
    _rule_greeting,
    classify,
)

# ---------------------------------------------------------------------------
# Rule: greeting -- whole-message match only
# ---------------------------------------------------------------------------


def test_greeting_rule_matches_bare_greetings():
    assert _rule_greeting("Hi!") == "greeting"
    assert _rule_greeting("hello there") == "greeting"
    assert _rule_greeting("Thanks so much!") == "greeting"
    assert _rule_greeting("bye") == "greeting"


def test_greeting_rule_abstains_when_greeting_has_extra_content():
    assert _rule_greeting("hi, when's HackFest?") is None
    assert _rule_greeting("thank you for the help") is None


# ---------------------------------------------------------------------------
# Rule: action_request -- self-referential imperative only
# ---------------------------------------------------------------------------


def test_action_request_rule_matches_self_referential_commands():
    assert _rule_action_request("Register me for HackFest 2025.") == "action_request"
    assert _rule_action_request("Sign me up for the Cloud Study Jam.") == "action_request"
    assert _rule_action_request("I'd like to submit feedback about the workshop.") == (
        "action_request"
    )
    assert _rule_action_request("Remind me about HackFest.") == "action_request"


def test_action_request_rule_abstains_on_informational_phrasing():
    """Topic overlap with an action word alone must not qualify -- these are
    questions, not commands."""
    assert _rule_action_request("How do I register for events?") is None
    assert _rule_action_request("Is registration open for HackFest?") is None
    assert _rule_action_request("Can I still sign up for HackFest?") is None


# ---------------------------------------------------------------------------
# Rule: event_inquiry -- named event, no competing weak action cue
# ---------------------------------------------------------------------------


def test_event_inquiry_rule_matches_named_events():
    assert _rule_event_inquiry("When is HackFest 2025?") == "event_inquiry"
    assert _rule_event_inquiry("Is the Cloud Study Jam still upcoming?") == "event_inquiry"
    assert _rule_event_inquiry("Has Flutter Forward already happened?") == "event_inquiry"


def test_event_inquiry_rule_is_correct_on_the_open_for_registration_example():
    """The example from the brief: no self-referential verb -> resolves to
    event_inquiry (informational status question), not action_request."""
    assert _rule_event_inquiry("Is HackFest still open for registration?") == "event_inquiry"


def test_event_inquiry_rule_abstains_when_message_also_carries_a_weak_action_cue():
    """"Can I sign up for HackFest?" is ambiguous between event_inquiry and
    action_request -- the rule must abstain rather than guess."""
    assert _rule_event_inquiry("Can I still sign up for HackFest?") is None


def test_event_inquiry_rule_abstains_without_a_named_event():
    assert _rule_event_inquiry("What events are upcoming this semester?") is None


# ---------------------------------------------------------------------------
# Rule: faq -- non-event KB topic vocabulary
# ---------------------------------------------------------------------------


def test_faq_rule_matches_team_and_general_club_topics():
    assert _rule_faq("Who leads the AIML team?") == "faq"
    assert _rule_faq("What are the club's rules for staying active?") == "faq"
    assert _rule_faq("Tell me about the recruitment process.") == "faq"
    assert _rule_faq("How many members are in the Web Dev team?") == "faq"


def test_faq_rule_abstains_on_unrelated_topics():
    assert _rule_faq("What's the weather like today?") is None


# ---------------------------------------------------------------------------
# classify() orchestrator
# ---------------------------------------------------------------------------


def test_classify_returns_rule_path_when_a_rule_matches():
    with patch("backend.intent.llm_client.classify_intent") as mock_classify:
        result = classify("Who leads the AIML team?")

    mock_classify.assert_not_called()
    assert isinstance(result, IntentResult)
    assert result.label == "faq"
    assert result.path == "rule"


def test_classify_falls_through_to_llm_when_every_rule_abstains():
    with patch(
        "backend.intent.llm_client.classify_intent", return_value="out_of_scope"
    ) as mock_classify:
        result = classify("Can you help me with my calculus homework?")

    mock_classify.assert_called_once_with("Can you help me with my calculus homework?")
    assert result.label == "out_of_scope"
    assert result.path == "llm"


def test_classify_blank_query_never_calls_a_rule_or_the_llm():
    with patch("backend.intent.llm_client.classify_intent") as mock_classify:
        result = classify("   ")

    mock_classify.assert_not_called()
    assert result.label == config.DEFAULT_INTENT_ON_LLM_FAILURE
    assert result.path == "rule"


def test_classify_action_request_wins_over_event_inquiry_when_both_topics_present():
    """"Register me for HackFest" contains both an action verb and a named
    event -- action_request (checked first) must win, not event_inquiry."""
    with patch("backend.intent.llm_client.classify_intent") as mock_classify:
        result = classify("Register me for HackFest 2025.")

    mock_classify.assert_not_called()
    assert result.label == "action_request"
    assert result.path == "rule"
