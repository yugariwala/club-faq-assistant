"""Interactive REPL for manually verifying `answer_question`.

Usage: uv run python -m backend.cli
Requires GEMINI_API_KEY (default provider) or ANTHROPIC_API_KEY (when
LLM_PROVIDER=anthropic) in the environment for grounded answers; refusals
never call the LLM so they work without one.

`.env` (see .env.example) is loaded here, once, at this application entry
point -- library modules (backend.llm_client, backend.config, ...) only
ever read from `os.environ` and never load `.env` themselves, so any other
entry point (a future dashboard, a script) must call `load_dotenv()` too.
"""

import logging
import uuid

from dotenv import load_dotenv

load_dotenv()

# Route logs (including the full tracebacks `qa.py` logs on an LLM-call
# failure) to a file instead of the console -- a REPL/demo session should
# show the user-facing degraded-answer message, not a raw stack trace,
# while still keeping the detail available for debugging.
logging.basicConfig(
    filename="club_faq_assistant.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from backend import llm_client  # noqa: E402 -- after load_dotenv(), see module docstring
from backend.qa import answer_question  # noqa: E402 -- after load_dotenv(), see module docstring


def main() -> None:
    missing_var = llm_client.missing_api_key_var()
    if missing_var:
        print(
            f"WARNING: {missing_var} is not set, so grounded answers will fail "
            "with a quota/error message until it's set in .env (see README.md "
            "Setup). Refusals (no KB match) still work without it.\n"
        )

    # One session_id per REPL run, so this run's whole conversation shares
    # one bounded history (spec: Code Map -> "generate one session_id per
    # REPL run; pass it to every answer_question call").
    session_id = uuid.uuid4().hex

    print("GDG On Campus Club FAQ Assistant (Slice 2: multi-turn memory)")
    print("Ask a question about the club, or type 'quit' / 'exit' to stop.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue

        result = answer_question(query, session_id)

        if result.rewritten_query and result.rewritten_query != query:
            print("[rewritten: {}]".format(result.rewritten_query))

        print(result.answer)
        if result.refused:
            print("[refused | no source | score={:.3f}]".format(result.score))
        else:
            print(
                "[source={} | score={:.3f}]".format(result.source_section, result.score)
            )
        print()


if __name__ == "__main__":
    main()
