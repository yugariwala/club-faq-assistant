"""Interactive REPL for manually verifying `answer_question`.

Usage: uv run python -m backend.cli
Requires GEMINI_API_KEY (default provider) or ANTHROPIC_API_KEY (when
LLM_PROVIDER=anthropic) in the environment for grounded answers; refusals
never call the LLM so they work without one.
"""

from backend.qa import answer_question


def main() -> None:
    print("GDG On Campus Club FAQ Assistant (Slice 1: KB-grounded Q&A)")
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

        result = answer_question(query)

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
