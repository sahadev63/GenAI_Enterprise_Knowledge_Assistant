from app.rag.generator import generate_answer


CONTEXT = """
Employees are entitled to 24 days of annual leave every year.
"""


def run_question(context: str, question: str) -> str:
    """Run a generator case and print the result."""
    answer = generate_answer(
        question=question,
        context=context,
    )

    print("\n" + "=" * 60)
    print("Question:")
    print(question)

    print("\nContext:")
    print(context)

    print("\nAnswer:")
    print(answer)

    return answer


def test_question_generation():
    """Smoke/regression test for grounded generation.

    The test intentionally uses the project's 24-day annual-leave policy.
    Unsupported questions must not be turned into unsupported factual claims.
    """
    answer1 = run_question(
        CONTEXT,
        "How many annual leave days are employees eligible for?",
    )
    assert "24" in answer1

    answer2 = run_question(
        CONTEXT,
        "How many sick leave days are employees eligible for?",
    )
    assert "24 days of sick leave" not in answer2.lower()
    assert "sick leave" not in answer2.lower() or "could not find" in answer2.lower()

    answer3 = run_question(
        CONTEXT,
        "Can employees take annual leave every month?",
    )
    # The evidence establishes an annual entitlement, not a monthly schedule.
    assert "every month" not in answer3.lower()


if __name__ == "__main__":
    # Preserve the original manual-script behavior:
    # python app/rag/test_generator.py
    run_question(
        CONTEXT,
        "How many annual leave days are employees eligible for?",
    )
    run_question(
        CONTEXT,
        "How many sick leave days are employees eligible for?",
    )
    run_question(
        CONTEXT,
        "Can employees take annual leave every month?",
    )
