from app.rag.generator import generate_answer


def test_question(context: str, question: str):
    answer = generate_answer(
        question=question,
        context=context
    )

    print("\n" + "=" * 60)
    print("Question:")
    print(question)

    print("\nContext:")
    print(context)

    print("\nAnswer:")
    print(answer)


context = """
Employees are eligible for 20 days of annual leave per calendar year.
"""


# Test 1: Answer exists directly in context
test_question(
    context,
    "How many annual leave days are employees eligible for?"
)


# Test 2: Answer does not exist in context
test_question(
    context,
    "How many sick leave days are employees eligible for?"
)


# Test 3: Semantically related but different question
test_question(
    context,
    "Can employees take annual leave every month?"
)