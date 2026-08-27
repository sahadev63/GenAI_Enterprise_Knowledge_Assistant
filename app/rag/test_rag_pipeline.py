from app.rag.rag_pipeline import answer_question


TEST_CASES = [
    {
        "name": "Known Question",
        "question": "How many days of annual leave are employees entitled to?",
        "expected": "24 days",
    },
    {
        "name": "Paraphrased Known Question",
        "question": "How much annual leave does an employee receive each year?",
        "expected": "24 days",
    },
    {
        "name": "Unknown Question",
        "question": "What is the company's laptop replacement policy?",
        "expected": "could not find",
    },
    {
        "name": "Partial Information",
        "question": (
            "What are the annual leave and maternity leave policies?"
        ),
        "expected": "24 days",
        "also_expected": "maternity leave policies was not found",
    },
    {
        "name": "Unrelated Question",
        "question": "What is the capital of France?",
        "expected": "could not find",
    },
]


passed = 0

print("=" * 70)
print("RAG PIPELINE EVALUATION")
print("=" * 70)

for index, test in enumerate(TEST_CASES, start=1):
    answer = answer_question(test["question"])

    checks = [
        test["expected"].lower() in answer.lower()
    ]

    if test.get("also_expected"):
        checks.append(
            test["also_expected"].lower() in answer.lower()
        )

    result = all(checks)

    if result:
        passed += 1

    print(f"\nTest {index}: {test['name']}")
    print(f"Question: {test['question']}")
    print(f"Answer: {answer}")
    print(f"Result: {'PASS' if result else 'FAIL'}")

print()
print("=" * 70)
print(f"Passed: {passed}/{len(TEST_CASES)}")
print(f"Success Rate: {passed / len(TEST_CASES) * 100:.2f}%")
print("=" * 70)
