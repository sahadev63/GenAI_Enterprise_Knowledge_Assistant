from app.rag.rag_pipeline import answer_question


test_cases = [
    {
        "question": "How many days of annual leave are employees entitled to?",
        "expected": "24 days",
        "answerable": True
    },
    {
        "question": "How many working days before planned leave should employees submit their request?",
        "expected": "5 working days",
        "answerable": True
    },
    {
        "question": "What is the company's laptop replacement policy?",
        "expected": "I could not find the answer in the provided documents.",
        "answerable": False
    },
    {
        "question": "What is the company's work-from-home allowance?",
        "expected": "I could not find the answer in the provided documents.",
        "answerable": False
    }
]


for index, test_case in enumerate(test_cases, start=1):

    question = test_case["question"]

    answer = answer_question(question)

    print("=" * 60)
    print(f"Test Case: {index}")
    print(f"Question: {question}")
    print(f"Expected: {test_case['expected']}")
    print(f"Actual:   {answer}")

    if test_case["answerable"]:
        if test_case["expected"].lower() in answer.lower():
            print("Result: PASS")
        else:
            print("Result: FAIL")
    else:
        if "could not find the answer" in answer.lower():
            print("Result: PASS")
        else:
            print("Result: FAIL")