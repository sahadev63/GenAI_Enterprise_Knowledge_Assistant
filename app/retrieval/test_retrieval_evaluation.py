from app.ingestion.embedding import generate_embedding
from app.retrieval.vector_store import search_documents
from app.config import RAG_DISTANCE_THRESHOLD, RAG_TOP_K


test_cases = [
    {
        "question": "How many days of annual leave are employees entitled to?",
        "expected_file": "test_policy.pdf",
        "should_retrieve": True,
    },
    {
        "question": "How many working days before planned leave should employees submit their request?",
        "expected_file": "test_policy.pdf",
        "should_retrieve": True,
    },
    {
        "question": "What is the company's laptop replacement policy?",
        "expected_file": "test_policy.pdf",
        "should_retrieve": False,
    },
    {
        "question": "What is the company's work-from-home allowance?",
        "expected_file": "test_policy.pdf",
        "should_retrieve": False,
    },
]


DISTANCE_THRESHOLD = RAG_DISTANCE_THRESHOLD
passed_count = 0
total_count = len(test_cases)

for index, test_case in enumerate(test_cases, start=1):

    question = test_case["question"]

    query_embedding = generate_embedding(question)

    results = search_documents(
        query_embedding=query_embedding,
        n_results=RAG_TOP_K,
        distance_threshold=DISTANCE_THRESHOLD,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = len(documents) > 0

    if test_case["should_retrieve"]:
        passed = (
            retrieved
            and test_case["expected_file"]
            in [metadata["file_name"] for metadata in metadatas]
        )
    else:
        passed = not retrieved

    print("=" * 60)
    print(f"Test Case: {index}")
    print(f"Question: {question}")
    print(f"Distances: {distances}")
    print(f"Retrieved: {retrieved}")

    if passed:
        print("Result: PASS")
        passed_count += 1
    else:
        print("Result: FAIL")

print()
print("=" * 60)
print("RAG RETRIEVAL EVALUATION")
print("=" * 60)
print(f"Total Tests : {total_count}")
print(f"Passed      : {passed_count}")
print(f"Failed      : {total_count - passed_count}")

success_rate = (passed_count / total_count) * 100

print(f"Success Rate: {success_rate:.2f}%")
print("=" * 60)