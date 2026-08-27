from app.evaluation.evaluate_retrieval import (
    load_evaluation_questions,
    evaluate_question,
    calculate_metrics,
)


questions = load_evaluation_questions()

results = []

print("=" * 60)
print("RAG RETRIEVAL EVALUATION")
print("=" * 60)

for index, item in enumerate(questions, start=1):

    question = item["question"]
    expected_retrieval = item["expected_retrieval"]

    result = evaluate_question(question)

    result["expected_retrieval"] = expected_retrieval

    results.append(result)

    print()
    print(f"Test Case: {index}")
    print(f"Question: {question}")
    print(f"Expected Retrieval: {expected_retrieval}")
    print(f"Retrieved: {result['retrieved']}")
    print(f"Distances: {result['distances']}")


metrics = calculate_metrics(results)


print()
print("=" * 60)
print("RAG RETRIEVAL METRICS")
print("=" * 60)

print(f"True Positives  : {metrics['true_positive']}")
print(f"True Negatives  : {metrics['true_negative']}")
print(f"False Positives : {metrics['false_positive']}")
print(f"False Negatives : {metrics['false_negative']}")

print(f"Precision       : {metrics['precision'] * 100:.2f}%")
print(f"Recall          : {metrics['recall'] * 100:.2f}%")
print(f"F1 Score        : {metrics['f1_score'] * 100:.2f}%")

print("=" * 60)