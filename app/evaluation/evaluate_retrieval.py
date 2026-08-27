import json

from app.ingestion.embedding import generate_embedding
from app.retrieval.vector_store import search_documents
from app.config import RAG_DISTANCE_THRESHOLD, RAG_TOP_K


EVALUATION_FILE = "data/evaluation/evaluation_questions.json"
DISTANCE_THRESHOLD = RAG_DISTANCE_THRESHOLD
TOP_K = RAG_TOP_K


def load_evaluation_questions():
    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_question(question):
    query_embedding = generate_embedding(question)

    results = search_documents(
        query_embedding=query_embedding,
        n_results=TOP_K,
        distance_threshold=DISTANCE_THRESHOLD,
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    return {
        "question": question,
        "retrieved": len(documents) > 0,
        "documents": documents,
        "distances": distances,
        "metadatas": metadatas,
    }

def calculate_metrics(results):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for result in results:

        expected_retrieval = result["expected_retrieval"]
        actual_retrieval = result["retrieved"]

        if expected_retrieval and actual_retrieval:
            true_positive += 1

        elif not expected_retrieval and not actual_retrieval:
            true_negative += 1

        elif not expected_retrieval and actual_retrieval:
            false_positive += 1

        elif expected_retrieval and not actual_retrieval:
            false_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }