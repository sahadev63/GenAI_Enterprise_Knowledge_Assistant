from app.agents.orchestrator import AgentOrchestrator


FALLBACK = "I could not find the answer in the provided documents."


def answer_question_with_trace(
    question: str,
    n_results: int = 3,
    distance_threshold: float | None = 1.0,
):
    """Run the full agentic RAG workflow and return its traceable response."""
    orchestrator = AgentOrchestrator(
        n_results=n_results,
        distance_threshold=distance_threshold,
    )
    return orchestrator.run(question)


def answer_question(
    question: str,
    n_results: int = 3,
    distance_threshold: float | None = 1.0,
) -> str:
    """Backward-compatible API returning only the final answer."""
    return answer_question_with_trace(
        question,
        n_results=n_results,
        distance_threshold=distance_threshold,
    ).answer
