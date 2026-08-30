from __future__ import annotations

from app.agents.common import context_supports_question, focused_context
from app.agents.models import RetrievedEvidence
from app.config import (
    AGENT_MAX_RETRIEVAL_RETRIES,
    RAG_DISTANCE_THRESHOLD,
    RAG_TOP_K,
)
from app.generation.llm import generate_json
from app.ingestion.embedding import generate_embedding
from app.retrieval.vector_store import search_documents


class RetrieverReasonerAgent:
    """Retrieves evidence, reasons about sufficiency, and retries when needed."""

    def __init__(
        self,
        n_results: int = RAG_TOP_K,
        distance_threshold: float | None = RAG_DISTANCE_THRESHOLD,
        max_retries: int = AGENT_MAX_RETRIEVAL_RETRIES,
    ):
        self.n_results = n_results
        self.distance_threshold = distance_threshold
        self.max_retries = max(0, max_retries)

    def retrieve(self, question: str) -> RetrievedEvidence:
        queries = self._query_variants(question)
        last = RetrievedEvidence(question=question, query_used=question)

        for attempt, query in enumerate(queries[: self.max_retries + 1], start=1):
            embedding = generate_embedding(query)
            results = search_documents(
                query_embedding=embedding,
                n_results=self.n_results,
                distance_threshold=self.distance_threshold,
            )

            documents = list(dict.fromkeys((results.get("documents") or [[]])[0]))
            metadatas = ((results.get("metadatas") or [[]])[0] or [])
            distances = ((results.get("distances") or [[]])[0] or [])
            context = "\n\n".join(documents)

            heuristic = bool(documents) and context_supports_question(question, context)
            sufficient, reason = self._reason_about_sufficiency(question, context, heuristic)

            last = RetrievedEvidence(
                question=question,
                query_used=query,
                documents=documents,
                metadatas=metadatas,
                distances=distances,
                sufficient=sufficient,
                reason=reason,
                attempts=attempt,
            )

            if sufficient:
                # Narrow context before handing it to the generator.
                last.documents = [focused_context(question, context)] if focused_context(question, context) else documents
                return last

        return last

    @staticmethod
    def _query_variants(question: str) -> list[str]:
        variants = [question.strip()]
        terms = question.strip().rstrip("?")
        if terms:
            variants.append(f"{terms} policy requirements entitlement eligibility rules")
            variants.append(f"{terms} procedure allowance days amount")
        return list(dict.fromkeys(variants))

    @staticmethod
    def _reason_about_sufficiency(question: str, context: str, heuristic: bool) -> tuple[bool, str]:
        if not context or not heuristic:
            return False, "Retrieved context did not contain a meaningful concept from the question."

        prompt = f"""
You are the retrieval-reasoner agent. Decide whether the evidence is sufficient
to answer the QUESTION without outside knowledge.

Return ONLY valid JSON:
{{"sufficient": true, "reason": "brief reason"}}

QUESTION:
{question}

EVIDENCE:
{context}

Rules:
- sufficient=true only when the evidence contains the facts needed to answer.
- Do not use outside knowledge.
- If the evidence is merely related but does not answer the question, use false.
"""

        try:
            data = generate_json(prompt)
            sufficient = bool(data.get("sufficient"))
            reason = str(data.get("reason") or "LLM retrieval reasoning completed.")
            return sufficient, reason
        except Exception as error:
            # A reasoning failure must never be interpreted as successful
            # validation. Fail closed so the orchestrator can retry retrieval
            # or return an unverified fallback.
            return heuristic, (
                f"LLM sufficiency reasoning failed ({type(error).__name__}); "
                f"deterministic grounding check returned {heuristic}."
            )
