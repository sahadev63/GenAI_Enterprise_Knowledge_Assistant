from __future__ import annotations

from app.agents.common import FALLBACK, focused_context
from app.agents.models import RetrievedEvidence
from app.generation.llm import generate_answer


class GeneratorAgent:
    """Generates an answer strictly from retrieved evidence."""

    def generate(self, question: str, evidence: RetrievedEvidence) -> str:
        if not evidence.sufficient or not evidence.context:
            return FALLBACK

        context = focused_context(question, evidence.context)
        prompt = f"""
You are the generator agent in an enterprise knowledge assistant.
Answer ONLY from the supplied evidence.

Rules:
- Never use outside knowledge.
- Never guess or invent facts.
- Answer only what the question asks.
- If the evidence does not support the answer, return exactly:
{FALLBACK}
- Keep the answer concise.

QUESTION:
{question}

EVIDENCE:
{context}

ANSWER:
"""

        try:
            answer = generate_answer(prompt).strip()
        except Exception:
            return FALLBACK

        return answer or FALLBACK
