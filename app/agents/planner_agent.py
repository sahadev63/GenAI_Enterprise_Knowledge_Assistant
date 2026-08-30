from __future__ import annotations

import re

from app.agents.common import split_multi_part_question
from app.agents.models import QueryPlan
from app.generation.llm import generate_json


class PlannerAgent:
    """Plans the user's task before retrieval begins.

    The question decomposition is deterministic so an LLM cannot replace a
    precise user question with vague workflow steps such as "identify policy"
    or "provide total".  The local LLM may still provide a strategy, but the
    actual retrieval subtasks always preserve the user's wording and intent.
    This makes retrieval, partial-information handling, and validation
    deterministic and reproducible.
    """

    def plan(self, question: str) -> QueryPlan:
        question = (question or "").strip()
        fallback_subtasks = split_multi_part_question(question)

        # The user's question is the source of truth for retrieval subtasks.
        # Do not allow the planning LLM to invent meta-level tasks: those can
        # retrieve generic policy text and cause valid questions to be rejected
        # by the evidence validator.
        strategy = "retrieve each question part, check evidence sufficiency, generate, then validate"

        prompt = f"""
You are the planning agent for an enterprise document assistant.
Provide ONLY a short execution strategy for this user question.

Return ONLY valid JSON:
{{"strategy": "short description"}}

Rules:
- Do not create subtasks.
- Do not answer the question.
- Do not invent document facts.
- The strategy must describe retrieval, evidence checking, generation, and validation.

User question:
{question}
"""

        try:
            data = generate_json(prompt)
            candidate = data.get("strategy")
            if isinstance(candidate, str) and candidate.strip():
                strategy = candidate.strip()
        except Exception:
            # Deterministic strategy is sufficient when the local model is
            # unavailable or returns malformed output.
            pass

        return QueryPlan(
            original_question=question,
            subtasks=fallback_subtasks[:4],
            strategy=strategy,
        )

    @staticmethod
    def _fallback_plan(question: str) -> QueryPlan:
        """Backward-compatible deterministic plan helper."""
        subtasks = split_multi_part_question(question)
        if len(subtasks) == 1 and ";" in question:
            subtasks = [
                part.strip() + "?"
                for part in re.split(r";", question)
                if part.strip()
            ]
        return QueryPlan(
            original_question=question,
            subtasks=subtasks[:4],
            strategy="retrieve each subtask, check evidence sufficiency, generate, then validate",
        )
