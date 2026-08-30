from __future__ import annotations

import re

from app.agents.common import split_multi_part_question
from app.agents.models import QueryPlan
from app.generation.llm import generate_json


class PlannerAgent:
    """Plans the user's task before retrieval begins.

    The LLM is used when available; a deterministic planner is retained as a
    safe fallback so the application remains usable when the local model is
    unavailable or returns malformed JSON.
    """

    def plan(self, question: str) -> QueryPlan:
        question = question.strip()
        fallback = self._fallback_plan(question)

        prompt = f"""
You are the planning agent for an enterprise document assistant.
Create a short execution plan for the user's question.

Return ONLY valid JSON with this schema:
{{
  "subtasks": ["question or sub-question"],
  "strategy": "short description"
}}

Rules:
- Split independent requests into separate subtasks.
- Do not answer the question.
- Do not invent document facts.
- Use at most 4 subtasks.

User question:
{question}
"""

        try:
            data = generate_json(prompt)
            subtasks = data.get("subtasks")
            strategy = data.get("strategy") or fallback.strategy
            if isinstance(subtasks, list):
                clean = [str(item).strip() for item in subtasks if str(item).strip()]
                if clean:
                    return QueryPlan(question, clean[:4], str(strategy))
        except Exception:
            pass

        return fallback

    @staticmethod
    def _fallback_plan(question: str) -> QueryPlan:
        subtasks = split_multi_part_question(question)
        # A second deterministic split handles common semicolon-separated requests.
        if len(subtasks) == 1 and ";" in question:
            subtasks = [part.strip() + "?" for part in re.split(r";", question) if part.strip()]
        return QueryPlan(
            original_question=question,
            subtasks=subtasks[:4],
            strategy="retrieve each subtask, check evidence sufficiency, generate, then validate",
        )
