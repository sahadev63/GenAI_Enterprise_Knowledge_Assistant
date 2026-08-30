from __future__ import annotations

import re

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

            # Prefer a deterministic answerability check for clear evidence.
            # The local LLM is useful for ambiguous cases, but it can
            # incorrectly reject an exact policy statement.  A direct factual
            # match must therefore not be downgraded by the LLM judge.
            deterministic_sufficient, deterministic_reason = (
                self._deterministic_answerability(question, context)
            )

            if deterministic_sufficient:
                sufficient, reason = True, deterministic_reason
            else:
                sufficient, reason = self._reason_about_sufficiency(
                    question, context, heuristic
                )

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
    def _deterministic_answerability(question: str, context: str) -> tuple[bool, str]:
        """Detect clear answer-bearing evidence without depending on the LLM.

        This protects known policy facts from false-negative LLM sufficiency
        judgments while remaining conservative for unrelated questions.
        """
        if not question.strip() or not context.strip():
            return False, "Question or retrieved context was empty."

        from app.agents.common import sentence_list, specific_query_terms, tokens

        question_terms = specific_query_terms(question)
        if not question_terms:
            return False, "No meaningful question terms were available."

        question_lower = question.lower()
        context_sentences = sentence_list(context)

        # Question-type requirements make the check more precise.
        needs_number = bool(
            re.search(r"\b(how many|how much|number of|days|amount)\b", question_lower)
        )
        needs_person_or_role = bool(
            re.search(r"\b(who|responsible|approv)\b", question_lower)
        )
        needs_condition = bool(
            re.search(r"\b(when|condition|require|requirement|eligible|eligibility)\b", question_lower)
        )

        for sentence in context_sentences:
            sentence_terms = tokens(sentence)
            overlap = question_terms.intersection(sentence_terms)

            # One distinctive domain term can be sufficient for a topic
            # question (for example, "annual leave"), while question-type
            # checks below still require the actual fact for numeric/timing/
            # responsibility questions.  Generic words are removed by
            # specific_query_terms(), so this does not rely on "policy",
            # "employee", "leave", etc. alone.
            if len(overlap) < 1:
                continue

            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", sentence)
            if needs_number and not numbers:
                continue

            # For "who" questions, evidence should contain a likely role/person
            # indicator rather than merely repeating the topic.
            if needs_person_or_role and not re.search(
                r"\b(manager|managers|supervisor|supervisors|hr|human resources|owner|team|department|responsible)\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                continue

            # For timing/condition questions, require an action/condition
            # concept in the same evidence sentence.
            if needs_condition and not re.search(
                r"\b(before|after|when|required|request|submit|eligible|eligibility|condition|requirement|responsible|approve|approving|approved)\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                continue

            return (
                True,
                "Deterministic answerability check found direct evidence in the retrieved policy text.",
            )

        return False, "Retrieved evidence did not contain enough direct terms to answer deterministically."

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
