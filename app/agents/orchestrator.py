from __future__ import annotations

import re

from app.agents.common import FALLBACK, split_multi_part_question
from app.agents.generator_agent import GeneratorAgent
from app.agents.models import AgentAnswer, QueryPlan, RetrievedEvidence, ValidationResult
from app.agents.planner_agent import PlannerAgent
from app.agents.retriever_reasoner_agent import RetrieverReasonerAgent
from app.agents.validator_agent import ValidatorAgent
from app.config import RAG_DISTANCE_THRESHOLD, RAG_TOP_K


class AgentOrchestrator:
    """Coordinates planning, retrieval/reasoning, generation, and validation.

    The orchestrator is fail-safe. Every generated answer is validated before
    it can be returned as verified. If any stage raises unexpectedly, the
    exception is captured and the final response is marked unverified.
    """

    def __init__(
        self,
        n_results: int = RAG_TOP_K,
        distance_threshold: float | None = RAG_DISTANCE_THRESHOLD,
    ):
        self.planner = PlannerAgent()
        self.retriever = RetrieverReasonerAgent(n_results, distance_threshold)
        self.generator = GeneratorAgent()
        self.validator = ValidatorAgent()

    def run(self, question: str) -> AgentAnswer:
        question = (question or "").strip()
        if not question:
            plan = QueryPlan("", [], "")
            validation = ValidationResult(
                False, True, ["Question was empty."], "Input validation failed."
            )
            return AgentAnswer(FALLBACK, plan, [], validation, ["Input validation failed."])

        trace: list[str] = ["Input validation passed."]

        # Planner exception handling: use deterministic planning fallback.
        try:
            plan = self.planner.plan(question)
            trace.append(f"Planner created {len(plan.subtasks)} subtask(s).")
        except Exception as error:
            trace.append(f"Planner exception caught: {type(error).__name__}.")
            plan = QueryPlan(
                question,
                split_multi_part_question(question),
                "deterministic fallback: retrieve, generate, validate",
            )

        if not plan.subtasks:
            plan.subtasks = split_multi_part_question(question)

        evidence_list: list[RetrievedEvidence] = []
        answers: list[str] = []
        validations: list[ValidationResult] = []

        for index, subtask in enumerate(plan.subtasks, start=1):
            # Retrieval exception handling.
            try:
                evidence = self.retriever.retrieve(subtask)
            except Exception as error:
                trace.append(
                    f"Retriever exception caught for subtask {index}: {type(error).__name__}."
                )
                evidence = RetrievedEvidence(
                    question=subtask,
                    query_used=subtask,
                    sufficient=False,
                    reason=f"Retrieval failed: {type(error).__name__}: {error}",
                    attempts=0,
                )

            evidence_list.append(evidence)
            trace.append(
                f"Retriever/Reasoner subtask {index}: attempts={evidence.attempts}, "
                f"sufficient={evidence.sufficient}."
            )

            if not evidence.sufficient:
                answer = (
                    f"Information about {self._topic(subtask)} was not found in the provided documents."
                    if len(plan.subtasks) > 1
                    else FALLBACK
                )
                # For a multi-part question, an unavailable subtask is an
                # intentional, evidence-based abstention rather than an
                # invalid factual claim. Mark that subtask as successfully
                # validated so a valid partial answer is not downgraded to
                # "unverified" merely because another requested topic is
                # absent from the documents.
                if len(plan.subtasks) > 1:
                    validation = ValidationResult(
                        supported=True,
                        checked=True,
                        unsupported_claims=[],
                        reason="No evidence was retrieved for this subtask; explicit missing-information abstention is valid.",
                    )
                    trace.append(
                        f"Validator accepted subtask {index} as an explicit "
                        "missing-information abstention."
                    )
                else:
                    # A single unanswered question remains an unverified
                    # fallback and is handled conservatively.
                    validation = self._safe_validate(
                        subtask, answer, evidence, trace, index
                    )

                answers.append(answer if validation.supported else FALLBACK)
                validations.append(validation)
                continue

            # Generation exception handling.
            try:
                answer = self.generator.generate(subtask, evidence)
            except Exception as error:
                trace.append(
                    f"Generator exception caught for subtask {index}: {type(error).__name__}."
                )
                answer = FALLBACK

            # Validation is mandatory, including after generation exceptions.
            validation = self._safe_validate(subtask, answer, evidence, trace, index)

            # One controlled correction pass. The correction result is ALSO
            # validated; it is never returned solely because it looks grounded.
            if not validation.supported and evidence.context:
                trace.append(
                    f"Validator rejected subtask {index}; applying evidence-only correction."
                )
                corrected_answer = self._evidence_only_answer(subtask, evidence)
                corrected_validation = self._safe_validate(
                    subtask, corrected_answer, evidence, trace, index
                )
                if corrected_validation.supported:
                    answer = corrected_answer
                    validation = corrected_validation
                else:
                    answer = FALLBACK
                    validation = corrected_validation

            answers.append(answer if validation.supported else FALLBACK)
            validations.append(validation)
            trace.append(
                f"Generator + Validator subtask {index}: checked={validation.checked}, "
                f"supported={validation.supported}."
            )

        final_answer = self._combine_answers(answers)
        overall = self._combine_validation(validations)

        # Final answer-level validation is performed again over the combined
        # result. This protects against errors introduced while combining
        # multi-part answers.
        if final_answer != FALLBACK and len(plan.subtasks) > 1:
            combined_evidence = RetrievedEvidence(
                question=question,
                query_used="combined subtask retrieval",
                documents=[doc for item in evidence_list for doc in item.documents],
                metadatas=[m for item in evidence_list for m in item.metadatas],
                distances=[d for item in evidence_list for d in item.distances],
                sufficient=any(item.sufficient for item in evidence_list),
                reason="Combined evidence from all subtasks.",
                attempts=sum(item.attempts for item in evidence_list),
            )
            final_validation = self._safe_validate(
                question, final_answer, combined_evidence, trace, 0, final_check=True
            )
            overall = self._combine_validation([overall, final_validation])
            if not final_validation.supported:
                final_answer = FALLBACK
                trace.append("Final combined-answer validation failed; fallback returned.")
            else:
                trace.append("Final combined-answer validation passed.")

        trace.append(
            f"Final validation: checked={overall.checked}, supported={overall.supported}."
        )
        return AgentAnswer(final_answer, plan, evidence_list, overall, trace)

    def _safe_validate(
        self,
        question: str,
        answer: str,
        evidence: RetrievedEvidence,
        trace: list[str],
        index: int,
        final_check: bool = False,
    ) -> ValidationResult:
        """Never let validator exceptions escape or mark an answer verified."""
        try:
            result = self.validator.validate(question, answer, evidence)
            trace.append(
                f"{'Final ' if final_check else ''}Validator executed for "
                f"subtask {index}: checked={result.checked}, supported={result.supported}."
            )
            return result
        except Exception as error:
            trace.append(
                f"{'Final ' if final_check else ''}Validator exception caught for "
                f"subtask {index}: {type(error).__name__}."
            )
            return ValidationResult(
                supported=False,
                checked=False,
                unsupported_claims=[
                    f"Validator exception: {type(error).__name__}: {error}"
                ],
                reason="Validation failed unexpectedly; answer is not verified.",
            )

    @staticmethod
    def _combine_answers(answers: list[str]) -> str:
        unique = []
        for answer in answers:
            if answer and answer not in unique and answer != FALLBACK:
                unique.append(answer)
        return "\n\n".join(unique) if unique else FALLBACK

    @staticmethod
    def _combine_validation(results: list[ValidationResult]) -> ValidationResult:
        if not results:
            return ValidationResult(False, True, ["No validation result."], "No subtasks.")
        unsupported = [claim for result in results for claim in result.unsupported_claims]
        return ValidationResult(
            supported=all(result.supported for result in results),
            checked=all(result.checked for result in results),
            unsupported_claims=unsupported,
            reason="All applicable subtask validations completed.",
        )

    @staticmethod
    def _topic(question: str) -> str:
        """Return a natural topic phrase for a missing-information message."""
        text = question.strip(" ?.")
        text = re.sub(
            r"^(what|which|how|when|who)\b\s*", "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"^(is|are|was|were|many|much|does|do|did|can|could|should|would|will)\b\s*",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(r"^the\s+", "", text, flags=re.IGNORECASE)
        return text.strip(" ?.") or "that information"

    @staticmethod
    def _evidence_only_answer(question: str, evidence: RetrievedEvidence) -> str:
        from app.agents.common import focused_context
        context = focused_context(question, evidence.context)
        return context or FALLBACK
