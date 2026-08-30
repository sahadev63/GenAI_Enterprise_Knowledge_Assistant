from __future__ import annotations

from app.agents.common import (
    FALLBACK,
    context_supports_question,
    sentence_list,
    specific_query_terms,
    tokens,
)
from app.agents.models import RetrievedEvidence, ValidationResult
from app.generation.llm import generate_json


class ValidatorAgent:
    """Checks generated claims against retrieved evidence.

    Validation is fail-safe: an exception in the LLM validator never allows an
    unvalidated answer to pass through. A deterministic evidence check is used
    as the fallback, and an exception in that check produces an explicit
    validation failure.
    """

    def validate(
        self,
        question: str,
        answer: str,
        evidence: RetrievedEvidence,
    ) -> ValidationResult:
        # Always return a validation result, even for invalid/empty inputs.
        try:
            if not answer or answer.strip() == FALLBACK:
                return ValidationResult(
                    supported=False,
                    checked=True,
                    unsupported_claims=["No generated answer was available to validate."],
                    reason="Validation completed: no answer was available.",
                )

            if not evidence.context:
                return ValidationResult(
                    supported=False,
                    checked=True,
                    unsupported_claims=["No retrieved evidence was available to validate the answer."],
                    reason="Validation completed: retrieved evidence was empty.",
                )

            prompt = f"""
You are the validation agent for an enterprise knowledge assistant.
Check whether every factual claim in ANSWER is supported by EVIDENCE.

Return ONLY valid JSON:
{{
  "supported": true,
  "unsupported_claims": [],
  "reason": "brief reason"
}}

Do not use outside knowledge. If any material claim is not supported,
supported=false.

QUESTION:
{question}

EVIDENCE:
{evidence.context}

ANSWER:
{answer}
"""

            try:
                data = generate_json(prompt)
                supported = data.get("supported")
                if not isinstance(supported, bool):
                    raise ValueError("Validator response did not contain a boolean 'supported' field.")

                claims = data.get("unsupported_claims") or []
                if not isinstance(claims, list):
                    claims = [str(claims)]
                claims = [str(item).strip() for item in claims if str(item).strip()]
                reason = str(data.get("reason") or "LLM validator completed.")

                # Never trust a malformed positive validation response.
                if supported and claims:
                    return ValidationResult(
                        supported=False,
                        checked=True,
                        unsupported_claims=claims,
                        reason="Validator returned conflicting positive/unsupported-claim results.",
                    )

                return ValidationResult(
                    supported=supported,
                    checked=True,
                    unsupported_claims=claims,
                    reason=reason,
                )

            except Exception as error:
                # Validator LLM failure is handled explicitly. The answer is
                # NOT considered validated merely because the LLM failed.
                fallback_result = self.validate_deterministic(question, answer, evidence)
                fallback_result.reason = (
                    f"LLM validation failed ({type(error).__name__}); "
                    f"deterministic validation fallback used. {fallback_result.reason}"
                )
                return fallback_result

        except Exception as error:
            # Last-resort fail-closed behavior. This result is consumed by the
            # orchestrator and prevents an answer from being reported as verified.
            return ValidationResult(
                supported=False,
                checked=False,
                unsupported_claims=[
                    f"Validation failed unexpectedly: {type(error).__name__}: {error}"
                ],
                reason="Validation exception was caught; answer is not considered verified.",
            )

    @staticmethod
    def validate_deterministic(
        question: str,
        answer: str,
        evidence: RetrievedEvidence,
    ) -> ValidationResult:
        """Fail-safe lexical validation used when the validator LLM fails."""
        try:
            if not answer or not evidence.context:
                return ValidationResult(
                    supported=False,
                    checked=True,
                    unsupported_claims=["Answer or evidence was empty."],
                    reason="Deterministic validation could not run with empty input.",
                )

            question_terms = specific_query_terms(question)
            evidence_terms = tokens(evidence.context)
            answer_sentences = sentence_list(answer)

            if not question_terms.intersection(evidence_terms):
                return ValidationResult(
                    supported=False,
                    checked=True,
                    unsupported_claims=["The answer topic is not grounded in the retrieved evidence."],
                    reason="Deterministic validation found no meaningful question/evidence overlap.",
                )

            # Each factual sentence must share meaningful terms with the
            # retrieved evidence. This is intentionally conservative.
            unsupported = []
            for sentence in answer_sentences:
                sentence_terms = tokens(sentence)
                if sentence_terms and not sentence_terms.intersection(evidence_terms):
                    unsupported.append(sentence)

            # Also require the answer to remain on-topic.
            answer_on_topic = any(
                specific_query_terms(question).intersection(tokens(sentence))
                for sentence in answer_sentences
            )

            supported = answer_on_topic and not unsupported
            return ValidationResult(
                supported=supported,
                checked=True,
                unsupported_claims=unsupported
                if unsupported
                else ([] if supported else ["The generated answer could not be deterministically grounded."]),
                reason="Deterministic evidence-grounding validation completed.",
            )

        except Exception as error:
            return ValidationResult(
                supported=False,
                checked=False,
                unsupported_claims=[
                    f"Deterministic validation failed: {type(error).__name__}: {error}"
                ],
                reason="Both LLM and deterministic validation failed; answer is not verified.",
            )
