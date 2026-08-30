from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.agents.common import (
    FALLBACK,
    sentence_list,
    specific_query_terms,
    tokens,
)
from app.agents.models import RetrievedEvidence, ValidationResult
from app.generation.llm import generate_json


class ValidatorAgent:
    """Validate generated claims against the retrieved evidence.

    Validation is fail-safe, but it also avoids false negatives from the LLM
    judge.  A deterministic evidence check is performed first for clear
    evidence matches (including close paraphrases).  The LLM judge is used for
    cases where deterministic validation cannot make a confident decision.
    """

    def validate(
        self,
        question: str,
        answer: str,
        evidence: RetrievedEvidence,
    ) -> ValidationResult:
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

            # First accept clear evidence-grounded answers deterministically.
            # This is important because an LLM judge can incorrectly reject an
            # answer even when the answer is directly present in the source.
            deterministic = self.validate_deterministic(question, answer, evidence)
            if deterministic.supported:
                deterministic.reason = (
                    "Answer is directly or strongly supported by retrieved evidence. "
                    "Deterministic grounding check passed."
                )
                return deterministic

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

A standard abstention statement such as "Information about X was not found
in the provided documents" is not itself an unsupported factual claim. Judge
the factual answer claims against EVIDENCE.

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
                fallback_result = deterministic
                fallback_result.reason = (
                    f"LLM validation failed ({type(error).__name__}); "
                    f"deterministic validation fallback used. {fallback_result.reason}"
                )
                return fallback_result

        except Exception as error:
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
        """Ground answer claims against evidence without relying on an LLM.

        The check deliberately favors false negatives over false positives, but
        recognizes exact evidence sentences and close paraphrases.  Standard
        missing-information statements are treated as abstentions rather than
        hallucinated claims.
        """
        try:
            if not answer or not evidence.context:
                return ValidationResult(
                    supported=False,
                    checked=True,
                    unsupported_claims=["Answer or evidence was empty."],
                    reason="Deterministic validation could not run with empty input.",
                )

            context_sentences = sentence_list(evidence.context)
            answer_sentences = sentence_list(answer)
            if not answer_sentences or not context_sentences:
                return ValidationResult(
                    supported=False,
                    checked=True,
                    unsupported_claims=["No factual answer/evidence sentences were available."],
                    reason="Deterministic validation found no usable sentences.",
                )

            question_terms = specific_query_terms(question)
            evidence_terms = tokens(evidence.context)

            if not question_terms.intersection(evidence_terms):
                return ValidationResult(
                    supported=False,
                    checked=True,
                    unsupported_claims=["The answer topic is not grounded in the retrieved evidence."],
                    reason="Deterministic validation found no meaningful question/evidence overlap.",
                )

            factual_sentences = []
            for sentence in answer_sentences:
                if ValidatorAgent._is_abstention_statement(sentence):
                    continue
                factual_sentences.append(sentence)

            # An answer containing only an explicit abstention is valid only if
            # the evidence does not actually provide an answer.
            if not factual_sentences:
                return ValidationResult(
                    supported=False,
                    checked=True,
                    unsupported_claims=[],
                    reason="Answer contains only an information-not-found statement.",
                )

            unsupported: list[str] = []
            for sentence in factual_sentences:
                if not ValidatorAgent._sentence_supported(sentence, context_sentences):
                    unsupported.append(sentence)

            answer_on_topic = any(
                question_terms.intersection(tokens(sentence))
                for sentence in factual_sentences
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

    @staticmethod
    def _sentence_supported(answer_sentence: str, evidence_sentences: list[str]) -> bool:
        """Return true for exact or strong lexical/paraphrase evidence matches."""
        normalized_answer = ValidatorAgent._normalize(answer_sentence)
        answer_terms = ValidatorAgent._claim_terms(answer_sentence)
        if not answer_terms:
            return False

        for evidence_sentence in evidence_sentences:
            normalized_evidence = ValidatorAgent._normalize(evidence_sentence)

            # Strongest case: the generated claim is copied from the source,
            # possibly with minor punctuation/casing differences.
            if normalized_answer and normalized_answer in normalized_evidence:
                return True
            if normalized_evidence and normalized_evidence in normalized_answer:
                return True

            evidence_terms = ValidatorAgent._claim_terms(evidence_sentence)
            if not evidence_terms:
                continue

            shared_terms = answer_terms.intersection(evidence_terms)
            overlap = len(shared_terms) / len(answer_terms)
            reverse_overlap = len(shared_terms) / len(evidence_terms)
            similarity = SequenceMatcher(None, normalized_answer, normalized_evidence).ratio()

            # Common safe paraphrases in policy answers should not cause a
            # false negative merely because the generator changed wording
            # such as ``entitled to`` -> ``receive``.
            paraphrase_pairs = (("entitled", "receive"), ("entitled", "allowed"),
                                ("entitlement", "receive"), ("request", "submit"),
                                ("submitting", "submit"))
            paraphrase_match = any(
                (a in answer_terms and b in evidence_terms) or
                (b in answer_terms and a in evidence_terms)
                for a, b in paraphrase_pairs
            )

            # Numeric claims are especially important: the same number must be
            # present in the evidence before a numerical answer is accepted.
            answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", answer_sentence))
            evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence_sentence))
            if answer_numbers and not answer_numbers.issubset(evidence_numbers):
                continue

            # High overlap accepts close paraphrases while avoiding a single
            # shared topic word being treated as proof.
            if (overlap >= 0.65 and reverse_overlap >= 0.35) or similarity >= 0.80:
                return True

            # Permit one controlled paraphrase only when the remaining claim
            # terms are still strongly anchored in the evidence.
            if paraphrase_match and overlap >= 0.55 and reverse_overlap >= 0.35:
                return True

        return False

    @staticmethod
    def _claim_terms(text: str) -> set[str]:
        # Keep domain words that the generic question tokenizer removes, such
        # as "leave" and "days", because they are useful when comparing a
        # generated claim directly with source evidence.
        raw = set(re.findall(r"[a-z0-9]+", text.lower()))
        generic = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "what", "which", "who", "when", "where", "why", "how", "many", "much",
            "does", "do", "did", "can", "could", "should", "would", "will", "shall",
            "of", "for", "to", "in", "on", "at", "by", "from", "and", "or", "with",
            "this", "that", "it", "they", "their", "each", "every", "per", "please",
            "tell", "me", "about",
        }
        return {term for term in raw if term not in generic}

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()

    @staticmethod
    def _is_abstention_statement(sentence: str) -> bool:
        normalized = ValidatorAgent._normalize(sentence)
        patterns = (
            "information about",
            "was not found in the provided documents",
            "were not found in the provided documents",
            "could not find the answer in the provided documents",
            "could not find the information in the provided documents",
        )
        return any(pattern in normalized for pattern in patterns)
