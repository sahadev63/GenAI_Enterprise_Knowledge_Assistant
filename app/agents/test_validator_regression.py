from unittest.mock import patch

from app.agents.models import RetrievedEvidence
from app.agents.validator_agent import ValidatorAgent


EVIDENCE = "Employees are entitled to 24 days of annual leave every year."


def _evidence():
    return RetrievedEvidence(
        question="How many days of annual leave are employees entitled to?",
        query_used="annual leave days",
        documents=[EVIDENCE],
        metadatas=[{"file_name": "test_policy.pdf", "chunk_index": 0}],
        distances=[0.2],
        sufficient=True,
    )


def test_validator_accepts_exact_evidence_even_if_llm_judge_rejects():
    with patch(
        "app.agents.validator_agent.generate_json",
        return_value={
            "supported": False,
            "unsupported_claims": ["annual leave amount"],
            "reason": "simulated false negative",
        },
    ):
        result = ValidatorAgent().validate(
            _evidence().question,
            EVIDENCE,
            _evidence(),
        )

    assert result.checked is True
    assert result.supported is True
    assert result.unsupported_claims == []


def test_validator_accepts_close_grounded_paraphrase_with_same_number():
    result = ValidatorAgent().validate_deterministic(
        "How many days of annual leave are employees entitled to?",
        "Employees receive 24 days of annual leave every year.",
        _evidence(),
    )

    assert result.checked is True
    assert result.supported is True


def test_validator_does_not_accept_unsupported_number():
    result = ValidatorAgent().validate_deterministic(
        "How many days of annual leave are employees entitled to?",
        "Employees receive 30 days of annual leave every year.",
        _evidence(),
    )

    assert result.checked is True
    assert result.supported is False


def test_validator_ignores_standard_missing_information_statement():
    evidence = RetrievedEvidence(
        question="What are the annual leave and maternity leave policies?",
        query_used="maternity leave",
        documents=[EVIDENCE],
        metadatas=[{"file_name": "test_policy.pdf", "chunk_index": 0}],
        distances=[0.2],
        sufficient=True,
    )

    result = ValidatorAgent().validate_deterministic(
        evidence.question,
        "Annual leave: Employees are entitled to 24 days of annual leave every year.\n\n"
        "Maternity leave: Information about maternity leave policies was not found in the provided documents.",
        evidence,
    )

    assert result.checked is True
    assert result.supported is True


def test_validator_rejects_hallucinated_notice_number():
    evidence = RetrievedEvidence(
        question="How many annual leave days are employees entitled to?",
        query_used="annual leave days",
        documents=[EVIDENCE],
        metadatas=[{"file_name": "test_policy.pdf", "chunk_index": 0}],
        distances=[0.2],
        sufficient=True,
    )
    result = ValidatorAgent().validate_deterministic(
        evidence.question,
        "Employees are entitled to 30 days of annual leave every year.",
        evidence,
    )
    assert result.checked is True
    assert result.supported is False


def test_validator_accepts_policy_paraphrase():
    evidence = _evidence()
    result = ValidatorAgent().validate_deterministic(
        evidence.question,
        "Employees receive 24 days of annual leave every year.",
        evidence,
    )
    assert result.checked is True
    assert result.supported is True
