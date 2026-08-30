from unittest.mock import patch

from app.agents.models import QueryPlan, RetrievedEvidence, ValidationResult
from app.agents.planner_agent import PlannerAgent
from app.agents.orchestrator import AgentOrchestrator
from app.agents.validator_agent import ValidatorAgent


def test_planner_fallback_splits_multi_part_question():
    with patch("app.agents.planner_agent.generate_json", side_effect=RuntimeError("offline")):
        plan = PlannerAgent().plan("What is annual leave and what is sick leave?")

    assert len(plan.subtasks) == 2


def test_validator_uses_safe_fallback_when_llm_unavailable():
    evidence = RetrievedEvidence(
        question="How many annual leave days are employees entitled to?",
        query_used="annual leave days",
        documents=["Employees are entitled to 24 days of annual leave every year."],
        metadatas=[{"file_name": "policy.pdf", "chunk_index": 0}],
        distances=[0.2],
        sufficient=True,
    )

    with patch("app.agents.validator_agent.generate_json", side_effect=RuntimeError("offline")):
        result = ValidatorAgent().validate(
            evidence.question,
            "Employees are entitled to 24 days of annual leave every year.",
            evidence,
        )

    assert result.checked is True
    assert result.supported is True


def test_orchestrator_returns_traceable_answer_without_llm_planner():
    evidence = RetrievedEvidence(
        question="How many annual leave days are employees entitled to?",
        query_used="annual leave days",
        documents=["Employees are entitled to 24 days of annual leave every year."],
        metadatas=[{"file_name": "policy.pdf", "chunk_index": 0}],
        distances=[0.2],
        sufficient=True,
    )

    orchestrator = AgentOrchestrator()
    with patch.object(orchestrator.planner, "plan", return_value=QueryPlan(
        "How many annual leave days are employees entitled to?",
        ["How many annual leave days are employees entitled to?"],
    )), patch.object(orchestrator.retriever, "retrieve", return_value=evidence), patch.object(
        orchestrator.generator, "generate", return_value="Employees are entitled to 24 days of annual leave every year."
    ), patch.object(orchestrator.validator, "validate", return_value=ValidationResult(True, True)):
        result = orchestrator.run(evidence.question)

    assert "24 days" in result.answer
    assert result.validation.supported is True
    assert result.trace
