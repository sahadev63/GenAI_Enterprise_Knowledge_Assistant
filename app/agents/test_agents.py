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


def test_retriever_deterministic_answerability_accepts_direct_policy_evidence():
    from app.agents.retriever_reasoner_agent import RetrieverReasonerAgent

    agent = RetrieverReasonerAgent()
    sufficient, reason = agent._deterministic_answerability(
        "How many working days before planned leave should employees submit their request?",
        "Employees should submit their leave request at least 5 working days before the planned leave.",
    )

    assert sufficient is True
    assert "direct evidence" in reason.lower()


def test_retriever_deterministic_answerability_rejects_unrelated_topic():
    from app.agents.retriever_reasoner_agent import RetrieverReasonerAgent

    agent = RetrieverReasonerAgent()
    sufficient, _ = agent._deterministic_answerability(
        "How many sick leave days are employees entitled to?",
        "Employees are entitled to 24 days of annual leave every year.",
    )

    assert sufficient is False


def test_planner_preserves_simple_question_as_single_retrieval_subtask():
    with patch(
        "app.agents.planner_agent.generate_json",
        return_value={"strategy": "retrieve and validate"},
    ):
        question = "How many working days before planned leave should employees submit their request?"
        plan = PlannerAgent().plan(question)

    assert plan.subtasks == [question]


def test_planner_preserves_partial_information_as_two_question_subtasks():
    with patch(
        "app.agents.planner_agent.generate_json",
        return_value={"strategy": "retrieve both topics and validate each"},
    ):
        plan = PlannerAgent().plan(
            "What are the annual leave and maternity leave policies?"
        )

    assert plan.subtasks == [
        "What are the annual leave?",
        "What are the maternity leave policies?",
    ]


def test_orchestrator_keeps_valid_partial_answer_verified():
    annual = RetrievedEvidence(
        question="What are the annual leave?",
        query_used="annual leave",
        documents=["Employees are entitled to 24 days of annual leave every year."],
        metadatas=[{"file_name": "policy.pdf", "chunk_index": 0}],
        distances=[0.2],
        sufficient=True,
    )
    maternity = RetrievedEvidence(
        question="What are the maternity leave policies?",
        query_used="maternity leave",
        documents=[],
        metadatas=[],
        distances=[],
        sufficient=False,
    )

    orchestrator = AgentOrchestrator()
    with patch.object(
        orchestrator.planner,
        "plan",
        return_value=QueryPlan(
            "What are the annual leave and maternity leave policies?",
            ["What are the annual leave?", "What are the maternity leave policies?"],
        ),
    ), patch.object(
        orchestrator.retriever,
        "retrieve",
        side_effect=[annual, maternity],
    ), patch.object(
        orchestrator.generator,
        "generate",
        return_value="Employees are entitled to 24 days of annual leave every year.",
    ), patch.object(
        orchestrator.validator,
        "validate",
        return_value=ValidationResult(True, True),
    ):
        result = orchestrator.run(
            "What are the annual leave and maternity leave policies?"
        )

    assert "24 days" in result.answer
    assert "maternity leave" in result.answer.lower()
    assert result.validation.checked is True
    assert result.validation.supported is True


def test_multi_part_question_reports_supported_and_missing_topics():
    annual = RetrievedEvidence(
        question="What are the annual leave?",
        query_used="annual leave",
        documents=["Employees are entitled to 24 days of annual leave every year."],
        metadatas=[{"file_name": "policy.pdf", "chunk_index": 0}],
        distances=[0.2],
        sufficient=True,
    )
    maternity = RetrievedEvidence(
        question="What are the maternity leave policies?",
        query_used="maternity leave",
        documents=[], metadatas=[], distances=[], sufficient=False,
    )
    orchestrator = AgentOrchestrator()
    with patch.object(orchestrator.planner, "plan", return_value=QueryPlan(
        "What are the annual leave and maternity leave policies?",
        ["What are the annual leave?", "What are the maternity leave policies?"],
    )), patch.object(orchestrator.retriever, "retrieve", side_effect=[annual, maternity]), patch.object(
        orchestrator.generator, "generate", return_value="Employees are entitled to 24 days of annual leave every year."
    ):
        result = orchestrator.run("What are the annual leave and maternity leave policies?")

    assert "24 days" in result.answer
    assert "maternity leave" in result.answer.lower()
    assert "not found" in result.answer.lower()
    assert result.validation.checked is True
    assert result.validation.supported is True
