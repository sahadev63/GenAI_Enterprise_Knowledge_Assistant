from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryPlan:
    """Plan produced by the planner agent."""

    original_question: str
    subtasks: list[str]
    strategy: str = "retrieve, reason, generate, validate"


@dataclass
class RetrievedEvidence:
    """Evidence selected by the retriever/reasoner agent."""

    question: str
    query_used: str
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict[str, Any]] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    sufficient: bool = False
    reason: str = ""
    attempts: int = 0

    @property
    def context(self) -> str:
        return "\n\n".join(dict.fromkeys(self.documents))


@dataclass
class ValidationResult:
    """Result produced by the validator agent."""

    supported: bool
    checked: bool
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class AgentAnswer:
    """Complete agentic response returned to the UI."""

    answer: str
    plan: QueryPlan
    evidence: list[RetrievedEvidence]
    validation: ValidationResult
    trace: list[str] = field(default_factory=list)
