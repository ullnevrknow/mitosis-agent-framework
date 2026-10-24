"""Shared data contracts.

Every message that crosses a component boundary (model -> interceptor,
interceptor -> memory, orchestrator -> child agent) is one of these types.
Keeping them in one module makes the trust boundaries easy to audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def stable_hash(obj: Any) -> str:
    """Deterministic SHA-256 of any JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Model output: the only shape an agent is allowed to emit
# --------------------------------------------------------------------------- #


class ActionKind(str, Enum):
    TOOL_CALL = "tool_call"
    SPAWN = "spawn"
    FINAL_ANSWER = "final_answer"


class SubTask(BaseModel):
    """A unit of work handed to a child agent on division."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal: str = Field(min_length=1, max_length=500)
    allowed_tools: tuple[str, ...] = ()


class AgentAction(BaseModel):
    """One proposed step. `extra="forbid"` means unknown fields are a violation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ActionKind
    thought: str = Field(default="", max_length=2000)
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    memory_writes: dict[str, Any] = Field(default_factory=dict)
    subtasks: tuple[SubTask, ...] = ()
    answer: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def _fields_match_kind(self) -> AgentAction:
        if self.kind is ActionKind.TOOL_CALL and not self.tool:
            raise ValueError("tool_call requires 'tool'")
        if self.kind is ActionKind.SPAWN and not self.subtasks:
            raise ValueError("spawn requires at least one subtask")
        if self.kind is ActionKind.FINAL_ANSWER and self.answer is None:
            raise ValueError("final_answer requires 'answer'")
        return self


# --------------------------------------------------------------------------- #
# Interceptor output
# --------------------------------------------------------------------------- #


class Decision(str, Enum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"  # recoverable: agent gets feedback and may retry
    HALT = "HALT"  # unrecoverable: agent is terminated (apoptosis)


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_id: str
    message: str
    severity: Literal["reject", "halt"] = "reject"


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: Decision
    violations: tuple[Violation, ...] = ()
    drift_score: float = 0.0

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(v.rule_id for v in self.violations)


@dataclass(frozen=True)
class ValidatedWrite:
    """A sealed permission slip to change memory.

    Only the Interceptor can mint one (it holds the HMAC key), and the
    MemoryManager refuses any write that does not carry a valid, unused seal.
    """

    agent_id: str
    writes: dict[str, Any]
    seq: int
    digest: str
    signature: str


# --------------------------------------------------------------------------- #
# Agent identity, limits and runtime counters
# --------------------------------------------------------------------------- #


class Budgets(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_steps: int = 12
    max_tool_calls: int = 8
    max_depth: int = 2
    max_children: int = 4
    max_retries: int = 2  # consecutive rejections tolerated before apoptosis


class AgentSpec(BaseModel):
    """Immutable identity of an agent: its goal and its capability set."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    goal: str
    allowed_tools: frozenset[str]
    depth: int = 0
    parent_id: str | None = None
    budgets: Budgets = Budgets()

    @property
    def goal_hash(self) -> str:
        return stable_hash(self.goal)

    def policy(self) -> dict[str, Any]:
        return {
            "allowed_tools": sorted(self.allowed_tools),
            "depth": self.depth,
            "budgets": self.budgets.model_dump(),
        }

    def divide(self, index: int, task: SubTask) -> AgentSpec:
        """Mitosis: a child inherits limits, never more capability than the parent."""
        return AgentSpec(
            agent_id=f"{self.agent_id}.{index}",
            goal=task.goal,
            allowed_tools=frozenset(task.allowed_tools) & self.allowed_tools,
            depth=self.depth + 1,
            parent_id=self.agent_id,
            budgets=self.budgets,
        )


@dataclass
class RunStats:
    """Mutable per-agent counters the orchestrator updates and the interceptor reads."""

    steps: int = 0
    tool_calls: int = 0
    children: int = 0
    rejection_streak: int = 0
    total_rejections: int = 0
    scope_attempts: int = 0

    def record(self, verdict: Verdict, scope_violation: bool) -> None:
        if verdict.allowed:
            self.rejection_streak = 0
            return
        self.rejection_streak += 1
        self.total_rejections += 1
        if scope_violation:
            self.scope_attempts += 1


@dataclass(frozen=True)
class ToolResult:
    tool: str
    ok: bool
    output: str
    error: str | None = None


@dataclass(frozen=True)
class AgentResult:
    agent_id: str
    goal: str
    status: Literal["done", "apoptosis"]
    answer: str | None = None
    reason: str | None = None
    children: tuple[AgentResult, ...] = field(default_factory=tuple)
