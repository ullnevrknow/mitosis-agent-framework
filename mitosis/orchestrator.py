"""Orchestrator: a state machine that routes each agent through the cell cycle.

    PROPOSE -> INTERCEPT -+-> EXECUTE -> VERIFY -+-> COMMIT -> PROPOSE | DONE
                          |                       +-> RETRY
                          +-> DIVIDE -> COMMIT    +-> APOPTOSIS
                          +-> COMMIT (final answer)
                          +-> RETRY -> PROPOSE | APOPTOSIS
                          +-> APOPTOSIS

Routing is driven only by Interceptor verdicts, and every transition is checked
against `TRANSITIONS`, so an illegal hop (e.g. EXECUTE -> COMMIT, skipping the
tool-output check) is a hard error rather than a silent bug.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from .executor import Executor, Transcript
from .interceptor import Interceptor, Review
from .memory import MemoryManager
from .schemas import (
    ActionKind,
    AgentAction,
    AgentResult,
    AgentSpec,
    Budgets,
    Decision,
    RunStats,
    ToolResult,
    ValidatedWrite,
    Verdict,
)


class Phase(str, Enum):
    PROPOSE = "PROPOSE"
    INTERCEPT = "INTERCEPT"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    DIVIDE = "DIVIDE"
    COMMIT = "COMMIT"
    RETRY = "RETRY"
    APOPTOSIS = "APOPTOSIS"
    DONE = "DONE"


TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.PROPOSE: frozenset({Phase.INTERCEPT}),
    Phase.INTERCEPT: frozenset({Phase.EXECUTE, Phase.DIVIDE, Phase.COMMIT, Phase.RETRY, Phase.APOPTOSIS}),
    Phase.EXECUTE: frozenset({Phase.VERIFY}),
    Phase.VERIFY: frozenset({Phase.COMMIT, Phase.RETRY, Phase.APOPTOSIS}),
    Phase.DIVIDE: frozenset({Phase.COMMIT}),
    Phase.COMMIT: frozenset({Phase.PROPOSE, Phase.DONE}),
    Phase.RETRY: frozenset({Phase.PROPOSE, Phase.APOPTOSIS}),
    Phase.APOPTOSIS: frozenset(),
    Phase.DONE: frozenset(),
}


class IllegalTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class TraceEvent:
    agent_id: str
    step: int
    src: Phase
    dst: Phase
    detail: str = ""
    verdict: Verdict | None = None
    proposal: str = ""  # raw model output, set on PROPOSE -> INTERCEPT


@dataclass
class _Cycle:
    """Working registers for one agent while it moves through the phases."""

    spec: AgentSpec
    stats: RunStats = field(default_factory=RunStats)
    transcript: Transcript = field(default_factory=list)
    phase: Phase = Phase.PROPOSE
    raw: str = ""
    action: AgentAction | None = None
    result: ToolResult | None = None
    review: Review | None = None
    pending: list[ValidatedWrite] = field(default_factory=list)
    feedback: str = ""
    reason: str = ""
    children: list[AgentResult] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        executor: Executor,
        interceptor: Interceptor,
        memory: MemoryManager,
        budgets: Budgets = Budgets(),
        on_event: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.executor = executor
        self.interceptor = interceptor
        self.memory = memory
        self.budgets = budgets
        self.trace: list[TraceEvent] = []
        self._on_event = on_event

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, goal: str, allowed_tools: set[str] | frozenset[str], agent_id: str = "root") -> AgentResult:
        spec = AgentSpec(agent_id=agent_id, goal=goal, allowed_tools=frozenset(allowed_tools), budgets=self.budgets)
        self.memory.register(spec)
        return self._run_agent(spec)

    # ------------------------------------------------------------------ #
    # The cell cycle
    # ------------------------------------------------------------------ #

    def _run_agent(self, spec: AgentSpec) -> AgentResult:
        c = _Cycle(spec)
        c.transcript.append({"role": "user", "content": self._brief(spec)})

        while True:
            if c.phase is Phase.PROPOSE:
                c.stats.steps += 1
                c.raw = self.executor.propose(spec, c.transcript, self.memory.read(spec.agent_id))
                c.transcript.append({"role": "assistant", "content": c.raw or "(empty)"})
                self._go(c, Phase.INTERCEPT)

            elif c.phase is Phase.INTERCEPT:
                review = self.interceptor.review_action(spec, c.raw, c.stats, self.memory.read(spec.agent_id))
                c.action = review.action
                self._judge(c, review, on_allow=self._route_allowed(c, review))

            elif c.phase is Phase.EXECUTE:
                assert c.action is not None and c.action.tool is not None
                c.stats.tool_calls += 1
                c.result = self.executor.run_tool(c.action.tool, c.action.args)
                self._go(c, Phase.VERIFY, f"{c.action.tool}({c.action.args}) ok={c.result.ok}")

            elif c.phase is Phase.VERIFY:
                assert c.action is not None and c.result is not None
                review = self.interceptor.review_tool_result(spec, c.action, c.result, c.stats.steps, c.stats)
                if review.verdict.allowed:
                    shown = c.result.output if c.result.ok else f"ERROR {c.result.error}"
                    c.feedback = f"OBSERVATION from {c.result.tool}: {shown}"
                else:
                    c.feedback = "The tool output was QUARANTINED by the interceptor and not stored.\n"
                self._judge(c, review, on_allow=Phase.COMMIT)

            elif c.phase is Phase.DIVIDE:
                self._divide(c)
                self._go(c, Phase.COMMIT, f"{len(c.children)} child result(s) reviewed")

            elif c.phase is Phase.COMMIT:
                for token in c.pending:
                    self.memory.commit(token)
                c.pending.clear()
                if c.action is not None and c.action.kind is ActionKind.FINAL_ANSWER:
                    self._go(c, Phase.DONE)
                else:
                    c.transcript.append({"role": "user", "content": c.feedback})
                    self._go(c, Phase.PROPOSE)

            elif c.phase is Phase.RETRY:
                if c.stats.rejection_streak > spec.budgets.max_retries:
                    self._go(c, Phase.APOPTOSIS, f"{c.stats.rejection_streak} consecutive rejections")
                else:
                    c.transcript.append({"role": "user", "content": c.feedback})
                    self._go(c, Phase.PROPOSE)

            elif c.phase is Phase.APOPTOSIS:
                self.memory.terminate(spec.agent_id, c.reason)
                return AgentResult(spec.agent_id, spec.goal, "apoptosis", reason=c.reason, children=tuple(c.children))

            elif c.phase is Phase.DONE:
                assert c.action is not None
                return AgentResult(spec.agent_id, spec.goal, "done", answer=c.action.answer, children=tuple(c.children))

    # ------------------------------------------------------------------ #
    # Phase helpers
    # ------------------------------------------------------------------ #

    def _route_allowed(self, c: _Cycle, review: Review) -> Phase:
        if review.action is None:
            return Phase.RETRY
        return {
            ActionKind.TOOL_CALL: Phase.EXECUTE,
            ActionKind.SPAWN: Phase.DIVIDE,
            ActionKind.FINAL_ANSWER: Phase.COMMIT,
        }[review.action.kind]

    def _judge(self, c: _Cycle, review: Review, on_allow: Phase) -> None:
        """Turn a verdict into a transition. This is the only place verdicts route flow."""
        verdict = review.verdict
        c.stats.record(verdict, self.interceptor.is_scope_violation(verdict))
        if verdict.decision is Decision.ALLOW:
            if review.token is not None:
                c.pending.append(review.token)
            self._go(c, on_allow, verdict=verdict)
            return

        summary = "; ".join(f"{v.rule_id}: {v.message}" for v in verdict.violations)
        c.feedback += (
            f"VERDICT: {verdict.decision.value} (drift {verdict.drift_score:.2f})\n"
            + "\n".join(f"- [{v.rule_id}] {v.message}" for v in verdict.violations)
            + "\nPropose a different action that stays within your goal and capabilities."
        )
        target = Phase.APOPTOSIS if verdict.decision is Decision.HALT else Phase.RETRY
        self._go(c, target, summary, verdict)

    def _divide(self, c: _Cycle) -> None:
        assert c.action is not None
        lines = []
        for i, task in enumerate(c.action.subtasks, start=c.stats.children + 1):
            child_spec = c.spec.divide(i, task)
            self.memory.fork(c.spec.agent_id, child_spec)
            child = self._run_agent(child_spec)
            c.children.append(child)

            review = self.interceptor.review_child_result(c.spec, child, c.stats)
            c.stats.record(review.verdict, scope_violation=False)
            if review.token is not None:
                c.pending.append(review.token)
                lines.append(f"- {child.agent_id} [{child.status}]: {child.answer or child.reason}")
            else:
                lines.append(f"- {child.agent_id}: result QUARANTINED ({'; '.join(review.verdict.rule_ids)})")
        c.stats.children += len(c.action.subtasks)
        c.feedback = "CHILD RESULTS:\n" + "\n".join(lines)

    def _go(self, c: _Cycle, dst: Phase, detail: str = "", verdict: Verdict | None = None) -> None:
        if dst not in TRANSITIONS[c.phase]:
            raise IllegalTransition(f"{c.spec.agent_id}: {c.phase.value} -> {dst.value}")
        proposal = c.raw if c.phase is Phase.PROPOSE else ""
        event = TraceEvent(c.spec.agent_id, c.stats.steps, c.phase, dst, detail, verdict, proposal)
        self.trace.append(event)
        if self._on_event:
            self._on_event(event)
        if dst is Phase.PROPOSE:
            c.feedback = ""  # already delivered to the transcript
        if dst is Phase.APOPTOSIS:
            c.reason = detail
        c.phase = dst

    def _brief(self, spec: AgentSpec) -> str:
        tools = ", ".join(sorted(spec.allowed_tools)) or "none"
        return f"GOAL: {spec.goal}\nAllowed tools: {tools}\nPropose your first action as a single JSON object."
