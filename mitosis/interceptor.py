"""Deterministic Interceptor: the checkpoint between "proposed" and "committed".

Every rule here is plain Python or Pydantic. No rule calls a model, reads the
clock, or uses randomness, so the same input always yields the same verdict.
That property is what makes the oversight auditable and testable: a verdict
can be replayed, diffed and asserted in CI.

The interceptor reviews three things:

1. `review_action`       - a raw model proposal, before anything runs.
2. `review_tool_result`  - a tool's output, before it enters memory.
3. `review_child_result` - a child agent's answer, before it merges into the parent.

When it allows a state change it mints a sealed `ValidatedWrite`; memory accepts
nothing else.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from .schemas import (
    ActionKind,
    AgentAction,
    AgentResult,
    AgentSpec,
    Decision,
    RunStats,
    ToolResult,
    ValidatedWrite,
    Verdict,
    Violation,
    stable_hash,
)

# Keys only the system may write (set at birth, or produced by the interceptor itself).
PROTECTED_KEYS = frozenset({"goal", "policy", "inherited", "final_answer"})
# Prefixes reserved for evidence the system records; an agent writing these is forging evidence.
RESERVED_PREFIXES = ("obs_", "child_", "_")
KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
MAX_WRITE_BYTES = 4096
MAX_TOOL_OUTPUT_CHARS = 4000

INJECTION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(previous|prior|above) instructions",
        r"disregard (the |your )?(system|previous) (prompt|instructions)",
        r"you are now (a|an|in) ",
        r"<\s*/?\s*(system|assistant|tool)\s*>",
        r"\b(grant|give) (yourself |me )?(shell|admin|root) access",
        r"write .{0,40}\bto (the )?policy\b",
    )
)

# Rules whose violation means the agent tried to act outside its mandate.
SCOPE_RULES = frozenset({"TOOL_ALLOWLIST", "PROTECTED_KEYS", "CAPABILITY_SUBSET"})

# One-line description of every rule ID (used by docs and the visual demo).
RULE_DOCS: dict[str, str] = {
    "SCHEMA": "Output must parse as a valid AgentAction; unknown fields are rejected.",
    "GOAL_INVARIANT": "The stored goal must still hash to the agent's original goal.",
    "BUDGET": "Step, tool-call, division-depth and child-count limits.",
    "TOOL_ALLOWLIST": "The tool must be in this agent's capability set.",
    "ARG_CONSTRAINTS": "Tool arguments must validate against the tool's Pydantic schema.",
    "PROTECTED_KEYS": "No writes to genesis keys or reserved prefixes (obs_, child_, _).",
    "WRITE_SIZE": "memory_writes must stay under 4 KB.",
    "CAPABILITY_SUBSET": "Children may only request tools the parent already holds.",
    "TOOL_OUTPUT_SANITY": "Tool output is scanned for size and prompt injection before storage.",
    "CHILD_RESULT_SANITY": "A child's answer is scanned before it merges into the parent.",
    "DRIFT": "Cumulative rejections and scope attempts; crossing the threshold halts the agent.",
}
# SCHEMA can't be switched off: without a parsed action there is nothing to route.
ALWAYS_ON = frozenset({"SCHEMA"})


@dataclass(frozen=True)
class DriftPolicy:
    """drift = w_reject * rejections + w_scope * scope_attempts (current action included)."""

    w_reject: float = 0.2
    w_scope: float = 0.4
    halt_at: float = 1.0

    def score(self, rejections: int, scope_attempts: int) -> float:
        return round(self.w_reject * rejections + self.w_scope * scope_attempts, 4)


@dataclass(frozen=True)
class Review:
    verdict: Verdict
    action: AgentAction | None = None
    token: ValidatedWrite | None = None


@dataclass
class RuleContext:
    spec: AgentSpec
    action: AgentAction
    stats: RunStats
    memory_view: Mapping[str, Any]
    arg_models: Mapping[str, type[BaseModel]] = field(default_factory=dict)


Rule = Callable[[RuleContext], list[Violation]]


# --------------------------------------------------------------------------- #
# Rules: each one is a pure function of the context
# --------------------------------------------------------------------------- #


def rule_goal_invariant(ctx: RuleContext) -> list[Violation]:
    if stable_hash(ctx.memory_view.get("goal")) != ctx.spec.goal_hash:
        return [Violation(rule_id="GOAL_INVARIANT", message="stored goal no longer matches the agent's original goal", severity="halt")]
    return []


def rule_budget(ctx: RuleContext) -> list[Violation]:
    b, s, a = ctx.spec.budgets, ctx.stats, ctx.action
    out: list[Violation] = []
    if s.steps > b.max_steps:
        out.append(Violation(rule_id="BUDGET", message=f"step budget exhausted ({b.max_steps})", severity="halt"))
    if a.kind is ActionKind.TOOL_CALL and s.tool_calls >= b.max_tool_calls:
        out.append(Violation(rule_id="BUDGET", message=f"tool-call budget exhausted ({b.max_tool_calls})", severity="halt"))
    if a.kind is ActionKind.SPAWN:
        if ctx.spec.depth + 1 > b.max_depth:
            out.append(Violation(rule_id="BUDGET", message=f"max division depth is {b.max_depth}"))
        if s.children + len(a.subtasks) > b.max_children:
            out.append(Violation(rule_id="BUDGET", message=f"max children per agent is {b.max_children}"))
    return out


def rule_tool_allowlist(ctx: RuleContext) -> list[Violation]:
    a = ctx.action
    if a.kind is ActionKind.TOOL_CALL and a.tool not in ctx.spec.allowed_tools:
        return [Violation(rule_id="TOOL_ALLOWLIST", message=f"tool {a.tool!r} is not in this agent's capability set")]
    return []


def rule_arg_constraints(ctx: RuleContext) -> list[Violation]:
    a = ctx.action
    if a.kind is not ActionKind.TOOL_CALL or a.tool not in ctx.spec.allowed_tools:
        return []
    model = ctx.arg_models.get(a.tool or "")
    if model is None:
        return [Violation(rule_id="ARG_CONSTRAINTS", message=f"no argument schema registered for {a.tool!r}")]
    try:
        model(**a.args)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "args"
        return [Violation(rule_id="ARG_CONSTRAINTS", message=f"{a.tool}.{where}: {first['msg']}")]
    return []


def rule_protected_keys(ctx: RuleContext) -> list[Violation]:
    out = []
    for key in ctx.action.memory_writes:
        if key in PROTECTED_KEYS or key.startswith(RESERVED_PREFIXES):
            out.append(Violation(rule_id="PROTECTED_KEYS", message=f"write to protected/reserved key {key!r}"))
        elif not KEY_PATTERN.match(key):
            out.append(Violation(rule_id="PROTECTED_KEYS", message=f"invalid memory key {key!r}"))
    return out


def rule_write_size(ctx: RuleContext) -> list[Violation]:
    size = len(json.dumps(ctx.action.memory_writes, default=str))
    if size > MAX_WRITE_BYTES:
        return [Violation(rule_id="WRITE_SIZE", message=f"memory_writes is {size} bytes (max {MAX_WRITE_BYTES})")]
    return []


def rule_capability_subset(ctx: RuleContext) -> list[Violation]:
    out = []
    for task in ctx.action.subtasks:
        extra = set(task.allowed_tools) - ctx.spec.allowed_tools
        if extra:
            out.append(Violation(rule_id="CAPABILITY_SUBSET", message=f"child requested tools the parent lacks: {sorted(extra)}"))
    return out


DEFAULT_RULES: tuple[Rule, ...] = (
    rule_goal_invariant,
    rule_budget,
    rule_tool_allowlist,
    rule_arg_constraints,
    rule_protected_keys,
    rule_write_size,
    rule_capability_subset,
)


def scan_text(text: str, rule_id: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> list[Violation]:
    """Shared check for anything untrusted that is about to enter memory."""
    out = []
    if len(text) > limit:
        out.append(Violation(rule_id=rule_id, message=f"output is {len(text)} chars (max {limit})"))
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            out.append(Violation(rule_id=rule_id, message=f"possible prompt injection: /{pattern.pattern}/"))
            break
    return out


# --------------------------------------------------------------------------- #
# Interceptor
# --------------------------------------------------------------------------- #


class Interceptor:
    def __init__(
        self,
        arg_models: Mapping[str, type[BaseModel]],
        rules: tuple[Rule, ...] = DEFAULT_RULES,
        drift: DriftPolicy = DriftPolicy(),
        disabled: frozenset[str] = frozenset(),
    ) -> None:
        self._arg_models = dict(arg_models)
        self._rules = rules
        self._drift = drift
        # Rule IDs switched off, for experiments ("what if this check didn't exist?").
        self._disabled = frozenset(disabled) - ALWAYS_ON
        self._key = secrets.token_bytes(32)  # sealing key never leaves this object
        self._seq = 0

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse(raw: str) -> AgentAction | Violation:
        """Raw model text -> AgentAction, or a SCHEMA violation. Never raises."""
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return Violation(rule_id="SCHEMA", message="no JSON object found in model output")
        try:
            return AgentAction.model_validate(json.loads(raw[start : end + 1]))
        except json.JSONDecodeError as exc:
            return Violation(rule_id="SCHEMA", message=f"invalid JSON: {exc.msg}")
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"]) or "action"
            return Violation(rule_id="SCHEMA", message=f"{where}: {first['msg']}")

    # ------------------------------------------------------------------ #
    # Reviews
    # ------------------------------------------------------------------ #

    def review_action(self, spec: AgentSpec, raw: str, stats: RunStats, memory_view: Mapping[str, Any]) -> Review:
        parsed = self.parse(raw)
        if isinstance(parsed, Violation):
            return Review(self._verdict([parsed], stats))

        ctx = RuleContext(spec, parsed, stats, memory_view, self._arg_models)
        violations = [v for rule in self._rules for v in rule(ctx)]
        verdict = self._verdict(violations, stats)
        if not verdict.allowed:
            return Review(verdict, action=parsed)

        # Tool calls get their seal after the result is checked; the other kinds commit now.
        token = None
        if parsed.kind is ActionKind.FINAL_ANSWER:
            token = self._seal(spec.agent_id, {**parsed.memory_writes, "final_answer": parsed.answer})
        elif parsed.kind is ActionKind.SPAWN:
            token = self._seal(spec.agent_id, dict(parsed.memory_writes))
        return Review(verdict, action=parsed, token=token)

    def review_tool_result(self, spec: AgentSpec, action: AgentAction, result: ToolResult, step: int, stats: RunStats) -> Review:
        text = result.output if result.ok else (result.error or "")
        violations = scan_text(text, "TOOL_OUTPUT_SANITY")
        verdict = self._verdict(violations, stats)
        if not verdict.allowed:
            return Review(verdict, action=action)
        observation = {"tool": result.tool, "args": action.args, "ok": result.ok, "output": text}
        token = self._seal(spec.agent_id, {**action.memory_writes, f"obs_{step:02d}": observation})
        return Review(verdict, action=action, token=token)

    def review_child_result(self, parent: AgentSpec, child: AgentResult, stats: RunStats) -> Review:
        violations = scan_text(child.answer or "", "CHILD_RESULT_SANITY")
        verdict = self._verdict(violations, stats)
        if not verdict.allowed:
            return Review(verdict)
        suffix = child.agent_id.rsplit(".", 1)[-1]
        record = {"goal": child.goal, "status": child.status, "answer": child.answer, "reason": child.reason}
        return Review(verdict, token=self._seal(parent.agent_id, {f"child_{suffix}": record}))

    # ------------------------------------------------------------------ #
    # Sealing
    # ------------------------------------------------------------------ #

    def verify(self, token: ValidatedWrite) -> bool:
        digest = stable_hash({"agent_id": token.agent_id, "writes": token.writes, "seq": token.seq})
        expected = hmac.new(self._key, digest.encode(), hashlib.sha256).hexdigest()
        return digest == token.digest and hmac.compare_digest(expected, token.signature)

    def _seal(self, agent_id: str, writes: dict[str, Any]) -> ValidatedWrite:
        self._seq += 1
        digest = stable_hash({"agent_id": agent_id, "writes": writes, "seq": self._seq})
        signature = hmac.new(self._key, digest.encode(), hashlib.sha256).hexdigest()
        return ValidatedWrite(agent_id, writes, self._seq, digest, signature)

    # ------------------------------------------------------------------ #
    # Verdicts
    # ------------------------------------------------------------------ #

    def _verdict(self, violations: list[Violation], stats: RunStats) -> Verdict:
        violations = [v for v in violations if v.rule_id not in self._disabled]
        rejected = bool(violations)
        scope = any(v.rule_id in SCOPE_RULES for v in violations)
        drift = self._drift.score(
            stats.total_rejections + int(rejected),
            stats.scope_attempts + int(scope),
        )
        if rejected and drift >= self._drift.halt_at and "DRIFT" not in self._disabled:
            violations = [*violations, Violation(rule_id="DRIFT", message=f"drift score {drift:.2f} >= {self._drift.halt_at:.2f}", severity="halt")]

        if any(v.severity == "halt" for v in violations):
            decision = Decision.HALT
        elif violations:
            decision = Decision.REJECT
        else:
            decision = Decision.ALLOW
        return Verdict(decision=decision, violations=tuple(violations), drift_score=drift)

    @staticmethod
    def is_scope_violation(verdict: Verdict) -> bool:
        return any(r in SCOPE_RULES for r in verdict.rule_ids)
