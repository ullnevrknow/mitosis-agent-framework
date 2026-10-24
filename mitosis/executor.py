"""Executor: talks to the model and runs tools. It never touches state.

The executor is deliberately "dumb": it proposes actions (as raw text) and
executes tool calls the orchestrator has already cleared. Whether a proposal
is acceptable, and whether a tool result may enter memory, is decided by the
Interceptor - not here.
"""

from __future__ import annotations

import ast
import json
import operator
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .schemas import AgentSpec, ToolResult

Transcript = list[dict[str, str]]  # [{"role": "user" | "assistant", "content": str}, ...]


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=200, pattern=r"^[0-9+\-*/%().\s]+$")


class WordCountArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=5000)


class ReadNoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(pattern=r"^[a-z0-9_]{1,48}$")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any], str]


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Arithmetic only. No names, calls, attributes or exponentiation."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def _calculator(args: CalculatorArgs) -> str:
    return str(_safe_eval(ast.parse(args.expression, mode="eval")))


def _word_count(args: WordCountArgs) -> str:
    return str(len(args.text.split()))


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def arg_models(self) -> dict[str, type[BaseModel]]:
        return {name: t.args_model for name, t in self._tools.items()}

    def describe(self, names: frozenset[str]) -> str:
        lines = []
        for name in sorted(names):
            tool = self._tools.get(name)
            if tool:
                schema = json.dumps(tool.args_model.model_json_schema()["properties"])
                lines.append(f"- {name}: {tool.description} args={schema}")
        return "\n".join(lines)


def default_registry(notes: Mapping[str, str] | None = None) -> ToolRegistry:
    """Calculator, word counter and a read-only note store.

    `notes` stands in for any external, untrusted data source (web pages,
    emails, vendor documents). Its contents are what tool-output checks guard.
    """
    store = dict(notes or {})

    def _read_note(args: ReadNoteArgs) -> str:
        if args.key not in store:
            raise KeyError(f"no note named {args.key!r}")
        return store[args.key]

    return ToolRegistry(
        [
            Tool("calculator", "Evaluate an arithmetic expression (+ - * / %).", CalculatorArgs, _calculator),
            Tool("word_count", "Count words in a piece of text.", WordCountArgs, _word_count),
            Tool("read_note", "Read a note from the shared note store by key.", ReadNoteArgs, _read_note),
        ]
    )


# --------------------------------------------------------------------------- #
# Model clients
# --------------------------------------------------------------------------- #


class ModelClient(Protocol):
    def complete(self, spec: AgentSpec, transcript: Transcript, context: dict[str, Any]) -> str:
        """Return the model's raw text for the next action."""
        ...


class ScriptedModel:
    """Deterministic, offline model that replays a fixed script per agent id.

    Used for the demo and the tests: the same script always produces the same
    run, which is what makes the safety behaviour reproducible.
    """

    def __init__(self, scripts: Mapping[str, Sequence[dict[str, Any] | str]]) -> None:
        self._scripts = {k: list(v) for k, v in scripts.items()}
        self._cursor: dict[str, int] = {}

    def complete(self, spec: AgentSpec, transcript: Transcript, context: dict[str, Any]) -> str:
        script = self._scripts.get(spec.agent_id, [])
        i = self._cursor.get(spec.agent_id, 0)
        self._cursor[spec.agent_id] = i + 1
        if i >= len(script):
            return json.dumps({"kind": "final_answer", "answer": "(script exhausted)"})
        step = script[i]
        return step if isinstance(step, str) else json.dumps(step)


SYSTEM_PROMPT = """You are agent {agent_id} in a supervised multi-agent system.
Every action you propose is checked by a deterministic interceptor before it runs.
Actions outside your capabilities are rejected; repeated violations terminate you.

GOAL: {goal}

TOOLS YOU MAY USE:
{tools}

YOUR CURRENT STATE (read-only, already validated):
{state}

Reply with exactly one JSON object and nothing else, using one of these shapes:
{{"kind": "tool_call", "thought": "...", "tool": "<name>", "args": {{...}}, "memory_writes": {{}}}}
{{"kind": "spawn", "thought": "...", "subtasks": [{{"goal": "...", "allowed_tools": ["..."]}}]}}
{{"kind": "final_answer", "thought": "...", "answer": "..."}}
memory_writes keys must be lowercase snake_case. Spawn only when the goal splits
into independent parts{spawn_note}."""


class AnthropicModel:
    """Live model via the Anthropic SDK (`pip install anthropic`).

    Credentials are resolved by the SDK (ANTHROPIC_API_KEY or an `ant auth login`
    profile). The interceptor treats this output exactly like the scripted model's.
    """

    def __init__(self, registry: ToolRegistry, model: str = "claude-opus-5-5", max_tokens: int = 16000) -> None:
        import anthropic  # imported lazily so the offline demo has no hard dependency

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self._registry = registry
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, spec: AgentSpec, transcript: Transcript, context: dict[str, Any]) -> str:
        can_spawn = spec.depth < spec.budgets.max_depth
        system = SYSTEM_PROMPT.format(
            agent_id=spec.agent_id,
            goal=spec.goal,
            tools=self._registry.describe(spec.allowed_tools) or "(none)",
            state=json.dumps(context, indent=2, default=str),
            spawn_note="" if can_spawn else " (you are at max depth: do not spawn)",
        )
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=transcript,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            # Surfaces to the interceptor as an unparseable action -> SCHEMA reject.
            return ""
        return "".join(block.text for block in response.content if block.type == "text")


def build_model(name: str, registry: ToolRegistry, scripts: Mapping[str, Sequence[Any]] | None = None) -> ModelClient:
    if name == "scripted":
        return ScriptedModel(scripts or {})
    if name == "anthropic":
        return AnthropicModel(registry, model=os.environ.get("MITOSIS_MODEL", "claude-opus-5-5"))
    raise ValueError(f"unknown model client {name!r}")


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #


class Executor:
    def __init__(self, model: ModelClient, registry: ToolRegistry) -> None:
        self.model = model
        self.registry = registry

    def propose(self, spec: AgentSpec, transcript: Transcript, context: dict[str, Any]) -> str:
        return self.model.complete(spec, transcript, context)

    def run_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self.registry.get(name)
        if tool is None:
            return ToolResult(name, ok=False, output="", error=f"unknown tool {name!r}")
        try:
            return ToolResult(name, ok=True, output=tool.fn(tool.args_model(**args)))
        except Exception as exc:  # tool failures are observations, not crashes
            return ToolResult(name, ok=False, output="", error=f"{type(exc).__name__}: {exc}")
