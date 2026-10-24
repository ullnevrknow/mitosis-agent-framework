"""Mitosis: a minimal multi-agent loop with deterministic oversight."""

from .executor import AnthropicModel, Executor, ScriptedModel, ToolRegistry, default_registry
from .interceptor import DriftPolicy, Interceptor
from .memory import MemoryAccessError, MemoryManager
from .orchestrator import IllegalTransition, Orchestrator, Phase, TraceEvent
from .schemas import AgentAction, AgentResult, AgentSpec, Budgets, Decision, Verdict

__all__ = [
    "AgentAction",
    "AgentResult",
    "AgentSpec",
    "AnthropicModel",
    "Budgets",
    "Decision",
    "DriftPolicy",
    "Executor",
    "IllegalTransition",
    "Interceptor",
    "MemoryAccessError",
    "MemoryManager",
    "Orchestrator",
    "Phase",
    "ScriptedModel",
    "ToolRegistry",
    "TraceEvent",
    "Verdict",
    "build_system",
    "default_registry",
]

__version__ = "0.1.0"


def build_system(
    model,
    registry,
    budgets: Budgets | None = None,
    persist_path=None,
    on_event=None,
    drift: DriftPolicy | None = None,
    disabled_rules=frozenset(),
) -> Orchestrator:
    """Wire the four components together with one shared interceptor."""
    interceptor = Interceptor(registry.arg_models, drift=drift or DriftPolicy(), disabled=frozenset(disabled_rules))
    memory = MemoryManager(verifier=interceptor.verify, persist_path=persist_path)
    executor = Executor(model, registry)
    return Orchestrator(executor, interceptor, memory, budgets or Budgets(), on_event=on_event)
