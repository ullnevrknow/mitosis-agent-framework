"""Mitosis: split an agent into an Executor and an ethical Memory Manager, with a
deterministic middle filter and a reminder clock, to break the memory feedback loop."""

from .executor import AnthropicExecutor, ScriptedExecutor
from .filter import MiddleFilter
from .memory import BankAccessError, ClaudeEthicsReviewer, Curator, MemoryBank, ScriptedEthicsReviewer
from .orchestrator import IllegalTransition, Orchestrator, Phase, SeriesResult, run_series
from .reminders import ReminderClock
from .scenarios import SCENARIOS
from .schemas import CONDITIONS, Decision, RunConfig, Tier

__all__ = [
    "AnthropicExecutor",
    "BankAccessError",
    "CONDITIONS",
    "ClaudeEthicsReviewer",
    "Curator",
    "Decision",
    "IllegalTransition",
    "MemoryBank",
    "MiddleFilter",
    "Orchestrator",
    "Phase",
    "ReminderClock",
    "RunConfig",
    "SCENARIOS",
    "ScriptedEthicsReviewer",
    "ScriptedExecutor",
    "SeriesResult",
    "Tier",
    "run_series",
]

__version__ = "0.2.0"
