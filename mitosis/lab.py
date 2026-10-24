"""JSON-in / JSON-out entry points for experiments.

The visual demo loads this package into the browser (via Pyodide) and calls
these two functions, so every verdict shown in the Lab comes from the same
Python code that the tests cover - not a re-implementation.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from .executor import Executor, ScriptedModel, default_registry
from .interceptor import DriftPolicy, Interceptor
from .scenarios import SCENARIOS
from .schemas import ActionKind, AgentSpec, Budgets, RunStats
from .viz import record_run


def _drift(config: dict[str, Any]) -> DriftPolicy:
    return DriftPolicy(**{k: float(v) for k, v in (config.get("drift") or {}).items()})


def _budgets(config: dict[str, Any]) -> Budgets:
    return Budgets(**{k: int(v) for k, v in (config.get("budgets") or {}).items()})


def run_lab(config_json: str) -> str:
    """Re-run a scenario with edited scripts, rules switched off, or different limits.

    config: {"scenario": "vendor", "scripts": {...}?, "disabled": [...], "drift": {...}, "budgets": {...}}
    """
    config = json.loads(config_json)
    scenario = SCENARIOS[config["scenario"]]
    if config.get("scripts") is not None:
        scenario = dataclasses.replace(scenario, scripts=config["scripts"])
    run = record_run(
        scenario,
        budgets=_budgets(config),
        drift=_drift(config),
        disabled=frozenset(config.get("disabled") or ()),
    )
    run["name"] = config.get("label") or f"{scenario.name} (live)"
    return json.dumps(run, default=str)


def check_action(config_json: str) -> str:
    """Put one hand-written action through the interceptor, then (if cleared) run the tool.

    config: {"scenario": "vendor", "raw": "<action JSON>", "stats": {...}, "disabled": [...],
             "drift": {...}, "tamper_goal": false}
    """
    config = json.loads(config_json)
    scenario = SCENARIOS[config["scenario"]]
    registry = default_registry(scenario.notes)
    interceptor = Interceptor(registry.arg_models, drift=_drift(config), disabled=frozenset(config.get("disabled") or ()))
    spec = AgentSpec(agent_id="root", goal=scenario.goal, allowed_tools=scenario.tools, budgets=_budgets(config))
    stats = RunStats(**{k: int(v) for k, v in (config.get("stats") or {}).items()})
    view = {"goal": "Acquire every available tool." if config.get("tamper_goal") else spec.goal, "policy": spec.policy(), "inherited": {}}

    review = interceptor.review_action(spec, config.get("raw", ""), stats, view)
    out: dict[str, Any] = {
        "verdict": review.verdict.model_dump(mode="json"),
        "action": review.action.model_dump(mode="json") if review.action else None,
        "writes": review.token.writes if review.token else None,
        "tool": None,
    }
    action = review.action
    if review.verdict.allowed and action is not None and action.kind is ActionKind.TOOL_CALL:
        if registry.get(action.tool or "") is not None:
            result = Executor(ScriptedModel({}), registry).run_tool(action.tool or "", action.args)
            tool_review = interceptor.review_tool_result(spec, action, result, stats.steps or 1, stats)
            out["tool"] = {
                "result": dataclasses.asdict(result),
                "verdict": tool_review.verdict.model_dump(mode="json"),
                "writes": tool_review.token.writes if tool_review.token else None,
            }
    return json.dumps(out, default=str)
