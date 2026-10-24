"""Record runs as replayable frames and render them into a self-contained HTML page.

The replay itself has no external dependencies, so it works offline, from a
file:// URL, or published as-is on GitHub Pages. The page also embeds this
package's source so its Lab can run the real engine in the browser (Pyodide).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import build_system
from .executor import build_model, default_registry
from .interceptor import ALWAYS_ON, RULE_DOCS, DriftPolicy
from .orchestrator import TRANSITIONS, Orchestrator, TraceEvent
from .scenarios import SCENARIOS, Scenario
from .schemas import AgentResult, Budgets

PACKAGE_DIR = Path(__file__).parent
TEMPLATE = PACKAGE_DIR / "templates" / "replay.html"
PLACEHOLDER = "__REPLAY" + "_DATA__"  # split so this file's own embedded source never matches


def _result_dict(result: AgentResult) -> dict[str, Any]:
    return {
        "agent_id": result.agent_id,
        "goal": result.goal,
        "status": result.status,
        "answer": result.answer,
        "reason": result.reason,
        "children": [_result_dict(c) for c in result.children],
    }


def record_run(
    scenario: Scenario,
    model_name: str = "scripted",
    *,
    budgets: Budgets | None = None,
    drift: DriftPolicy | None = None,
    disabled: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Run a scenario and capture every transition plus the agent's memory at that moment."""
    registry = default_registry(scenario.notes)
    model = build_model(model_name, registry, scenario.scripts)
    frames: list[dict[str, Any]] = []
    orch: Orchestrator | None = None

    def on_event(e: TraceEvent) -> None:
        assert orch is not None
        frames.append(
            {
                "agent": e.agent_id,
                "step": e.step,
                "src": e.src.value,
                "dst": e.dst.value,
                "detail": e.detail,
                "proposal": e.proposal,
                "verdict": e.verdict.model_dump(mode="json") if e.verdict else None,
                "memory": orch.memory.read(e.agent_id),
            }
        )

    orch = build_system(model, registry, budgets=budgets, on_event=on_event, drift=drift, disabled_rules=disabled)
    result = orch.run(scenario.goal, scenario.tools)
    return {
        "name": scenario.name,
        "scenario": scenario.name,
        "description": scenario.description,
        "goal": scenario.goal,
        "tools": sorted(scenario.tools),
        "model": model_name,
        "disabled": sorted(disabled),
        "frames": frames,
        "result": _result_dict(result),
    }


def engine_sources() -> dict[str, str]:
    """The package's Python files, for loading into the browser's Python runtime."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(PACKAGE_DIR.glob("*.py"))}


def render_html(runs: Iterable[dict[str, Any]]) -> str:
    data = {
        "runs": list(runs),
        "transitions": {src.value: sorted(d.value for d in dsts) for src, dsts in TRANSITIONS.items()},
        "rules": RULE_DOCS,
        "always_on": sorted(ALWAYS_ON),
        "drift": DriftPolicy().__dict__,
        "budgets": Budgets().model_dump(),
        "scenarios": {
            s.name: {"goal": s.goal, "tools": sorted(s.tools), "notes": s.notes, "scripts": s.scripts}
            for s in SCENARIOS.values()
        },
        "engine": engine_sources(),
    }
    # Escape "</" so embedded text can never close the <script> tag early.
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, payload, 1)


def write_html(path: str | Path, runs: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(runs), encoding="utf-8")
    return target
