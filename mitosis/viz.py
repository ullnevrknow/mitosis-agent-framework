"""Record runs as replayable frames and render them into a self-contained HTML page.

The page has no external dependencies, so it works offline, from a file:// URL,
or published as-is on GitHub Pages.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import build_system
from .executor import build_model, default_registry
from .orchestrator import TRANSITIONS, Orchestrator, TraceEvent
from .scenarios import Scenario
from .schemas import AgentResult

TEMPLATE = Path(__file__).with_name("templates") / "replay.html"


def _result_dict(result: AgentResult) -> dict[str, Any]:
    return {
        "agent_id": result.agent_id,
        "goal": result.goal,
        "status": result.status,
        "answer": result.answer,
        "reason": result.reason,
        "children": [_result_dict(c) for c in result.children],
    }


def record_run(scenario: Scenario, model_name: str = "scripted") -> dict[str, Any]:
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

    orch = build_system(model, registry, on_event=on_event)
    result = orch.run(scenario.goal, scenario.tools)
    return {
        "name": scenario.name,
        "description": scenario.description,
        "goal": scenario.goal,
        "tools": sorted(scenario.tools),
        "model": model_name,
        "frames": frames,
        "result": _result_dict(result),
    }


def render_html(runs: Iterable[dict[str, Any]]) -> str:
    data = {
        "runs": list(runs),
        "transitions": {src.value: sorted(d.value for d in dsts) for src, dsts in TRANSITIONS.items()},
    }
    # Escape "</" so embedded text can never close the <script> tag early.
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("__REPLAY_DATA__", payload)


def write_html(path: str | Path, runs: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(runs), encoding="utf-8")
    return target
