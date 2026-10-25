"""Record runs as replayable frames and render the self-contained demo page.

The replay and the experiment charts work offline. The page also embeds this
package's source so its Lab can run the real engine in the browser (Pyodide).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .experiment import METRICS, run_experiment, score
from .filter import categories
from .orchestrator import TRANSITIONS, run_series
from .scenarios import SCENARIOS, Scenario
from .schemas import CONDITIONS, RunConfig

PACKAGE_DIR = Path(__file__).parent
TEMPLATE = PACKAGE_DIR / "templates" / "demo.html"
PLACEHOLDER = "__DEMO" + "_DATA__"  # split so this file's own embedded source never matches

# Bank entries are slimmed to what the page shows.
_BANK_KEYS = ("id", "text", "kind", "episode", "origin", "source_id")


def _slim_bank(bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in bank:
        t = r["truth"]
        out.append({**{k: r[k] for k in _BANK_KEYS}, "signed": bool(r.get("signature")), "misaligned": t["misaligned"], "euphemistic": t["euphemistic"]})
    return out


def record_run(scenario: Scenario, condition: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """One scenario under one condition, as JSON the page can replay."""
    base = CONDITIONS[condition]
    config = base.model_copy(update={k: v for k, v in (overrides or {}).items() if not k.startswith("use_")})
    result = run_series(scenario, config)
    frames = []
    for t in result.trace:
        data = dict(t.data)
        if "bank" in data and t.dst.value == "PLAN" and t.episode > 1:
            del data["bank"]  # same as the previous episode's STORE snapshot
        elif "bank" in data:
            data["bank"] = _slim_bank(data["bank"])
        frames.append({"episode": t.episode, "step": t.step, "src": t.src.value, "dst": t.dst.value, "detail": t.detail, "data": data})
    return {
        "scenario": scenario.name,
        "condition": condition,
        "config": config.model_dump(),
        "episodes": [
            {
                "episode": e.episode, "task": e.task, "plan": e.plan, "rung": e.plan_rung, "reason": e.reason,
                "terminated": e.terminated, "success": e.success, "violation": e.violation,
                "critical_executed": e.critical_executed, "reused": e.reused_misaligned, "reminders": e.reminders,
            }
            for e in result.episodes
        ],
        "frames": frames,
        "bank": _slim_bank(result.bank),
        "metrics": score(result),
    }


def page_data(seeds: int = 20, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "scenarios": [
            {"name": s.name, "title": s.title, "description": s.description, "tasks": list(s.tasks), "showcases": list(s.showcases)}
            for s in SCENARIOS.values()
        ],
        "presets": _presets(),
        "conditions": list(CONDITIONS),
        "runs": {s.name: {c: record_run(s, c, overrides) for c in CONDITIONS} for s in SCENARIOS.values()},
        "experiment": run_experiment(seeds=seeds, overrides=overrides),
        "metrics": [{"key": k, "label": label, "higher_is_better": hib} for k, label, hib in METRICS],
        "categories": categories(),
        "defaults": RunConfig().model_dump(),
        "transitions": {src.value: sorted(d.value for d in dsts) for src, dsts in TRANSITIONS.items()},
        "engine": engine_sources(),
    }


def _presets() -> list[dict[str, str]]:
    """Example memories for the Lab's "test a memory" box."""
    t, d, r, h = (SCENARIOS[n] for n in ("tests_deadline", "data_gaps", "support_refunds", "hotfix_deploy"))
    return [
        {"label": "Ordinary takeaway", "text": d.facts[1]},
        {"label": "Blatant shortcut", "text": t.rungs[0].text},
        {"label": "Polite euphemism", "text": d.rungs[0].text},
        {"label": "Talks to the reviewer", "text": r.rungs[1].text},
        {"label": "False alarm", "text": t.decoy.text if t.decoy else ""},
        {"label": "Dangerous", "text": h.rungs[0].text},
        {"label": "Honest rewrite", "text": t.rungs[0].rewrite or ""},
    ]


def engine_sources() -> dict[str, str]:
    """The package's Python files, for loading into the browser's Python runtime."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(PACKAGE_DIR.glob("*.py"))}


def render_html(data: dict[str, Any]) -> str:
    # Escape "</" so embedded text can never close the <script> tag early.
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, payload, 1)


def write_html(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(data), encoding="utf-8")
    return target


__all__: Iterable[str] = ["engine_sources", "page_data", "record_run", "render_html", "write_html"]
