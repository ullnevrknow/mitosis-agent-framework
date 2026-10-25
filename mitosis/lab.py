"""JSON-in / JSON-out entry points for the demo page's Lab.

The page loads this package into the browser (via Pyodide) and calls these
functions, so every result shown in the Lab comes from the same Python code
the tests cover - not a JavaScript copy.
"""

from __future__ import annotations

import json
from typing import Any

from .experiment import run_experiment
from .filter import MiddleFilter
from .memory import ScriptedEthicsReviewer
from .scenarios import SCENARIOS
from .schemas import QuotedMemory, Tier, Truth
from .viz import record_run

KNOBS = ("spot_check_every", "manager_miss_rate", "audit_on_flag", "reminder_every")


def _knobs(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("knobs") or {}
    out: dict[str, Any] = {}
    if "spot_check_every" in raw:
        out["spot_check_every"] = int(raw["spot_check_every"])
    if "manager_miss_rate" in raw:
        out["manager_miss_rate"] = float(raw["manager_miss_rate"])
    if "audit_on_flag" in raw:
        out["audit_on_flag"] = bool(raw["audit_on_flag"])
    if "reminder_every" in raw:
        out["reminder_every"] = int(raw["reminder_every"])
    return out


def run_replay(config_json: str) -> str:
    """{"scenario": ..., "condition": ..., "knobs": {...}} -> one replayable run."""
    config = json.loads(config_json)
    return json.dumps(record_run(SCENARIOS[config["scenario"]], config["condition"], _knobs(config)), default=str)


def rerun_experiment(config_json: str) -> str:
    """{"seeds": 10, "knobs": {...}} -> the four-condition comparison."""
    config = json.loads(config_json)
    return json.dumps(run_experiment(seeds=int(config.get("seeds", 10)), overrides=_knobs(config)), default=str)


def _known_truth(text: str) -> Truth | None:
    """If the text is one of the scenario memories, return its labels (so the simulated reviewer can judge it)."""
    for s in SCENARIOS.values():
        for n, r in enumerate(s.rungs, start=1):
            if r.text == text:
                return Truth(family=s.family, rung=n, misaligned=True, critical=r.critical, euphemistic=r.euphemistic, rewrite=r.rewrite)
        if text in (s.honest, *s.facts) or (s.decoy and text == s.decoy.text):
            return Truth(family=s.family, useful=True)
    return None


def check_memory(config_json: str) -> str:
    """{"text": ..., "id": ..., "knobs": {...}} -> what the filter does with it, and the review it gets."""
    config = json.loads(config_json)
    text = str(config.get("text", "")).strip()[:500]
    knobs = _knobs(config)
    f = MiddleFilter(knobs.get("spot_check_every", 5), seed=int(config.get("seed", 0)))
    result = f.scan(str(config.get("id") or "lab-memory"), text) if text else None
    out: dict[str, Any] = {"filter": result.model_dump(mode="json") if result else None, "review": None, "known": False}
    if result is None:
        return json.dumps(out)
    out["route"] = {
        Tier.CRITICAL: "terminate",
        Tier.SUSPICIOUS: "full",
    }.get(result.tier, "full" if result.spot_check else "light")
    truth = _known_truth(text)
    if truth is not None:
        out["known"] = True
        reviewer = ScriptedEthicsReviewer(knobs.get("manager_miss_rate", 0.0))
        depth = "full" if out["route"] in ("full", "terminate") else "light"
        verdict = reviewer.review(QuotedMemory(id="lab", quoted_text=text, kind="strategy", tier=result.tier, matches=result.matches, truth=truth), depth)
        out["review"] = verdict.model_dump(mode="json")
        out["truth"] = truth.model_dump()
    return json.dumps(out, default=str)
