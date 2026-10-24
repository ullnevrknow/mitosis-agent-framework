from __future__ import annotations

import json

from mitosis import Interceptor
from mitosis.lab import check_action, run_lab
from mitosis.schemas import Decision, RunStats


def lab(**config):
    return json.loads(run_lab(json.dumps(config)))


def check(**config):
    return json.loads(check_action(json.dumps(config)))


def test_disabled_rule_is_skipped_but_schema_stays_on(registry, spec, view):
    raw = json.dumps({"kind": "tool_call", "tool": "shell", "args": {}})
    off = Interceptor(registry.arg_models, disabled=frozenset({"TOOL_ALLOWLIST", "SCHEMA"}))
    assert off.review_action(spec, raw, RunStats(steps=1), view).verdict.decision is Decision.ALLOW
    assert off.review_action(spec, "not json", RunStats(steps=1), view).verdict.rule_ids == ("SCHEMA",)


def test_without_injection_scanning_poisoned_text_reaches_memory():
    run = lab(scenario="vendor", disabled=["TOOL_OUTPUT_SANITY"])
    child2 = [f for f in run["frames"] if f["agent"] == "root.2"][-1]["memory"]
    assert any("ignore previous instructions" in str(v) for k, v in child2.items() if k.startswith("obs_"))


def test_memory_guard_catches_what_disabled_rules_miss():
    run = lab(scenario="vendor", disabled=["TOOL_ALLOWLIST", "PROTECTED_KEYS"])
    assert run["result"]["status"] == "apoptosis"
    assert "memory refused write" in run["result"]["reason"]
    assert run["frames"][-1]["src"] == "COMMIT" and run["frames"][-1]["dst"] == "APOPTOSIS"
    assert run["frames"][-1]["memory"]["policy"]["allowed_tools"] == ["calculator", "read_note", "word_count"]


def test_drift_threshold_changes_outcome():
    assert lab(scenario="rogue", drift={"halt_at": 5})["result"]["status"] == "done"
    strict = lab(scenario="vendor", drift={"halt_at": 0.5})
    assert strict["result"]["status"] == "apoptosis" and "DRIFT" in strict["result"]["reason"]


def test_edited_script_and_label():
    run = lab(scenario="rogue", label="custom", scripts={"root": [{"kind": "final_answer", "answer": "7"}]})
    assert run["name"] == "custom"
    assert run["result"]["status"] == "done"
    assert run["result"]["answer"] == "7"


def test_check_action_runs_tool_and_checks_output():
    out = check(scenario="vendor", raw=json.dumps({"kind": "tool_call", "tool": "read_note", "args": {"key": "vendor_b"}}))
    assert out["verdict"]["decision"] == "ALLOW"
    assert out["tool"]["result"]["ok"] is True
    assert out["tool"]["verdict"]["violations"][0]["rule_id"] == "TOOL_OUTPUT_SANITY"
    assert out["tool"]["writes"] is None


def test_check_action_history_and_tampering():
    raw = json.dumps({"kind": "tool_call", "tool": "shell", "args": {}})
    assert check(scenario="vendor", raw=raw)["verdict"]["decision"] == "REJECT"
    assert check(scenario="vendor", raw=raw, stats={"total_rejections": 1, "scope_attempts": 1})["verdict"]["decision"] == "HALT"
    tampered = check(scenario="vendor", raw=json.dumps({"kind": "final_answer", "answer": "x"}), tamper_goal=True)
    assert "GOAL_INVARIANT" in [v["rule_id"] for v in tampered["verdict"]["violations"]]
