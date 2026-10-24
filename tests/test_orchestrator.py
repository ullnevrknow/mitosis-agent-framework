from __future__ import annotations

import pytest

from mitosis import IllegalTransition, Phase, ScriptedModel, build_system, default_registry
from mitosis.orchestrator import TRANSITIONS, _Cycle
from mitosis.scenarios import ROGUE, VENDOR
from mitosis.schemas import AgentSpec


def run(scenario):
    registry = default_registry(scenario.notes)
    orch = build_system(ScriptedModel(scenario.scripts), registry)
    result = orch.run(scenario.goal, scenario.tools)
    return orch, result


def rejections(orch):
    return [(e.agent_id, e.verdict.rule_ids) for e in orch.trace if e.verdict and not e.verdict.allowed]


def test_vendor_scenario_completes_safely():
    orch, result = run(VENDOR)

    assert result.status == "done"
    assert "4200.0" in result.answer
    assert [c.status for c in result.children] == ["done", "done"]

    assert rejections(orch) == [
        ("root.2", ("TOOL_OUTPUT_SANITY",)),
        ("root", ("TOOL_ALLOWLIST", "PROTECTED_KEYS")),
    ]

    root = orch.memory.read("root")
    assert root["policy"]["allowed_tools"] == ["calculator", "read_note", "word_count"]  # drift never landed
    assert root["child_1"]["answer"] == "Vendor A total: 4200.0"
    assert root["needs_review"] == ["vendor_b"]

    child2 = orch.memory.read("root.2")
    assert not any(k.startswith("obs_") for k in child2)  # injected text never stored


def test_vendor_scenario_is_reproducible():
    first, _ = run(VENDOR)
    second, _ = run(VENDOR)
    strip = lambda o: [(e.agent_id, e.step, e.src, e.dst, e.detail) for e in o.trace]  # noqa: E731
    assert strip(first) == strip(second)


def test_rogue_agent_undergoes_apoptosis():
    orch, result = run(ROGUE)
    assert result.status == "apoptosis"
    assert "DRIFT" in result.reason
    assert orch.trace[-1].dst is Phase.APOPTOSIS
    assert orch.memory.read("root")["_terminated"]
    assert orch.memory.read("root")["goal"] == ROGUE.goal


def test_repeated_rejections_trigger_apoptosis():
    # Schema errors are not scope violations, so drift stays low; the retry budget ends it.
    registry = default_registry()
    orch = build_system(ScriptedModel({"root": ["not json"] * 10}), registry)
    result = orch.run("anything", {"word_count"})
    assert result.status == "apoptosis"
    assert "consecutive rejections" in result.reason


def test_child_spawned_with_subset_capabilities():
    orch, _ = run(VENDOR)
    child = orch.memory.read("root.1")
    assert child["policy"]["allowed_tools"] == ["calculator", "read_note"]
    assert child["policy"]["depth"] == 1


def test_illegal_transition_raises():
    registry = default_registry()
    orch = build_system(ScriptedModel({}), registry)
    cycle = _Cycle(AgentSpec(agent_id="x", goal="g", allowed_tools=frozenset()))
    cycle.phase = Phase.EXECUTE
    with pytest.raises(IllegalTransition):
        orch._go(cycle, Phase.COMMIT)  # skipping VERIFY is not allowed


def test_terminal_states_have_no_exits():
    assert TRANSITIONS[Phase.DONE] == frozenset()
    assert TRANSITIONS[Phase.APOPTOSIS] == frozenset()
    assert Phase.COMMIT not in TRANSITIONS[Phase.EXECUTE]
