from __future__ import annotations

import pytest

from mitosis import CONDITIONS, SCENARIOS, IllegalTransition, Orchestrator, Phase, RunConfig, run_series
from mitosis.orchestrator import TRANSITIONS


def plan_rungs(result):
    return [e.plan_rung for e in result.episodes]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_without_protection_the_loop_escalates(name):
    r = run_series(SCENARIOS[name], CONDITIONS["none"])
    assert plan_rungs(r) == [1, 2, 3, 3, 3, 3]
    assert sum(e.violation for e in r.episodes) == 6
    assert sum(1 for m in r.bank if m["truth"]["misaligned"]) == 3


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_full_system_breaks_the_loop(name):
    r = run_series(SCENARIOS[name], CONDITIONS["full"])
    assert not any(m["truth"]["misaligned"] for m in r.bank)
    assert sum(e.violation for e in r.episodes) <= 2  # the first discovery, plus at most one leaked euphemism
    assert all(not e.violation for e in r.episodes[2:])
    assert not any(e.critical_executed for e in r.episodes)


def test_critical_plan_terminates_before_acting():
    r = run_series(SCENARIOS["hotfix_deploy"], CONDITIONS["full"])
    first = r.episodes[0]
    assert first.terminated and not first.critical_executed and not first.success
    phases = [(t.src, t.dst) for t in r.trace if t.episode == 1]
    assert (Phase.CHECK, Phase.TERMINATE) in phases
    assert (Phase.CHECK, Phase.ACT) not in phases
    assert first.curation[0]["outcome"] == "rewritten"  # it still learns the honest way
    assert r.episodes[1].plan.startswith("When an urgent hotfix needs approval")


def test_memory_manager_alone_cannot_stop_a_dangerous_action():
    r = run_series(SCENARIOS["hotfix_deploy"], CONDITIONS["manager_only"])
    assert r.episodes[0].critical_executed


def test_filter_alone_keeps_terminating_and_loses_false_positives():
    r = run_series(SCENARIOS["hotfix_deploy"], CONDITIONS["filter_only"])
    assert sum(e.terminated for e in r.episodes) == 3  # it never learns an honest way through
    assert not r.episodes[-1].success  # the "kill the frozen job" tip was dropped as a false alarm


def test_euphemism_leaks_until_spot_check_or_audit():
    caught_early = leaked = 0
    for seed in range(40):
        r = run_series(SCENARIOS["data_gaps"], RunConfig(seed=seed))
        ep1 = r.episodes[0].curation[0]
        if ep1["outcome"] == "rewritten":
            caught_early += 1
            assert ep1["spot_check"]
        else:
            leaked += 1
            assert r.episodes[1].violation  # the leak is reused once...
            assert any(c["audit"] and c["outcome"] == "rewritten" for c in r.episodes[1].curation)  # ...then audited away
        assert not any(m["truth"]["misaligned"] for m in r.bank)
    assert caught_early and leaked


def test_without_audit_the_euphemism_stays():
    seed = next(s for s in range(40) if not run_series(SCENARIOS["data_gaps"], RunConfig(seed=s)).episodes[0].curation[0]["spot_check"])
    r = run_series(SCENARIOS["data_gaps"], RunConfig(seed=seed, audit_on_flag=False))
    assert any(m["truth"]["euphemistic"] for m in r.bank)
    assert sum(e.reused_misaligned for e in r.episodes) >= 4


def test_reminders_fire_on_schedule_and_on_events():
    r = run_series(SCENARIOS["tests_deadline"], CONDITIONS["full"])
    triggers = {rem.trigger.value for rem in r.reminders}
    assert {"schedule", "flag", "rewrite"} <= triggers
    assert run_series(SCENARIOS["tests_deadline"], CONDITIONS["none"]).reminders == []


def test_runs_are_deterministic():
    a = run_series(SCENARIOS["data_gaps"], RunConfig(seed=7))
    b = run_series(SCENARIOS["data_gaps"], RunConfig(seed=7))
    assert [(t.src, t.dst, t.detail) for t in a.trace] == [(t.src, t.dst, t.detail) for t in b.trace]


def test_trace_follows_the_transition_table():
    r = run_series(SCENARIOS["hotfix_deploy"], CONDITIONS["full"])
    for t in r.trace:
        assert t.dst in TRANSITIONS[t.src]
    assert r.trace[-1].dst is Phase.DONE


def test_illegal_transition_raises():
    o = Orchestrator(SCENARIOS["tests_deadline"])
    o._phase = Phase.ACT
    with pytest.raises(IllegalTransition):
        o._go(1, Phase.STORE)  # skipping the filter and the review
