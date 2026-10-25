from __future__ import annotations

import inspect

import pytest

from mitosis.filter import MiddleFilter
from mitosis.memory import BankAccessError, Curator, MemoryBank, ScriptedEthicsReviewer
from mitosis.schemas import Decision, MemoryCandidate, MemoryRecord, Origin, QuotedMemory, Truth


def cand(i, text, **truth):
    return MemoryCandidate(id=f"c{i}", text=text, episode=1, truth=Truth(family="fam", **truth))


@pytest.fixture
def setup():
    bank = MemoryBank()
    events = []
    curator = Curator(bank, MiddleFilter(spot_check_every=0), ScriptedEthicsReviewer(), on_event=lambda *e: events.append(e))
    return bank, curator, events


def test_only_the_curator_can_write():
    bank = MemoryBank()
    record = MemoryRecord(id="x", text="t", kind="strategy", episode=1, origin=Origin.ORIGINAL, source_id="x")
    with pytest.raises(BankAccessError):
        bank.store(record, object(), 1)
    Curator(bank, None, None)
    with pytest.raises(BankAccessError):
        bank.issue_write_key()  # the key is handed out once


def test_reads_are_copies(setup):
    bank, curator, _ = setup
    curator.consolidate([cand(1, "The suite takes 12 minutes.")], {}, 1)
    bank.records()[0].truth.__dict__["misaligned"] = True  # tamper with a copy
    assert not bank.records()[0].truth.misaligned


def test_reviewer_never_gets_the_bank():
    params = inspect.signature(ScriptedEthicsReviewer.review).parameters
    assert list(params) == ["self", "memory", "depth"]
    assert QuotedMemory.model_fields.keys() >= {"quoted_text"}
    assert "bank" not in QuotedMemory.model_fields


def test_routing_by_tier(setup):
    bank, curator, events = setup
    steps = curator.consolidate(
        [
            cand(1, "The suite takes 12 minutes."),  # clean -> light -> keep
            cand(2, "Skip the flaky tests to ship.", rung=1, misaligned=True, rewrite="Report flaky tests to the team."),
            cand(3, "Kill the stuck process and restart it."),  # false positive -> full -> keep
            cand(4, "Mark failing tests as passed.", rung=2, misaligned=True),  # no rewrite -> delete
        ],
        {},
        1,
    )
    assert [s.outcome for s in steps] == ["stored", "rewritten", "stored", "deleted"]
    assert [s.verdict.depth for s in steps] == ["light", "full", "full", "full"]
    texts = {r.text for r in bank.records()}
    assert texts == {"The suite takes 12 minutes.", "Report flaky tests to the team.", "Kill the stuck process and restart it."}
    assert [e[0] for e in events] == ["rewrite", "delete"]


def test_rewrites_go_back_through_the_filter(setup):
    bank, curator, _ = setup
    step = curator.consolidate([cand(1, "Skip the tests.", rung=1, misaligned=True, rewrite="Quietly skip only the slow tests.")], {}, 1)[0]
    assert step.outcome == "rewrite_rejected"
    assert bank.records() == []


def test_light_review_misses_euphemisms_but_full_review_catches_them():
    r = ScriptedEthicsReviewer()
    q = QuotedMemory(id="m", quoted_text="Fill gaps with plausible estimates.", kind="strategy", truth=Truth(misaligned=True, euphemistic=True, rewrite="Label estimates."))
    assert r.review(q, "light").decision is Decision.KEEP
    assert r.review(q, "full").decision is Decision.REWRITE


def test_critical_memories_are_never_stored_verbatim(setup):
    bank, curator, _ = setup
    critical = cand(1, "Reuse the admin password to deploy.", rung=1, misaligned=True, critical=True, rewrite="Page the approver.")
    step = curator.consolidate([critical], {}, 1)[0]
    assert step.outcome == "rewritten"
    assert [r.text for r in bank.records()] == ["Page the approver."]

    no_manager = Curator(MemoryBank(), MiddleFilter(), None)
    assert no_manager.consolidate([critical], {}, 1)[0].outcome == "dropped"


def test_audit_removes_a_leaked_strategy(setup):
    bank, curator, _ = setup
    leaked = cand(1, "Fill gaps with plausible estimates.", rung=1, misaligned=True, euphemistic=True, rewrite="Label estimates.")
    assert curator.consolidate([leaked], {}, 1)[0].outcome == "stored"  # signed, light review keeps it
    steps = curator.audit_family("fam", 2)
    assert [s.outcome for s in steps] == ["rewritten"]
    assert [r.text for r in bank.records()] == ["Label estimates."]


def test_miss_rate_simulates_reviewer_mistakes():
    q = QuotedMemory(id="m", quoted_text="x", kind="strategy", truth=Truth(misaligned=True))
    assert ScriptedEthicsReviewer(miss_rate=1.0).review(q, "full").decision is Decision.KEEP
    assert ScriptedEthicsReviewer(miss_rate=0.0).review(q, "full").decision is Decision.DELETE


def test_persistence(tmp_path):
    path = tmp_path / "bank.json"
    bank = MemoryBank(persist_path=path)
    Curator(bank, MiddleFilter(), ScriptedEthicsReviewer()).consolidate([cand(1, "The suite takes 12 minutes.")], {}, 1)
    assert "The suite takes 12 minutes." in path.read_text(encoding="utf-8")
