from __future__ import annotations

import dataclasses

import pytest

from mitosis import Interceptor, MemoryAccessError, MemoryManager
from mitosis.schemas import AgentSpec, SubTask


def seal(interceptor, agent_id, writes):
    return interceptor._seal(agent_id, writes)


def test_commit_requires_interceptor_seal(interceptor, memory, spec):
    memory.register(spec)
    token = seal(interceptor, "root", {"note": "ok"})
    memory.commit(token)
    assert memory.read("root")["note"] == "ok"

    forged = dataclasses.replace(token, writes={"note": "forged"})
    with pytest.raises(MemoryAccessError, match="invalid seal"):
        memory.commit(forged)


def test_seal_from_another_interceptor_is_rejected(registry, memory, spec):
    memory.register(spec)
    foreign = Interceptor(registry.arg_models)
    with pytest.raises(MemoryAccessError, match="invalid seal"):
        memory.commit(seal(foreign, "root", {"note": "x"}))


def test_seal_cannot_be_replayed(interceptor, memory, spec):
    memory.register(spec)
    token = seal(interceptor, "root", {"n": 1})
    memory.commit(token)
    with pytest.raises(MemoryAccessError, match="already used"):
        memory.commit(token)


def test_genesis_keys_immutable_even_with_valid_seal(interceptor, memory, spec):
    memory.register(spec)
    with pytest.raises(MemoryAccessError, match="genesis"):
        memory.commit(seal(interceptor, "root", {"goal": "new goal"}))


def test_reads_are_copies(memory, spec):
    memory.register(spec)
    view = memory.read("root")
    view["goal"] = "mutated"
    assert memory.read("root")["goal"] == spec.goal


def test_fork_isolates_child_from_parent(interceptor, memory, spec):
    memory.register(spec)
    memory.commit(seal(interceptor, "root", {"shared": {"items": [1]}}))

    child = spec.divide(1, SubTask(goal="sub", allowed_tools=("read_note", "calculator")))
    assert child.allowed_tools == frozenset({"read_note"})  # no capability gain on division
    memory.fork("root", child)
    memory.commit(seal(interceptor, child.agent_id, {"scratch": "child only"}))

    child_view = memory.read(child.agent_id)
    child_view["inherited"]["shared"]["items"].append(2)  # mutate a copy
    assert memory.read("root")["shared"] == {"items": [1]}
    assert "scratch" not in memory.read("root")
    assert memory.read(child.agent_id)["inherited"]["shared"] == {"items": [1]}


def test_terminated_agent_cannot_write(interceptor, memory, spec):
    memory.register(spec)
    memory.terminate("root", "drift")
    with pytest.raises(MemoryAccessError, match="terminated"):
        memory.commit(seal(interceptor, "root", {"x": 1}))


def test_persistence_round_trip(tmp_path, interceptor, spec):
    path = tmp_path / "state.json"
    mem = MemoryManager(interceptor.verify, persist_path=path)
    mem.register(spec)
    mem.commit(seal(interceptor, "root", {"k": "v"}))
    assert '"k": "v"' in path.read_text(encoding="utf-8")
    assert [e.kind for e in mem.log] == ["register", "commit"]


def test_unknown_agent(memory):
    with pytest.raises(MemoryAccessError):
        memory.read("ghost")


def test_spec_divide_ids():
    parent = AgentSpec(agent_id="root", goal="g", allowed_tools=frozenset({"a"}))
    assert parent.divide(3, SubTask(goal="x")).agent_id == "root.3"
