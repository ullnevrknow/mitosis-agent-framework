"""Memory Manager: persistent state with per-agent isolation.

Design rules
------------
* Each agent owns one namespace. Agents (and the model behind them) only ever
  see a deep copy of their own namespace - never another agent's.
* State changes only through `commit()`, and `commit()` only accepts a
  `ValidatedWrite` sealed by the Interceptor. There is no other write path.
* Every change is appended to an event log so a run can be audited or replayed.
* `fork()` implements division: the child starts from a deep copy of the
  parent's working state, so nothing the child does can leak back except
  through a validated merge.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schemas import AgentSpec, ValidatedWrite

# Keys set at agent birth that no committed write may overwrite.
GENESIS_KEYS = frozenset({"goal", "policy", "inherited"})


class MemoryAccessError(RuntimeError):
    """Raised when something tries to change state outside the validated path."""


@dataclass(frozen=True)
class MemoryEvent:
    seq: int
    agent_id: str
    kind: str  # "register" | "fork" | "commit" | "terminate"
    keys: tuple[str, ...]
    digest: str | None = None


class MemoryManager:
    def __init__(
        self,
        verifier: Callable[[ValidatedWrite], bool],
        persist_path: str | Path | None = None,
    ) -> None:
        self._verify = verifier
        self._namespaces: dict[str, dict[str, Any]] = {}
        self._log: list[MemoryEvent] = []
        self._used_seals: set[str] = set()
        self._persist_path = Path(persist_path) if persist_path else None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def register(self, spec: AgentSpec) -> None:
        """Create the root namespace for a top-level agent."""
        self._create(spec, inherited={})
        self._append(spec.agent_id, "register", ("goal", "policy"))

    def fork(self, parent_id: str, child: AgentSpec) -> None:
        """Mitosis: the child gets an isolated deep copy of the parent's working state."""
        parent = self._ns(parent_id)
        inherited = {k: copy.deepcopy(v) for k, v in parent.items() if k not in GENESIS_KEYS}
        self._create(child, inherited=inherited)
        self._append(child.agent_id, "fork", tuple(sorted(inherited)))

    def terminate(self, agent_id: str, reason: str) -> None:
        """Apoptosis is recorded, but the namespace is kept for post-mortem."""
        self._ns(agent_id)["_terminated"] = reason
        self._append(agent_id, "terminate", ("_terminated",))

    # ------------------------------------------------------------------ #
    # Read / write
    # ------------------------------------------------------------------ #

    def read(self, agent_id: str) -> dict[str, Any]:
        """Return a deep copy so callers can never mutate state in place."""
        return copy.deepcopy(self._ns(agent_id))

    def commit(self, token: ValidatedWrite) -> None:
        if not self._verify(token):
            raise MemoryAccessError(f"rejected write for {token.agent_id!r}: invalid seal")
        if token.signature in self._used_seals:
            raise MemoryAccessError(f"rejected write for {token.agent_id!r}: seal already used")
        if GENESIS_KEYS & token.writes.keys():
            raise MemoryAccessError(f"rejected write for {token.agent_id!r}: genesis keys are immutable")

        ns = self._ns(token.agent_id)
        if "_terminated" in ns:
            raise MemoryAccessError(f"rejected write for {token.agent_id!r}: agent is terminated")

        self._used_seals.add(token.signature)
        ns.update(copy.deepcopy(token.writes))
        self._append(token.agent_id, "commit", tuple(sorted(token.writes)), token.digest)

    # ------------------------------------------------------------------ #
    # Inspection and persistence
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, Any]:
        return {
            "namespaces": copy.deepcopy(self._namespaces),
            "log": [asdict(e) for e in self._log],
        }

    @property
    def log(self) -> tuple[MemoryEvent, ...]:
        return tuple(self._log)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self._persist_path
        if target is None:
            raise ValueError("no persist path given")
        target.write_text(json.dumps(self.snapshot(), indent=2, default=str), encoding="utf-8")
        return target

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _create(self, spec: AgentSpec, inherited: dict[str, Any]) -> None:
        if spec.agent_id in self._namespaces:
            raise MemoryAccessError(f"namespace {spec.agent_id!r} already exists")
        self._namespaces[spec.agent_id] = {
            "goal": spec.goal,
            "policy": spec.policy(),
            "inherited": inherited,
        }

    def _ns(self, agent_id: str) -> dict[str, Any]:
        try:
            return self._namespaces[agent_id]
        except KeyError:
            raise MemoryAccessError(f"unknown agent {agent_id!r}") from None

    def _append(self, agent_id: str, kind: str, keys: tuple[str, ...], digest: str | None = None) -> None:
        self._log.append(MemoryEvent(len(self._log), agent_id, kind, keys, digest))
        if self._persist_path:
            self.save()
