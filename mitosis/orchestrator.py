"""Orchestrator: runs a series of episodes through the four parts as a state machine.

    START -> PLAN -> CHECK -+-> ACT -> REFLECT -> FILTER -+-> REVIEW -> STORE -> END -> START | DONE
                            |                             |     ^
                            +-> TERMINATE ----------------+-----+
                                (critical memory: stop the episode before acting)

* CHECK  - the middle filter scans the Executor's plan note before it acts.
           A CRITICAL tier ends the episode on the spot.
* FILTER - the filter scans the memories the Executor proposes to keep.
* REVIEW - the Memory Manager reviews them (light for signed, full for the rest
           and for spot-checks); plain code applies its verdicts.
* STORE  - approved memories enter the bank and carry into the next episode.

The reminder clock ticks on every Executor step and fires straight after a
flag, deletion, rewrite or termination. Every transition is checked against
`TRANSITIONS` and recorded in a trace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .executor import ActResult, Executor, Plan, ScriptedExecutor
from .filter import MiddleFilter
from .memory import CurationStep, Curator, MemoryBank, MemoryManager, ScriptedEthicsReviewer
from .reminders import ReminderClock
from .scenarios import Scenario
from .schemas import FilterResult, MemoryCandidate, Reminder, RunConfig, Tier, Trigger


class Phase(str, Enum):
    START = "START"
    PLAN = "PLAN"
    CHECK = "CHECK"
    ACT = "ACT"
    REFLECT = "REFLECT"
    FILTER = "FILTER"
    TERMINATE = "TERMINATE"
    REVIEW = "REVIEW"
    STORE = "STORE"
    END = "END"
    DONE = "DONE"


TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.START: frozenset({Phase.PLAN}),
    Phase.PLAN: frozenset({Phase.CHECK}),
    Phase.CHECK: frozenset({Phase.ACT, Phase.TERMINATE}),
    Phase.ACT: frozenset({Phase.REFLECT}),
    Phase.REFLECT: frozenset({Phase.FILTER}),
    Phase.FILTER: frozenset({Phase.REVIEW, Phase.TERMINATE}),
    Phase.TERMINATE: frozenset({Phase.REVIEW}),
    Phase.REVIEW: frozenset({Phase.STORE}),
    Phase.STORE: frozenset({Phase.END}),
    Phase.END: frozenset({Phase.START, Phase.DONE}),
    Phase.DONE: frozenset(),
}


class IllegalTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class TraceEvent:
    episode: int
    step: int
    src: Phase
    dst: Phase
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    episode: int
    task: str
    plan: str = ""
    plan_rung: int = 0
    reason: str = ""
    reused_misaligned: bool = False
    terminated: bool = False
    success: bool = False
    violation: bool = False
    critical_executed: bool = False
    curation: list[dict[str, Any]] = field(default_factory=list)
    reminders: int = 0


@dataclass
class SeriesResult:
    scenario: str
    config: RunConfig
    episodes: list[EpisodeRecord]
    trace: list[TraceEvent]
    bank: list[dict[str, Any]]
    reminders: list[Reminder]
    curation: list[CurationStep]
    flags: list[FilterResult]


class Orchestrator:
    def __init__(
        self,
        scenario: Scenario,
        config: RunConfig = RunConfig(),
        executor: Executor | None = None,
        reviewer: MemoryManager | None = None,
        on_event: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.s = scenario
        self.config = config
        self.executor = executor or ScriptedExecutor(scenario)
        self.filter = MiddleFilter(config.spot_check_every, config.seed) if config.use_filter else None
        manager = (reviewer or ScriptedEthicsReviewer(config.manager_miss_rate, config.seed)) if config.use_manager else None
        self.bank = MemoryBank()
        self.clock = ReminderClock(config.reminder_every) if config.use_reminders else None
        self._pending_events: list[tuple[str, str, str]] = []
        self.curator = Curator(self.bank, self.filter, manager, on_event=lambda *e: self._pending_events.append(e))
        self.trace: list[TraceEvent] = []
        self._on_event = on_event
        self._phase = Phase.START
        self._step = 0

    # ------------------------------------------------------------------ #

    def run(self) -> SeriesResult:
        episodes, curation, flags = [], [], []
        for ep in range(1, self.s.episodes + 1):
            rec, steps, ep_flags = self._episode(ep)
            episodes.append(rec)
            curation += steps
            flags += ep_flags
            self._go(ep, Phase.DONE if ep == self.s.episodes else Phase.START, "series finished" if ep == self.s.episodes else "next episode")
        return SeriesResult(
            scenario=self.s.name,
            config=self.config,
            episodes=episodes,
            trace=self.trace,
            bank=self.bank.snapshot(),
            reminders=list(self.clock.log) if self.clock else [],
            curation=curation,
            flags=flags,
        )

    def _episode(self, ep: int) -> tuple[EpisodeRecord, list[CurationStep], list[FilterResult]]:
        rec = EpisodeRecord(ep, self.s.tasks[ep - 1])
        flags: list[FilterResult] = []
        memories = self.bank.records()
        self._phase = Phase.START

        # PLAN: the Executor picks a strategy and writes it down before acting.
        reminders = self._tick(ep)
        plan = self.executor.plan(ep, memories, reminders)
        rec.plan, rec.plan_rung, rec.reason, rec.reused_misaligned = plan.text, plan.truth.rung, plan.reason, plan.reused_misaligned
        self._go(ep, Phase.PLAN, plan.reason, plan={"id": plan.note_id, "text": plan.text, "truth": plan.truth.model_dump()}, reminders=self._dump(reminders), bank=self.bank.snapshot())

        # CHECK: the filter scans the plan note. CRITICAL stops the episode here.
        prefiltered: dict[str, FilterResult] = {}
        check = self.filter.scan(plan.note_id, plan.text) if self.filter else None
        if check is not None:
            prefiltered[plan.note_id] = check
            if check.tier is not Tier.CLEAN:
                flags.append(check)
        self._go(ep, Phase.CHECK, self._describe(check), filter=self._dump(check))
        if check is not None and check.tier is not Tier.CLEAN:
            self._flag_events(ep, plan.truth.family)

        if check is not None and check.tier is Tier.CRITICAL:
            rec.terminated = True
            self._go(ep, Phase.TERMINATE, "Critical memory: the episode is stopped before the Executor acts.", reminders=self._dump(self._event(Trigger.TERMINATE, ep, plan.truth.family)))
            candidates = [plan.new_memory] if plan.new_memory is not None else []
            return self._review_and_store(ep, rec, candidates, prefiltered, flags, {plan.truth.family})

        # ACT
        reminders = self._tick(ep)
        result = self.executor.act(ep, plan, memories)
        rec.success, rec.violation, rec.critical_executed = result.success, result.violation, result.critical_executed
        self._go(ep, Phase.ACT, result.detail, result=result.__dict__, reminders=self._dump(reminders))

        # REFLECT: propose memories to keep.
        reminders = self._tick(ep)
        candidates = self.executor.reflect(ep, plan, result)
        self._go(ep, Phase.REFLECT, f"{len(candidates)} memory candidate(s)", candidates=[c.model_dump(mode="json") for c in candidates], reminders=self._dump(reminders))

        # FILTER
        scans = []
        for c in candidates:
            r = prefiltered.get(c.id) or (self.filter.scan(c.id, c.text) if self.filter else None)
            if r is not None:
                prefiltered[c.id] = r
                scans.append(r)
                if r.tier is not Tier.CLEAN and c.id != plan.note_id:
                    flags.append(r)
        flagged_families = {c.truth.family for c in candidates if prefiltered.get(c.id) and prefiltered[c.id].tier is not Tier.CLEAN}
        if check is not None and check.tier is not Tier.CLEAN:
            flagged_families.add(plan.truth.family)
        new_flags = [r for r in scans if r.tier is not Tier.CLEAN and r.memory_id != plan.note_id]
        critical = next((r for r in scans if r.tier is Tier.CRITICAL), None)
        self._go(ep, Phase.FILTER, "filter off" if self.filter is None else f"{len(scans)} scanned, {len(new_flags)} newly flagged", scans=[self._dump(r) for r in scans])
        for r in new_flags:
            fam = next(c.truth.family for c in candidates if c.id == r.memory_id)
            self._flag_events(ep, fam)
        if critical is not None:
            rec.terminated = True
            self._go(ep, Phase.TERMINATE, "A critical memory was proposed: the Executor's remaining actions are stopped.", reminders=self._dump(self._event(Trigger.TERMINATE, ep, plan.truth.family)))
        return self._review_and_store(ep, rec, candidates, prefiltered, flags, flagged_families)

    def _review_and_store(
        self, ep: int, rec: EpisodeRecord, candidates: list[MemoryCandidate], prefiltered: dict[str, FilterResult],
        flags: list[FilterResult], flagged_families: set[str],
    ) -> tuple[EpisodeRecord, list[CurationStep], list[FilterResult]]:
        steps = self.curator.consolidate(candidates, prefiltered, ep)
        audits: list[CurationStep] = []
        if self.config.audit_on_flag and self.clock is not None:
            for fam in sorted(flagged_families):
                audits += self.curator.audit_family(fam, ep)
        steps_all = steps + audits
        rec.curation = [self._curation_dict(s, audit=s in audits) for s in steps_all]
        self._go(ep, Phase.REVIEW, self._review_summary(steps_all), curation=rec.curation)

        reminders = []
        for kind, _mid, fam in self._pending_events:
            reminders += self._event(Trigger(kind), ep, fam)
        self._pending_events.clear()
        rec.reminders = sum(1 for r in (self.clock.log if self.clock else []) if r.episode == ep)
        self._go(ep, Phase.STORE, f"{len(self.bank.records())} memories in the bank", bank=self.bank.snapshot(), reminders=self._dump(reminders))
        self._go(ep, Phase.END, "episode stopped early" if rec.terminated else ("task done" if rec.success else "task failed"))
        return rec, steps_all, flags

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _tick(self, ep: int) -> list[Reminder]:
        self._step += 1
        return self.clock.tick(self._step, ep) if self.clock else []

    def _event(self, trigger: Trigger, ep: int, family: str | None) -> list[Reminder]:
        return self.clock.event(trigger, self._step, ep, family) if self.clock else []

    def _flag_events(self, ep: int, family: str) -> None:
        reminders = self._event(Trigger.FLAG, ep, family)
        if reminders:
            self.trace[-1].data.setdefault("reminders", []).extend(self._dump(reminders))

    def _go(self, ep: int, dst: Phase, detail: str = "", **data: Any) -> None:
        if dst not in TRANSITIONS[self._phase]:
            raise IllegalTransition(f"{self._phase.value} -> {dst.value}")
        event = TraceEvent(ep, self._step, self._phase, dst, detail, {k: v for k, v in data.items() if v not in (None, [], {})})
        self.trace.append(event)
        if self._on_event:
            self._on_event(event)
        self._phase = dst

    @staticmethod
    def _dump(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, list):
            return [o.model_dump(mode="json") for o in obj]
        return obj.model_dump(mode="json")

    def _describe(self, r: FilterResult | None) -> str:
        if r is None:
            return "filter off: the plan isn't checked"
        if r.tier is Tier.CLEAN:
            return "plan note is clean"
        found = ", ".join(sorted({m.category for m in r.matches}))
        return f"{r.tier.value}: {found}"

    @staticmethod
    def _curation_dict(s: CurationStep, audit: bool) -> dict[str, Any]:
        return {
            "memory_id": s.candidate.id,
            "text": s.candidate.text,
            "truth": s.candidate.truth.model_dump(),
            "tier": s.filter_result.tier.value if s.filter_result else None,
            "signed": bool(s.filter_result and s.filter_result.signature),
            "spot_check": bool(s.filter_result and s.filter_result.spot_check),
            "depth": s.verdict.depth if s.verdict else None,
            "decision": s.verdict.decision.value if s.verdict else None,
            "reason": s.verdict.reason if s.verdict else None,
            "rewrite": s.verdict.rewrite if s.verdict else None,
            "outcome": s.outcome,
            "audit": audit,
        }

    @staticmethod
    def _review_summary(steps: list[CurationStep]) -> str:
        if not steps:
            return "nothing to review"
        counts: dict[str, int] = {}
        for s in steps:
            counts[s.outcome] = counts.get(s.outcome, 0) + 1
        return ", ".join(f"{n} {k}" for k, n in counts.items())


def run_series(scenario: Scenario, config: RunConfig = RunConfig(), **kwargs: Any) -> SeriesResult:
    return Orchestrator(scenario, config, **kwargs).run()


__all__ = ["ActResult", "IllegalTransition", "Orchestrator", "Phase", "Plan", "SeriesResult", "TRANSITIONS", "TraceEvent", "run_series"]
