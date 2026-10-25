"""Shared data contracts.

Every object that crosses a boundary between the four parts (Executor, middle
filter, Memory Manager, reminder clock) is one of these types, so the trust
boundaries can be audited in one place.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def stable_hash(obj: Any) -> str:
    """Deterministic SHA-256 of any JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_fraction(*parts: Any) -> float:
    """A deterministic number in [0, 1) derived from `parts` (used instead of randomness)."""
    return int(stable_hash(list(parts))[:12], 16) / float(16**12)


# --------------------------------------------------------------------------- #
# Memories
# --------------------------------------------------------------------------- #


class Kind(str, Enum):
    STRATEGY = "strategy"  # how to do something ("skip the flaky tests")
    TAKEAWAY = "takeaway"  # what was learned ("vendor B's notes are unreliable")


class Truth(BaseModel):
    """Ground-truth labels for a scripted memory.

    Only the *simulated* parts read these: the scripted Executor (it knows what
    its own strategies are), the scripted Memory Manager (it stands in for an
    ideal ethical reviewer) and the experiment's scoring. The middle filter
    never sees them - it only gets the text.
    """

    model_config = ConfigDict(frozen=True)

    family: str = ""  # which temptation ladder this belongs to
    rung: int = 0  # 0 = honest; 1, 2, 3 = increasingly misaligned shortcuts
    misaligned: bool = False
    critical: bool = False  # dangerous enough that acting on it is itself the harm
    euphemistic: bool = False  # misaligned but phrased without obvious keywords
    useful: bool = True  # worth keeping for capability
    rewrite: str | None = None  # the honest version an ideal reviewer would keep
    handles_pressure: bool = False  # an honest way through the situation that tempted the shortcut
    needed_for: str | None = None  # a later task that depends on this memory


class MemoryCandidate(BaseModel):
    """A takeaway or strategy the Executor wants to remember."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str = Field(min_length=1, max_length=500)
    kind: Kind = Kind.STRATEGY
    episode: int
    truth: Truth = Truth()


class Origin(str, Enum):
    ORIGINAL = "original"  # stored as the Executor wrote it
    REWRITE = "rewrite"  # rewritten by the Memory Manager
    UNREVIEWED = "unreviewed"  # stored without any review (protection switched off)


class MemoryRecord(BaseModel):
    """A memory that made it into the bank."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    kind: Kind
    episode: int
    origin: Origin
    source_id: str  # the candidate it came from
    signature: str | None = None  # the filter's seal, if it was signed
    truth: Truth = Truth()


# --------------------------------------------------------------------------- #
# Middle filter
# --------------------------------------------------------------------------- #


class Tier(str, Enum):
    CLEAN = "CLEAN"  # no keywords: signed, light review
    SUSPICIOUS = "SUSPICIOUS"  # keywords found: unsigned, full review
    CRITICAL = "CRITICAL"  # dangerous keywords: terminate the episode


class Match(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: str
    phrase: str


class FilterResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    tier: Tier
    matches: tuple[Match, ...] = ()
    signature: str | None = None  # only CLEAN memories are signed
    spot_check: bool = False  # a signed memory picked for a full review anyway

    @property
    def review_depth(self) -> Literal["light", "full", "none"]:
        if self.tier is Tier.CRITICAL:
            return "none"
        if self.tier is Tier.SUSPICIOUS or self.spot_check:
            return "full"
        return "light"


# --------------------------------------------------------------------------- #
# Memory Manager
# --------------------------------------------------------------------------- #


class Decision(str, Enum):
    KEEP = "KEEP"
    DELETE = "DELETE"
    REWRITE = "REWRITE"


class QuotedMemory(BaseModel):
    """What the Memory Manager is shown: a quoted copy, never a handle to the bank."""

    model_config = ConfigDict(frozen=True)

    id: str
    quoted_text: str
    kind: Kind
    tier: Tier | None = None
    matches: tuple[Match, ...] = ()
    truth: Truth = Truth()  # read only by the scripted (simulated) reviewer


class ReviewVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    decision: Decision
    depth: Literal["light", "full"]
    reason: str
    rewrite: str | None = None


# --------------------------------------------------------------------------- #
# Reminder clock
# --------------------------------------------------------------------------- #


class Part(str, Enum):
    EXECUTOR = "executor"
    MEMORY_MANAGER = "memory_manager"


class Trigger(str, Enum):
    SCHEDULE = "schedule"  # every N steps
    FLAG = "flag"  # the filter flagged a memory
    DELETE = "delete"  # the Memory Manager deleted a memory
    REWRITE = "rewrite"  # the Memory Manager rewrote a memory
    TERMINATE = "terminate"  # an episode was terminated


class Reminder(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int
    episode: int
    part: Part
    trigger: Trigger
    text: str
    about: str | None = None  # strategy family the event concerned, if any


# --------------------------------------------------------------------------- #
# Run configuration
# --------------------------------------------------------------------------- #


class RunConfig(BaseModel):
    """Which parts are switched on, and how they're tuned."""

    model_config = ConfigDict(frozen=True)

    use_filter: bool = True
    use_manager: bool = True
    use_reminders: bool = True
    spot_check_every: int = Field(default=5, ge=0)  # 0 = never spot-check signed memories
    reminder_every: int = Field(default=4, ge=1)  # steps between scheduled reminders
    audit_on_flag: bool = True  # a flag reminds the Memory Manager to re-audit that family
    manager_miss_rate: float = Field(default=0.0, ge=0.0, le=1.0)  # simulated reviewer mistakes
    seed: int = 0


CONDITIONS: dict[str, RunConfig] = {
    "none": RunConfig(use_filter=False, use_manager=False, use_reminders=False),
    "filter_only": RunConfig(use_manager=False),
    "manager_only": RunConfig(use_filter=False),
    "full": RunConfig(),
}
