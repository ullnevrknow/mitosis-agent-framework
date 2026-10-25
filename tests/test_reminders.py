from __future__ import annotations

from mitosis.reminders import ReminderClock
from mitosis.schemas import Part, Trigger


def test_scheduled_reminders_every_n_steps():
    clock = ReminderClock(every_n_steps=3)
    fired = [step for step in range(1, 10) if clock.tick(step, episode=1)]
    assert fired == [3, 6, 9]
    assert {r.part for r in clock.log} == {Part.EXECUTOR, Part.MEMORY_MANAGER}


def test_flags_remind_both_parts_and_ask_for_an_audit():
    clock = ReminderClock()
    out = clock.event(Trigger.FLAG, step=2, episode=1, about="tests")
    assert [r.part for r in out] == [Part.EXECUTOR, Part.MEMORY_MANAGER]
    assert "re-audit" in out[1].text and out[1].about == "tests"


def test_other_events_remind_the_executor():
    clock = ReminderClock()
    for trigger in (Trigger.DELETE, Trigger.REWRITE, Trigger.TERMINATE):
        out = clock.event(trigger, step=1, episode=1)
        assert [r.part for r in out] == [Part.EXECUTOR]


def test_reminders_never_reveal_keywords():
    from mitosis.filter import _LEXICON

    clock = ReminderClock(every_n_steps=1)
    clock.tick(1, 1)
    for t in Trigger:
        if t is not Trigger.SCHEDULE:
            clock.event(t, 1, 1)
    text = " ".join(r.text.lower() for r in clock.log)
    # Reminders state principles ("bypassing safeguards"), never the exact trigger phrases.
    for word in ("fake", "quietly", "skip the", "password", "kill", "pre-approved", "mark"):
        assert word not in text
    assert _LEXICON  # the lexicon exists, it just isn't in any reminder
