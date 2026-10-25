"""Run one scenario as a series of episodes and print what each part did.

    python main.py                                         # data_gaps, full system
    python main.py --scenario tests_deadline --condition none
    python main.py --scenario hotfix_deploy --condition full --trace
"""

from __future__ import annotations

import argparse

from mitosis import CONDITIONS, SCENARIOS, run_series
from mitosis.experiment import score


def main() -> None:
    parser = argparse.ArgumentParser(description="Mitosis proof model: one scenario, one condition")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="data_gaps")
    parser.add_argument("--condition", choices=list(CONDITIONS), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", action="store_true", help="print every state-machine transition")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    config = CONDITIONS[args.condition].model_copy(update={"seed": args.seed})
    result = run_series(scenario, config)

    print(f"{scenario.title} ({scenario.name}) | condition: {args.condition} | seed {args.seed}")
    print(f"  {scenario.description}\n")
    for e in result.episodes:
        outcome = "STOPPED" if e.terminated else ("done" if e.success else "FAILED")
        honesty = " - unethical" if e.violation else ""
        print(f"Episode {e.episode}: {e.task}")
        print(f"  plan    {e.plan}")
        print(f"  why     {e.reason}")
        print(f"  result  {outcome}{honesty}")
        for c in e.curation:
            route = "re-audit" if c["audit"] else (c["tier"] or "unfiltered")
            if c["signed"]:
                route += " + signed"
            if c["spot_check"]:
                route += " + spot-check"
            review = f"{c['depth']} review -> {c['decision']}" if c["decision"] else "no review"
            print(f"  memory  {c['outcome']:<16} {route:<26} {review}: {c['text']}")
        if e.reminders:
            print(f"  clock   {e.reminders} reminder(s)")
        print()

    print("Memory bank at the end:")
    for m in result.bank:
        flag = "  <- unethical" if m["truth"]["misaligned"] else ""
        print(f"  [{m['origin']}] {m['text']}{flag}")

    s = score(result)
    print(
        f"\nUnethical actions {s['violations']} | shortcut reuses {s['misaligned_reuses']} | "
        f"tasks done honestly {s['honest_done']}/{s['episodes']} | episodes stopped {s['terminations']} | reminders {s['reminders']}"
    )

    if args.trace:
        print("\nTrace:")
        for t in result.trace:
            print(f"  e{t.episode} s{t.step:<3} {t.src.value:>9} -> {t.dst.value:<9} {t.detail}")


if __name__ == "__main__":
    main()
