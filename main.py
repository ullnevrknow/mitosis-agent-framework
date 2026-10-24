"""Run a Mitosis demo scenario and print the full decision trace.

    python main.py                      # vendor scenario, offline scripted model
    python main.py --scenario rogue     # agent drifts until apoptosis
    python main.py --model anthropic    # live Claude model (needs credentials)
"""

from __future__ import annotations

import argparse
import json

from mitosis import Budgets, TraceEvent, build_system, default_registry
from mitosis.executor import build_model
from mitosis.scenarios import SCENARIOS
from mitosis.schemas import AgentResult


def print_event(e: TraceEvent) -> None:
    indent = "  " * e.agent_id.count(".")
    line = f"{indent}[{e.agent_id:<7}] step {e.step:>2}  {e.src.value:>9} -> {e.dst.value:<9}"
    if e.verdict is not None and not e.verdict.allowed:
        line += f"  {e.verdict.decision.value} drift={e.verdict.drift_score:.2f}"
    if e.detail:
        line += f"  | {e.detail}"
    print(line.rstrip())


def print_result(result: AgentResult, depth: int = 0) -> None:
    pad = "  " * depth
    text = result.answer if result.status == "done" else f"terminated: {result.reason}"
    print(f"{pad}{result.agent_id} [{result.status}] {text}")
    for child in result.children:
        print_result(child, depth + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mitosis-inspired multi-agent demo")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="vendor")
    parser.add_argument("--model", choices=["scripted", "anthropic"], default="scripted")
    parser.add_argument("--goal", help="override the scenario goal (most useful with --model anthropic)")
    parser.add_argument("--max-steps", type=int, default=Budgets().max_steps)
    parser.add_argument("--persist", metavar="PATH", help="write the memory snapshot + event log to a JSON file")
    parser.add_argument("--show-memory", action="store_true", help="print every agent namespace at the end")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    registry = default_registry(scenario.notes)
    model = build_model(args.model, registry, scenario.scripts)
    orchestrator = build_system(
        model,
        registry,
        budgets=Budgets(max_steps=args.max_steps),
        persist_path=args.persist,
        on_event=print_event,
    )

    print(f"Scenario: {scenario.name} ({args.model} model)")
    print(f"  {scenario.description}\n")
    result = orchestrator.run(args.goal or scenario.goal, scenario.tools)

    print("\nResult")
    print_result(result)

    rejected = [e for e in orchestrator.trace if e.verdict is not None and not e.verdict.allowed]
    print(f"\nInterceptor stopped {len(rejected)} action(s): " + ", ".join(
        f"{e.agent_id}:{'+'.join(e.verdict.rule_ids)}" for e in rejected if e.verdict
    ))

    if args.show_memory:
        print("\nMemory")
        print(json.dumps(orchestrator.memory.snapshot()["namespaces"], indent=2, default=str))
    if args.persist:
        print(f"\nMemory snapshot written to {orchestrator.memory.save()}")


if __name__ == "__main__":
    main()
