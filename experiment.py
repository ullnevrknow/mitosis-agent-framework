"""Run the four-condition experiment and write the results.

    python experiment.py                     # 20 seeds, writes results/results.json and results/results.md
    python experiment.py --seeds 50
    python experiment.py --miss-rate 0.2     # a Memory Manager that misses 20% of problems
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mitosis.experiment import run_experiment, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare no protection, filter only, Memory Manager only and the full system")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--spot-check-every", type=int, default=5, help="1 in N signed memories gets a full review (0 = never)")
    parser.add_argument("--miss-rate", type=float, default=0.0, help="share of problems the simulated Memory Manager misses")
    parser.add_argument("--no-audit", action="store_true", help="don't re-audit a strategy family after a flag")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    overrides = {"spot_check_every": args.spot_check_every, "manager_miss_rate": args.miss_rate, "audit_on_flag": not args.no_audit}
    results = run_experiment(seeds=args.seeds, overrides=overrides)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = to_markdown(results)
    (out / "results.md").write_text(
        f"# Experiment results\n\n{len(results['scenarios'])} scenarios x {results['seeds']} seeds, settings {overrides}.\n"
        "Offline proof model: scripted Executor and Memory Manager.\n\n" + table + "\n",
        encoding="utf-8",
    )
    print(table)
    print(f"\nWrote {out / 'results.json'} and {out / 'results.md'}")


if __name__ == "__main__":
    main()
