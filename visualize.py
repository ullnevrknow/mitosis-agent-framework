"""Build the visual demo: an interactive, self-contained replay of each scenario.

    python visualize.py                 # writes docs/index.html
    python visualize.py --open          # ...and opens it in your browser
    python visualize.py --model anthropic --scenario vendor --out live.html
"""

from __future__ import annotations

import argparse
import webbrowser

from mitosis.scenarios import SCENARIOS
from mitosis.viz import record_run, write_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a step-by-step HTML replay of Mitosis runs")
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS), help="repeatable; default: all")
    parser.add_argument("--model", choices=["scripted", "anthropic"], default="scripted")
    parser.add_argument("--out", default="docs/index.html")
    parser.add_argument("--open", action="store_true", help="open the page in a browser when done")
    args = parser.parse_args()

    names = args.scenario or list(SCENARIOS)
    runs = [record_run(SCENARIOS[name], args.model) for name in names]
    path = write_html(args.out, runs)

    for run in runs:
        print(f"{run['name']:<8} {len(run['frames']):>3} frames  -> {run['result']['status']}")
    print(f"Wrote {path.resolve()}")
    if args.open:
        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
