"""Build the demo page: replays of every scenario and setup, the experiment charts, and the Lab.

    python visualize.py                 # writes docs/index.html
    python visualize.py --open          # ...and opens it in your browser
"""

from __future__ import annotations

import argparse
import webbrowser

from mitosis.viz import page_data, write_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Mitosis demo page")
    parser.add_argument("--seeds", type=int, default=20, help="seeds for the experiment charts")
    parser.add_argument("--out", default="docs/index.html")
    parser.add_argument("--open", action="store_true", help="open the page in a browser when done")
    args = parser.parse_args()

    data = page_data(seeds=args.seeds)
    path = write_html(args.out, data)
    runs = sum(len(v) for v in data["runs"].values())
    print(f"{runs} replays, experiment over {data['experiment']['seeds']} seeds")
    print(f"Wrote {path.resolve()}")
    if args.open:
        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
