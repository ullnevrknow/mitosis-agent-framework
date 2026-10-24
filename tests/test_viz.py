from __future__ import annotations

import json
import re

from mitosis.orchestrator import TRANSITIONS, Phase
from mitosis.scenarios import ROGUE, VENDOR
from mitosis.viz import record_run, render_html


def embedded_data(html: str) -> dict:
    match = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', html, re.S)
    assert match, "replay data block missing"
    return json.loads(match.group(1))


def test_record_run_captures_every_transition():
    run = record_run(VENDOR)
    frames = run["frames"]
    assert frames[0]["src"] == "PROPOSE" and frames[-1]["dst"] == "DONE"
    assert {f["agent"] for f in frames} == {"root", "root.1", "root.2"}
    for f in frames:
        assert Phase(f["dst"]) in TRANSITIONS[Phase(f["src"])]
        assert "goal" in f["memory"]
    assert all(f["proposal"] for f in frames if f["src"] == "PROPOSE")


def test_rendered_page_is_self_contained_and_round_trips():
    html = render_html([record_run(VENDOR), record_run(ROGUE)])
    assert html.count("__REPLAY" + "_DATA__") == 0
    assert not re.search(r'<(script|link)[^>]+(src|href)="https?://', html)  # no external assets
    data = embedded_data(html)
    assert [r["name"] for r in data["runs"]] == ["vendor", "rogue"]
    assert data["runs"][1]["result"]["status"] == "apoptosis"


def test_script_tags_in_data_cannot_break_out():
    run = record_run(VENDOR)
    run["description"] = "</script><script>alert(1)</script>"
    html = render_html([run])
    assert embedded_data(html)["runs"][0]["description"] == run["description"]
