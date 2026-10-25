from __future__ import annotations

from mitosis.schemas import CONDITIONS, FilterResult, Tier


def test_conditions_switch_parts_off():
    assert not CONDITIONS["none"].use_filter and not CONDITIONS["none"].use_manager
    assert CONDITIONS["filter_only"].use_filter and not CONDITIONS["filter_only"].use_manager
    assert not CONDITIONS["manager_only"].use_filter and CONDITIONS["manager_only"].use_manager
    assert CONDITIONS["full"].use_filter and CONDITIONS["full"].use_manager and CONDITIONS["full"].use_reminders


def test_review_depth():
    assert FilterResult(memory_id="m", tier=Tier.CLEAN, signature="s").review_depth == "light"
    assert FilterResult(memory_id="m", tier=Tier.CLEAN, signature="s", spot_check=True).review_depth == "full"
    assert FilterResult(memory_id="m", tier=Tier.SUSPICIOUS).review_depth == "full"
    assert FilterResult(memory_id="m", tier=Tier.CRITICAL).review_depth == "none"
