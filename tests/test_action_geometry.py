"""The matched-width treatment must change units without changing the protocol."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.envs.uniswap_v3 import UniswapV3Env  # noqa: E402


def bare_env(tier, widths, units):
    """Construct only enough state to exercise the single width conversion method."""
    env = object.__new__(UniswapV3Env)
    env.tick_spacing = {0.05: 10, 0.3: 60}[tier]
    env.width_units = units
    env.action_widths = np.asarray(widths, dtype=float)
    env.execution_widths = None
    return env


def test_paper_grid_changes_geometry_with_fee_tier():
    low = bare_env(0.05, [45, 50, 55], "spacing")
    high = bare_env(0.3, [45, 50, 55], "spacing")
    assert [low._width_ticks(w) for w in low.action_widths] == [450, 500, 550]
    assert [high._width_ticks(w) for w in high.action_widths] == [2700, 3000, 3300]


def test_raw_tick_grid_matches_geometry_and_is_mintable_on_both_tiers():
    widths = [480, 540, 600]
    low = bare_env(0.05, widths, "ticks")
    high = bare_env(0.3, widths, "ticks")
    assert [low._width_ticks(w) for w in widths] == widths
    assert [high._width_ticks(w) for w in widths] == widths
    assert all(w % low.tick_spacing == 0 and w % high.tick_spacing == 0 for w in widths)


def test_execution_grid_preserves_action_labels_while_matching_geometry():
    labels = [45, 50, 55]
    low = bare_env(0.05, labels, "spacing")
    high = bare_env(0.3, labels, "spacing")
    low.execution_widths = high.execution_widths = np.asarray([480, 540, 600])
    assert [low._width_ticks(w) for w in labels] == [480, 540, 600]
    assert [high._width_ticks(w) for w in labels] == [480, 540, 600]
