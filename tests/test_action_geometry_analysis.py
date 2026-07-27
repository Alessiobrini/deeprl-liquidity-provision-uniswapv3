import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.experiments import action_geometry as G  # noqa: E402


def test_strategy_set_contains_every_manuscript_and_new_algorithm_arm():
    assert set(G.STRATEGIES) == {
        "Passive", "PassiveWidthSweep", "VolProportionalWidth", "ILMinimizer",
        "ReactiveRecentering", "RecentreWhenOut", "PPO", "A2C", "DQN", "QRDQN",
        "RECURRENTPPO",
    }


def test_tags_distinguish_geometry_and_preserve_primary_tag():
    assert G._tag("ppo", [45, 50, 55], "spacing") == "cb1edae2"
    assert G._tag("ppo", [45, 50, 55], "spacing", [480, 540, 600]) != "cb1edae2"


def test_incomplete_collection_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "CORE", ["pool"])
    tag = G._tag("ppo", [45, 50, 55], "spacing")
    row = {"unit": {"pool": "pool", "step": 0}, "provenance": {"config": {
        "algo": "ppo", "widths": [45, 50, 55]}}, "arms": {}}
    (tmp_path / f"pool__step000__{tag}.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="1/144"):
        G._read_algo(tmp_path, "ppo", [45, 50, 55], "spacing")
