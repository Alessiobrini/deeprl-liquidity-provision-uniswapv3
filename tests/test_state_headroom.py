import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.experiments.state_headroom import (  # noqa: E402
    action_disagrees,
    action_returns,
    fit_reward_stump,
)
from src.deeprl_liquidity_provision_uniswapv3.experiments.validate_state_headroom import (  # noqa: E402
    audit,
)
from tests.test_uniswap_env import make_panel, paper_env  # noqa: E402


def test_action_returns_uses_full_fixed_horizon_without_mutating_source():
    env = paper_env(panel=make_panel(n=220, drift=0.001), gas_usd=5.0)
    env.reset()
    before = env.i
    got = action_returns(env, (3, 7))
    assert env.i == before
    assert set(got) == {3, 7}
    assert all(values.shape == (4,) for values in got.values())

    branch = copy.copy(env)
    expected = 0.0
    for elapsed in range(1, 8):
        _, _, _, _, info = branch.step(2 if elapsed == 1 else 0)
        expected += info["reward_true"]
    assert got[7][2] == pytest.approx(expected)


def test_reward_stump_directly_maximizes_realized_action_rewards():
    x = np.column_stack([np.arange(8, dtype=float), np.ones(8)])
    rewards = np.zeros((8, 4))
    rewards[:4, 1] = 10.0
    rewards[4:, 2] = 12.0
    weights = np.ones(8)
    stump = fit_reward_stump(x, rewards, weights, min_leaf=2)
    assert stump.feature == 0
    assert stump.threshold == pytest.approx(3.5)
    assert stump.left_action == 1
    assert stump.right_action == 2
    assert stump(np.array([1.0, 1.0]), None) == 1
    assert stump(np.array([6.0, 1.0]), None) == 2


def test_reward_stump_stays_constant_when_conditioning_adds_no_reward():
    x = np.arange(18, dtype=float).reshape(6, 3)
    rewards = np.zeros((6, 4))
    rewards[:, 3] = 4.0
    stump = fit_reward_stump(x, rewards, np.ones(6), min_leaf=2)
    assert stump.feature is None
    assert stump.left_action == stump.right_action == 3


def test_action_disagreement_counts_tied_fixed_action_as_agreement():
    q = np.array([1.0, 3.0, 3.0, 2.0])
    assert not action_disagrees(q, 1)
    assert not action_disagrees(q, 2)
    assert action_disagrees(q, 0)


def test_state_headroom_slurm_wrapper_has_no_site_specific_partition():
    body = (
        Path(__file__).resolve().parents[1]
        / "scripts/slurm/state_headroom_array.sh"
    ).read_text()
    assert "#SBATCH --partition=" not in body


def test_state_headroom_audit_rejects_incomplete_collection(tmp_path):
    result = audit(tmp_path)
    assert result["status"] == "INVALID"
    assert "0/144 files" in result["problems"][0]
