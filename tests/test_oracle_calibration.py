import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.experiments.oracle_calibration import (  # noqa: E402
    HORIZONS,
    analyze,
    block_wild_signs,
    calibrate_matrix,
    horizon_block_states,
    null_action_values,
    paired_block_ratio_interval,
    two_way_decompose,
)
from src.deeprl_liquidity_provision_uniswapv3.experiments.validate_oracle_calibration import (  # noqa: E402
    audit,
)


def test_two_way_decomposition_has_zero_interaction_margins():
    q = np.arange(32, dtype=float).reshape(8, 4)
    grand, state, action, residual = two_way_decompose(q)
    rebuilt = grand + state[:, None] + action[None, :] + residual
    assert rebuilt == pytest.approx(q)
    assert residual.mean(axis=0) == pytest.approx(np.zeros(4), abs=1e-12)
    assert residual.mean(axis=1) == pytest.approx(np.zeros(8), abs=1e-12)


def test_null_draws_preserve_fitted_state_and_action_means():
    rng = np.random.default_rng(5)
    q = rng.normal(size=(12, 4)) + np.array([0.0, 2.0, -1.0, 4.0])
    draws = null_action_values(q, samples=25, block=3, rng=np.random.default_rng(9))
    assert draws.mean(axis=1) == pytest.approx(
        np.broadcast_to(q.mean(axis=0), (25, 4)), abs=1e-12
    )
    assert draws.mean(axis=2) == pytest.approx(
        np.broadcast_to(q.mean(axis=1), (25, 12)), abs=1e-12
    )


def test_additive_matrix_has_no_calibrated_excess():
    state = np.linspace(-3.0, 3.0, 10)
    action = np.array([0.0, 1.0, 4.0, 2.0])
    q = state[:, None] + action[None, :]
    got = calibrate_matrix(q, fixed_action=1, samples=100, block=4, seed=7)
    assert got["observed_headroom"] == pytest.approx(3.0)
    assert got["null_headroom"]["mean"] == pytest.approx(3.0)
    assert got["headroom_excess"] == pytest.approx(0.0)
    assert got["observed_disagreement"] == pytest.approx(1.0)
    assert got["null_disagreement"]["mean"] == pytest.approx(1.0)
    assert got["disagreement_excess"] == pytest.approx(0.0)
    assert got["observed_selection_premium"] == pytest.approx(0.0)
    assert got["null_selection_premium"]["mean"] == pytest.approx(0.0)
    assert got["selection_premium_excess"] == pytest.approx(0.0)


def test_symmetric_four_winner_matrix_has_75pct_disagreement_without_excess():
    base = np.array([3.0, 2.0, 1.0, 0.0])
    q = np.stack([np.roll(base, shift) for shift in range(4)])
    got = calibrate_matrix(q, fixed_action=0, samples=100, block=4, seed=17)
    assert got["observed_disagreement"] == pytest.approx(0.75)
    assert got["disagreement_excess"] == pytest.approx(0.0)
    assert got["selection_premium_excess"] == pytest.approx(0.0)


def test_selection_premium_is_invariant_to_intact_vector_permutation():
    q = np.random.default_rng(21).normal(size=(19, 4))
    permutation = np.random.default_rng(22).permutation(len(q))
    original = calibrate_matrix(q, fixed_action=0, samples=10, block=2, seed=23)
    permuted = calibrate_matrix(
        q[permutation], fixed_action=0, samples=10, block=2, seed=23
    )
    assert permuted["observed_selection_premium"] == pytest.approx(
        original["observed_selection_premium"]
    )


def test_paired_block_ratio_interval_preserves_exact_ratio():
    denominator = np.arange(1.0, 25.0)
    numerator = 0.5 * denominator
    lo, hi = paired_block_ratio_interval(
        numerator, denominator, np.random.default_rng(24), 100, 4
    )
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_horizon_block_spans_one_reward_horizon():
    hours = np.array([0, 24, 48, 72, 96])
    assert horizon_block_states(hours, 168) == 7
    assert horizon_block_states(hours, 24) == 1


def test_block_wild_signs_are_seeded_and_binary():
    first = block_wild_signs(17, 10, 4, np.random.default_rng(11))
    second = block_wild_signs(17, 10, 4, np.random.default_rng(11))
    assert np.array_equal(first, second)
    assert set(np.unique(first)) <= {-1.0, 1.0}


def test_oracle_calibration_slurm_wrapper_has_no_site_partition():
    body = (
        Path(__file__).resolve().parents[1]
        / "scripts/slurm/oracle_calibration_array.sh"
    ).read_text()
    assert "#SBATCH --partition=" not in body


def test_oracle_calibration_audit_rejects_incomplete_collection(tmp_path):
    result = audit(tmp_path)
    assert result["status"] == "INVALID"
    assert "0/144 files" in result["problems"][0]


def test_complete_aggregate_reports_all_descriptive_excesses(monkeypatch, tmp_path):
    rows = {}
    for pool in (
        "usdc_weth_005", "usdc_weth_030", "wbtc_weth_005",
        "wbtc_weth_030", "weth_usdt_005", "weth_usdt_030",
    ):
        for step in range(24):
            diagnostics = {}
            for horizon in HORIZONS:
                diagnostics[str(horizon)] = {
                    "observed_headroom": 3.0,
                    "observed_disagreement": 0.75,
                    "observed_selection_premium": 2.0,
                    "null_headroom": {"mean": 2.0},
                    "null_disagreement": {"mean": 0.5},
                    "null_selection_premium": {"mean": 1.0},
                    "headroom_excess": 1.0,
                    "disagreement_excess": 0.25,
                    "selection_premium_excess": 1.0,
                    "null_state_block": 4,
                }
            rows[(pool, step)] = {"fixed_action": 0, "diagnostics": diagnostics}
    monkeypatch.setattr(
        "src.deeprl_liquidity_provision_uniswapv3.experiments.oracle_calibration._load_results",
        lambda _: (rows, []),
    )
    report = tmp_path / "aggregate"
    text = analyze(tmp_path, report, samples=100)
    assert text.startswith("COMPLETE - FINAL COLLECTION")
    assert "excess=+1.00" in text
    assert (report / "headline.csv").is_file()
    headline = (report / "headline.csv").read_text()
    assert "null_explained_fraction_168h_ci_low" in headline
    assert "selection_premium_excess_168h_p_raw" in headline
