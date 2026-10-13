import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.experiments.oracle_pool_null import (  # noqa: E402
    aggregate,
    pool_randomization_summary,
)


def test_pool_randomization_summary_counts_one_sided_exceedances():
    observed = np.array([2.0, 4.0])
    null = np.array([[1.0, 3.0, 5.0], [3.0, 3.0, 5.0]])
    got = pool_randomization_summary(observed, null)
    assert got["observed_selection_premium"] == pytest.approx(3.0)
    assert got["exceedances"] == 2
    assert got["p_exceedance"] == pytest.approx(2 / 3)
    assert got["p_exceedance_plus_one"] == pytest.approx(3 / 4)


def test_pool_randomization_summary_rejects_mismatched_windows():
    with pytest.raises(ValueError, match="expected observed"):
        pool_randomization_summary(np.ones(3), np.ones((2, 10)))


@pytest.mark.parametrize("observed,null", [
    (np.empty(0), np.empty((0, 3))),
    (np.ones(2), np.empty((2, 0))),
    (np.array([np.nan]), np.ones((1, 3))),
    (np.ones(1), np.array([[np.inf]])),
])
def test_pool_randomization_summary_rejects_empty_or_nonfinite_data(observed, null):
    with pytest.raises(ValueError):
        pool_randomization_summary(observed, null)


def test_pool_randomization_summary_includes_ties_and_reports_exact_fraction():
    got = pool_randomization_summary(
        np.array([2.0, 4.0]),
        np.array([[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]]),
    )
    # Pool draws are [1, 3, 5], compared against the window mean of 3.
    assert got["exceedances"] == 2
    assert got["null_explained_fraction"] == pytest.approx(1.0)
    assert got["null_fraction_q025"] == pytest.approx(1.1 / 3)
    assert got["null_fraction_q975"] == pytest.approx(4.9 / 3)


def test_zero_observed_premium_has_no_explained_fraction():
    got = pool_randomization_summary(np.zeros(2), np.zeros((2, 4)))
    assert got["p_exceedance"] == 1.0
    assert got["null_explained_fraction"] is None
    assert got["null_fraction_q025"] is None
    assert got["null_fraction_q975"] is None


def test_no_exceedances_retains_positive_monte_carlo_p_value():
    got = pool_randomization_summary(np.array([3.0]), np.array([[0.0, 1.0, 2.0]]))
    assert got["p_exceedance"] == 0.0
    assert got["p_exceedance_plus_one"] == pytest.approx(0.25)


def test_partial_collection_does_not_write_a_final_report(tmp_path):
    report = tmp_path / "summary.csv"
    with pytest.raises(FileNotFoundError):
        aggregate(tmp_path, report)
    assert not report.exists()
