import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.experiments.retail_frontier import (  # noqa: E402
    adjusted, config_tag,
)
from src.deeprl_liquidity_provision_uniswapv3.experiments.retail_frontier_summary import (  # noqa: E402
    summarize_frontier_csv,
)


def test_cost_counterfactual_is_exact_linear_accounting():
    result = {"reward": 100.0, "n_rebalances": 3,
              "rebalance_notional": 1_000.0}
    assert adjusted(result, gas=5, rate=0.001) == 84.0


def test_frontier_tag_changes_with_design():
    a = config_tag(["p"], [1_500], [45, 50, 55])
    assert a != config_tag(["p"], [5_000], [45, 50, 55])
    assert a != config_tag(["q"], [1_500], [45, 50, 55])


def test_frontier_summary_exercises_sign_and_holm_counts(tmp_path):
    path = tmp_path / "aggregate_block4" / "frontier.csv"
    path.parent.mkdir()
    rows = []
    designs = {
        "usdc_weth_005": (10, 5, 7),
        "usdc_weth_030": (20, 2, 5),
    }
    for pool, (positive, significant_positive, significant_negative) in designs.items():
        for cell in range(25):
            is_positive = cell < positive
            significant = (
                cell < significant_positive if is_positive
                else cell - positive < significant_negative
            )
            rows.append({
                "pool": pool, "regime": "conversion",
                "capital": 1_500 * (cell // 5 + 1), "gas": cell % 5,
                "difference": 1.0 if is_positive else -1.0,
                "p_holm": 0.01 if significant else 0.20,
            })
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_frontier_csv(path)
    keyed = {(row["scope"], row["label"]): row for row in summary}
    assert keyed[("pool", "usdc_weth_005")] == {
        "block": 4, "scope": "pool", "label": "usdc_weth_005",
        "regime": "conversion", "cells": 25, "positive_cells": 10,
        "negative_cells": 15, "holm_favorable": 5, "holm_adverse": 7,
    }
    assert keyed[("all", "all")]["positive_cells"] == 30
    assert keyed[("all", "all")]["holm_favorable"] == 7
    assert keyed[("all", "all")]["holm_adverse"] == 12


def test_retail_slurm_wrapper_has_no_site_specific_partition():
    body = (
        Path(__file__).resolve().parents[1]
        / "scripts/slurm/retail_frontier_array.sh"
    ).read_text()
    assert "#SBATCH --partition=" not in body
