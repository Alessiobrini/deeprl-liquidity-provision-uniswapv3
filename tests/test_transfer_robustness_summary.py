import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.experiments.transfer_robustness_summary import (  # noqa: E402
    summarize_algorithm,
)


def _write_block(root: Path, algorithm: str, block: int, rows: list[dict]) -> None:
    path = root / f"aggregate_{algorithm}_block{block}" / "comparisons.csv"
    path.parent.mkdir()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_transfer_summary_counts_only_directions_surviving_every_block(tmp_path):
    for block in (2, 4, 6, 8):
        _write_block(tmp_path, "ppo", block, [
            {"target": "a", "source": "b", "difference": 2, "p_holm": 0.01},
            {"target": "a", "source": "c", "difference": -3, "p_holm": 0.02},
            {"target": "b", "source": "a", "difference": 4,
             "p_holm": 0.01 if block != 6 else 0.20},
        ])

    assert summarize_algorithm(tmp_path, "ppo") == {
        "algorithm": "ppo", "tested_directions": 3,
        "robust_favorable": 1, "robust_adverse": 1,
        "blocks": "2 4 6 8", "alpha": 0.05,
    }


def test_transfer_summary_rejects_misaligned_direction_sets(tmp_path):
    for block in (2, 4, 6, 8):
        rows = [
            {"target": "a", "source": "b", "difference": 2, "p_holm": 0.01}
        ]
        if block == 8:
            rows.append(
                {"target": "a", "source": "c", "difference": 1, "p_holm": 0.01}
            )
        _write_block(tmp_path, "a2c", block, rows)

    with pytest.raises(ValueError, match="directions differ"):
        summarize_algorithm(tmp_path, "a2c")
