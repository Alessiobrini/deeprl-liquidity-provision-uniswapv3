"""Recompute retail-frontier headline counts from aggregate frontier CSV files."""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from ..data.pools import POOLS

REQUIRED = {
    "pool", "regime", "capital", "gas", "difference", "p_holm",
}
SCOPE_ORDER = {"pool": 0, "fee_tier": 1, "all": 2}


def _block_length(path: Path) -> int:
    match = re.fullmatch(r"aggregate_block(\d+)", path.parent.name)
    if match is None:
        raise ValueError(
            f"{path}: parent directory must be named aggregate_block<N>"
        )
    return int(match.group(1))


def _counts(rows: list[dict], alpha: float) -> dict[str, int]:
    differences = [float(row["difference"]) for row in rows]
    adjusted = [float(row["p_holm"]) for row in rows]
    return {
        "cells": len(rows),
        "positive_cells": sum(value > 0 for value in differences),
        "negative_cells": sum(value < 0 for value in differences),
        "holm_favorable": sum(
            value > 0 and p_value < alpha
            for value, p_value in zip(differences, adjusted)
        ),
        "holm_adverse": sum(
            value < 0 and p_value < alpha
            for value, p_value in zip(differences, adjusted)
        ),
    }


def summarize_frontier_csv(path: Path, alpha: float = 0.05) -> list[dict]:
    """Return pool, fee-tier, and overall counts for one bootstrap block length."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)

    keys = [
        (row["pool"], row["regime"], row["capital"], row["gas"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate pool/regime/capital/gas cells")

    by_pool_regime: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        pool = row["pool"]
        if pool not in POOLS:
            raise ValueError(f"{path}: unknown pool {pool!r}")
        by_pool_regime[(pool, row["regime"])].append(row)
    incomplete = {
        key: len(group)
        for key, group in by_pool_regime.items()
        if len(group) != 25
    }
    if incomplete:
        raise ValueError(f"{path}: expected 25 cells per pool/regime; got {incomplete}")

    block = _block_length(path)
    output: list[dict] = []
    for regime in sorted({row["regime"] for row in rows}):
        regime_rows = [row for row in rows if row["regime"] == regime]
        for pool in sorted({row["pool"] for row in regime_rows}):
            group = [row for row in regime_rows if row["pool"] == pool]
            output.append({
                "block": block, "scope": "pool", "label": pool,
                "regime": regime, **_counts(group, alpha),
            })
        for tier in sorted({POOLS[row["pool"]].fee_tier_pct for row in regime_rows}):
            group = [
                row for row in regime_rows
                if POOLS[row["pool"]].fee_tier_pct == tier
            ]
            output.append({
                "block": block, "scope": "fee_tier", "label": f"{tier:g}%",
                "regime": regime, **_counts(group, alpha),
            })
        output.append({
            "block": block, "scope": "all", "label": "all",
            "regime": regime, **_counts(regime_rows, alpha),
        })
    return output


def summarize_root(root: Path, alpha: float = 0.05) -> list[dict]:
    paths = sorted(
        root.glob("aggregate_block*/frontier.csv"),
        key=_block_length,
    )
    if not paths:
        raise ValueError(f"{root}: no aggregate_block*/frontier.csv files")
    rows = [
        row
        for path in paths
        for row in summarize_frontier_csv(path, alpha)
    ]
    return sorted(
        rows,
        key=lambda row: (
            row["block"], row["regime"], SCOPE_ORDER[row["scope"]], row["label"],
        ),
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict], alpha: float) -> str:
    lines = [
        "Retail-frontier counts recomputed from frontier.csv",
        "Positive/negative cells use the sign of paired mean differences; "
        f"Holm counts additionally require p_holm < {alpha:g}.",
        "",
        f"{'block':>5} {'regime':<12} {'scope':<9} {'label':<18} "
        f"{'positive':>9} {'Holm +':>7} {'Holm -':>7}",
        "-" * 76,
    ]
    for row in rows:
        lines.append(
            f"{row['block']:>5} {row['regime']:<12} {row['scope']:<9} "
            f"{row['label']:<18} "
            f"{row['positive_cells']:>3}/{row['cells']:<5} "
            f"{row['holm_favorable']:>7} {row['holm_adverse']:>7}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between zero and one")
    rows = summarize_root(args.root, args.alpha)
    if args.output is not None:
        _write_csv(args.output, rows)
    print(_render(rows, args.alpha), end="")


if __name__ == "__main__":
    main()
