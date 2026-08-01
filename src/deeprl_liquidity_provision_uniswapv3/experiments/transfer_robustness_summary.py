"""Summarize transfer directions significant at every declared block length."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


REQUIRED = {"target", "source", "difference", "p_holm"}


def _read(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    keyed = {
        (row["target"], row["source"]):
            (float(row["difference"]), float(row["p_holm"]))
        for row in rows
    }
    if len(keyed) != len(rows):
        raise ValueError(f"{path}: duplicate target/source directions")
    return keyed


def summarize_algorithm(
    root: Path,
    algorithm: str,
    blocks: tuple[int, ...] = (2, 4, 6, 8),
    alpha: float = 0.05,
) -> dict:
    """Count directions whose Holm result survives every block specification."""
    by_block = {
        block: _read(
            root / f"aggregate_{algorithm}_block{block}" / "comparisons.csv"
        )
        for block in blocks
    }
    direction_sets = {block: set(rows) for block, rows in by_block.items()}
    reference = direction_sets[blocks[0]]
    if any(directions != reference for directions in direction_sets.values()):
        counts = {block: len(directions) for block, directions in direction_sets.items()}
        raise ValueError(
            f"{algorithm}: transfer directions differ across blocks: {counts}"
        )

    favorable = adverse = 0
    for direction in reference:
        estimates = [by_block[block][direction][0] for block in blocks]
        adjusted = [by_block[block][direction][1] for block in blocks]
        if not all(p_value < alpha for p_value in adjusted):
            continue
        if all(estimate > 0 for estimate in estimates):
            favorable += 1
        elif all(estimate < 0 for estimate in estimates):
            adverse += 1
        else:
            raise ValueError(
                f"{algorithm} {direction}: significant estimate changes sign across blocks"
            )
    return {
        "algorithm": algorithm,
        "tested_directions": len(reference),
        "robust_favorable": favorable,
        "robust_adverse": adverse,
        "blocks": " ".join(str(block) for block in blocks),
        "alpha": alpha,
    }


def summarize_root(
    root: Path,
    algorithms: tuple[str, ...] | None = None,
    blocks: tuple[int, ...] = (2, 4, 6, 8),
    alpha: float = 0.05,
) -> list[dict]:
    if algorithms is None:
        pattern = re.compile(r"aggregate_(.+)_block\d+")
        algorithms = tuple(sorted({
            match.group(1)
            for path in root.glob("aggregate_*_block*")
            if (match := pattern.fullmatch(path.name)) is not None
        }))
    if not algorithms:
        raise ValueError(f"{root}: no transfer aggregate directories")
    return [
        summarize_algorithm(root, algorithm, blocks, alpha)
        for algorithm in algorithms
    ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(rows: list[dict]) -> str:
    lines = [
        "Transfer directions surviving Holm correction at every declared block length",
        "",
        f"{'algorithm':<15} {'tested':>6} {'favorable':>10} {'adverse':>8}",
        "-" * 43,
    ]
    for row in rows:
        lines.append(
            f"{row['algorithm']:<15} {row['tested_directions']:>6} "
            f"{row['robust_favorable']:>10} {row['robust_adverse']:>8}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--algorithms", nargs="*")
    parser.add_argument("--blocks", nargs="*", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between zero and one")
    rows = summarize_root(
        args.root,
        tuple(args.algorithms) if args.algorithms else None,
        tuple(args.blocks),
        args.alpha,
    )
    if args.output is not None:
        _write_csv(args.output, rows)
    print(_render(rows), end="")


if __name__ == "__main__":
    main()
