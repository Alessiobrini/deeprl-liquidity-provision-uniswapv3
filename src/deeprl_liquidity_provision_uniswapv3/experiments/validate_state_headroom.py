"""Independent structural audit for the state-contingent headroom collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ..data.pools import CORE
from .state_headroom import (
    EXECUTION_WIDTHS,
    HORIZONS,
    MIN_LEAVES,
    TRAIN_STATE_SPACING,
    WIDTHS,
)

EXPECTED = {(pool, step) for pool in CORE for step in range(24)}


def collection_hash(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def audit(root: Path, ignored: set[Path] | None = None) -> dict:
    ignored = {path.resolve() for path in (ignored or set())}
    files = sorted(
        path for path in root.glob("*.json")
        if path.resolve() not in ignored
    )
    problems, rows, seen = [], [], set()
    for path in files:
        try:
            row = json.loads(path.read_text())
        except Exception as exc:
            problems.append(f"{path.name}: invalid JSON: {exc}")
            continue
        rows.append((path, row))
        key = (row.get("unit", {}).get("pool"), row.get("unit", {}).get("step"))
        if key in seen:
            problems.append(f"{path.name}: duplicate {key}")
        seen.add(key)
    if len(files) != 144 or seen != EXPECTED:
        problems.append(
            f"{len(files)}/144 files; missing={len(EXPECTED-seen)} "
            f"extra={len(seen-EXPECTED)}"
        )

    for path, row in rows:
        pool = row.get("unit", {}).get("pool")
        step = row.get("unit", {}).get("step")
        expected_split = {
            "train": list(range(step, step + 4)),
            "val": [step + 4],
            "test": [step + 5],
        } if isinstance(step, int) else None
        if row.get("split") != expected_split:
            problems.append(f"{path.name}: split mismatch")
        provenance = row.get("provenance", {})
        expected_provenance = {
            "widths": list(WIDTHS),
            "execution_widths": list(EXECUTION_WIDTHS),
            "horizons": list(HORIZONS),
            "min_leaves": list(MIN_LEAVES),
            "train_state_spacing": TRAIN_STATE_SPACING,
        }
        for key, expected in expected_provenance.items():
            if provenance.get(key) != expected:
                problems.append(
                    f"{path.name}: provenance {key}={provenance.get(key)!r}, "
                    f"expected {expected!r}"
                )

        fixed, contextual = row.get("fixed", {}), row.get("contextual", {})
        if fixed.get("action") not in range(4):
            problems.append(f"{path.name}: invalid fixed action")
        if len(fixed.get("validation_rewards", [])) != 4:
            problems.append(f"{path.name}: fixed validation score count")
        if contextual.get("selected_min_leaf") not in MIN_LEAVES:
            problems.append(f"{path.name}: invalid selected min leaf")
        for label, value in (
            ("fixed reward", fixed.get("reward")),
            ("contextual reward", contextual.get("reward")),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"{path.name}: non-finite {label}")

        diagnostics = row.get("diagnostics", {})
        if set(diagnostics) != {str(horizon) for horizon in HORIZONS}:
            problems.append(f"{path.name}: diagnostic horizons mismatch")
            continue
        for horizon in HORIZONS:
            result = diagnostics[str(horizon)]
            n = result.get("n_states", 0)
            disagreements = result.get("n_disagreements", -1)
            fraction = result.get("disagreement_fraction", float("nan"))
            if not isinstance(n, int) or n <= 0:
                problems.append(f"{path.name}: invalid {horizon}h state count")
            if not isinstance(disagreements, int) or not 0 <= disagreements <= n:
                problems.append(f"{path.name}: invalid {horizon}h disagreements")
            if not math.isclose(fraction, disagreements / n, abs_tol=1e-12):
                problems.append(f"{path.name}: inconsistent {horizon}h fraction")
            for key in ("mean_headroom", "mean_contextual_regret"):
                value = result.get(key, float("nan"))
                if not math.isfinite(value) or value < -1e-8:
                    problems.append(f"{path.name}: invalid {horizon}h {key}")
        if pool not in CORE:
            problems.append(f"{path.name}: unknown pool")

    return {
        "status": "COMPLETE" if not problems else "INVALID",
        "problems": problems,
        "count": len(files),
        "sha256": collection_hash(files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, {args.output})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
