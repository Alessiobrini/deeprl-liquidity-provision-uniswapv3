"""Independent structural audit for the oracle-calibration collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ..data.pools import CORE
from .oracle_calibration import (
    NULL_METHOD,
    NULL_SAMPLES,
    NULL_BLOCK_RULE,
    NULL_SEED,
)
from .state_headroom import EXECUTION_WIDTHS, HORIZONS, WIDTHS

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
        path for path in root.glob("*__step*.json")
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

    expected_provenance = {
        "widths": list(WIDTHS),
        "execution_widths": list(EXECUTION_WIDTHS),
        "horizons": list(HORIZONS),
        "null_method": NULL_METHOD,
        "null_samples": NULL_SAMPLES,
        "null_block_rule": NULL_BLOCK_RULE,
        "null_block_override": 0,
        "null_seed": NULL_SEED,
    }
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
        if row.get("fixed_action") not in range(4):
            problems.append(f"{path.name}: invalid fixed action")
        if len(row.get("fixed_validation_rewards", [])) != 4:
            problems.append(f"{path.name}: fixed validation score count")
        provenance = row.get("provenance", {})
        for key, expected in expected_provenance.items():
            if provenance.get(key) != expected:
                problems.append(
                    f"{path.name}: provenance {key}={provenance.get(key)!r}, "
                    f"expected {expected!r}"
                )
        diagnostics = row.get("diagnostics", {})
        if set(diagnostics) != {str(horizon) for horizon in HORIZONS}:
            problems.append(f"{path.name}: diagnostic horizons mismatch")
            continue
        for horizon in HORIZONS:
            result = diagnostics[str(horizon)]
            if not isinstance(result.get("n_states"), int) or result["n_states"] <= 0:
                problems.append(f"{path.name}: invalid {horizon}h state count")
            if result.get("fixed_action") != row.get("fixed_action"):
                problems.append(f"{path.name}: inconsistent {horizon}h fixed action")
            for key in (
                "observed_headroom",
                "observed_disagreement",
                "observed_selection_premium",
                "headroom_excess",
                "disagreement_excess",
                "selection_premium_excess",
            ):
                if not math.isfinite(result.get(key, float("nan"))):
                    problems.append(f"{path.name}: non-finite {horizon}h {key}")
            if result.get("observed_headroom", -1.0) < -1e-8:
                problems.append(f"{path.name}: negative {horizon}h headroom")
            if not 0.0 <= result.get("observed_disagreement", -1.0) <= 1.0:
                problems.append(f"{path.name}: invalid {horizon}h disagreement")
            if not isinstance(result.get("null_state_block"), int) \
                    or result["null_state_block"] <= 0:
                problems.append(f"{path.name}: invalid {horizon}h null block")
            for label in (
                "null_headroom", "null_disagreement", "null_selection_premium"
            ):
                summary = result.get(label, {})
                values = [summary.get(key, float("nan")) for key in (
                    "mean", "q025", "q975", "p_two_sided"
                )]
                if not all(math.isfinite(value) for value in values):
                    problems.append(f"{path.name}: invalid {horizon}h {label}")
                elif summary["q025"] > summary["q975"]:
                    problems.append(f"{path.name}: unordered {horizon}h {label} quantiles")
                elif not 0.0 < summary["p_two_sided"] <= 1.0:
                    problems.append(f"{path.name}: invalid {horizon}h {label} p")
                if label == "null_disagreement" and not (
                    0.0 <= summary.get("q025", -1.0)
                    <= summary.get("q975", 2.0) <= 1.0
                ):
                    problems.append(
                        f"{path.name}: invalid {horizon}h disagreement interval"
                    )
            for metric in ("headroom", "disagreement", "selection_premium"):
                expected_excess = (
                    result.get(f"observed_{metric}", float("nan"))
                    - result.get(f"null_{metric}", {}).get("mean", float("nan"))
                )
                if not math.isclose(
                    result.get(f"{metric}_excess", float("nan")),
                    expected_excess,
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):
                    problems.append(
                        f"{path.name}: inconsistent {horizon}h {metric} excess"
                    )
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
