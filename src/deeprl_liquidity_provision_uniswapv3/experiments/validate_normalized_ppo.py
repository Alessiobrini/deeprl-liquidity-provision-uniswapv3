"""Independent structural audit for the gated reward-normalized PPO collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .normalized_ppo import (
    EXECUTION_WIDTHS,
    NORMALIZATION,
    POOLS,
    SEEDS,
    WIDTHS,
)

EXPECTED = {(pool, step) for pool in POOLS for step in range(24)}


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
    problems, seen = [], set()
    for path in files:
        try:
            row = json.loads(path.read_text())
        except Exception as exc:
            problems.append(f"{path.name}: invalid JSON: {exc}")
            continue
        unit = row.get("unit", {})
        pool, step = unit.get("pool"), unit.get("step")
        key = (pool, step)
        if key in seen:
            problems.append(f"{path.name}: duplicate {key}")
        seen.add(key)
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
            "schedule": "event_driven",
            "steps": 20_000,
            "seeds": list(SEEDS),
            "normalization": NORMALIZATION,
        }
        for name, expected in expected_provenance.items():
            if provenance.get(name) != expected:
                problems.append(
                    f"{path.name}: provenance {name}={provenance.get(name)!r}, "
                    f"expected {expected!r}"
                )
        result = row.get("normalized_ppo", {})
        for name in ("test", "validation"):
            value = result.get(name, float("nan"))
            if not math.isfinite(value):
                problems.append(f"{path.name}: non-finite {name}")
        if result.get("n_actions") not in range(1, 5):
            problems.append(f"{path.name}: invalid n_actions")
        config = result.get("selected_config", {})
        if config.get("learning_rate") not in (3e-4, 1e-3):
            problems.append(f"{path.name}: invalid learning rate")
        if config.get("ent_coef") not in (0.0, 0.01):
            problems.append(f"{path.name}: invalid entropy coefficient")
        if config.get("net_arch") not in ([4, 2], [64, 64]):
            problems.append(f"{path.name}: invalid network")

    if len(files) != 48 or seen != EXPECTED:
        problems.append(
            f"{len(files)}/48 files; missing={len(EXPECTED-seen)} "
            f"extra={len(seen-EXPECTED)}"
        )
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
