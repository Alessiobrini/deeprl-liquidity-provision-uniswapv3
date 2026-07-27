"""Independent completeness and consistency audit for final experiment artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..data.pools import CORE
from .sensitivity import EXPECTED_VARIANTS, variant_name

ALGOS = ("ppo", "a2c", "dqn", "qrdqn", "recurrentppo")


def _read(directory: Path) -> list[tuple[Path, dict]]:
    return [(p, json.loads(p.read_text())) for p in sorted(directory.glob("*.json"))]


def _digest(rows: list[tuple[Path, dict]]) -> str:
    h = hashlib.sha256()
    for path, _ in rows:
        h.update(path.name.encode()); h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def audit(primary_dir: Path, transfer_dir: Path, sensitivity_dir: Path) -> dict:
    problems: list[str] = []
    primary_rows, transfer_rows, sensitivity_rows = (
        _read(primary_dir), _read(transfer_dir), _read(sensitivity_dir))

    primary = {}
    for path, row in primary_rows:
        cfg = row.get("provenance", {}).get("config", {})
        algo = str(cfg.get("algo", "")).lower()
        if algo not in ALGOS:
            continue
        key = (algo, row["unit"]["pool"], int(row["unit"]["step"]))
        if key in primary:
            problems.append(f"duplicate primary key {key}")
        primary[key] = float(row["arms"][algo.upper()]["test"])
    expected_primary = {(a, p, s) for a in ALGOS for p in CORE for s in range(24)}
    if set(primary) != expected_primary:
        problems.append(f"primary keys {len(primary)}/{len(expected_primary)}")

    transfer = {}
    native_mismatches = seed_mean_mismatches = target_mismatches = split_mismatches = 0
    for path, row in transfer_rows:
        cfg = row.get("provenance", {}).get("config", {})
        key = (str(cfg.get("algo", "")).lower(), row["unit"]["source"],
               int(row["unit"]["step"]))
        if key in transfer:
            problems.append(f"duplicate transfer key {key}")
        transfer[key] = row
        if set(row["targets"]) != set(CORE):
            target_mismatches += 1
        if row["split"].get("test") != [key[2] + 5]:
            split_mismatches += 1
        for target in CORE:
            if target not in row["targets"]:
                continue
            result = row["targets"][target]
            seeds = result.get("per_seed", [])
            if len(seeds) != 2 or not np.isclose(np.mean(seeds), result["test"]):
                seed_mean_mismatches += 1
        native = row["targets"].get(key[1], {}).get("test")
        if key in primary and not np.isclose(native, primary[key], rtol=1e-12, atol=1e-9):
            native_mismatches += 1
    expected_transfer = expected_primary
    if set(transfer) != expected_transfer:
        problems.append(f"transfer keys {len(transfer)}/{len(expected_transfer)}")
    for label, count in (("target-set", target_mismatches),
                         ("split", split_mismatches),
                         ("seed-mean", seed_mean_mismatches),
                         ("native-primary", native_mismatches)):
        if count:
            problems.append(f"{label} mismatches {count}")

    sensitivity = {}
    sensitivity_split_mismatches = 0
    for path, row in sensitivity_rows:
        cfg = row.get("provenance", {}).get("config", {})
        name = variant_name(cfg)
        key = (name, row["unit"]["pool"], int(row["unit"]["step"]))
        if key in sensitivity:
            problems.append(f"duplicate sensitivity key {key}")
        sensitivity[key] = row
        if row["split"].get("test") != [key[2] + 5]:
            sensitivity_split_mismatches += 1
    expected_sensitivity = {(v, p, s) for v in EXPECTED_VARIANTS
                            for p in CORE for s in range(24)}
    if set(sensitivity) != expected_sensitivity:
        problems.append(f"sensitivity keys {len(sensitivity)}/{len(expected_sensitivity)}")
    if sensitivity_split_mismatches:
        problems.append(f"sensitivity split mismatches {sensitivity_split_mismatches}")

    return {
        "status": "COMPLETE" if not problems else "INCOMPLETE",
        "problems": problems,
        "counts": {"primary": len(primary), "transfer": len(transfer),
                   "sensitivity": len(sensitivity)},
        "sha256": {"primary": _digest(primary_rows), "transfer": _digest(transfer_rows),
                   "sensitivity": _digest(sensitivity_rows)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--transfer", type=Path, required=True)
    ap.add_argument("--sensitivity", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = audit(args.primary, args.transfer, args.sensitivity)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
