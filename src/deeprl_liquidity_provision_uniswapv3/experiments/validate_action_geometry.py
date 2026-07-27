"""Independent structural audit for the matched action-geometry collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ..data.pools import CORE
from .action_geometry import ALGOS, HEURISTICS, LEARNED, _tag


def collection_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.name):
        h.update(path.name.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def audit(root: Path, ignored: set[Path] | None = None) -> dict:
    problems, used = [], []
    ignored = {p.resolve() for p in (ignored or set())}
    all_json = {p for p in root.glob("*.json") if p.resolve() not in ignored}
    expected_keys = {(pool, step) for pool in CORE for step in range(24)}
    heuristic_reference = {}
    for algo in ALGOS:
        tag = _tag(algo, [45.0, 50.0, 55.0], "spacing", [480.0, 540.0, 600.0])
        files = sorted(root.glob(f"*__{tag}.json"))
        used.extend(files)
        rows, seen = [], set()
        for path in files:
            try:
                row = json.loads(path.read_text())
            except Exception as exc:
                problems.append(f"{path.name}: invalid JSON: {exc}")
                continue
            rows.append((path, row))
            key = (row.get("unit", {}).get("pool"), row.get("unit", {}).get("step"))
            if key in seen:
                problems.append(f"{algo}: duplicate {key}")
            seen.add(key)
        if len(files) != 144 or seen != expected_keys:
            problems.append(f"{algo}: {len(files)}/144 files; "
                            f"missing={len(expected_keys-seen)} extra={len(seen-expected_keys)}")
        for path, row in rows:
            unit, cfg = row.get("unit", {}), row.get("provenance", {}).get("config", {})
            pool, step = unit.get("pool"), unit.get("step")
            expected_split = {"train": list(range(step, step + 4)),
                              "val": [step + 4], "test": [step + 5]}
            if row.get("split") != expected_split:
                problems.append(f"{path.name}: split mismatch")
            expected_cfg = {"algo": algo, "widths": [45.0, 50.0, 55.0],
                            "width_units": "spacing",
                            "execution_widths": [480.0, 540.0, 600.0],
                            "schedule": "event_driven",
                            "steps": 20_000, "seeds": [42, 123], "shaping": "none",
                            "paper_extractor": False}
            for key, value in expected_cfg.items():
                if cfg.get(key) != value:
                    problems.append(f"{path.name}: config {key}={cfg.get(key)!r}, "
                                    f"expected {value!r}")
            expected_arms = set(HEURISTICS) | {LEARNED[algo]}
            arms = row.get("arms", {})
            if set(arms) != expected_arms:
                problems.append(f"{path.name}: arm set mismatch")
                continue
            for arm, result in arms.items():
                if not math.isfinite(result.get("test", float("nan"))):
                    problems.append(f"{path.name}: non-finite {arm} test")
                if not math.isfinite(result.get("val", float("nan"))):
                    problems.append(f"{path.name}: non-finite {arm} validation")
                if not 1 <= result.get("n_actions", 0) <= 4:
                    problems.append(f"{path.name}: invalid {arm} n_actions")
            for arm in HEURISTICS:
                hkey = (pool, step, arm)
                value = arms[arm]
                if hkey in heuristic_reference and heuristic_reference[hkey] != value:
                    problems.append(f"{path.name}: repeated heuristic {arm} differs")
                heuristic_reference[hkey] = value
    unexpected = sorted(all_json - set(used), key=lambda p: p.name)
    if unexpected:
        problems.append(f"unexpected JSON files: {[p.name for p in unexpected[:5]]}")
    return {"status": "COMPLETE" if not problems else "INVALID",
            "problems": problems, "count": len(used), "sha256": collection_hash(used)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    # The report may intentionally live inside the collection. Ignore that exact
    # path so validation is idempotent; every other unexpected JSON remains invalid.
    result = audit(args.root, {args.output})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
