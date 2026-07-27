"""Independent structural and cross-collection audit of the retail frontier."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ..data.pools import CORE
from .retail_frontier import CAPITALS, WIDTHS, config_tag


def audit(root: Path, primary: Path) -> dict:
    tag = config_tag(CORE, CAPITALS, WIDTHS)
    files = sorted(root.glob(f"*__{tag}.json"))
    expected = {(pool, step, capital) for pool in CORE for step in range(24)
                for capital in CAPITALS}
    seen, rows, problems = set(), {}, []
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
        try:
            row = json.loads(path.read_text())
        except Exception as exc:
            problems.append(f"{path.name}: invalid JSON: {exc}")
            continue
        u = row.get("unit", {})
        key = (u.get("pool"), u.get("step"), u.get("capital"))
        if key in seen:
            problems.append(f"duplicate {key}")
        seen.add(key)
        rows[key] = row
        step = u.get("step")
        split = {"train": list(range(step, step + 4)), "val": [step + 4],
                 "test": [step + 5]}
        if row.get("split") != split:
            problems.append(f"{path.name}: split mismatch")
        if set(row.get("recentre", {})) != {str(w) for w in WIDTHS}:
            problems.append(f"{path.name}: width set mismatch")
        for result in [row.get("passive_test", {})] + [
                d[s] for d in row.get("recentre", {}).values() for s in ("val", "test")]:
            if not math.isfinite(result.get("reward", float("nan"))):
                problems.append(f"{path.name}: non-finite reward")
            if result.get("n_rebalances", -1) < 0 or result.get("rebalance_notional", -1) < 0:
                problems.append(f"{path.name}: invalid cost diagnostic")
    if len(files) != 720 or seen != expected:
        problems.append(f"{len(files)}/720 files; missing={len(expected-seen)} "
                        f"extra={len(seen-expected)}")

    # At the paper's exact $30k/$5/no-conversion setting, the cached frontier must
    # reproduce the completed RecentreWhenOut and Passive arms window by window.
    from .action_geometry import ALGOS, _tag
    ptag = _tag("ppo", [45, 50, 55], "spacing")
    primary_rows = {}
    for path in primary.glob(f"*__{ptag}.json"):
        row = json.loads(path.read_text())
        primary_rows[(row["unit"]["pool"], row["unit"]["step"])] = row
    if len(primary_rows) != 144:
        problems.append(f"primary reference has {len(primary_rows)}/144 PPO rows")
    else:
        for pool in CORE:
            for step in range(24):
                row = rows.get((pool, step, 30_000))
                if row is None:
                    continue
                best = max(WIDTHS, key=lambda w: row["recentre"][str(w)]["val"]["reward"]
                           - 5 * row["recentre"][str(w)]["val"]["n_rebalances"])
                active = row["recentre"][str(best)]["test"]
                reward = active["reward"] - 5 * active["n_rebalances"]
                ref = primary_rows[(pool, step)]["arms"]
                if abs(reward - ref["RecentreWhenOut"]["test"]) > 1e-8:
                    problems.append(f"{pool}/step{step}: active primary mismatch")
                if abs(row["passive_test"]["reward"] - ref["Passive"]["test"]) > 1e-8:
                    problems.append(f"{pool}/step{step}: passive primary mismatch")
    return {"status": "COMPLETE" if not problems else "INVALID",
            "problems": problems, "count": len(files), "sha256": h.hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.root, args.primary)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
