"""Capital/gas viability frontier for recentering only after leaving the range."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..data.pools import CORE, POOLS
from ..policies.baselines import Passive, RecentreWhenOut
from .bakeoff import build_env, load_panel, score_details
from .rolling import holm_adjust, paired_block_inference, rolling_steps

CAPITALS = (1_500, 5_000, 10_000, 30_000, 100_000)
GAS_LEVELS = (1, 5, 25, 100, 250)
WIDTHS = (45, 50, 55)


def config_tag(pools, capitals, widths) -> str:
    payload = json.dumps({"pools": list(pools), "capitals": list(capitals),
                          "widths": list(widths), "fee_model": "per_swap",
                          "features": "legacy"}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class Unit:
    pool: str
    step: int
    capital: int
    tag: str

    @property
    def name(self) -> str:
        return (f"{self.pool}__step{self.step:03d}__capital{self.capital:06d}"
                f"__{self.tag}")


def work_units(pools, capitals, tag) -> list[Unit]:
    return [Unit(pool, step, int(capital), tag)
            for pool in sorted(pools)
            for step in range(len(rolling_steps(len(load_panel(pool)) // 1500)))
            for capital in capitals]


def _evaluate(pool, window, capital, policy) -> dict:
    env = build_env(pool, window, WIDTHS, capital_usd=capital, gas_usd=0.0,
                    swap_fee_frac=0.0, slippage_frac=0.0)
    return score_details(env, policy)


def run_unit(unit: Unit) -> dict:
    split = rolling_steps(len(load_panel(unit.pool)) // 1500)[unit.step]
    candidates = {}
    for width in WIDTHS:
        candidates[str(width)] = {
            "val": _evaluate(unit.pool, split.val[0], unit.capital,
                             RecentreWhenOut(width)),
            "test": _evaluate(unit.pool, split.test[0], unit.capital,
                              RecentreWhenOut(width)),
        }
    return {"unit": asdict(unit),
            "split": {"train": split.train, "val": split.val, "test": split.test},
            "passive_test": _evaluate(unit.pool, split.test[0], unit.capital, Passive()),
            "recentre": candidates,
            "provenance": {"host": platform.node(), "python": platform.python_version(),
                           "config": {"widths": list(WIDTHS), "gas_usd": 0.0,
                                      "swap_fee_frac": 0.0, "slippage_frac": 0.0}}}


def adjusted(result: dict, gas: float, rate: float) -> float:
    """Exact under reward-channel costs: position value is not debited."""
    return (result["reward"] - gas * result["n_rebalances"]
            - rate * result["rebalance_notional"])


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], report_dir: Path, seed=20260722,
              samples=10_000, block=4) -> str:
    by_key = {(r["unit"]["pool"], r["unit"]["step"], r["unit"]["capital"]): r
              for r in rows}
    expected = {(p, i, c) for p in CORE for i in range(24) for c in CAPITALS}
    if len(rows) != 720 or set(by_key) != expected:
        raise ValueError(f"PARTIAL — NOT FINAL: {len(rows)}/720 rows; "
                         f"missing {len(expected-set(by_key))}")

    output = []
    for pool_i, pool in enumerate(CORE):
        family = []
        for regime in ("gas_only", "conversion"):
            rate = 0.0 if regime == "gas_only" else POOLS[pool].fee_tier_pct / 100 + 0.0005
            for capital_i, capital in enumerate(CAPITALS):
                for gas_i, gas in enumerate(GAS_LEVELS):
                    active, passive, rebalances, chosen = [], [], [], []
                    for step in range(24):
                        row = by_key[(pool, step, capital)]
                        best = max(WIDTHS, key=lambda w: adjusted(
                            row["recentre"][str(w)]["val"], gas, rate))
                        result = row["recentre"][str(best)]["test"]
                        active.append(adjusted(result, gas, rate))
                        passive.append(row["passive_test"]["reward"])
                        rebalances.append(result["n_rebalances"])
                        chosen.append(best)
                    delta = np.asarray(active) - np.asarray(passive)
                    rng = np.random.default_rng(
                        seed + 1000 * pool_i + 100 * (regime == "conversion")
                        + 10 * capital_i + gas_i)
                    lo, hi, p = paired_block_inference(delta, rng, samples, block)
                    family.append({"pool": pool, "regime": regime, "capital": capital,
                                   "gas": gas, "active_mean": float(np.mean(active)),
                                   "passive_mean": float(np.mean(passive)),
                                   "difference": float(delta.mean()),
                                   "wins": float((delta > 0).mean()),
                                   "ci_low": lo, "ci_high": hi, "p_raw": p,
                                   "mean_rebalances": float(np.mean(rebalances)),
                                   "modal_width": max(WIDTHS, key=chosen.count)})
        for row, ph in zip(family, holm_adjust([r["p_raw"] for r in family])):
            row["p_holm"] = ph
            output.append(row)

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "frontier.csv", output)
    lines = ["COMPLETE — FINAL COLLECTION", "720/720 pool-window-capital units.",
             "Positive differences favor recentering after exit over leaving the "
             "initial position unchanged.", ""]
    for pool in CORE:
        lines.append(f"POOL {pool}")
        for regime in ("gas_only", "conversion"):
            subset = [r for r in output if r["pool"] == pool and r["regime"] == regime]
            viable = [(r["capital"], r["gas"]) for r in subset if r["difference"] > 0]
            lines.append(f"  {regime}: {len(viable)}/25 cells positive; {viable}")
        lines.append("")
    text = "\n".join(lines)
    (report_dir / "report.txt").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=CORE)
    ap.add_argument("--capitals", nargs="*", type=int, default=list(CAPITALS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--bootstrap-seed", type=int, default=20260722)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--bootstrap-block", type=int, default=4)
    args = ap.parse_args()
    tag = config_tag(args.pools, args.capitals, WIDTHS)
    units = work_units(args.pools, args.capitals, tag)
    if args.aggregate:
        files = sorted(args.out.glob(f"*__{tag}.json"))
        rows = [json.loads(p.read_text()) for p in files]
        print(aggregate(rows, args.report_dir or args.out / "aggregate",
                        args.bootstrap_seed, args.bootstrap_samples,
                        args.bootstrap_block))
        return
    if not 0 <= args.shard < args.of:
        raise ValueError(f"shard {args.shard} out of range for {args.of}")
    args.out.mkdir(parents=True, exist_ok=True)
    mine = units[args.shard::args.of]
    print(f"shard {args.shard}/{args.of}: {len(mine)} of {len(units)} units")
    for unit in mine:
        path = args.out / f"{unit.name}.json"
        if path.exists():
            print(f"  {unit.name} skip (done)")
            continue
        result = run_unit(unit)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=1))
        tmp.rename(path)
        print(f"  {unit.name} done")


if __name__ == "__main__":
    main()
