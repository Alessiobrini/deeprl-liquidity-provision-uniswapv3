"""Paired analysis of paper-scaled versus economically matched action widths.

The paper convention multiplies an action by pool tick spacing. The matched treatment
uses the same raw tick half-widths on both fee tiers. This module only reads completed
walk-forward JSONs; it never trains or evaluates on market data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..data.pools import CORE, POOLS
from .rolling import config_tag, holm_adjust, paired_block_inference

ALGOS = ("ppo", "a2c", "dqn", "qrdqn", "recurrentppo")
LEARNED = {a: a.upper() for a in ALGOS}
HEURISTICS = ("Passive", "PassiveWidthSweep", "VolProportionalWidth",
              "ILMinimizer", "ReactiveRecentering", "RecentreWhenOut")
STRATEGIES = HEURISTICS + tuple(LEARNED.values())
PAIRS = tuple(sorted({POOLS[p].pair for p in CORE}))


def _tag(algo: str, widths: list[float], units: str,
         execution_widths: list[float] | None = None) -> str:
    args = SimpleNamespace(algo=algo, schedule="event_driven", widths=widths,
                           width_units=units, shaping="none", steps=20_000,
                           seeds=[42, 123], paper_extractor=False,
                           execution_widths=execution_widths)
    return config_tag(args)


def _read_algo(root: Path, algo: str, widths: list[float], units: str,
               execution_widths: list[float] | None = None) -> dict:
    tag = _tag(algo, widths, units, execution_widths)
    rows = [json.loads(p.read_text()) for p in sorted(root.glob(f"*__{tag}.json"))]
    expected = {(pool, step) for pool in CORE for step in range(24)}
    by_key = {(r["unit"]["pool"], r["unit"]["step"]): r for r in rows}
    if len(rows) != 144 or set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        raise ValueError(f"{root} {algo}/{units}: {len(rows)}/144 units; "
                         f"missing {missing[:5]}")
    for r in rows:
        cfg = r["provenance"]["config"]
        actual_units = cfg.get("width_units", "spacing")
        actual_execution = cfg.get("execution_widths")
        if (cfg["algo"] != algo or actual_units != units or cfg["widths"] != widths
                or actual_execution != execution_widths):
            raise ValueError(f"provenance mismatch in {r['unit']}")
    return by_key


def load_design(paper: Path, matched: Path) -> dict:
    """Return geometry/pool/step/strategy records after strict collection checks."""
    collections = {
        "paper": {a: _read_algo(paper, a, [45, 50, 55], "spacing") for a in ALGOS},
        "matched": {a: _read_algo(matched, a, [45.0, 50.0, 55.0], "spacing",
                                   [480.0, 540.0, 600.0]) for a in ALGOS},
    }
    out = {}
    for geometry, algos in collections.items():
        for pool in CORE:
            for step in range(24):
                base = algos["ppo"][(pool, step)]["arms"]
                record = {h: base[h] for h in HEURISTICS}
                for algo, label in LEARNED.items():
                    record[label] = algos[algo][(pool, step)]["arms"][label]
                # Heuristics are deterministic and repeated in each algorithm file.
                for algo in ALGOS[1:]:
                    arms = algos[algo][(pool, step)]["arms"]
                    for h in HEURISTICS:
                        if arms[h] != base[h]:
                            raise ValueError(f"heuristic mismatch: {geometry}/{pool}/"
                                             f"step{step}/{algo}/{h}")
                out[(geometry, pool, step)] = record
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(data: dict, report_dir: Path, seed: int = 20260722,
            samples: int = 10_000, block: int = 4) -> str:
    summary, comparisons, did = [], [], []
    lines = ["COMPLETE — FINAL COLLECTION",
             "Economically matched raw-tick grid versus paper spacing-unit grid.",
             "One observation is one seed-averaged paired test window."]

    deltas = {}
    for pool_i, pool in enumerate(CORE):
        raw = []
        lines.extend(["", f"POOL {pool}",
                      f"{'strategy':<24} {'paper':>9} {'matched':>9} {'diff':>9} "
                      f"{'95% block CI':>23} {'p Holm':>9}", "-" * 88])
        for strategy_i, strategy in enumerate(STRATEGIES):
            paper = np.array([data[("paper", pool, i)][strategy]["test"]
                              for i in range(24)], dtype=float)
            matched = np.array([data[("matched", pool, i)][strategy]["test"]
                                for i in range(24)], dtype=float)
            delta = matched - paper
            deltas[(pool, strategy)] = delta
            rng = np.random.default_rng(seed + 100 * pool_i + strategy_i)
            lo, hi, p = paired_block_inference(delta, rng, samples, block)
            raw.append({"pool": pool, "strategy": strategy,
                        "paper_mean": float(paper.mean()),
                        "matched_mean": float(matched.mean()),
                        "difference": float(delta.mean()),
                        "wins": float((delta > 0).mean()), "ci_low": lo,
                        "ci_high": hi, "p_raw": p,
                        "paper_actions": float(np.mean([
                            data[("paper", pool, i)][strategy]["n_actions"]
                            for i in range(24)])),
                        "matched_actions": float(np.mean([
                            data[("matched", pool, i)][strategy]["n_actions"]
                            for i in range(24)]))})
        adjusted = holm_adjust([r["p_raw"] for r in raw])
        for row, ph in zip(raw, adjusted):
            row["p_holm"] = ph
            comparisons.append(row)
            summary.append({k: row[k] for k in
                            ("pool", "strategy", "paper_mean", "matched_mean",
                             "difference", "paper_actions", "matched_actions")})
            ci = f"[{row['ci_low']:+,.0f}, {row['ci_high']:+,.0f}]"
            lines.append(f"{row['strategy']:<24} {row['paper_mean']:>9,.0f} "
                         f"{row['matched_mean']:>9,.0f} {row['difference']:>+9,.0f} "
                         f"{ci:>23} {ph:>9.4f}")

    # Exploratory difference-in-differences: geometry effect at 0.30% minus the
    # geometry effect at 0.05%, paired on the panel's common clock.
    for strategy_i, strategy in enumerate(STRATEGIES):
        raw = []
        for pair_i, pair in enumerate(PAIRS):
            pools = [p for p in CORE if POOLS[p].pair == pair]
            low = next(p for p in pools if POOLS[p].fee_tier_pct == 0.05)
            high = next(p for p in pools if POOLS[p].fee_tier_pct == 0.3)
            x = deltas[(high, strategy)] - deltas[(low, strategy)]
            rng = np.random.default_rng(seed + 10_000 + 100 * strategy_i + pair_i)
            lo, hi, p = paired_block_inference(x, rng, samples, block)
            raw.append({"strategy": strategy, "pair": pair,
                        "difference_in_differences": float(x.mean()),
                        "ci_low": lo, "ci_high": hi, "p_raw": p})
        for row, ph in zip(raw, holm_adjust([r["p_raw"] for r in raw])):
            row["p_holm"] = ph
            did.append(row)

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "summary.csv", summary)
    _write_csv(report_dir / "comparisons.csv", comparisons)
    _write_csv(report_dir / "tier_did.csv", did)
    lines.extend(["", "Holm correction is applied across 11 strategies within each "
                  "pool. Difference-in-differences are exploratory and Holm-adjusted "
                  "across three pairs within each strategy."])
    text = "\n".join(lines) + "\n"
    (report_dir / "report.txt").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", type=Path, default=Path("outputs/algos_v1"))
    ap.add_argument("--matched", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--bootstrap-seed", type=int, default=20260722)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--bootstrap-block", type=int, default=4)
    args = ap.parse_args()
    data = load_design(args.paper, args.matched)
    print(analyze(data, args.report_dir, args.bootstrap_seed,
                  args.bootstrap_samples, args.bootstrap_block), end="")


if __name__ == "__main__":
    main()
