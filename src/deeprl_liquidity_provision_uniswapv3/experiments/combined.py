"""Combine completed rolling runs without retraining or duplicating baselines.

One JSON is one (algorithm, pool, test window), and already contains the mean test
reward across seeds.  This reader aligns those JSONs by (pool, step), verifies that
the repeated heuristic arms agree across algorithms, then reports two pre-specified
families per pool:

* H1: active heuristics versus Passive.
* H2: learned algorithms versus the heuristic selected by mean VALIDATION reward.

Inference uses the circular moving-block bootstrap from ``rolling`` so adjacent
walk-forward windows are not treated as independent. Holm correction is applied
separately to H1 and H2 within each pool.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..data.pools import CORE
from .rolling import holm_adjust, paired_block_inference

ALGOS = ("ppo", "a2c", "dqn", "qrdqn", "recurrentppo")
ALGO_LABEL = {a: a.upper() for a in ALGOS}
HEURISTICS = ("Passive", "PassiveWidthSweep", "VolProportionalWidth",
              "ILMinimizer", "ReactiveRecentering", "RecentreWhenOut")
ACTIVE_HEURISTICS = tuple(h for h in HEURISTICS if h != "Passive")
EXPECTED_WINDOWS = 24


def load_runs(out_dir: Path) -> tuple[dict, bool, list[str]]:
    """Load and align result JSONs; return data, completeness, and diagnostics."""
    data, diagnostics = {}, []
    for path in sorted(out_dir.glob("*.json")):
        row = json.loads(path.read_text())
        cfg = row.get("provenance", {}).get("config", {})
        algo = str(cfg.get("algo", "")).lower()
        if algo not in ALGOS:
            continue
        key = (row["unit"]["pool"], int(row["unit"]["step"]))
        if algo in data.setdefault(key, {}):
            raise ValueError(f"duplicate {algo} result for {key}")
        data[key][algo] = row

    expected = {(p, s) for p in CORE for s in range(EXPECTED_WINDOWS)}
    for key in sorted(expected):
        missing = sorted(set(ALGOS) - set(data.get(key, {})))
        if missing:
            diagnostics.append(f"{key[0]} step {key[1]} missing {missing}")
    extra = sorted(set(data) - expected)
    if extra:
        diagnostics.append(f"unexpected pool/steps: {extra[:5]}")

    # Heuristics are recomputed in every algorithm unit. They must agree exactly
    # enough to be one observation, not five repeated observations.
    for key, by_algo in data.items():
        rows = list(by_algo.values())
        if not rows:
            continue
        for arm in HEURISTICS:
            vals = [r["arms"][arm]["test"] for r in rows]
            val_scores = [r["arms"][arm]["val"] for r in rows]
            if not np.allclose(vals, vals[0], rtol=0, atol=1e-9):
                raise ValueError(f"repeated {arm} test values disagree at {key}: {vals}")
            if not np.allclose(val_scores, val_scores[0], rtol=0, atol=1e-9):
                raise ValueError(f"repeated {arm} validation values disagree at {key}")
    return data, not diagnostics, diagnostics


def pool_arrays(data: dict, pool: str) -> tuple[dict[str, np.ndarray], dict[str, float],
                                                 dict[str, float]]:
    """One aligned vector per strategy; heuristic copies are taken exactly once."""
    rewards = {k: [] for k in HEURISTICS + tuple(ALGO_LABEL.values())}
    validation = {k: [] for k in HEURISTICS}
    actions = {k: [] for k in rewards}
    for step in range(EXPECTED_WINDOWS):
        by_algo = data[(pool, step)]
        first = by_algo[ALGOS[0]]
        for arm in HEURISTICS:
            rewards[arm].append(first["arms"][arm]["test"])
            validation[arm].append(first["arms"][arm]["val"])
            actions[arm].append(first["arms"][arm]["n_actions"])
        for algo in ALGOS:
            label = ALGO_LABEL[algo]
            arm = next(k for k in by_algo[algo]["arms"] if k not in HEURISTICS)
            rewards[label].append(by_algo[algo]["arms"][arm]["test"])
            actions[label].append(by_algo[algo]["arms"][arm]["n_actions"])
    return ({k: np.asarray(v, dtype=float) for k, v in rewards.items()},
            {k: float(np.mean(v)) for k, v in validation.items()},
            {k: float(np.mean(v)) for k, v in actions.items()})


def comparisons(rewards: dict[str, np.ndarray], names: tuple[str, ...], reference: str,
                seed: int, samples: int, block: int) -> list[dict]:
    rng, rows = np.random.default_rng(seed), []
    for name in names:
        d = rewards[name] - rewards[reference]
        lo, hi, p = paired_block_inference(d, rng, samples, block)
        rows.append({"strategy": name, "reference": reference, "diff": float(d.mean()),
                     "wins": float((d > 0).mean()), "ci_low": lo, "ci_high": hi,
                     "p_raw": p})
    for row, adjusted in zip(rows, holm_adjust([r["p_raw"] for r in rows])):
        row["p_holm"] = adjusted
    return rows


def render(out_dir: Path, report_dir: Path, seed: int, samples: int, block: int) -> str:
    data, complete, diagnostics = load_runs(out_dir)
    status = "COMPLETE — FINAL COLLECTION" if complete else "PARTIAL — NOT A FINAL RESULT"
    lines = [status, f"Source: {out_dir} ({len(list(out_dir.glob('*.json')))} JSON files)"]
    if not complete:
        lines.extend(f"MISSING: {d}" for d in diagnostics[:20])
        return "\n".join(lines) + "\n"

    report_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, comparison_rows = [], []
    for pool_i, pool in enumerate(CORE):
        rewards, validation, actions = pool_arrays(data, pool)
        best_h = max(ACTIVE_HEURISTICS, key=lambda h: validation[h])
        order = sorted(rewards, key=lambda k: -rewards[k].mean())
        passive = rewards["Passive"].mean()
        lines.extend(["", f"POOL {pool} — COMPLETE — {EXPECTED_WINDOWS}/{EXPECTED_WINDOWS} windows",
                      f"Validation-selected heuristic reference: {best_h}",
                      f"{'strategy':<24} {'kind':<10} {'TEST':>9} {'vs passive':>11} {'distinct':>8}",
                      "-" * 68])
        for name in order:
            kind = "rl" if name in ALGO_LABEL.values() else "heuristic"
            mean = float(rewards[name].mean())
            lines.append(f"{name:<24} {kind:<10} {mean:>9,.0f} "
                         f"{mean-passive:>+11,.0f} {actions[name]:>8.1f}")
            summary_rows.append({"pool": pool, "strategy": name, "kind": kind,
                                 "mean_test": mean, "vs_passive": mean-passive,
                                 "mean_actions": actions[name],
                                 "mean_validation": validation.get(name, ""),
                                 "heuristic_reference": best_h})

        families = (("H1 heuristic vs Passive", ACTIVE_HEURISTICS, "Passive"),
                    ("H2 RL vs validation-selected heuristic",
                     tuple(ALGO_LABEL.values()), best_h))
        for family_i, (family, names, reference) in enumerate(families):
            # Distinct deterministic seeds by pool/family, while remaining reproducible.
            rows = comparisons(rewards, names, reference,
                               seed + 100 * pool_i + family_i, samples, block)
            lines.extend(["", f"{family}; paired seed-averaged windows; reference={reference}",
                          f"Block bootstrap: block={block}, resamples={samples:,}; Holm within family.",
                          f"{'strategy':<24} {'diff':>9} {'wins':>6} {'95% block CI':>23} "
                          f"{'p raw':>9} {'p Holm':>9}"])
            for row in rows:
                ci = f"[{row['ci_low']:+,.0f}, {row['ci_high']:+,.0f}]"
                lines.append(f"{row['strategy']:<24} {row['diff']:>+9,.0f} "
                             f"{row['wins']:>5.0%} {ci:>23} {row['p_raw']:>9.4f} "
                             f"{row['p_holm']:>9.4f}")
                comparison_rows.append({"pool": pool, "family": family, **row})

    lines.extend(["", "No pooled significance test: pools are heterogeneous and adjacent "
                   "windows are serially dependent."])
    _write_csv(report_dir / "summary.csv", summary_rows)
    _write_csv(report_dir / "comparisons.csv", comparison_rows)
    text = "\n".join(lines) + "\n"
    (report_dir / "report.txt").write_text(text)
    return text


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="rolling JSON directory")
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--bootstrap-seed", type=int, default=20260716)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--bootstrap-block", type=int, default=4)
    args = ap.parse_args()
    report_dir = args.report_dir or args.out / "combined"
    print(render(args.out, report_dir, args.bootstrap_seed,
                 args.bootstrap_samples, args.bootstrap_block), end="")


if __name__ == "__main__":
    main()
