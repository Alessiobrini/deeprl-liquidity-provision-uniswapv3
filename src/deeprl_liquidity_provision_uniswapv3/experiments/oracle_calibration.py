"""Calibrate local-oracle statistics for maximization over noisy action returns.

The local oracle is future-aware. Its raw reward gap is positive under a no-signal
null because it selects the largest of four realized returns. This module keeps the
state and action main effects of each held-out action-value matrix and applies a
block-wild null to its residual state--action interaction. One JSON remains one
held-out rolling window.

The frozen protocol is ``reports/ORACLE_CALIBRATION_PROTOCOL.md``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from ..data.pools import CORE
from ..envs.schedule import EventDriven
from .bakeoff import load_panel
from .rolling import holm_adjust, paired_block_inference, rolling_steps
from .state_headroom import (
    EXECUTION_WIDTHS,
    HORIZONS,
    PRIMARY_HORIZON,
    ROBUSTNESS_HORIZON,
    WIDTHS,
    _build_matched_env,
    _full_horizon,
    action_returns,
    select_fixed,
)

EXPECTED_WINDOWS = 24
NULL_METHOD = "two_way_additive_block_wild"
NULL_SAMPLES = 2_000
NULL_BLOCK_RULE = "ceil_horizon_over_median_decision_gap"
NULL_SEED = 20260802
FAMILY = (
    "selection_premium_excess_168h",
    "disagreement_excess_168h",
)


def two_way_decompose(q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Balanced additive projection into grand, state, action, and interaction."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] != 4:
        raise ValueError("expected a non-empty (states, 4 actions) matrix")
    if not np.isfinite(q).all():
        raise ValueError("action returns must be finite")
    grand = float(q.mean())
    state = q.mean(axis=1) - grand
    action = q.mean(axis=0) - grand
    residual = q - grand - state[:, None] - action[None, :]
    return grand, state, action, residual


def block_wild_signs(n_states: int, samples: int, block: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Rademacher signs constant on randomly rotated, non-overlapping blocks."""
    if n_states < 1 or samples < 1 or block < 1:
        raise ValueError("states, samples, and block must be positive")
    block = min(block, n_states)
    n_blocks = int(np.ceil(n_states / block))
    signs = np.repeat(
        rng.choice(np.array([-1.0, 1.0]), size=(samples, n_blocks)),
        block,
        axis=1,
    )[:, :n_states]
    offsets = rng.integers(0, block, size=samples)
    for row, offset in enumerate(offsets):
        signs[row] = np.roll(signs[row], int(offset))
    return signs


def null_action_values(q: np.ndarray, *, samples: int, block: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Generate block-wild null matrices while preserving fitted main effects."""
    grand, state, action, residual = two_way_decompose(q)
    signs = block_wild_signs(len(q), samples, block, rng)
    wild = signs[:, :, None] * residual[None, :, :]
    # Block signs retain zero row means. Re-projecting columns to zero preserves the
    # observed action means exactly; the column corrections sum to zero, so state
    # means remain unchanged as well.
    wild -= wild.mean(axis=1, keepdims=True)
    return grand + state[None, :, None] + action[None, None, :] + wild


def _metrics(q: np.ndarray, fixed_action: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    optimum = q.max(axis=-1)
    fixed = q[..., fixed_action]
    headroom = optimum - fixed
    disagreement = ~np.isclose(fixed, optimum, rtol=0.0, atol=1e-9)
    selection_premium = optimum.mean(axis=-1) - q.mean(axis=-2).max(axis=-1)
    return headroom, disagreement.astype(float), selection_premium


def calibrate_matrix(q: np.ndarray, fixed_action: int, *, samples: int = NULL_SAMPLES,
                     block: int,
                     seed: int = NULL_SEED) -> dict:
    """Observed, null, and calibrated oracle statistics for one action-value matrix."""
    q = np.asarray(q, dtype=float)
    if fixed_action not in range(4):
        raise ValueError("fixed_action must be in 0..3")
    two_way_decompose(q)  # validate before allocating null draws
    observed_h, observed_d, observed_p = _metrics(q, fixed_action)
    null_q = null_action_values(
        q,
        samples=samples,
        block=block,
        rng=np.random.default_rng(seed),
    )
    null_h, null_d, null_p = _metrics(null_q, fixed_action)
    null_h = null_h.mean(axis=1)
    null_d = null_d.mean(axis=1)
    observed_headroom = float(observed_h.mean())
    observed_disagreement = float(observed_d.mean())
    observed_premium = float(observed_p)
    null_headroom = float(null_h.mean())
    null_disagreement = float(null_d.mean())
    null_premium = float(null_p.mean())

    def summary(values: np.ndarray, observed: float) -> dict:
        center = float(values.mean())
        tail = np.abs(values - center) >= abs(observed - center) - 1e-12
        return {
            "mean": center,
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "p_two_sided": float((np.count_nonzero(tail) + 1) / (len(values) + 1)),
        }

    return {
        "n_states": int(len(q)),
        "fixed_action": int(fixed_action),
        "observed_headroom": observed_headroom,
        "observed_disagreement": observed_disagreement,
        "observed_selection_premium": observed_premium,
        "null_headroom": summary(null_h, observed_headroom),
        "null_disagreement": summary(null_d, observed_disagreement),
        "null_selection_premium": summary(null_p, observed_premium),
        "headroom_excess": observed_headroom - null_headroom,
        "disagreement_excess": observed_disagreement - null_disagreement,
        "selection_premium_excess": observed_premium - null_premium,
    }


def test_action_matrices(pool: str, window: int, fixed_action: int
                         ) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Action returns on the held-out fixed-action trajectory, in decision order."""
    wrapper = EventDriven(_build_matched_env(pool, window, scheduled=False))
    wrapper.reset()
    inner = wrapper.unwrapped
    rows = {horizon: [] for horizon in HORIZONS}
    hours = []
    done = trunc = False
    while not (done or trunc):
        if _full_horizon(inner, PRIMARY_HORIZON):
            hours.append(int(inner.i))
            returns = action_returns(inner, HORIZONS)
            for horizon, values in returns.items():
                rows[horizon].append(values)
        _, _, done, trunc, _ = wrapper.step(fixed_action)
    if any(not values for values in rows.values()):
        raise RuntimeError(f"{pool} window {window}: no complete action horizons")
    return (
        {horizon: np.asarray(values, dtype=float) for horizon, values in rows.items()},
        np.asarray(hours, dtype=int),
    )


def horizon_block_states(hours: np.ndarray, horizon: int) -> int:
    """Decision-state block spanning approximately one overlapping reward horizon."""
    hours = np.asarray(hours, dtype=int)
    if hours.ndim != 1 or len(hours) < 1:
        raise ValueError("hours must be a non-empty vector")
    if len(hours) == 1:
        return 1
    gaps = np.diff(hours)
    if (gaps <= 0).any():
        raise ValueError("decision hours must be strictly increasing")
    return max(1, int(np.ceil(float(horizon) / float(np.median(gaps)))))


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_unit(pool: str, step: int, *, null_samples: int = NULL_SAMPLES,
             null_block: int = 0, null_seed: int = NULL_SEED) -> dict:
    split = rolling_steps(len(load_panel(pool)) // 1500)[step]
    fixed_action, fixed_val_scores = select_fixed(pool, split.val[0])
    matrices, hours = test_action_matrices(pool, split.test[0], fixed_action)
    diagnostics = {}
    for horizon_i, horizon in enumerate(HORIZONS):
        # A stable unit-specific seed makes shards and resumed collections identical.
        pool_i = list(CORE).index(pool)
        seed = null_seed + 10_000 * pool_i + 100 * step + horizon_i
        block = null_block or horizon_block_states(hours, horizon)
        diagnostics[str(horizon)] = calibrate_matrix(
            matrices[horizon],
            fixed_action,
            samples=null_samples,
            block=block,
            seed=seed,
        )
        diagnostics[str(horizon)]["null_state_block"] = block
    return {
        "unit": {"pool": pool, "step": step},
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "fixed_action": fixed_action,
        "fixed_validation_rewards": fixed_val_scores,
        "diagnostics": diagnostics,
        "provenance": {
            "git_sha": _git_sha(),
            "protocol": "reports/ORACLE_CALIBRATION_PROTOCOL.md",
            "widths": list(WIDTHS),
            "execution_widths": list(EXECUTION_WIDTHS),
            "horizons": list(HORIZONS),
            "null_method": NULL_METHOD,
            "null_samples": null_samples,
            "null_block_rule": NULL_BLOCK_RULE,
            "null_block_override": null_block,
            "null_seed": null_seed,
        },
    }


def expected_units(pools: list[str]) -> list[tuple[str, int]]:
    return [(pool, step) for pool in sorted(pools) for step in range(EXPECTED_WINDOWS)]


def _load_results(root: Path) -> tuple[dict[tuple[str, int], dict], list[tuple[str, int]]]:
    rows = {}
    for path in sorted(root.glob("*__step*.json")):
        row = json.loads(path.read_text())
        key = (row["unit"]["pool"], int(row["unit"]["step"]))
        if key in rows:
            raise ValueError(f"duplicate result for {key}")
        rows[key] = row
    expected = set(expected_units(list(CORE)))
    extra = set(rows) - expected
    if extra:
        raise ValueError(f"unexpected result units: {sorted(extra)}")
    return rows, sorted(expected - set(rows))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_block_ratio_interval(numerator: np.ndarray, denominator: np.ndarray,
                                rng: np.random.Generator, n_boot: int,
                                block_length: int) -> tuple[float, float]:
    """Percentile interval for a ratio of paired, dependent window means."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if numerator.ndim != 1 or denominator.shape != numerator.shape or len(numerator) < 1:
        raise ValueError("numerator and denominator must be equal non-empty vectors")
    n = len(numerator)
    block_length = min(max(1, block_length), n)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block_length)) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    denominator_means = denominator[idx].mean(axis=1)
    if np.any(np.isclose(denominator_means, 0.0)):
        raise ValueError("bootstrap denominator mean is zero")
    ratios = numerator[idx].mean(axis=1) / denominator_means
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return float(lo), float(hi)


def analyze(root: Path, report_dir: Path, *, seed: int = NULL_SEED,
            samples: int = 10_000, block: int = 4) -> str:
    """Window-level calibrated inference for the complete six-pool collection."""
    rows, missing = _load_results(root)
    if missing:
        return (
            "PARTIAL - NOT A FINAL RESULT\n"
            f"{len(rows)}/144 units; missing {len(missing)}: {missing[:8]}\n"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    window_rows, headline_rows, contrast_rows = [], [], []
    lines = [
        "COMPLETE - FINAL COLLECTION",
        "Oracle maximization-bias calibration; one observation is one held-out window.",
        "H5 Holm family: two calibrated-excess contrasts within each pool.",
    ]
    metric_map = {
        "selection_premium_excess_168h": (
            PRIMARY_HORIZON, "selection_premium_excess"
        ),
        "disagreement_excess_168h": (PRIMARY_HORIZON, "disagreement_excess"),
    }
    for pool_i, pool in enumerate(CORE):
        arrays = {name: [] for name in FAMILY}
        observed = {
            h: {"headroom": [], "disagreement": [], "selection_premium": []}
            for h in HORIZONS
        }
        null = {
            h: {"headroom": [], "disagreement": [], "selection_premium": []}
            for h in HORIZONS
        }
        for step in range(EXPECTED_WINDOWS):
            row = rows[(pool, step)]
            out = {"pool": pool, "step": step, "fixed_action": row["fixed_action"]}
            for horizon in HORIZONS:
                d = row["diagnostics"][str(horizon)]
                observed[horizon]["headroom"].append(d["observed_headroom"])
                observed[horizon]["disagreement"].append(d["observed_disagreement"])
                observed[horizon]["selection_premium"].append(
                    d["observed_selection_premium"]
                )
                null[horizon]["headroom"].append(d["null_headroom"]["mean"])
                null[horizon]["disagreement"].append(d["null_disagreement"]["mean"])
                null[horizon]["selection_premium"].append(
                    d["null_selection_premium"]["mean"]
                )
                out.update({
                    f"observed_headroom_{horizon}h": d["observed_headroom"],
                    f"null_headroom_{horizon}h": d["null_headroom"]["mean"],
                    f"headroom_excess_{horizon}h": d["headroom_excess"],
                    f"observed_disagreement_{horizon}h": d["observed_disagreement"],
                    f"null_disagreement_{horizon}h": d["null_disagreement"]["mean"],
                    f"disagreement_excess_{horizon}h": d["disagreement_excess"],
                    f"observed_selection_premium_{horizon}h":
                        d["observed_selection_premium"],
                    f"null_selection_premium_{horizon}h":
                        d["null_selection_premium"]["mean"],
                    f"selection_premium_excess_{horizon}h":
                        d["selection_premium_excess"],
                    f"null_state_block_{horizon}h": d["null_state_block"],
                })
            for name, (horizon, key) in metric_map.items():
                arrays[name].append(row["diagnostics"][str(horizon)][key])
            window_rows.append(out)

        raw = []
        for contrast_i, name in enumerate(FAMILY):
            values = np.asarray(arrays[name], dtype=float)
            lo, hi, p = paired_block_inference(
                values,
                np.random.default_rng(seed + 1000 * pool_i + contrast_i),
                samples,
                block,
            )
            raw.append({
                "pool": pool,
                "family": "H5 oracle calibration",
                "contrast": name,
                "difference": float(values.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "p_raw": p,
            })
        adjusted = holm_adjust([row["p_raw"] for row in raw])
        for row, p_holm in zip(raw, adjusted):
            row["p_holm"] = p_holm
            contrast_rows.append(row)

        primary = {row["contrast"]: row for row in raw}
        headline = {"pool": pool, "windows": EXPECTED_WINDOWS}
        for horizon in HORIZONS:
            for metric in ("headroom", "disagreement", "selection_premium"):
                headline[f"observed_{metric}_{horizon}h"] = float(
                    np.mean(observed[horizon][metric])
                )
                headline[f"null_{metric}_{horizon}h"] = float(
                    np.mean(null[horizon][metric])
                )
                headline[f"{metric}_excess_{horizon}h"] = (
                    headline[f"observed_{metric}_{horizon}h"]
                    - headline[f"null_{metric}_{horizon}h"]
                )
        for name in FAMILY:
            row = primary[name]
            headline[name] = row["difference"]
            headline[f"{name}_ci_low"] = row["ci_low"]
            headline[f"{name}_ci_high"] = row["ci_high"]
            headline[f"{name}_p_raw"] = row["p_raw"]
            headline[f"{name}_p_holm"] = adjusted[FAMILY.index(name)]
        observed_premium = np.asarray(
            observed[PRIMARY_HORIZON]["selection_premium"], dtype=float
        )
        null_premium = np.asarray(
            null[PRIMARY_HORIZON]["selection_premium"], dtype=float
        )
        ratio_lo, ratio_hi = paired_block_ratio_interval(
            null_premium,
            observed_premium,
            np.random.default_rng(seed + 1000 * pool_i + 97),
            samples,
            block,
        )
        headline[f"null_explained_fraction_{PRIMARY_HORIZON}h"] = float(
            null_premium.mean() / observed_premium.mean()
        )
        headline[f"null_explained_fraction_{PRIMARY_HORIZON}h_ci_low"] = ratio_lo
        headline[f"null_explained_fraction_{PRIMARY_HORIZON}h_ci_high"] = ratio_hi
        headline[f"windows_null_fraction_above_one_{PRIMARY_HORIZON}h"] = int(
            np.count_nonzero(null_premium > observed_premium)
        )
        headline_rows.append(headline)
        h = PRIMARY_HORIZON
        lines.extend([
            "",
            f"POOL {pool}",
            (
                f"  {h}h headroom observed={headline[f'observed_headroom_{h}h']:+,.2f} "
                f"null={headline[f'null_headroom_{h}h']:+,.2f} "
                f"excess={headline[f'headroom_excess_{h}h']:+,.2f}"
            ),
            (
                f"  {h}h disagreement observed="
                f"{headline[f'observed_disagreement_{h}h']:.1%} "
                f"null={headline[f'null_disagreement_{h}h']:.1%} "
                f"excess={headline[f'disagreement_excess_{h}h']:+.1%}"
            ),
            (
                f"  {h}h selection premium observed="
                f"{headline[f'observed_selection_premium_{h}h']:+,.2f} "
                f"null={headline[f'null_selection_premium_{h}h']:+,.2f} "
                f"excess={headline[f'selection_premium_excess_{h}h']:+,.2f}"
            ),
        ])

    _write_csv(report_dir / "windows.csv", window_rows)
    _write_csv(report_dir / "headline.csv", headline_rows)
    _write_csv(report_dir / "contrasts.csv", contrast_rows)
    lines.extend([
        "",
        f"Window bootstrap: seed={seed}, resamples={samples:,}, block={block}.",
        "Pools are not combined.",
    ])
    text = "\n".join(lines) + "\n"
    (report_dir / "report.txt").write_text(text)
    return text


def collection_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*__step*.json")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pools", nargs="*", default=CORE)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--null-samples", type=int, default=NULL_SAMPLES)
    parser.add_argument(
        "--null-block", type=int, default=0,
        help="decision-state block override; zero uses the frozen horizon/gap rule",
    )
    parser.add_argument("--null-seed", type=int, default=NULL_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-block", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        report_dir = args.report_dir or args.out / "aggregate_block4"
        print(analyze(
            args.out,
            report_dir,
            seed=args.null_seed,
            samples=args.bootstrap_samples,
            block=args.bootstrap_block,
        ), end="")
        return
    if not 0 <= args.shard < args.of:
        parser.error(f"shard {args.shard} is outside 0..{args.of - 1}")
    units = expected_units(args.pools)
    mine = units[args.shard::args.of]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard}/{args.of}: {len(mine)} of {len(units)} units")
    for pool, step in mine:
        path = args.out / f"{pool}__step{step:03d}.json"
        if path.exists() and not args.force:
            print(f"  {pool} step {step:03d} skip (done)")
            continue
        result = run_unit(
            pool,
            step,
            null_samples=args.null_samples,
            null_block=args.null_block,
            null_seed=args.null_seed,
        )
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=1))
        temp.rename(path)
        primary = result["diagnostics"][str(PRIMARY_HORIZON)]
        print(
            f"  {pool} step {step:03d} fixed={result['fixed_action']} "
            f"headroom={primary['observed_headroom']:+,.2f} "
            f"null={primary['null_headroom']['mean']:+,.2f}"
        )
    print(f"collection sha256 {collection_hash(args.out)}")


if __name__ == "__main__":
    main()
