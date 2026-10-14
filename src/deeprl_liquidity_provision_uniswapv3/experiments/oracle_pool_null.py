"""Pool-level conditional randomization test for the oracle selection premium.

See reproducibility/protocols.md#pool-level-oracle-null. This conditional
block-wild reference is not a test of feature predictability.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..data.pools import CORE
from .oracle_calibration import (
    EXPECTED_WINDOWS,
    NULL_SAMPLES,
    NULL_SEED,
    _metrics,
    horizon_block_states,
    null_action_values,
    test_action_matrices,
)
from .state_headroom import PRIMARY_HORIZON


def run_unit(oracle_root: Path, out: Path, pool: str, step: int,
             samples: int = NULL_SAMPLES, seed: int = NULL_SEED) -> None:
    """Rebuild one action matrix and retain its full null-premium draws."""
    source = json.loads((oracle_root / f"{pool}__step{step:03d}.json").read_text())
    matrices, hours = test_action_matrices(
        pool, source["split"]["test"][0], source["fixed_action"]
    )
    q = matrices[PRIMARY_HORIZON]
    block = horizon_block_states(hours, PRIMARY_HORIZON)
    unit_seed = seed + 10_000 * list(CORE).index(pool) + 100 * step
    draws = null_action_values(
        q,
        samples=samples,
        block=block,
        rng=np.random.default_rng(unit_seed),
    )
    null_premium = _metrics(draws, source["fixed_action"])[2]
    observed = float(_metrics(q, source["fixed_action"])[2])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{pool}__step{step:03d}.npz",
        observed_selection_premium=observed,
        null_selection_premium=null_premium,
        null_state_block=block,
        seed=unit_seed,
    )


def pool_randomization_summary(observed: np.ndarray, null: np.ndarray) -> dict:
    """One-sided pool test from aligned, independently generated window draws."""
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    if observed.ndim != 1 or null.ndim != 2 or null.shape[0] != len(observed):
        raise ValueError("expected observed (windows,) and null (windows, draws)")
    if len(observed) == 0 or null.shape[1] == 0:
        raise ValueError("windows and draws must be non-empty")
    if not np.isfinite(observed).all() or not np.isfinite(null).all():
        raise ValueError("observed and null premiums must be finite")
    pool_observed = float(observed.mean())
    pool_null = null.mean(axis=0)
    exceedances = int(np.count_nonzero(pool_null >= pool_observed))
    draws = int(len(pool_null))
    # A zero observed premium has no defined explained fraction. Its tail
    # probability is still well-defined, including a completely additive matrix.
    fraction = pool_null / pool_observed if pool_observed != 0 else None
    return {
        "windows": int(len(observed)),
        "draws": draws,
        "observed_selection_premium": pool_observed,
        "null_selection_premium_mean": float(pool_null.mean()),
        "null_selection_premium_q025": float(np.quantile(pool_null, 0.025)),
        "null_selection_premium_q975": float(np.quantile(pool_null, 0.975)),
        "exceedances": exceedances,
        "p_exceedance": float(exceedances / draws),
        "p_exceedance_plus_one": float((exceedances + 1) / (draws + 1)),
        "null_explained_fraction": float(pool_null.mean() / pool_observed) if fraction is not None else None,
        "null_fraction_q025": float(np.quantile(fraction, 0.025)) if fraction is not None else None,
        "null_fraction_q975": float(np.quantile(fraction, 0.975)) if fraction is not None else None,
    }


def aggregate(root: Path, report: Path) -> list[dict]:
    """Combine 24 window draws for each pool without treating states as observations."""
    rows = []
    for pool in CORE:
        observed, null = [], []
        for step in range(EXPECTED_WINDOWS):
            path = root / f"{pool}__step{step:03d}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as unit:
                observed.append(float(unit["observed_selection_premium"]))
                null.append(np.asarray(unit["null_selection_premium"], dtype=float))
        summary = pool_randomization_summary(np.asarray(observed), np.stack(null))
        rows.append({"pool": pool, **summary})
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pools", nargs="+", choices=CORE, default=CORE)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--samples", type=int, default=NULL_SAMPLES)
    parser.add_argument("--seed", type=int, default=NULL_SEED)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.of < 1 or not 0 <= args.shard < args.of:
        parser.error("--shard must satisfy 0 <= shard < of")
    if args.aggregate:
        report = args.report or args.out / "pool_randomization.csv"
        for row in aggregate(args.out, report):
            print(row)
        return
    units = [(pool, step) for pool in args.pools for step in range(EXPECTED_WINDOWS)]
    for pool, step in units[args.shard::args.of]:
        path = args.out / f"{pool}__step{step:03d}.npz"
        if path.is_file():
            expected_seed = args.seed + 10_000 * list(CORE).index(pool) + 100 * step
            with np.load(path) as cached:
                if (len(cached["null_selection_premium"]) != args.samples
                        or int(cached["seed"]) != expected_seed):
                    raise ValueError(f"{path}: cached draw count or seed differs from requested configuration")
            continue
        run_unit(args.oracle_root, args.out, pool, step, args.samples, args.seed)
        print(f"{pool} step {step:03d}", flush=True)


if __name__ == "__main__":
    main()
