"""Gated reward-normalization diagnostic for matched-geometry PPO.

Training uses reward-only VecNormalize; validation and test always report raw dollar
reward from an unwrapped evaluation environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..agents.registry import make_agent
from .action_geometry import _read_algo
from .agent_arm import AgentPolicy, score_agent, train_env
from .bakeoff import Split, build_env, load_panel, score
from .rolling import grid_for, paired_block_inference, rolling_steps
from .state_headroom import EXECUTION_WIDTHS, WIDTHS, _bootstrap_ci

POOLS = ("usdc_weth_005", "usdc_weth_030")
SEEDS = (42, 123)
EXPECTED_WINDOWS = 24
NORMALIZATION = {
    "norm_obs": False,
    "norm_reward": True,
    "clip_reward": 10.0,
    "gamma": 0.99,
    "epsilon": 1e-8,
}


def normalized_train_env(pool: str, split: Split, schedule: str = "event_driven"):
    """Reward-normalized training env; observations remain exactly unchanged."""
    base = train_env(
        pool,
        WIDTHS,
        split,
        schedule,
        width_units="spacing",
        execution_widths=EXECUTION_WIDTHS,
    )
    vector = DummyVecEnv([lambda: base])
    return VecNormalize(vector, training=True, **NORMALIZATION)


def normalized_agent_step(pool: str, split: Split, steps: int) -> dict:
    """Select on validation, refit on train+validation, and read test once."""
    best_config, best_validation = None, -np.inf
    for config in grid_for("ppo"):
        validation = []
        for seed in SEEDS:
            env = normalized_train_env(pool, split)
            model = make_agent("ppo", env, seed=seed, **config)
            model.learn(total_timesteps=steps)
            validation.append(float(score_agent(
                pool,
                model,
                WIDTHS,
                split.val,
                "event_driven",
                width_units="spacing",
                execution_widths=EXECUTION_WIDTHS,
            ).mean()))
            env.close()
        score_value = float(np.mean(validation))
        if score_value > best_validation:
            best_config, best_validation = config, score_value

    fit = Split(train=split.train + split.val, val=[], test=split.test)
    models, envs = [], []
    for seed in SEEDS:
        env = normalized_train_env(pool, fit)
        model = make_agent("ppo", env, seed=seed, **best_config)
        model.learn(total_timesteps=steps)
        envs.append(env)
        models.append(model)
    per_seed = [
        score_agent(
            pool,
            model,
            WIDTHS,
            split.test,
            "event_driven",
            width_units="spacing",
            execution_widths=EXECUTION_WIDTHS,
        )
        for model in models
    ]
    evaluation = build_env(
        pool,
        split.test[0],
        WIDTHS,
        schedule="event_driven",
        width_units="spacing",
        execution_widths=EXECUTION_WIDTHS,
    )
    n_actions = score(evaluation, AgentPolicy(models[0], "normalized_ppo"))[1]
    for env in envs:
        env.close()
    return {
        "test": float(np.mean([values.mean() for values in per_seed])),
        "validation": best_validation,
        "selected_config": best_config,
        "n_actions": int(n_actions),
    }


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_unit(pool: str, step: int, steps: int) -> dict:
    split = rolling_steps(len(load_panel(pool)) // 1500)[step]
    result = normalized_agent_step(pool, split, steps)
    return {
        "unit": {"pool": pool, "step": step},
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "normalized_ppo": result,
        "provenance": {
            "git_sha": _git_sha(),
            "widths": list(WIDTHS),
            "execution_widths": list(EXECUTION_WIDTHS),
            "schedule": "event_driven",
            "steps": int(steps),
            "seeds": list(SEEDS),
            "normalization": NORMALIZATION,
        },
    }


def _load_results(root: Path) -> tuple[dict, list[tuple[str, int]]]:
    rows = {}
    for path in sorted(root.glob("*__step*.json")):
        row = json.loads(path.read_text())
        key = (row["unit"]["pool"], int(row["unit"]["step"]))
        if key in rows:
            raise ValueError(f"duplicate result for {key}")
        rows[key] = row
    expected = {(pool, step) for pool in POOLS for step in range(EXPECTED_WINDOWS)}
    extra = sorted(set(rows) - expected)
    if extra:
        raise ValueError(f"unexpected result units: {extra[:5]}")
    return rows, sorted(expected - set(rows))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path, standard_root: Path, report_dir: Path, *,
            seed: int = 20260728, samples: int = 10_000, block: int = 4) -> str:
    rows, missing = _load_results(root)
    if missing:
        return (
            f"PARTIAL - NOT A FINAL RESULT\n"
            f"{len(rows)}/48 units; missing {len(missing)}: {missing[:8]}\n"
        )
    standard = _read_algo(
        standard_root,
        "ppo",
        list(WIDTHS),
        "spacing",
        list(EXECUTION_WIDTHS),
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    window_rows, summary_rows = [], []
    lines = [
        "COMPLETE - FINAL COLLECTION",
        "Gated reward-normalized PPO diagnostic.",
        "One observation is one seed-averaged held-out window.",
    ]
    for pool_i, pool in enumerate(POOLS):
        normalized, reference, normalized_actions, standard_actions = [], [], [], []
        entropy = []
        for step in range(EXPECTED_WINDOWS):
            treatment = rows[(pool, step)]["normalized_ppo"]
            control = standard[(pool, step)]["arms"]["PPO"]
            normalized.append(float(treatment["test"]))
            reference.append(float(control["test"]))
            normalized_actions.append(int(treatment["n_actions"]))
            standard_actions.append(int(control["n_actions"]))
            entropy.append(float(treatment["selected_config"]["ent_coef"]))
            window_rows.append({
                "pool": pool,
                "step": step,
                "standard_reward": control["test"],
                "normalized_reward": treatment["test"],
                "difference": treatment["test"] - control["test"],
                "standard_n_actions": control["n_actions"],
                "normalized_n_actions": treatment["n_actions"],
                "selected_entropy": treatment["selected_config"]["ent_coef"],
                "selected_learning_rate":
                    treatment["selected_config"]["learning_rate"],
                "selected_net_arch":
                    json.dumps(treatment["selected_config"]["net_arch"]),
            })
        normalized = np.asarray(normalized)
        reference = np.asarray(reference)
        difference = normalized - reference
        lo, hi, p = paired_block_inference(
            difference,
            np.random.default_rng(seed + pool_i),
            samples,
            block,
        )
        action_difference = np.asarray(normalized_actions) - np.asarray(standard_actions)
        action_lo, action_hi = _bootstrap_ci(
            action_difference,
            np.random.default_rng(seed + 100 + pool_i),
            samples,
            block,
        )
        row = {
            "pool": pool,
            "standard_mean": float(reference.mean()),
            "normalized_mean": float(normalized.mean()),
            "difference": float(difference.mean()),
            "ci_low": lo,
            "ci_high": hi,
            "p_raw": p,
            "p_holm": p,  # one-member H4 family
            "wins": float((difference > 0).mean()),
            "standard_multi_action_windows":
                float((np.asarray(standard_actions) > 1).mean()),
            "normalized_multi_action_windows":
                float((np.asarray(normalized_actions) > 1).mean()),
            "mean_action_difference": float(action_difference.mean()),
            "action_difference_ci_low": action_lo,
            "action_difference_ci_high": action_hi,
            "entropy_001_selected": float((np.asarray(entropy) == 0.01).mean()),
        }
        summary_rows.append(row)
        lines.extend([
            "",
            f"POOL {pool}",
            (
                f"  standard {row['standard_mean']:,.0f}; "
                f"normalized {row['normalized_mean']:,.0f}"
            ),
            (
                f"  normalized-standard {row['difference']:+,.0f} "
                f"[{lo:+,.0f}, {hi:+,.0f}], p={p:.4f}"
            ),
            (
                f"  multi-action windows: standard "
                f"{row['standard_multi_action_windows']:.0%}, normalized "
                f"{row['normalized_multi_action_windows']:.0%}"
            ),
            f"  entropy 0.01 selected: {row['entropy_001_selected']:.0%}",
        ])

    _write_csv(report_dir / "windows.csv", window_rows)
    _write_csv(report_dir / "summary.csv", summary_rows)
    lines.extend([
        "",
        f"Bootstrap: seed={seed}, resamples={samples:,}, block={block}.",
        "H4 has one reward contrast per pool; Holm equals raw p.",
    ])
    text = "\n".join(lines) + "\n"
    (report_dir / "report.txt").write_text(text)
    return text


def expected_units(pools: list[str]) -> list[tuple[str, int]]:
    return [(pool, step) for pool in pools for step in range(EXPECTED_WINDOWS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pools", nargs="*", choices=POOLS, default=list(POOLS))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--standard-root", type=Path,
                        default=Path("outputs/action_geometry_v1"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-block", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        report_dir = args.report_dir or args.out / "aggregate_block4"
        print(analyze(
            args.out,
            args.standard_root,
            report_dir,
            seed=args.bootstrap_seed,
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
        result = run_unit(pool, step, args.steps)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=1))
        temp.rename(path)
        arm = result["normalized_ppo"]
        print(
            f"  {pool} step {step:03d} "
            f"test={arm['test']:,.0f} actions={arm['n_actions']} "
            f"cfg={arm['selected_config']}"
        )


if __name__ == "__main__":
    main()
