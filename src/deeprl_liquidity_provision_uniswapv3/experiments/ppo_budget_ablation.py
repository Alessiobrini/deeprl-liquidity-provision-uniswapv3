"""Validation-only PPO training-budget ablation on a matched fee-tier pair.

The treatment changes only the number of PPO environment steps: 20k, 100k, or
500k.  Every budget independently selects the same eight-config PPO grid on the
validation window, refits the selected config on train plus validation, and reads
the held-out test window once.  Both refit seeds are retained as behavior records,
but their rewards are averaged within a window before inference.

One work unit is one (budget, pool, rolling step).  Final refit checkpoints and a
self-describing JSON are written per unit.  The two pools are the matched-geometry
USDC/WETH 0.05% and 0.30% pools.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..agents.registry import make_agent
from .agent_arm import score_agent, train_env
from .bakeoff import Split, build_env, load_panel
from .rolling import grid_for, holm_adjust, paired_block_inference, rolling_steps

POOLS = ("usdc_weth_005", "usdc_weth_030")
BUDGETS = (20_000, 100_000, 500_000)
SEEDS = (42, 123)
EXPECTED_WINDOWS = 24
SCHEDULE = "event_driven"
WIDTHS = (45.0, 50.0, 55.0)
EXECUTION_WIDTHS = (480.0, 540.0, 600.0)
ACTION_LABELS = ("hold", "width_45", "width_50", "width_55")


@dataclass(frozen=True)
class BudgetUnit:
    pool: str
    step: int
    budget: int

    @property
    def name(self) -> str:
        return f"{self.pool}__step{self.step:03d}__budget{self.budget:06d}"


def expected_units(
    pools: tuple[str, ...] | list[str] = POOLS,
    budgets: tuple[int, ...] | list[int] = BUDGETS,
) -> list[BudgetUnit]:
    """Deterministic order shared by local shards and the Slurm array."""
    return [
        BudgetUnit(pool, step, int(budget))
        for budget in budgets
        for pool in pools
        for step in range(EXPECTED_WINDOWS)
    ]


def _matched_train_env(pool: str, split: Split):
    return train_env(
        pool,
        WIDTHS,
        split,
        SCHEDULE,
        width_units="spacing",
        execution_widths=EXECUTION_WIDTHS,
    )


def select_config(pool: str, split: Split, budget: int) -> tuple[dict, float, list[dict]]:
    """Select only on validation; the test indices are never passed into this loop."""
    search, best_config, best_validation = [], None, -np.inf
    for config in grid_for("ppo"):
        seed_values = []
        for seed in SEEDS:
            env = _matched_train_env(pool, split)
            try:
                model = make_agent("ppo", env, seed=seed, **config)
                model.learn(total_timesteps=budget)
                values = score_agent(
                    pool,
                    model,
                    WIDTHS,
                    split.val,
                    SCHEDULE,
                    width_units="spacing",
                    execution_widths=EXECUTION_WIDTHS,
                )
                seed_values.append(float(values.mean()))
            finally:
                env.close()
        validation = float(np.mean(seed_values))
        search.append({
            "config": config,
            "seed_validation": [
                {"seed": seed, "reward": value}
                for seed, value in zip(SEEDS, seed_values)
            ],
            "mean_validation": validation,
        })
        # grid_for is prespecified and puts the smaller network first. Retaining the
        # first exact tie makes selection deterministic without reading test.
        if validation > best_validation:
            best_config, best_validation = config, validation
    if best_config is None:  # pragma: no cover - the shipped PPO grid is nonempty
        raise RuntimeError("PPO configuration grid is empty")
    return best_config, best_validation, search


def _policy_probabilities(model, observation) -> np.ndarray:
    """Categorical PPO probabilities for one observation."""
    model.policy.set_training_mode(False)
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        probabilities = distribution.distribution.probs
    out = probabilities.detach().cpu().numpy().reshape(-1)
    if out.shape != (len(ACTION_LABELS),):
        raise ValueError(
            f"expected {len(ACTION_LABELS)} action probabilities, got {out.shape}"
        )
    return out.astype(float)


def evaluate_behavior(pool: str, window: int, model) -> dict:
    """Score one deterministic raw-dollar test trace and retain behavior diagnostics."""
    env = build_env(
        pool,
        window,
        WIDTHS,
        schedule=SCHEDULE,
        width_units="spacing",
        execution_widths=EXECUTION_WIDTHS,
    )
    if env is None:  # pragma: no cover - complete rolling windows are prechecked
        raise ValueError(f"pool {pool} window {window} is incomplete")
    try:
        observation, _ = env.reset()
        total, terminated, truncated = 0.0, False, False
        actions, probability_rows, entropies = [], [], []
        info = {}
        while not (terminated or truncated):
            probabilities = _policy_probabilities(model, observation)
            action = int(np.argmax(probabilities))
            actions.append(action)
            probability_rows.append(probabilities)
            positive = probabilities[probabilities > 0]
            entropies.append(float(-(positive * np.log(positive)).sum()))
            observation, _, terminated, truncated, info = env.step(action)
            total += float(info["reward_true"])
    finally:
        env.close()

    matrix = np.asarray(probability_rows, dtype=float)
    mean_probability = matrix.mean(axis=0)
    # Mean total-variation distance from the trace's mean categorical policy. It is
    # zero for a state-invariant stochastic policy and bounded by one.
    state_dependence = float(
        np.mean(0.5 * np.abs(matrix - mean_probability).sum(axis=1))
    )
    histogram = np.bincount(actions, minlength=len(ACTION_LABELS))
    return {
        "reward": total,
        "n_decisions": len(actions),
        "n_actions": int(np.count_nonzero(histogram)),
        "action_histogram": {
            label: int(histogram[index])
            for index, label in enumerate(ACTION_LABELS)
        },
        "n_rebalances": int(info.get("n_rebalances", 0)),
        "mean_policy_entropy": float(np.mean(entropies)),
        "normalized_policy_entropy": float(np.mean(entropies) / math.log(len(ACTION_LABELS))),
        "mean_action_probability": {
            label: float(mean_probability[index])
            for index, label in enumerate(ACTION_LABELS)
        },
        "std_action_probability": {
            label: float(matrix[:, index].std())
            for index, label in enumerate(ACTION_LABELS)
        },
        "state_dependence_tv": state_dependence,
        "max_probability_range": float(np.ptp(matrix, axis=0).max()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip())
    except Exception:  # pragma: no cover - git is available in production runs
        sha, dirty = "unknown", True
    return {
        "git_sha": sha or "unknown",
        "git_dirty": dirty,
        "host": platform.node(),
        "python": platform.python_version(),
    }


def run_unit(unit: BudgetUnit, out: Path) -> dict:
    split = rolling_steps(len(load_panel(unit.pool)) // 1500)[unit.step]
    selected, validation, search = select_config(unit.pool, split, unit.budget)
    fit = Split(train=split.train + split.val, val=[], test=split.test)
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed in SEEDS:
        env = _matched_train_env(unit.pool, fit)
        try:
            model = make_agent("ppo", env, seed=seed, **selected)
            model.learn(total_timesteps=unit.budget)
            checkpoint = checkpoint_dir / f"{unit.name}__seed{seed}.zip"
            model.save(checkpoint)
            behavior = evaluate_behavior(unit.pool, split.test[0], model)
        finally:
            env.close()
        per_seed.append({
            "seed": seed,
            "checkpoint": str(checkpoint.relative_to(out)),
            "checkpoint_sha256": _sha256(checkpoint),
            **behavior,
        })
    return {
        "unit": asdict(unit),
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "ppo": {
            "test": float(np.mean([row["reward"] for row in per_seed])),
            "validation": validation,
            "selected_config": selected,
            "validation_search": search,
            "per_seed": per_seed,
        },
        "provenance": {
            **_git_provenance(),
            "widths": list(WIDTHS),
            "execution_widths": list(EXECUTION_WIDTHS),
            "width_units": "spacing",
            "schedule": SCHEDULE,
            "budget": unit.budget,
            "seeds": list(SEEDS),
            "selection": "mean validation reward; first prespecified config wins exact tie",
            "test_reads": "one deterministic trace per refit seed",
        },
    }


def _result_is_complete(path: Path, out: Path) -> bool:
    try:
        row = json.loads(path.read_text())
        checkpoints = [out / seed["checkpoint"] for seed in row["ppo"]["per_seed"]]
        return len(checkpoints) == len(SEEDS) and all(item.is_file() for item in checkpoints)
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _load_results(root: Path) -> tuple[dict[tuple[str, int, int], dict], list[BudgetUnit]]:
    rows = {}
    for path in sorted(root.glob("*__budget*.json")):
        row = json.loads(path.read_text())
        unit = row["unit"]
        key = (unit["pool"], int(unit["step"]), int(unit["budget"]))
        if key in rows:
            raise ValueError(f"duplicate result for {key}")
        rows[key] = row
    expected = {(u.pool, u.step, u.budget): u for u in expected_units()}
    extra = sorted(set(rows) - set(expected))
    if extra:
        raise ValueError(f"unexpected result units: {extra[:5]}")
    missing = [expected[key] for key in expected if key not in rows]
    return rows, missing


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    root: Path,
    report_dir: Path,
    *,
    seed: int = 20260802,
    samples: int = 10_000,
    block: int = 4,
) -> str:
    """Aggregate only a complete collection; incomplete collections are labeled."""
    rows, missing = _load_results(root)
    if missing:
        return (
            "PARTIAL - NOT A FINAL RESULT\n"
            f"{len(rows)}/{len(expected_units())} units; missing {len(missing)}: "
            f"{[unit.name for unit in missing[:8]]}\n"
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    windows, behaviors, searches, summaries = [], [], [], []
    lines = [
        "COMPLETE - FINAL COLLECTION",
        "Matched-geometry PPO training-budget ablation.",
        "One inferential observation is one two-seed-averaged held-out window.",
    ]
    for pool_index, pool in enumerate(POOLS):
        rewards = {}
        for budget in BUDGETS:
            rewards[budget] = np.asarray([
                rows[(pool, step, budget)]["ppo"]["test"]
                for step in range(EXPECTED_WINDOWS)
            ], dtype=float)

        raw_comparisons = []
        for budget_index, budget in enumerate(BUDGETS[1:]):
            difference = rewards[budget] - rewards[BUDGETS[0]]
            lo, hi, p = paired_block_inference(
                difference,
                np.random.default_rng(seed + 100 * pool_index + budget_index),
                samples,
                block,
            )
            raw_comparisons.append((budget, difference, lo, hi, p))
        adjusted = holm_adjust([item[4] for item in raw_comparisons])
        inference = {
            budget: (difference, lo, hi, p, p_holm)
            for (budget, difference, lo, hi, p), p_holm
            in zip(raw_comparisons, adjusted)
        }

        lines.extend(["", f"POOL {pool}"])
        for budget in BUDGETS:
            seed_traces = [
                seed_row
                for step in range(EXPECTED_WINDOWS)
                for seed_row in rows[(pool, step, budget)]["ppo"]["per_seed"]
            ]
            selected_validation = np.asarray([
                rows[(pool, step, budget)]["ppo"]["validation"]
                for step in range(EXPECTED_WINDOWS)
            ], dtype=float)
            summary = {
                "pool": pool,
                "budget": budget,
                "mean_test_reward": float(rewards[budget].mean()),
                "mean_selected_validation_reward": float(selected_validation.mean()),
                "difference_vs_20000": 0.0,
                "ci_low": "",
                "ci_high": "",
                "p_raw": "",
                "p_holm": "",
                "wins_vs_20000": "",
                "multi_action_seed_traces": float(np.mean([
                    item["n_actions"] > 1 for item in seed_traces
                ])),
                "mean_rebalances": float(np.mean([
                    item["n_rebalances"] for item in seed_traces
                ])),
                "mean_policy_entropy": float(np.mean([
                    item["mean_policy_entropy"] for item in seed_traces
                ])),
                "mean_normalized_policy_entropy": float(np.mean([
                    item["normalized_policy_entropy"] for item in seed_traces
                ])),
                "mean_state_dependence_tv": float(np.mean([
                    item["state_dependence_tv"] for item in seed_traces
                ])),
                "max_state_dependence_tv": float(np.max([
                    item["state_dependence_tv"] for item in seed_traces
                ])),
                "max_probability_range": float(np.max([
                    item["max_probability_range"] for item in seed_traces
                ])),
            }
            for label in ACTION_LABELS:
                summary[f"single_action_traces_{label}"] = sum(
                    item["n_actions"] == 1 and item["action_histogram"][label] > 0
                    for item in seed_traces
                )
            if budget != BUDGETS[0]:
                difference, lo, hi, p, p_holm = inference[budget]
                summary.update({
                    "difference_vs_20000": float(difference.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_raw": p,
                    "p_holm": p_holm,
                    "wins_vs_20000": float((difference > 0).mean()),
                })
                comparison = (
                    f"; vs 20k {difference.mean():+,.0f} "
                    f"[{lo:+,.0f}, {hi:+,.0f}], p_Holm={p_holm:.4f}"
                )
            else:
                comparison = ""
            summaries.append(summary)
            lines.append(
                f"  {budget:>6,d} steps: test {summary['mean_test_reward']:>,.0f}; "
                f"multi-action seed traces {summary['multi_action_seed_traces']:.0%}; "
                f"state-TV {summary['mean_state_dependence_tv']:.4f}{comparison}"
            )

            for step in range(EXPECTED_WINDOWS):
                result = rows[(pool, step, budget)]["ppo"]
                per_seed = result["per_seed"]
                for item in per_seed:
                    behavior = {
                        "pool": pool,
                        "step": step,
                        "budget": budget,
                        "seed": item["seed"],
                        "reward": item["reward"],
                        "n_decisions": item["n_decisions"],
                        "n_actions": item["n_actions"],
                        "n_rebalances": item["n_rebalances"],
                        "mean_policy_entropy": item["mean_policy_entropy"],
                        "normalized_policy_entropy": item["normalized_policy_entropy"],
                        "state_dependence_tv": item["state_dependence_tv"],
                        "max_probability_range": item["max_probability_range"],
                    }
                    for label in ACTION_LABELS:
                        behavior[f"count_{label}"] = item["action_histogram"][label]
                        behavior[f"mean_probability_{label}"] = (
                            item["mean_action_probability"][label]
                        )
                        behavior[f"std_probability_{label}"] = (
                            item["std_action_probability"][label]
                        )
                    behaviors.append(behavior)
                windows.append({
                    "pool": pool,
                    "step": step,
                    "budget": budget,
                    "test_reward_seed_mean": result["test"],
                    "validation_reward_seed_mean": result["validation"],
                    "seed_42_reward": per_seed[0]["reward"],
                    "seed_123_reward": per_seed[1]["reward"],
                    "seed_42_n_actions": per_seed[0]["n_actions"],
                    "seed_123_n_actions": per_seed[1]["n_actions"],
                    "seed_42_rebalances": per_seed[0]["n_rebalances"],
                    "seed_123_rebalances": per_seed[1]["n_rebalances"],
                    "mean_policy_entropy": float(np.mean([
                        item["mean_policy_entropy"] for item in per_seed
                    ])),
                    "mean_state_dependence_tv": float(np.mean([
                        item["state_dependence_tv"] for item in per_seed
                    ])),
                    "selected_learning_rate": result["selected_config"]["learning_rate"],
                    "selected_entropy": result["selected_config"]["ent_coef"],
                    "selected_net_arch": json.dumps(result["selected_config"]["net_arch"]),
                })
                for candidate_index, candidate in enumerate(result["validation_search"]):
                    searches.append({
                        "pool": pool,
                        "step": step,
                        "budget": budget,
                        "candidate_index": candidate_index,
                        "learning_rate": candidate["config"]["learning_rate"],
                        "entropy": candidate["config"]["ent_coef"],
                        "net_arch": json.dumps(candidate["config"]["net_arch"]),
                        "seed_42_validation": candidate["seed_validation"][0]["reward"],
                        "seed_123_validation": candidate["seed_validation"][1]["reward"],
                        "mean_validation": candidate["mean_validation"],
                        "selected": candidate["config"] == result["selected_config"],
                    })

    _write_csv(report_dir / "windows.csv", windows)
    _write_csv(report_dir / "behavior.csv", behaviors)
    _write_csv(report_dir / "validation_search.csv", searches)
    _write_csv(report_dir / "summary.csv", summaries)
    lines.extend([
        "",
        f"Bootstrap: seed={seed}, resamples={samples:,}, block={block}.",
        "Holm correction is across the 100k-vs-20k and 500k-vs-20k contrasts "
        "within each pool.",
    ])
    text = "\n".join(lines) + "\n"
    (report_dir / "report.txt").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pools", nargs="*", choices=POOLS, default=list(POOLS))
    parser.add_argument("--budgets", nargs="*", type=int, choices=BUDGETS,
                        default=list(BUDGETS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-block", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        report_dir = args.report_dir or args.out / "aggregate_block4"
        print(analyze(
            args.out,
            report_dir,
            seed=args.bootstrap_seed,
            samples=args.bootstrap_samples,
            block=args.bootstrap_block,
        ), end="")
        return

    if not 0 <= args.shard < args.of:
        parser.error(f"shard {args.shard} is outside 0..{args.of - 1}")
    units = expected_units(args.pools, args.budgets)
    mine = units[args.shard::args.of]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard}/{args.of}: {len(mine)} of {len(units)} units")
    for unit in mine:
        path = args.out / f"{unit.name}.json"
        if path.exists() and not args.force and _result_is_complete(path, args.out):
            print(f"  {unit.name} skip (complete)")
            continue
        result = run_unit(unit, args.out)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=1) + "\n")
        temporary.replace(path)
        print(
            f"  {unit.name} test={result['ppo']['test']:,.0f} "
            f"config={result['ppo']['selected_config']}"
        )


if __name__ == "__main__":
    main()
