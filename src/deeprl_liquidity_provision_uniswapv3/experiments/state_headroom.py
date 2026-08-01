"""State-contingent control headroom under the corrected CLEB environment.

The learned CLEB policies use one action per pool. This module distinguishes a
rational collapse from a learning failure by comparing:

* a validation-selected constant event-driven action;
* a deployable one-split policy tree fitted to realized 168-hour action rewards;
* a validation-selected RL algorithm from the completed matched-geometry run; and
* a non-deployable local clairvoyant action diagnostic.

The full prospective specification is frozen in
``reports/STATE_HEADROOM_PROTOCOL.md``. One JSON is one held-out rolling window.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..data.pools import CORE
from ..envs.features import LEGACY_NAMES
from ..envs.schedule import EventDriven
from ..policies.baselines import Policy
from .action_geometry import ALGOS, LEARNED, _read_algo
from .bakeoff import build_env, load_panel, score_details
from .rolling import (
    _block_bootstrap_mean,
    holm_adjust,
    paired_block_inference,
    rolling_steps,
)

WIDTHS = (45.0, 50.0, 55.0)
EXECUTION_WIDTHS = (480.0, 540.0, 600.0)
PRIMARY_HORIZON = 168
ROBUSTNESS_HORIZON = 24
HORIZONS = (PRIMARY_HORIZON, ROBUSTNESS_HORIZON)
MIN_LEAVES = (20, 10, 5)  # simpler model wins an exact validation tie
TRAIN_STATE_SPACING = 24
EXPECTED_WINDOWS = 24
FAMILY = (
    "local_headroom_168h",
    "contextual_minus_fixed",
    "selected_rl_minus_contextual",
    "local_contextual_regret_168h",
)


class ConstantAction(Policy):
    """Return one action at every event-driven policy consultation."""

    def __init__(self, action: int):
        self.action = int(action)
        self.name = f"ConstantAction({self.action})"

    def __call__(self, obs, env) -> int:
        return self.action


@dataclass(frozen=True)
class RewardStump(Policy):
    """One-split deterministic policy tree, or a constant when feature is ``None``."""

    feature: int | None
    threshold: float | None
    left_action: int
    right_action: int
    min_leaf: int
    training_objective: float

    @property
    def name(self) -> str:
        if self.feature is None:
            return f"RewardStump(constant={self.left_action},min_leaf={self.min_leaf})"
        return (
            f"RewardStump({LEGACY_NAMES[self.feature]}<={self.threshold:.6g}"
            f"?{self.left_action}:{self.right_action},min_leaf={self.min_leaf})"
        )

    def __call__(self, obs, env) -> int:
        if self.feature is None:
            return self.left_action
        return self.left_action if float(obs[self.feature]) <= float(self.threshold) \
            else self.right_action


def _build_matched_env(pool: str, window: int, *, scheduled: bool):
    schedule = "event_driven" if scheduled else None
    return build_env(
        pool,
        window,
        WIDTHS,
        schedule=schedule,
        width_units="spacing",
        execution_widths=EXECUTION_WIDTHS,
    )


def _full_horizon(env, horizon: int) -> bool:
    """Whether ``horizon`` complete hourly transitions remain."""
    return env.i + int(horizon) <= env.n - 2


def action_returns(env, horizons: tuple[int, ...] = HORIZONS) -> dict[int, np.ndarray]:
    """Counterfactual true rewards for every action over fixed, complete horizons.

    The environment is shallow-copied: large immutable market and swap arrays are
    shared, while every scalar position state is isolated. The first transition
    applies the candidate action and every remaining transition holds.
    """
    requested = tuple(sorted({int(h) for h in horizons}))
    if not requested or requested[0] <= 0:
        raise ValueError("horizons must contain positive integers")
    if not _full_horizon(env, requested[-1]):
        raise ValueError(
            f"state {env.i} has no complete {requested[-1]}-hour horizon"
        )
    out = {h: np.empty(env.action_space.n, dtype=float) for h in requested}
    for action in range(env.action_space.n):
        branch = copy.copy(env)
        total = 0.0
        for elapsed in range(1, requested[-1] + 1):
            _, _, term, trunc, info = branch.step(action if elapsed == 1 else 0)
            total += float(info["reward_true"])
            if elapsed in out:
                out[elapsed][action] = total
            if term or trunc:
                if elapsed != requested[-1]:
                    raise RuntimeError("counterfactual terminated before a full horizon")
                break
    return out


def _trajectory_rows(pool: str, window: int, reference_action: int,
                     state_spacing: int = TRAIN_STATE_SPACING) -> tuple[np.ndarray, np.ndarray]:
    """Training states and 168-hour action returns from one reference trajectory."""
    wrapper = EventDriven(_build_matched_env(pool, window, scheduled=False))
    obs, _ = wrapper.reset()
    inner = wrapper.unwrapped
    x_rows, y_rows = [], []
    last_kept = -10**9
    done = trunc = False
    while not (done or trunc):
        if (_full_horizon(inner, PRIMARY_HORIZON)
                and inner.i - last_kept >= state_spacing):
            x_rows.append(np.asarray(obs, dtype=float).copy())
            y_rows.append(action_returns(inner, (PRIMARY_HORIZON,))[PRIMARY_HORIZON])
            last_kept = inner.i
        obs, _, done, trunc, _ = wrapper.step(reference_action)
    if not x_rows:
        raise RuntimeError(f"{pool} window {window} action {reference_action}: no states")
    return np.asarray(x_rows), np.asarray(y_rows)


@lru_cache(maxsize=None)
def training_window(pool: str, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counterfactual training data with equal weight per reference trajectory."""
    xs, ys, weights = [], [], []
    for action in range(4):
        x, y = _trajectory_rows(pool, window, action)
        xs.append(x)
        ys.append(y)
        weights.append(np.full(len(x), 1.0 / len(x)))
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    w_all = np.concatenate(weights)
    # Each (window, reference action) trajectory has total weight one. Scaling all
    # weights by a common constant cannot change the fitted stump.
    return x_all, y_all, w_all


def combine_training(pool: str, windows: list[int]) -> tuple[np.ndarray, np.ndarray,
                                                              np.ndarray]:
    parts = [training_window(pool, window) for window in windows]
    return (
        np.concatenate([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
        np.concatenate([part[2] for part in parts]),
    )


def fit_reward_stump(x: np.ndarray, action_rewards: np.ndarray, weights: np.ndarray,
                     min_leaf: int) -> RewardStump:
    """Fit the exact one-split tree maximizing weighted realized action reward."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(action_rewards, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if x.ndim != 2 or y.shape != (len(x), 4) or weights.shape != (len(x),):
        raise ValueError("expected x=(n,p), action_rewards=(n,4), weights=(n,)")
    if len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("training data must be non-empty and finite")
    if (weights <= 0).any() or not np.isfinite(weights).all():
        raise ValueError("weights must be positive and finite")

    weighted = weights[:, None] * y
    totals = weighted.sum(axis=0)
    constant_action = int(np.argmax(totals))
    best_score = float(totals[constant_action])
    best = RewardStump(
        feature=None,
        threshold=None,
        left_action=constant_action,
        right_action=constant_action,
        min_leaf=int(min_leaf),
        training_objective=best_score / float(weights.sum()),
    )
    if len(x) < 2 * min_leaf:
        return best

    for feature in range(x.shape[1]):
        order = np.argsort(x[:, feature], kind="stable")
        values = x[order, feature]
        cumulative = np.cumsum(weighted[order], axis=0)
        for split in range(min_leaf, len(x) - min_leaf + 1):
            if values[split - 1] == values[split]:
                continue
            left = cumulative[split - 1]
            right = totals - left
            left_action = int(np.argmax(left))
            right_action = int(np.argmax(right))
            score = float(left[left_action] + right[right_action])
            # Strict improvement keeps deterministic, conservative tie handling.
            if score <= best_score + 1e-12:
                continue
            threshold = float(values[split - 1]
                              + (values[split] - values[split - 1]) / 2.0)
            best_score = score
            best = RewardStump(
                feature=feature,
                threshold=threshold,
                left_action=left_action,
                right_action=right_action,
                min_leaf=int(min_leaf),
                training_objective=score / float(weights.sum()),
            )
    return best


def score_policy(pool: str, window: int, policy: Policy) -> dict:
    return score_details(_build_matched_env(pool, window, scheduled=True), policy)


def select_fixed(pool: str, val_window: int) -> tuple[int, list[float]]:
    scores = [
        score_policy(pool, val_window, ConstantAction(action))["reward"]
        for action in range(4)
    ]
    return int(np.argmax(scores)), [float(score) for score in scores]


def fit_contextual(pool: str, train: list[int], val: int) -> tuple[RewardStump,
                                                                    list[dict]]:
    x, y, weights = combine_training(pool, train)
    candidates = []
    for min_leaf in MIN_LEAVES:
        stump = fit_reward_stump(x, y, weights, min_leaf)
        reward = score_policy(pool, val, stump)["reward"]
        candidates.append({"min_leaf": min_leaf, "validation_reward": float(reward),
                           "stump": stump})
    # MIN_LEAVES is ordered simple -> flexible, and argmax preserves the first tie.
    selected = candidates[int(np.argmax([row["validation_reward"] for row in candidates]))]
    x, y, weights = combine_training(pool, train + [val])
    refit = fit_reward_stump(x, y, weights, int(selected["min_leaf"]))
    return refit, candidates


def action_disagrees(q: np.ndarray, fixed_action: int) -> bool:
    """Whether ``fixed_action`` is outside the exact numerical argmax set."""
    q = np.asarray(q, dtype=float)
    optimum = float(np.max(q))
    best = np.isclose(q, optimum, rtol=0.0, atol=1e-9)
    return not bool(best[int(fixed_action)])


def local_diagnostics(pool: str, window: int, fixed_action: int,
                      contextual: RewardStump) -> dict[str, dict]:
    """Local oracle diagnostics on the fixed policy's held-out state trajectory."""
    wrapper = EventDriven(_build_matched_env(pool, window, scheduled=False))
    obs, _ = wrapper.reset()
    inner = wrapper.unwrapped
    values = {
        horizon: {"headroom": [], "regret": [], "disagreement": []}
        for horizon in HORIZONS
    }
    done = trunc = False
    while not (done or trunc):
        if _full_horizon(inner, PRIMARY_HORIZON):
            returns = action_returns(inner, HORIZONS)
            contextual_action = int(contextual(obs, inner))
            for horizon, q in returns.items():
                optimum = float(np.max(q))
                values[horizon]["headroom"].append(optimum - float(q[fixed_action]))
                values[horizon]["regret"].append(
                    optimum - float(q[contextual_action])
                )
                values[horizon]["disagreement"].append(
                    action_disagrees(q, fixed_action)
                )
        obs, _, done, trunc, _ = wrapper.step(fixed_action)

    out = {}
    for horizon, rows in values.items():
        n = len(rows["headroom"])
        if n == 0:
            raise RuntimeError(f"{pool} window {window}: no full-horizon test states")
        disagreements = int(np.count_nonzero(rows["disagreement"]))
        out[str(horizon)] = {
            "n_states": n,
            "n_disagreements": disagreements,
            "disagreement_fraction": disagreements / n,
            "mean_headroom": float(np.mean(rows["headroom"])),
            "mean_contextual_regret": float(np.mean(rows["regret"])),
        }
    return out


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_unit(pool: str, step: int) -> dict:
    split = rolling_steps(len(load_panel(pool)) // 1500)[step]
    fixed_action, fixed_val_scores = select_fixed(pool, split.val[0])
    contextual, candidates = fit_contextual(pool, split.train, split.val[0])
    fixed_test = score_policy(pool, split.test[0], ConstantAction(fixed_action))
    contextual_test = score_policy(pool, split.test[0], contextual)
    diagnostics = local_diagnostics(
        pool, split.test[0], fixed_action, contextual
    )
    return {
        "unit": {"pool": pool, "step": step},
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "fixed": {
            "action": fixed_action,
            "validation_rewards": fixed_val_scores,
            **fixed_test,
        },
        "contextual": {
            "selected_min_leaf": contextual.min_leaf,
            "validation_candidates": [
                {
                    "min_leaf": row["min_leaf"],
                    "validation_reward": row["validation_reward"],
                    "stump": asdict(row["stump"]),
                }
                for row in candidates
            ],
            "stump": asdict(contextual),
            **contextual_test,
        },
        "diagnostics": diagnostics,
        "provenance": {
            "git_sha": _git_sha(),
            "protocol": "reports/STATE_HEADROOM_PROTOCOL.md",
            "widths": list(WIDTHS),
            "execution_widths": list(EXECUTION_WIDTHS),
            "horizons": list(HORIZONS),
            "min_leaves": list(MIN_LEAVES),
            "train_state_spacing": TRAIN_STATE_SPACING,
        },
    }


def expected_units(pools: list[str]) -> list[tuple[str, int]]:
    return [
        (pool, step)
        for pool in sorted(pools)
        for step in range(len(rolling_steps(len(load_panel(pool)) // 1500)))
    ]


def _load_results(root: Path) -> tuple[dict, list[tuple[str, int]]]:
    rows = {}
    for path in sorted(root.glob("*__step*.json")):
        row = json.loads(path.read_text())
        key = (row["unit"]["pool"], int(row["unit"]["step"]))
        if key in rows:
            raise ValueError(f"duplicate result for {key}")
        rows[key] = row
    expected = {(pool, step) for pool in CORE for step in range(EXPECTED_WINDOWS)}
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    if extra:
        raise ValueError(f"unexpected units: {extra[:5]}")
    return rows, missing


def _load_matched_rl(root: Path) -> dict:
    return {
        algo: _read_algo(
            root, algo, list(WIDTHS), "spacing", list(EXECUTION_WIDTHS)
        )
        for algo in ALGOS
    }


def _selected_rl(rl: dict, pool: str, step: int) -> tuple[str, float, float]:
    choices = []
    for algo in ALGOS:
        label = LEARNED[algo]
        arm = rl[algo][(pool, step)]["arms"][label]
        choices.append((label, float(arm["val"]), float(arm["test"])))
    # ALGOS is the prospectively fixed tie order.
    return max(choices, key=lambda row: row[1])


def _bootstrap_ci(x: np.ndarray, rng: np.random.Generator, samples: int,
                  block: int) -> tuple[float, float]:
    boot = _block_bootstrap_mean(np.asarray(x, dtype=float), rng, samples, block)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(result_root: Path, rl_root: Path, report_dir: Path, *,
            seed: int = 20260728, samples: int = 10_000, block: int = 4) -> str:
    """Aggregate only a complete collection, with the frozen H3 Holm family."""
    rows, missing = _load_results(result_root)
    if missing:
        return (
            f"PARTIAL - NOT A FINAL RESULT\n"
            f"{len(rows)}/144 units; missing {len(missing)}: {missing[:8]}\n"
        )
    rl = _load_matched_rl(rl_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    window_rows, headline_rows, contrast_rows = [], [], []
    lines = [
        "COMPLETE - FINAL COLLECTION",
        "State-contingent headroom; one observation is one held-out window.",
        "H3 Holm family: four prospectively specified contrasts within each pool.",
    ]

    for pool_i, pool in enumerate(CORE):
        arrays = {name: [] for name in FAMILY}
        disagreement = {horizon: [] for horizon in HORIZONS}
        pooled_counts = {
            horizon: {"states": 0, "disagreements": 0}
            for horizon in HORIZONS
        }
        selected_algorithms = []
        for step in range(EXPECTED_WINDOWS):
            row = rows[(pool, step)]
            fixed = float(row["fixed"]["reward"])
            contextual = float(row["contextual"]["reward"])
            rl_name, rl_val, rl_test = _selected_rl(rl, pool, step)
            selected_algorithms.append(rl_name)
            d168 = row["diagnostics"][str(PRIMARY_HORIZON)]
            arrays["local_headroom_168h"].append(d168["mean_headroom"])
            arrays["contextual_minus_fixed"].append(contextual - fixed)
            arrays["selected_rl_minus_contextual"].append(rl_test - contextual)
            arrays["local_contextual_regret_168h"].append(
                d168["mean_contextual_regret"]
            )
            for horizon in HORIZONS:
                d = row["diagnostics"][str(horizon)]
                disagreement[horizon].append(d["disagreement_fraction"])
                pooled_counts[horizon]["states"] += d["n_states"]
                pooled_counts[horizon]["disagreements"] += d["n_disagreements"]
            window_rows.append({
                "pool": pool,
                "step": step,
                "fixed_action": row["fixed"]["action"],
                "fixed_reward": fixed,
                "contextual_reward": contextual,
                "contextual_minus_fixed": contextual - fixed,
                "contextual_min_leaf": row["contextual"]["selected_min_leaf"],
                "contextual_feature": row["contextual"]["stump"]["feature"],
                "selected_rl": rl_name,
                "selected_rl_validation": rl_val,
                "selected_rl_reward": rl_test,
                "selected_rl_minus_contextual": rl_test - contextual,
                "disagreement_168h": d168["disagreement_fraction"],
                "headroom_168h": d168["mean_headroom"],
                "regret_168h": d168["mean_contextual_regret"],
                "disagreement_24h":
                    row["diagnostics"][str(ROBUSTNESS_HORIZON)][
                        "disagreement_fraction"
                    ],
                "headroom_24h":
                    row["diagnostics"][str(ROBUSTNESS_HORIZON)]["mean_headroom"],
                "regret_24h":
                    row["diagnostics"][str(ROBUSTNESS_HORIZON)][
                        "mean_contextual_regret"
                    ],
            })

        raw_contrasts = []
        for contrast_i, name in enumerate(FAMILY):
            x = np.asarray(arrays[name], dtype=float)
            rng = np.random.default_rng(seed + 1000 * pool_i + contrast_i)
            lo, hi, p = paired_block_inference(x, rng, samples, block)
            raw_contrasts.append({
                "pool": pool,
                "family": "H3 state-contingency",
                "contrast": name,
                "difference": float(x.mean()),
                "wins": float((x > 0).mean()),
                "ci_low": lo,
                "ci_high": hi,
                "p_raw": p,
            })
        adjusted = holm_adjust([row["p_raw"] for row in raw_contrasts])
        for row, p_holm in zip(raw_contrasts, adjusted):
            row["p_holm"] = p_holm
            contrast_rows.append(row)

        ci = {}
        for horizon_i, horizon in enumerate(HORIZONS):
            x = np.asarray(disagreement[horizon], dtype=float)
            ci[horizon] = _bootstrap_ci(
                x,
                np.random.default_rng(seed + 1000 * pool_i + 100 + horizon_i),
                samples,
                block,
            )
        context = next(
            row for row in raw_contrasts
            if row["contrast"] == "contextual_minus_fixed"
        )
        primary_fraction = float(np.mean(disagreement[PRIMARY_HORIZON]))
        robust_fraction = float(np.mean(disagreement[ROBUSTNESS_HORIZON]))
        no_headroom = (
            ci[PRIMARY_HORIZON][1] < 0.10
            and ci[ROBUSTNESS_HORIZON][1] < 0.10
        )
        headline_rows.append({
            "pool": pool,
            "windows": EXPECTED_WINDOWS,
            "disagreement_168h": primary_fraction,
            "disagreement_168h_ci_low": ci[PRIMARY_HORIZON][0],
            "disagreement_168h_ci_high": ci[PRIMARY_HORIZON][1],
            "disagreement_168h_pooled":
                pooled_counts[PRIMARY_HORIZON]["disagreements"]
                / pooled_counts[PRIMARY_HORIZON]["states"],
            "states_168h": pooled_counts[PRIMARY_HORIZON]["states"],
            "disagreement_24h": robust_fraction,
            "disagreement_24h_ci_low": ci[ROBUSTNESS_HORIZON][0],
            "disagreement_24h_ci_high": ci[ROBUSTNESS_HORIZON][1],
            "disagreement_24h_pooled":
                pooled_counts[ROBUSTNESS_HORIZON]["disagreements"]
                / pooled_counts[ROBUSTNESS_HORIZON]["states"],
            "states_24h": pooled_counts[ROBUSTNESS_HORIZON]["states"],
            "no_headroom_both_horizons": no_headroom,
            "contextual_minus_fixed": context["difference"],
            "contextual_ci_low": context["ci_low"],
            "contextual_ci_high": context["ci_high"],
            "contextual_p_holm": context["p_holm"],
            "contextual_wins": context["wins"],
        })

        counts = {name: selected_algorithms.count(name) for name in LEARNED.values()}
        lines.extend([
            "",
            f"POOL {pool}",
            (
                f"  168h disagreement {primary_fraction:.1%} "
                f"[{ci[PRIMARY_HORIZON][0]:.1%}, {ci[PRIMARY_HORIZON][1]:.1%}]"
            ),
            (
                f"   24h disagreement {robust_fraction:.1%} "
                f"[{ci[ROBUSTNESS_HORIZON][0]:.1%}, "
                f"{ci[ROBUSTNESS_HORIZON][1]:.1%}]"
            ),
            (
                f"  contextual-fixed {context['difference']:+,.0f} "
                f"[{context['ci_low']:+,.0f}, {context['ci_high']:+,.0f}], "
                f"Holm p={context['p_holm']:.4f}"
            ),
            f"  no-headroom criterion at both horizons: {no_headroom}",
            f"  validation-selected RL counts: {counts}",
        ])

    _write_csv(report_dir / "windows.csv", window_rows)
    _write_csv(report_dir / "headline.csv", headline_rows)
    _write_csv(report_dir / "contrasts.csv", contrast_rows)
    lines.extend([
        "",
        f"Bootstrap: seed={seed}, resamples={samples:,}, block={block}.",
        "Pools are not combined. The 24h diagnostics are robustness estimates, "
        "not an additional null-hypothesis family.",
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
    parser.add_argument("--rl-root", type=Path, default=Path("outputs/action_geometry_v1"))
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
            args.rl_root,
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
        result = run_unit(pool, step)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=1))
        temp.rename(path)
        print(
            f"  {pool} step {step:03d} "
            f"fixed={result['fixed']['reward']:,.0f} "
            f"context={result['contextual']['reward']:,.0f} "
            f"disagree168="
            f"{result['diagnostics'][str(PRIMARY_HORIZON)]['disagreement_fraction']:.1%}"
        )
    print(f"collection sha256 {collection_hash(args.out)}")


if __name__ == "__main__":
    main()
