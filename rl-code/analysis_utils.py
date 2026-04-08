"""
Shared analysis helpers for paper-ready summaries and plots.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from statistics import NormalDist


STRATEGY_DISPLAY_NAMES = {
    "PPO": "PPO",
    "PassiveWidthSweep": "Passive Width",
    "VolProportionalWidth": "Vol-Proportional",
    "ILMinimizer": "IL Minimizer",
    "ReactiveRecentering": "Reactive Recentering",
}


STRATEGY_COLORS = {
    "PPO": "#c0392b",
    "PassiveWidthSweep": "#2980b9",
    "VolProportionalWidth": "#27ae60",
    "ILMinimizer": "#8e44ad",
    "ReactiveRecentering": "#f39c12",
}


def _normal_approx_two_sided_pvalue(statistic: float) -> float:
    return 2 * (1 - NormalDist().cdf(abs(statistic)))


def _paired_ttest(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(diffs)
    if n < 2:
        return np.nan, np.nan

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    if math.isclose(std_diff, 0.0):
        return np.nan, np.nan

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    return t_stat, _normal_approx_two_sided_pvalue(t_stat)


def _welch_ttest(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan

    x_var = np.var(x, ddof=1)
    y_var = np.var(y, ddof=1)
    denom = math.sqrt((x_var / len(x)) + (y_var / len(y)))
    if math.isclose(denom, 0.0):
        return np.nan, np.nan

    t_stat = (np.mean(x) - np.mean(y)) / denom
    return float(t_stat), _normal_approx_two_sided_pvalue(float(t_stat))


def get_display_name(strategy: str) -> str:
    return STRATEGY_DISPLAY_NAMES.get(strategy, strategy)


def get_color(strategy: str) -> str:
    return STRATEGY_COLORS.get(strategy, "#34495e")


def compute_strategy_statistics(
    detailed_results: pd.DataFrame,
    baseline_key: str = "PPO",
    n_bootstrap: int = 2000,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Recompute summary statistics directly from saved per-evaluation results.

    We use paired t-tests whenever the compared series have equal length.
    This is the natural choice for these saved outputs because each PPO run is
    evaluated against the same window-level deterministic baseline result.
    """
    rng = np.random.default_rng(seed)

    if baseline_key not in detailed_results.columns:
        raise KeyError(f"Missing baseline column: {baseline_key}")

    baseline = detailed_results[baseline_key].dropna().to_numpy(dtype=float)
    rows = []

    for strategy in detailed_results.columns:
        rewards = detailed_results[strategy].dropna().to_numpy(dtype=float)

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else np.nan
        median_reward = float(np.median(rewards))
        q1 = float(np.percentile(rewards, 25))
        q3 = float(np.percentile(rewards, 75))
        iqr = q3 - q1
        min_reward = float(np.min(rewards))
        max_reward = float(np.max(rewards))
        positive_rate = float(np.mean(rewards > 0))

        bootstrap_means = np.array(
            [np.mean(rng.choice(rewards, size=len(rewards), replace=True)) for _ in range(n_bootstrap)]
        )
        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))

        if strategy != baseline_key:
            if len(rewards) == len(baseline):
                t_stat, p_value = _paired_ttest(baseline, rewards)
            else:
                t_stat, p_value = _welch_ttest(baseline, rewards)
            wins = int(np.sum(baseline > rewards))
            ties = int(np.sum(np.isclose(baseline, rewards)))
        else:
            t_stat, p_value = np.nan, np.nan
            wins, ties = np.nan, np.nan

        rows.append(
            {
                "Strategy": strategy,
                "Display_Name": get_display_name(strategy),
                "Mean": mean_reward,
                "Std": std_reward,
                "Median": median_reward,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Min": min_reward,
                "Max": max_reward,
                "Positive_Rate": positive_rate,
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
                "T-statistic": t_stat,
                "P-value": p_value,
                f"Wins_vs_{baseline_key}": wins,
                f"Ties_vs_{baseline_key}": ties,
                "N": len(rewards),
            }
        )

    return pd.DataFrame(rows)


def add_display_columns(stats_df: pd.DataFrame) -> pd.DataFrame:
    enriched = stats_df.copy()
    enriched["Display_Name"] = enriched["Strategy"].map(get_display_name)
    enriched["Color"] = enriched["Strategy"].map(get_color)
    return enriched
