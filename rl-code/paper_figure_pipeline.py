"""
Generate paper-support figures and diagnostics from saved experiment outputs.

This script avoids retraining and instead rebuilds the publication assets from:
1. Saved result CSVs
2. Saved example PPO / benchmark window plots
3. Raw WETH hourly price data
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import compute_strategy_statistics, get_display_name


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "paper_figures"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 17,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})


def plot_weth_price_with_test_windows(output_dir: Path, window_size: int = 1500, train_windows: int = 5) -> None:
    df = pd.read_csv(BASE_DIR / "data_price_uni_h_time.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    num_full_windows = len(df) // window_size
    usable = df.iloc[: num_full_windows * window_size].copy()

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.plot(usable["timestamp"], usable["price"], color="#2f6b8a", linewidth=1.4, label="WETH/USDC hourly price")

    for test_idx in range(train_windows, num_full_windows):
        start = usable.iloc[test_idx * window_size]["timestamp"]
        end = usable.iloc[min((test_idx + 1) * window_size - 1, len(usable) - 1)]["timestamp"]
        color = "#dceef8" if (test_idx - train_windows) % 2 == 0 else "#edf5d9"
        ax.axvspan(start, end, color=color, alpha=0.7, linewidth=0)

    first_test_boundary = usable.iloc[train_windows * window_size]["timestamp"]
    ax.axvline(first_test_boundary, color="#c0392b", linestyle="--", linewidth=1.5, label="First out-of-sample boundary")

    ax.set_title("WETH/USDC Price Series with Rolling Out-of-Sample Windows")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USDC)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", framealpha=0.95)

    ax.text(
        0.995,
        0.02,
        f"{train_windows} training windows x {window_size} hours, then 1 test window x {window_size} hours per roll",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(output_dir / "weth_price_with_rolling_windows.png", bbox_inches="tight")
    plt.close(fig)


def _draw_image_grid(image_paths, titles, output_path: Path, nrows: int, ncols: int, figsize=(12, 8)) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for ax, path, title in zip(axes, image_paths, titles):
        image = mpimg.imread(path)
        ax.imshow(image)
        ax.set_title(title, fontsize=18, pad=10)
        ax.axis("off")

    for ax in axes[len(image_paths):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_example_window_panels(output_dir: Path) -> None:
    price_paths = [
        BASE_DIR / "plot" / "20241122_PPOUniswap_Benchmark" / "plt_rw_0_benchmark_price.png",
        BASE_DIR / "plot" / "20241122_PPOUniswap" / "plt_rw_0_price.png",
        BASE_DIR / "plot" / "20241122_PPOUniswap_Benchmark" / "plt_rw_7_benchmark_price.png",
        BASE_DIR / "plot" / "20241122_PPOUniswap" / "plt_rw_7_price.png",
    ]
    price_titles = [
        "Window 0: Benchmark range path",
        "Window 0: PPO range path",
        "Window 7: Benchmark range path",
        "Window 7: PPO range path",
    ]
    _draw_image_grid(
        price_paths,
        price_titles,
        output_dir / "example_price_paths.png",
        nrows=2,
        ncols=2,
        figsize=(12, 8.2),
    )

    reward_paths = [
        BASE_DIR / "plot" / "20241122_PPOUniswap_Benchmark" / "plt_rw_0_benchmark_cumulative.png",
        BASE_DIR / "plot" / "20241122_PPOUniswap" / "plt_rw_0_cumulative.png",
        BASE_DIR / "plot" / "20241122_PPOUniswap_Benchmark" / "plt_rw_7_benchmark_cumulative.png",
        BASE_DIR / "plot" / "20241122_PPOUniswap" / "plt_rw_7_cumulative.png",
    ]
    reward_titles = [
        "Window 0: Benchmark cumulative outcome",
        "Window 0: PPO cumulative outcome",
        "Window 7: Benchmark cumulative outcome",
        "Window 7: PPO cumulative outcome",
    ]
    _draw_image_grid(
        reward_paths,
        reward_titles,
        output_dir / "example_cumulative_outcomes.png",
        nrows=2,
        ncols=2,
        figsize=(12, 8.2),
    )


def _draw_vertical_pair(image_paths, titles, output_path: Path, figsize=(7.1, 8.9)) -> None:
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    for ax, path, title in zip(axes, image_paths, titles):
        image = mpimg.imread(path)
        ax.imshow(image)
        ax.set_title(title, fontsize=19, pad=10)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_column_friendly_examples(output_dir: Path) -> None:
    # Each output is intended for IEEE single-column placement.
    _draw_vertical_pair(
        [
            BASE_DIR / "plot" / "20241122_PPOUniswap_Benchmark" / "plt_rw_0_benchmark_price.png",
            BASE_DIR / "plot" / "20241122_PPOUniswap" / "plt_rw_0_price.png",
        ],
        [
            "Window 0 benchmark range path",
            "Window 0 PPO range path",
        ],
        output_dir / "window0_price_comparison_column.png",
    )

    _draw_vertical_pair(
        [
            BASE_DIR / "plot" / "20241122_PPOUniswap_Benchmark" / "plt_rw_7_benchmark_price.png",
            BASE_DIR / "plot" / "20241122_PPOUniswap" / "plt_rw_7_price.png",
        ],
        [
            "Window 7 benchmark range path",
            "Window 7 PPO range path",
        ],
        output_dir / "window7_price_comparison_column.png",
    )

    _draw_vertical_pair(
        [
            BASE_DIR / "plot" / "20241122_PPOUniswap_Benchmark" / "plt_rw_0_benchmark_cumulative.png",
            BASE_DIR / "plot" / "20241122_PPOUniswap" / "plt_rw_0_cumulative.png",
        ],
        [
            "Window 0 benchmark cumulative outcome",
            "Window 0 PPO cumulative outcome",
        ],
        output_dir / "window0_cumulative_comparison_column.png",
    )

    _draw_vertical_pair(
        [
            BASE_DIR / "plot" / "20241122_PPOUniswap_Benchmark" / "plt_rw_7_benchmark_cumulative.png",
            BASE_DIR / "plot" / "20241122_PPOUniswap" / "plt_rw_7_cumulative.png",
        ],
        [
            "Window 7 benchmark cumulative outcome",
            "Window 7 PPO cumulative outcome",
        ],
        output_dir / "window7_cumulative_comparison_column.png",
    )


def _load_pool_stats(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed = pd.read_csv(results_dir / "detailed_results_all_windows.csv")
    stats = compute_strategy_statistics(detailed)
    return stats, detailed


def export_pool_diagnostics(output_dir: Path) -> None:
    pool_dirs = {
        "WETH/USDC 0.05%": BASE_DIR / "output" / "baseline_comparison_optimized",
        "WBTC/USDC 0.3%": BASE_DIR / "output" / "baseline_comparison_wbtc",
    }

    stats_rows = []
    window_rows = []
    seeds_per_window = 5

    for pool_name, results_dir in pool_dirs.items():
        stats_df, detailed_df = _load_pool_stats(results_dir)
        stats_df = stats_df.copy()
        stats_df.insert(0, "Pool", pool_name)
        stats_rows.append(stats_df)

        ppo_values = detailed_df["PPO"].to_numpy()
        num_windows = math.ceil(len(detailed_df) / seeds_per_window)

        for window_idx in range(num_windows):
            sl = slice(window_idx * seeds_per_window, (window_idx + 1) * seeds_per_window)
            ppo_window = ppo_values[sl]

            for strategy in detailed_df.columns:
                if strategy == "PPO":
                    continue
                baseline_window = detailed_df[strategy].to_numpy()[sl]
                window_rows.append(
                    {
                        "Pool": pool_name,
                        "Window": window_idx,
                        "Strategy": strategy,
                        "Display_Name": get_display_name(strategy),
                        "Baseline_Reward": float(np.mean(baseline_window)),
                        "PPO_Mean": float(np.mean(ppo_window)),
                        "PPO_Median": float(np.median(ppo_window)),
                        "PPO_Positive_Count": int(np.sum(ppo_window > 0)),
                        "PPO_Beat_Baseline_Count": int(np.sum(ppo_window > baseline_window)),
                    }
                )

    pd.concat(stats_rows, ignore_index=True).to_csv(output_dir / "pool_strategy_diagnostics.csv", index=False)
    pd.DataFrame(window_rows).to_csv(output_dir / "window_level_diagnostics.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Writing paper-support outputs to: {OUTPUT_DIR}")
    plot_weth_price_with_test_windows(OUTPUT_DIR)
    export_example_window_panels(OUTPUT_DIR)
    export_column_friendly_examples(OUTPUT_DIR)
    export_pool_diagnostics(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
