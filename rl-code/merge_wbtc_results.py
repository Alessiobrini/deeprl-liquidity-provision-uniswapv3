"""
Merge WBTC training results from parallel GPU execution
Combines results from all 4 GPUs into unified statistics
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats

def compute_statistics(data_dict, baseline_key='PPO'):
    """Compute statistical summaries across all results."""
    results = []
    baseline_data = np.array(data_dict[baseline_key])

    for strategy, rewards in data_dict.items():
        rewards_arr = np.array(rewards)

        # Compute basic statistics
        mean_reward = np.mean(rewards_arr)
        std_reward = np.std(rewards_arr, ddof=1)
        median_reward = np.median(rewards_arr)

        # Bootstrap 95% confidence interval
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(rewards_arr, size=len(rewards_arr), replace=True)
            bootstrap_means.append(np.mean(sample))
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        # T-test vs baseline
        if strategy != baseline_key:
            t_stat, p_value = stats.ttest_ind(baseline_data, rewards_arr)
            wins = np.sum(baseline_data > rewards_arr)
        else:
            t_stat, p_value = np.nan, np.nan
            wins = np.nan

        results.append({
            'Strategy': strategy,
            'Mean': mean_reward,
            'Std': std_reward,
            'Median': median_reward,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'T-statistic': t_stat,
            'P-value': p_value,
            f'Wins_vs_{baseline_key}': wins,
            'N': len(rewards_arr)
        })

    return pd.DataFrame(results)

def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output" / "baseline_comparison_wbtc"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Merging WBTC Results from Parallel GPU Execution")
    print("="*70)

    # Collect results from all GPUs
    all_detailed_results = []
    all_window_stats = []

    for gpu_id in range(4):
        gpu_dir = base_dir / "output" / f"baseline_comparison_wbtc_gpu{gpu_id}"

        if not gpu_dir.exists():
            print(f"WARNING: GPU {gpu_id} directory not found: {gpu_dir}")
            continue

        print(f"\nGPU {gpu_id}: {gpu_dir}")

        # Load detailed results
        detailed_file = gpu_dir / f"gpu{gpu_id}_detailed_results.csv"
        if detailed_file.exists():
            df = pd.read_csv(detailed_file)
            all_detailed_results.append(df)
            print(f"  Loaded {len(df)} rows from detailed results")
        else:
            print(f"  WARNING: Missing {detailed_file}")

        # Load per-window statistics
        window_files = sorted(gpu_dir.glob("window_*_statistics.csv"))
        for wf in window_files:
            wdf = pd.read_csv(wf)
            all_window_stats.append(wdf)
        print(f"  Loaded {len(window_files)} window statistics files")

    if not all_detailed_results:
        print("\nERROR: No detailed results found! Check GPU directories.")
        return

    # Merge all detailed results
    print("\n" + "="*70)
    print("Combining Results")
    print("="*70)

    combined_detailed = pd.concat(all_detailed_results, ignore_index=True)
    print(f"\nTotal rows: {len(combined_detailed)}")
    print(f"Strategies: {combined_detailed.columns.tolist()}")

    # Compute overall statistics
    data_dict = {col: combined_detailed[col].tolist() for col in combined_detailed.columns}
    overall_stats = compute_statistics(data_dict, baseline_key='PPO')

    print("\n" + "="*70)
    print("OVERALL STATISTICS - ALL WINDOWS (WBTC/USDC 0.3%)")
    print("="*70)
    print(overall_stats.to_string(index=False))

    # Save merged results
    overall_stats.to_csv(output_dir / "overall_statistics.csv", index=False)
    combined_detailed.to_csv(output_dir / "detailed_results_all_windows.csv", index=False)

    # Combine window statistics
    if all_window_stats:
        combined_windows = pd.concat(all_window_stats, ignore_index=True)
        combined_windows = combined_windows.sort_values('Window')
        combined_windows.to_csv(output_dir / "all_windows_statistics.csv", index=False)
        print(f"\nSaved {len(combined_windows)} window statistics")

    # Generate LaTeX table
    latex_table = overall_stats[['Strategy', 'Mean', 'Std', 'P-value']].copy()
    latex_table['Mean±Std'] = latex_table.apply(
        lambda row: f"${row['Mean']:.2f} \\pm {row['Std']:.2f}$",
        axis=1
    )
    latex_table['P-value'] = latex_table['P-value'].apply(
        lambda x: f"${x:.4f}$" if pd.notna(x) else "—"
    )
    latex_str = latex_table[['Strategy', 'Mean±Std', 'P-value']].to_latex(
        index=False,
        escape=False,
        caption="WBTC/USDC 0.3\\% Pool: Comparison of PPO vs. Baseline Strategies",
        label="tab:baseline_comparison_wbtc"
    )

    with open(output_dir / "latex_table.tex", "w") as f:
        f.write(latex_str)

    print(f"\n{'='*70}")
    print("Results saved to:")
    print(f"  {output_dir / 'overall_statistics.csv'}")
    print(f"  {output_dir / 'detailed_results_all_windows.csv'}")
    print(f"  {output_dir / 'all_windows_statistics.csv'}")
    print(f"  {output_dir / 'latex_table.tex'}")
    print("="*70)

if __name__ == "__main__":
    main()
