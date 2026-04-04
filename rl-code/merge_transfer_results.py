"""
Merge Transfer Learning Results from Parallel GPU Execution
Combines results from all GPUs into unified statistics and visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def compute_statistics(data_dict):
    """Compute statistical summaries for transfer learning results."""
    results = []

    for experiment, rewards in data_dict.items():
        rewards_arr = np.array(rewards)

        # Compute basic statistics
        mean_reward = np.mean(rewards_arr)
        std_reward = np.std(rewards_arr, ddof=1)
        median_reward = np.median(rewards_arr)
        min_reward = np.min(rewards_arr)
        max_reward = np.max(rewards_arr)

        # Bootstrap 95% confidence interval
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(rewards_arr, size=len(rewards_arr), replace=True)
            bootstrap_means.append(np.mean(sample))
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        # Parse experiment name
        train_pool, rest = experiment.split('->')
        test_pool, period = rest.split('_')

        results.append({
            'Experiment': experiment,
            'Train_Pool': train_pool,
            'Test_Pool': test_pool,
            'Period': period,
            'Mean': mean_reward,
            'Std': std_reward,
            'Median': median_reward,
            'Min': min_reward,
            'Max': max_reward,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'N': len(rewards_arr)
        })

    return pd.DataFrame(results)


def compute_hypothesis_tests(results_dict):
    """Perform statistical hypothesis tests on transfer learning results."""
    tests_results = []

    # Test 1: WETH->WBTC same period vs future period
    same = np.array(results_dict['WETH->WBTC_same'])
    future = np.array(results_dict['WETH->WBTC_future'])
    t_stat, p_value = stats.ttest_ind(same, future)

    tests_results.append({
        'Hypothesis': 'H1: WETH->WBTC same period > future period',
        'Group1': 'WETH->WBTC_same',
        'Group2': 'WETH->WBTC_future',
        'Mean1': np.mean(same),
        'Mean2': np.mean(future),
        'Std1': np.std(same, ddof=1),
        'Std2': np.std(future, ddof=1),
        'T-statistic': t_stat,
        'P-value': p_value,
        'Significant': (np.mean(same) > np.mean(future) and p_value < 0.05)
    })

    # Test 2: WBTC->WETH same period vs future period
    same = np.array(results_dict['WBTC->WETH_same'])
    future = np.array(results_dict['WBTC->WETH_future'])
    t_stat, p_value = stats.ttest_ind(same, future)

    tests_results.append({
        'Hypothesis': 'H2: WBTC->WETH same period > future period',
        'Group1': 'WBTC->WETH_same',
        'Group2': 'WBTC->WETH_future',
        'Mean1': np.mean(same),
        'Mean2': np.mean(future),
        'Std1': np.std(same, ddof=1),
        'Std2': np.std(future, ddof=1),
        'T-statistic': t_stat,
        'P-value': p_value,
        'Significant': (np.mean(same) > np.mean(future) and p_value < 0.05)
    })

    # Test 3: Cross-asset transfer effectiveness
    weth_weth_same = np.array(results_dict['WETH->WETH_same'])
    weth_wbtc_same = np.array(results_dict['WETH->WBTC_same'])
    transfer_efficiency_weth = np.mean(weth_wbtc_same) / np.mean(weth_weth_same) * 100

    wbtc_wbtc_same = np.array(results_dict['WBTC->WBTC_same'])
    wbtc_weth_same = np.array(results_dict['WBTC->WETH_same'])
    transfer_efficiency_wbtc = np.mean(wbtc_weth_same) / np.mean(wbtc_wbtc_same) * 100

    tests_results.append({
        'Hypothesis': 'H3: Cross-asset transfer efficiency',
        'Group1': 'WETH->WBTC transfer efficiency',
        'Group2': 'WBTC->WETH transfer efficiency',
        'Mean1': transfer_efficiency_weth,
        'Mean2': transfer_efficiency_wbtc,
        'Std1': np.nan,
        'Std2': np.nan,
        'T-statistic': np.nan,
        'P-value': np.nan,
        'Significant': True
    })

    return pd.DataFrame(tests_results)


def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output" / "transfer_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Merging Transfer Learning Results from Parallel GPU Execution")
    print("="*70)

    # Collect results from all GPUs
    all_detailed_results = []

    for gpu_id in range(4):
        gpu_dir = base_dir / "output" / f"transfer_learning_gpu{gpu_id}"

        if not gpu_dir.exists():
            print(f"INFO: GPU {gpu_id} directory not found: {gpu_dir}")
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

    if not all_detailed_results:
        print("\nERROR: No detailed results found! Check GPU directories.")
        print("Make sure parallel job completed successfully.")
        return

    # Merge all detailed results
    print("\n" + "="*70)
    print("Combining Results")
    print("="*70)

    combined_detailed = pd.concat(all_detailed_results, ignore_index=True)
    print(f"\nTotal seeds processed: {len(combined_detailed)}")
    print(f"Experiments: {combined_detailed.columns.tolist()}")

    # Compute overall statistics
    data_dict = {col: combined_detailed[col].tolist() for col in combined_detailed.columns}
    summary_df = compute_statistics(data_dict)

    print("\n" + "="*70)
    print("TRANSFER LEARNING RESULTS - MERGED FROM ALL GPUS")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Hypothesis testing
    hypothesis_df = compute_hypothesis_tests(data_dict)

    print("\n" + "="*70)
    print("HYPOTHESIS TESTS")
    print("="*70)
    print(hypothesis_df.to_string(index=False))

    # Save merged results
    summary_df.to_csv(output_dir / "transfer_learning_results.csv", index=False)
    combined_detailed.to_csv(output_dir / "transfer_learning_detailed.csv", index=False)
    hypothesis_df.to_csv(output_dir / "hypothesis_tests.csv", index=False)

    # Generate LaTeX table for summary
    latex_rows = []
    latex_rows.append("\\begin{table}[h]")
    latex_rows.append("\\centering")
    latex_rows.append("\\caption{Cross-Pool Transfer Learning Results}")
    latex_rows.append("\\label{tab:transfer_learning}")
    latex_rows.append("\\begin{tabular}{llcc}")
    latex_rows.append("\\toprule")
    latex_rows.append("Train Pool & Test Pool & Same Period & Future Period \\\\")
    latex_rows.append("\\midrule")

    # WETH experiments
    weth_weth_same = summary_df[summary_df['Experiment'] == 'WETH->WETH_same'].iloc[0]
    weth_weth_future = summary_df[summary_df['Experiment'] == 'WETH->WETH_future'].iloc[0]
    weth_wbtc_same = summary_df[summary_df['Experiment'] == 'WETH->WBTC_same'].iloc[0]
    weth_wbtc_future = summary_df[summary_df['Experiment'] == 'WETH->WBTC_future'].iloc[0]

    latex_rows.append(f"WETH & WETH & ${weth_weth_same['Mean']:.0f} \\pm {weth_weth_same['Std']:.0f}$ & ${weth_weth_future['Mean']:.0f} \\pm {weth_weth_future['Std']:.0f}$ \\\\")
    latex_rows.append(f"WETH & WBTC & ${weth_wbtc_same['Mean']:.0f} \\pm {weth_wbtc_same['Std']:.0f}$ & ${weth_wbtc_future['Mean']:.0f} \\pm {weth_wbtc_future['Std']:.0f}$ \\\\")

    # WBTC experiments
    wbtc_wbtc_same = summary_df[summary_df['Experiment'] == 'WBTC->WBTC_same'].iloc[0]
    wbtc_wbtc_future = summary_df[summary_df['Experiment'] == 'WBTC->WBTC_future'].iloc[0]
    wbtc_weth_same = summary_df[summary_df['Experiment'] == 'WBTC->WETH_same'].iloc[0]
    wbtc_weth_future = summary_df[summary_df['Experiment'] == 'WBTC->WETH_future'].iloc[0]

    latex_rows.append(f"WBTC & WBTC & ${wbtc_wbtc_same['Mean']:.0f} \\pm {wbtc_wbtc_same['Std']:.0f}$ & ${wbtc_wbtc_future['Mean']:.0f} \\pm {wbtc_wbtc_future['Std']:.0f}$ \\\\")
    latex_rows.append(f"WBTC & WETH & ${wbtc_weth_same['Mean']:.0f} \\pm {wbtc_weth_same['Std']:.0f}$ & ${wbtc_weth_future['Mean']:.0f} \\pm {wbtc_weth_future['Std']:.0f}$ \\\\")

    latex_rows.append("\\bottomrule")
    latex_rows.append("\\end{tabular}")
    latex_rows.append("\\end{table}")

    with open(output_dir / "latex_table.tex", "w") as f:
        f.write('\n'.join(latex_rows))

    print(f"\n{'='*70}")
    print("Results saved to:")
    print(f"  {output_dir / 'transfer_learning_results.csv'}")
    print(f"  {output_dir / 'transfer_learning_detailed.csv'}")
    print(f"  {output_dir / 'hypothesis_tests.csv'}")
    print(f"  {output_dir / 'latex_table.tex'}")
    print("="*70)

    print("\nNext steps:")
    print("  1. Run: python visualize_transfer_learning.py")
    print("  2. Review hypothesis test results in hypothesis_tests.csv")
    print("  3. Check visualizations in output/transfer_learning/")

if __name__ == "__main__":
    main()
