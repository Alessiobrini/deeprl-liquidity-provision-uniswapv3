"""
WETH vs WBTC Pool Comparison Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

def load_results(pool_name):
    """Load results for a specific pool."""
    base_dir = Path(__file__).parent
    results_dir = base_dir / "output" / f"baseline_comparison_{pool_name}"

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    overall_stats = pd.read_csv(results_dir / "overall_statistics.csv")
    detailed_results = pd.read_csv(results_dir / "detailed_results_all_windows.csv")

    return overall_stats, detailed_results

def plot_strategy_comparison(weth_stats, wbtc_stats, output_dir):
    """
    Plot 1: Side-by-side comparison of strategy performance on both pools
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    strategies = weth_stats['Strategy'].tolist()

    # WETH performance
    means_weth = weth_stats['Mean'].values
    stds_weth = weth_stats['Std'].values
    x = np.arange(len(strategies))

    bars1 = ax1.bar(x, means_weth, yerr=stds_weth, capsize=5, alpha=0.7,
                    color=['#2ecc71' if s == 'PPO' else '#95a5a6' for s in strategies])
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('WETH/USDC 0.05% Pool')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)

    # WBTC performance
    means_wbtc = wbtc_stats['Mean'].values
    stds_wbtc = wbtc_stats['Std'].values

    bars2 = ax2.bar(x, means_wbtc, yerr=stds_wbtc, capsize=5, alpha=0.7,
                    color=['#3498db' if s == 'PPO' else '#95a5a6' for s in strategies])
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Mean Reward')
    ax2.set_title('WBTC/USDC 0.3% Pool')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pool_comparison_strategies.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'pool_comparison_strategies.png'}")
    plt.close()

def plot_ppo_advantage(weth_stats, wbtc_stats, output_dir):
    """
    Plot 2: PPO advantage over baselines (grouped bar chart)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate PPO advantage (PPO mean - baseline mean)
    baselines = weth_stats[weth_stats['Strategy'] != 'PPO']['Strategy'].tolist()

    ppo_mean_weth = weth_stats[weth_stats['Strategy'] == 'PPO']['Mean'].values[0]
    ppo_mean_wbtc = wbtc_stats[wbtc_stats['Strategy'] == 'PPO']['Mean'].values[0]

    advantages_weth = []
    advantages_wbtc = []

    for baseline in baselines:
        baseline_mean_weth = weth_stats[weth_stats['Strategy'] == baseline]['Mean'].values[0]
        baseline_mean_wbtc = wbtc_stats[wbtc_stats['Strategy'] == baseline]['Mean'].values[0]

        advantages_weth.append(ppo_mean_weth - baseline_mean_weth)
        advantages_wbtc.append(ppo_mean_wbtc - baseline_mean_wbtc)

    x = np.arange(len(baselines))
    width = 0.35

    bars1 = ax.bar(x - width/2, advantages_weth, width, label='WETH/USDC 0.05%',
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, advantages_wbtc, width, label='WBTC/USDC 0.3%',
                   color='#3498db', alpha=0.8)

    ax.set_xlabel('Baseline Strategy')
    ax.set_ylabel('PPO Advantage (Mean Reward Difference)')
    ax.set_title('PPO Performance Advantage Over Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'ppo_advantage_comparison.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'ppo_advantage_comparison.png'}")
    plt.close()

def plot_statistical_significance(weth_stats, wbtc_stats, output_dir):
    """
    Plot 3: P-values heatmap showing statistical significance
    """
    baselines = weth_stats[weth_stats['Strategy'] != 'PPO']['Strategy'].tolist()

    p_values = np.zeros((2, len(baselines)))

    for i, baseline in enumerate(baselines):
        p_values[0, i] = weth_stats[weth_stats['Strategy'] == baseline]['P-value'].values[0]
        p_values[1, i] = wbtc_stats[wbtc_stats['Strategy'] == baseline]['P-value'].values[0]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Create heatmap with log scale for better visualization
    im = ax.imshow(p_values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)

    ax.set_xticks(np.arange(len(baselines)))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(baselines, rotation=45, ha='right')
    ax.set_yticklabels(['WETH/USDC 0.05%', 'WBTC/USDC 0.3%'])

    # Add p-value text annotations
    for i in range(2):
        for j in range(len(baselines)):
            text = ax.text(j, i, f'{p_values[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title('P-values: PPO vs Baselines (Lower = More Significant)')

    # Add significance threshold line
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P-value', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_significance.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'statistical_significance.png'}")
    plt.close()

def plot_pool_characteristics(output_dir):
    """
    Plot 4: Pool characteristics comparison table as visualization
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    data = [
        ['Parameter', 'WETH/USDC 0.05%', 'WBTC/USDC 0.3%', 'Ratio'],
        ['Fee Tier', '0.05%', '0.30%', '6x'],
        ['Tick Spacing', '10', '60', '6x'],
        ['Action Values', '[0, 45, 50, 55]', '[0, 60, 120, 180]', '~2-3x'],
        ['Initial Quantity', '10 ETH (~$30k)', '0.15 BTC (~$6k)', '-'],
        ['Data Hours', '23,995', '40,801', '1.7x'],
        ['Training Windows', '~10', '~22', '2.2x']
    ]

    table = ax.table(cellText=data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('Pool Characteristics Comparison', fontsize=14, weight='bold', pad=20)
    plt.savefig(output_dir / 'pool_characteristics.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'pool_characteristics.png'}")
    plt.close()

def plot_reward_distributions(weth_detailed, wbtc_detailed, output_dir):
    """
    Plot 5: Distribution of rewards for PPO across both pools
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # WETH distribution
    weth_ppo = weth_detailed['PPO'].dropna()
    ax1.hist(weth_ppo, bins=30, alpha=0.7, color='#2ecc71', edgecolor='black')
    ax1.axvline(weth_ppo.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {weth_ppo.mean():.1f}')
    ax1.axvline(weth_ppo.median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {weth_ppo.median():.1f}')
    ax1.set_xlabel('PPO Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('WETH/USDC 0.05% - PPO Reward Distribution')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # WBTC distribution
    wbtc_ppo = wbtc_detailed['PPO'].dropna()
    ax2.hist(wbtc_ppo, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
    ax2.axvline(wbtc_ppo.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {wbtc_ppo.mean():.1f}')
    ax2.axvline(wbtc_ppo.median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {wbtc_ppo.median():.1f}')
    ax2.set_xlabel('PPO Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('WBTC/USDC 0.3% - PPO Reward Distribution')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'reward_distributions.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'reward_distributions.png'}")
    plt.close()

def create_summary_table(weth_stats, wbtc_stats, output_dir):
    """
    Create a comprehensive summary table for the paper
    """
    summary_data = []

    for strategy in weth_stats['Strategy']:
        weth_row = weth_stats[weth_stats['Strategy'] == strategy].iloc[0]
        wbtc_row = wbtc_stats[wbtc_stats['Strategy'] == strategy].iloc[0]

        summary_data.append({
            'Strategy': strategy,
            'WETH_Mean': weth_row['Mean'],
            'WETH_Std': weth_row['Std'],
            'WETH_P': weth_row['P-value'],
            'WBTC_Mean': wbtc_row['Mean'],
            'WBTC_Std': wbtc_row['Std'],
            'WBTC_P': wbtc_row['P-value']
        })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    summary_df.to_csv(output_dir / 'weth_wbtc_comparison_summary.csv', index=False)
    print(f"Saved: {output_dir / 'weth_wbtc_comparison_summary.csv'}")

    # Create LaTeX table
    latex_rows = []
    latex_rows.append("\\begin{table}[h]")
    latex_rows.append("\\centering")
    latex_rows.append("\\caption{Performance Comparison: WETH/USDC 0.05\\% vs WBTC/USDC 0.3\\%}")
    latex_rows.append("\\label{tab:pool_comparison}")
    latex_rows.append("\\begin{tabular}{lccc}")
    latex_rows.append("\\toprule")
    latex_rows.append("Strategy & WETH Mean$\\pm$Std & WBTC Mean$\\pm$Std & P-value (WETH, WBTC) \\\\")
    latex_rows.append("\\midrule")

    for _, row in summary_df.iterrows():
        strategy = row['Strategy']
        weth_str = f"${row['WETH_Mean']:.1f} \\pm {row['WETH_Std']:.1f}$"
        wbtc_str = f"${row['WBTC_Mean']:.1f} \\pm {row['WBTC_Std']:.1f}$"

        if pd.isna(row['WETH_P']):
            p_str = "—"
        else:
            p_str = f"${row['WETH_P']:.4f}$, ${row['WBTC_P']:.4f}$"

        latex_rows.append(f"{strategy} & {weth_str} & {wbtc_str} & {p_str} \\\\")

    latex_rows.append("\\bottomrule")
    latex_rows.append("\\end{tabular}")
    latex_rows.append("\\end{table}")

    with open(output_dir / 'comparison_latex_table.tex', 'w') as f:
        f.write('\n'.join(latex_rows))

    print(f"Saved: {output_dir / 'comparison_latex_table.tex'}")

def main():
    print("="*70)
    print("WETH vs WBTC Pool Comparison Visualization")
    print("="*70)

    # Load results
    print("\nLoading results...")
    try:
        weth_stats, weth_detailed = load_results('weth')
        print("  [OK] WETH results loaded")
    except FileNotFoundError:
        print("  [INFO] WETH results not found. Using optimized results...")
        weth_stats, weth_detailed = load_results('optimized')
        print("  [OK] WETH (optimized) results loaded")

    wbtc_stats, wbtc_detailed = load_results('wbtc')
    print("  [OK] WBTC results loaded")

    # Create output directory
    output_dir = Path(__file__).parent / "output" / "pool_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations...")
    print(f"Output directory: {output_dir}")
    print()

    # Generate all plots
    plot_strategy_comparison(weth_stats, wbtc_stats, output_dir)
    plot_ppo_advantage(weth_stats, wbtc_stats, output_dir)
    plot_statistical_significance(weth_stats, wbtc_stats, output_dir)
    plot_pool_characteristics(output_dir)
    plot_reward_distributions(weth_detailed, wbtc_detailed, output_dir)
    create_summary_table(weth_stats, wbtc_stats, output_dir)

    print("\n" + "="*70)
    print("All visualizations created successfully!")
    print("="*70)
    print(f"\nFiles saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. pool_comparison_strategies.png - Side-by-side strategy performance")
    print("  2. ppo_advantage_comparison.png - PPO advantage over baselines")
    print("  3. statistical_significance.png - P-value heatmap")
    print("  4. pool_characteristics.png - Pool comparison table")
    print("  5. reward_distributions.png - PPO reward distributions")
    print("  6. weth_wbtc_comparison_summary.csv - Summary statistics")
    print("  7. comparison_latex_table.tex - LaTeX table for paper")

if __name__ == "__main__":
    main()
