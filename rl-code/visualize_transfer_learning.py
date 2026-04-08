"""
Visualize Cross-Pool Transfer Learning Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'


def plot_transfer_matrix(summary_df, output_dir):
    """Plot transfer learning results as a heatmap matrix."""
    # Reshape data for heatmap
    matrix_data = []
    experiments = ['WETH->WETH_same', 'WETH->WETH_future', 'WETH->WBTC_same', 'WETH->WBTC_future',
                   'WBTC->WBTC_same', 'WBTC->WBTC_future', 'WBTC->WETH_same', 'WBTC->WETH_future']

    for exp in experiments:
        row = summary_df[summary_df['Experiment'] == exp].iloc[0]
        matrix_data.append(row['Mean'])

    matrix = np.array(matrix_data).reshape(2, 4)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Same Asset\nSame Period', 'Same Asset\nFuture Period',
                        'Diff Asset\nSame Period', 'Diff Asset\nFuture Period'])
    ax.set_yticklabels(['Train on WETH', 'Train on WBTC'])

    # Add value annotations
    for i in range(2):
        for j in range(4):
            text = ax.text(j, i, f'{matrix[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=11, weight='bold')

    ax.set_title('Transfer Learning Performance Matrix\n(Mean Reward Across Seeds)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Reward', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'transfer_matrix.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'transfer_matrix.png'}")
    plt.close()


def plot_same_vs_future(summary_df, output_dir):
    """Compare same-period vs future-period performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # WETH -> WBTC comparison
    weth_same = summary_df[summary_df['Experiment'] == 'WETH->WBTC_same'].iloc[0]
    weth_future = summary_df[summary_df['Experiment'] == 'WETH->WBTC_future'].iloc[0]

    x = [0, 1]
    means = [weth_same['Mean'], weth_future['Mean']]
    stds = [weth_same['Std'], weth_future['Std']]

    bars = ax1.bar(x, means, yerr=stds, capsize=10, alpha=0.7,
                   color=['#2ecc71', '#e74c3c'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Same Period', 'Future Period'])
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Train on WETH -> Test on WBTC')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=10, weight='bold')

    # WBTC -> WETH comparison
    wbtc_same = summary_df[summary_df['Experiment'] == 'WBTC->WETH_same'].iloc[0]
    wbtc_future = summary_df[summary_df['Experiment'] == 'WBTC->WETH_future'].iloc[0]

    means = [wbtc_same['Mean'], wbtc_future['Mean']]
    stds = [wbtc_same['Std'], wbtc_future['Std']]

    bars = ax2.bar(x, means, yerr=stds, capsize=10, alpha=0.7,
                   color=['#3498db', '#e67e22'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Same Period', 'Future Period'])
    ax2.set_ylabel('Mean Reward')
    ax2.set_title('Train on WBTC -> Test on WETH')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'same_vs_future_comparison.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'same_vs_future_comparison.png'}")
    plt.close()


def plot_transfer_efficiency(summary_df, output_dir):
    """Plot transfer efficiency (cross-asset / same-asset performance)."""
    # Calculate transfer efficiency
    weth_weth_same = summary_df[summary_df['Experiment'] == 'WETH->WETH_same'].iloc[0]['Mean']
    weth_wbtc_same = summary_df[summary_df['Experiment'] == 'WETH->WBTC_same'].iloc[0]['Mean']
    wbtc_wbtc_same = summary_df[summary_df['Experiment'] == 'WBTC->WBTC_same'].iloc[0]['Mean']
    wbtc_weth_same = summary_df[summary_df['Experiment'] == 'WBTC->WETH_same'].iloc[0]['Mean']

    transfer_eff_weth = (weth_wbtc_same / weth_weth_same) * 100
    transfer_eff_wbtc = (wbtc_weth_same / wbtc_wbtc_same) * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    x = [0, 1]
    efficiencies = [transfer_eff_weth, transfer_eff_wbtc]

    bars = ax.bar(x, efficiencies, alpha=0.7, color=['#9b59b6', '#16a085'])
    ax.set_xticks(x)
    ax.set_xticklabels(['WETH->WBTC', 'WBTC->WETH'])
    ax.set_ylabel('Transfer Efficiency (%)')
    ax.set_title('Cross-Asset Transfer Efficiency\n(% of Same-Asset Performance Retained)')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, label='100% (perfect transfer)')
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='80% (good transfer)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, weight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'transfer_efficiency.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'transfer_efficiency.png'}")
    plt.close()


def plot_detailed_results(detailed_df, output_dir):
    """Plot box plots of detailed results across seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # WETH-based experiments
    weth_data = [
        detailed_df['WETH->WETH_same'].values,
        detailed_df['WETH->WETH_future'].values,
        detailed_df['WETH->WBTC_same'].values,
        detailed_df['WETH->WBTC_future'].values,
    ]

    bp1 = axes[0, 0].boxplot(weth_data, labels=['WETH\nSame', 'WETH\nFuture', 'WBTC\nSame', 'WBTC\nFuture'])
    axes[0, 0].set_title('Train on WETH', fontsize=12, weight='bold')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # WBTC-based experiments
    wbtc_data = [
        detailed_df['WBTC->WBTC_same'].values,
        detailed_df['WBTC->WBTC_future'].values,
        detailed_df['WBTC->WETH_same'].values,
        detailed_df['WBTC->WETH_future'].values,
    ]

    bp2 = axes[0, 1].boxplot(wbtc_data, labels=['WBTC\nSame', 'WBTC\nFuture', 'WETH\nSame', 'WETH\nFuture'])
    axes[0, 1].set_title('Train on WBTC', fontsize=12, weight='bold')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Cross-asset comparison (same period)
    cross_same = [
        detailed_df['WETH->WETH_same'].values,
        detailed_df['WETH->WBTC_same'].values,
        detailed_df['WBTC->WBTC_same'].values,
        detailed_df['WBTC->WETH_same'].values,
    ]

    bp3 = axes[1, 0].boxplot(cross_same, labels=['WETH->\nWETH', 'WETH->\nWBTC', 'WBTC->\nWBTC', 'WBTC->\nWETH'])
    axes[1, 0].set_title('Same Time Period', fontsize=12, weight='bold')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Cross-asset comparison (future period)
    cross_future = [
        detailed_df['WETH->WETH_future'].values,
        detailed_df['WETH->WBTC_future'].values,
        detailed_df['WBTC->WBTC_future'].values,
        detailed_df['WBTC->WETH_future'].values,
    ]

    bp4 = axes[1, 1].boxplot(cross_future, labels=['WETH->\nWETH', 'WETH->\nWBTC', 'WBTC->\nWBTC', 'WBTC->\nWETH'])
    axes[1, 1].set_title('Future Time Period', fontsize=12, weight='bold')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.suptitle('Transfer Learning Results Distribution Across Seeds', fontsize=14, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'transfer_boxplots.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'transfer_boxplots.png'}")
    plt.close()


def main():
    """Generate all transfer learning visualizations."""
    print("="*70)
    print("Transfer Learning Visualization")
    print("="*70)

    # Load results
    base_dir = Path(__file__).parent
    input_dir = base_dir / "output" / "transfer_learning"

    if not input_dir.exists():
        print(f"\nERROR: Results directory not found: {input_dir}")
        print("Please run cross_pool_transfer.py first!")
        return

    summary_df = pd.read_csv(input_dir / "transfer_learning_results.csv")
    detailed_df = pd.read_csv(input_dir / "transfer_learning_detailed.csv")

    print(f"\nLoaded results from: {input_dir}")
    print(f"  Summary: {len(summary_df)} experiments")
    print(f"  Detailed: {len(detailed_df)} seeds\n")

    # Create visualizations
    output_dir = base_dir / "output" / "transfer_learning"

    print("Generating visualizations...")
    plot_transfer_matrix(summary_df, output_dir)
    plot_same_vs_future(summary_df, output_dir)
    plot_transfer_efficiency(summary_df, output_dir)
    plot_detailed_results(detailed_df, output_dir)

    print("\n" + "="*70)
    print("All visualizations created!")
    print("="*70)
    print(f"\nSaved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. transfer_matrix.png - Performance heatmap")
    print("  2. same_vs_future_comparison.png - Temporal comparison")
    print("  3. transfer_efficiency.png - Cross-asset transfer efficiency")
    print("  4. transfer_boxplots.png - Distribution across seeds")


if __name__ == "__main__":
    main()
