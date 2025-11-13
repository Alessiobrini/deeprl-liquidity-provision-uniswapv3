"""
Create publication-quality visualizations for baseline comparison results.
Generates figures suitable for academic papers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def load_results(output_dir):
    """Load comparison results from CSV files."""
    overall_stats = pd.read_csv(os.path.join(output_dir, "overall_statistics.csv"))
    detailed_results = pd.read_csv(os.path.join(output_dir, "detailed_results_all_windows.csv"))
    return overall_stats, detailed_results


def plot_mean_comparison(overall_stats, output_dir):
    """Bar plot comparing mean rewards with error bars (std)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = overall_stats['Strategy']
    means = overall_stats['Mean']
    stds = overall_stats['Std']

    # Color PPO differently
    colors = ['#d62728' if s == 'PPO' else '#1f77b4' for s in strategies]

    bars = ax.bar(range(len(strategies)), means, yerr=stds,
                   capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Strategy', fontweight='bold')
    ax.set_ylabel('Cumulative Reward (Mean ± Std)', fontweight='bold')
    ax.set_title('Performance Comparison: PPO vs. Baseline Strategies')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'mean_comparison.png'), bbox_inches='tight')
    plt.close()


def plot_boxplot_comparison(detailed_results, output_dir):
    """Box plot showing distribution of rewards across windows."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Reshape data for boxplot
    data_to_plot = []
    labels = []
    for col in detailed_results.columns:
        data_to_plot.append(detailed_results[col].dropna().values)
        labels.append(col)

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     showfliers=True, notch=True)

    # Color PPO box differently
    for patch, label in zip(bp['boxes'], labels):
        if label == 'PPO':
            patch.set_facecolor('#d62728')
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor('#1f77b4')
            patch.set_alpha(0.5)

    ax.set_xlabel('Strategy', fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontweight='bold')
    ax.set_title('Reward Distribution Across All Windows')
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'boxplot_comparison.png'), bbox_inches='tight')
    plt.close()


def plot_pvalue_heatmap(overall_stats, output_dir):
    """Heatmap visualization of p-values."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Extract strategies and p-values
    strategies = overall_stats[overall_stats['Strategy'] != 'PPO']['Strategy'].values
    pvalues = overall_stats[overall_stats['Strategy'] != 'PPO']['P-value'].values

    # Create matrix for heatmap
    n = len(strategies)
    pvalue_matrix = np.full((n, 1), np.nan)
    for i, pval in enumerate(pvalues):
        if pd.notna(pval):
            pvalue_matrix[i, 0] = pval

    # Plot heatmap
    im = ax.imshow(pvalue_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)

    # Set ticks and labels
    ax.set_yticks(range(n))
    ax.set_yticklabels(strategies)
    ax.set_xticks([0])
    ax.set_xticklabels(['PPO'])
    ax.set_xlabel('Baseline', fontweight='bold')
    ax.set_title('P-values: Strategy vs. PPO\n(Green = significant, Red = not significant)')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P-value', rotation=270, labelpad=15)

    # Add text annotations
    for i in range(n):
        if pd.notna(pvalue_matrix[i, 0]):
            text_color = 'white' if pvalue_matrix[i, 0] < 0.05 else 'black'
            ax.text(0, i, f'{pvalue_matrix[i, 0]:.4f}',
                   ha='center', va='center', color=text_color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pvalue_heatmap.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'pvalue_heatmap.png'), bbox_inches='tight')
    plt.close()


def plot_confidence_intervals(overall_stats, output_dir):
    """Plot with confidence intervals."""
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = overall_stats['Strategy']
    means = overall_stats['Mean']
    ci_lower = overall_stats['CI_Lower']
    ci_upper = overall_stats['CI_Upper']

    # Calculate error bars
    yerr_lower = means - ci_lower
    yerr_upper = ci_upper - means

    # Color PPO differently
    colors = ['#d62728' if s == 'PPO' else '#1f77b4' for s in strategies]

    ax.errorbar(range(len(strategies)), means,
                yerr=[yerr_lower, yerr_upper],
                fmt='o', markersize=8, capsize=5, capthick=2,
                linewidth=2, alpha=0.8)

    for i, (x, y, c) in enumerate(zip(range(len(strategies)), means, colors)):
        ax.plot(x, y, 'o', markersize=8, color=c, zorder=3)

    ax.set_xlabel('Strategy', fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontweight='bold')
    ax.set_title('Mean Rewards with 95% Bootstrap Confidence Intervals')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_intervals.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'confidence_intervals.png'), bbox_inches='tight')
    plt.close()


def plot_window_progression(output_dir):
    """Plot performance across windows over time."""
    # Load per-window statistics
    window_files = sorted([f for f in os.listdir(output_dir) if f.startswith('window_') and f.endswith('_statistics.csv')])

    if not window_files:
        print("No per-window statistics found. Skipping window progression plot.")
        return

    # Aggregate data
    progression_data = {}

    for wfile in window_files:
        window_idx = int(wfile.split('_')[1])
        df = pd.read_csv(os.path.join(output_dir, wfile))

        for _, row in df.iterrows():
            strategy = row['Strategy']
            if strategy not in progression_data:
                progression_data[strategy] = {'windows': [], 'means': [], 'stds': []}

            progression_data[strategy]['windows'].append(window_idx)
            progression_data[strategy]['means'].append(row['Mean'])
            progression_data[strategy]['stds'].append(row['Std'])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy, data in progression_data.items():
        if strategy == 'PPO':
            linestyle = '-'
            linewidth = 2.5
            alpha = 1.0
            marker = 'o'
        else:
            linestyle = '--'
            linewidth = 1.5
            alpha = 0.7
            marker = 's'

        ax.plot(data['windows'], data['means'], label=strategy,
               linestyle=linestyle, linewidth=linewidth, alpha=alpha,
               marker=marker, markersize=5)

    ax.set_xlabel('Window Index', fontweight='bold')
    ax.set_ylabel('Mean Cumulative Reward', fontweight='bold')
    ax.set_title('Performance Across Rolling Windows')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'window_progression.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'window_progression.png'), bbox_inches='tight')
    plt.close()


def main():
    """Generate all visualizations."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output", "baseline_comparison_optimized")

    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        print("Please run baseline_comparison_statistical.py first.")
        return

    print("Loading results...")
    overall_stats, detailed_results = load_results(output_dir)

    print("Generating visualizations...")

    print("  1. Mean comparison bar plot...")
    plot_mean_comparison(overall_stats, output_dir)

    print("  2. Box plot comparison...")
    plot_boxplot_comparison(detailed_results, output_dir)

    print("  3. P-value heatmap...")
    plot_pvalue_heatmap(overall_stats, output_dir)

    print("  4. Confidence intervals plot...")
    plot_confidence_intervals(overall_stats, output_dir)

    print("  5. Window progression plot...")
    plot_window_progression(output_dir)

    print(f"\n✓ All visualizations saved to: {output_dir}")
    print("  Generated files:")
    print("    - mean_comparison.pdf/png")
    print("    - boxplot_comparison.pdf/png")
    print("    - pvalue_heatmap.pdf/png")
    print("    - confidence_intervals.pdf/png")
    print("    - window_progression.pdf/png")


if __name__ == "__main__":
    main()
