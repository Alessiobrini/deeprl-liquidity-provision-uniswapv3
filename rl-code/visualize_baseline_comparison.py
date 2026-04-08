"""
Create publication-quality visualizations for baseline comparison results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from analysis_utils import compute_strategy_statistics, get_color, get_display_name

# LaTeX and publication-quality settings
plt.style.use('seaborn-v0_8-whitegrid')

# Paper formatting parameters
fsize = 16
params = {
    "text.usetex": False,  # Set True if you have LaTeX installed (MiKTeX)
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": fsize,
    "legend.fontsize": fsize - 1,
    "xtick.labelsize": fsize - 2,
    "ytick.labelsize": fsize - 2,
    "axes.titlesize": fsize + 3,
    "axes.labelsize": fsize + 1,
    "figure.figsize": (7, 4.5),
}
plt.rcParams.update(params)

def load_results(output_dir):
    """Load comparison results from CSV files."""
    detailed_results = pd.read_csv(os.path.join(output_dir, "detailed_results_all_windows.csv"))
    overall_stats = compute_strategy_statistics(detailed_results)
    return overall_stats, detailed_results


def plot_mean_comparison(overall_stats, output_dir):
    """Bar plot comparing mean rewards with error bars (std)."""
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    strategies = overall_stats['Strategy']
    display_names = [get_display_name(s) for s in strategies]
    means = overall_stats['Mean']
    stds = overall_stats['Std']

    # Use consistent colors for all strategies
    colors = [get_color(s) for s in strategies]

    bars = ax.bar(range(len(strategies)), means, yerr=stds,
                   capsize=5, alpha=0.85, color=colors, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Strategy')
    ax.set_ylabel('Mean Cumulative Reward +/- Std')
    ax.set_title('Performance Comparison: PPO vs. Baseline Strategies')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(display_names, rotation=20, ha='right')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'mean_comparison.png'), bbox_inches='tight')
    plt.close()


def plot_boxplot_comparison(detailed_results, output_dir):
    """Box plot showing distribution of rewards across windows."""
    fig, ax = plt.subplots(figsize=(10.2, 6.5))

    # Reshape data for boxplot
    data_to_plot = []
    labels = []
    original_names = []
    for col in detailed_results.columns:
        data_to_plot.append(detailed_results[col].dropna().values)
        labels.append(get_display_name(col))
        original_names.append(col)

    bp = ax.boxplot(
        data_to_plot,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True,
        notch=True,
        medianprops=dict(color='#2c3e50', linewidth=1.8),
        whiskerprops=dict(color='#555555'),
        capprops=dict(color='#555555'),
    )

    # Use consistent colors for all strategies
    for patch, orig_name in zip(bp['boxes'], original_names):
        patch.set_facecolor(get_color(orig_name))
        patch.set_alpha(0.7)

    for xpos, values, orig_name in zip(range(1, len(data_to_plot) + 1), data_to_plot, original_names):
        jitter = np.linspace(-0.09, 0.09, len(values))
        ax.scatter(
            xpos + jitter,
            values,
            color=get_color(orig_name),
            edgecolor='white',
            linewidth=0.4,
            alpha=0.55,
            s=34,
            zorder=3,
        )

    # Symlog keeps the negative values while making the extreme positive outlier readable.
    ax.set_yscale('symlog', linthresh=20000)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Cumulative Reward (symlog scale)')
    ax.set_title('Reward Distribution Across Test Windows')
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    ax.text(
        0.99,
        0.02,
        'Symlog scale used to keep the Reactive Recentering outlier visible.',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=11,
        color='#555555',
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'boxplot_comparison.png'), bbox_inches='tight')
    plt.close()


def plot_pvalue_heatmap(overall_stats, output_dir):
    """Heatmap visualization of p-values."""
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    # Extract strategies and p-values
    mask = overall_stats['Strategy'] != 'PPO'
    strategies = overall_stats[mask]['Strategy'].values
    display_names = [get_display_name(s) for s in strategies]
    pvalues = overall_stats[mask]['P-value'].values

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
    ax.set_yticklabels(display_names)
    ax.set_xticks([0])
    ax.set_xticklabels(['PPO'])
    ax.set_xlabel('Reference')
    ax.set_title('Statistical Significance (p-values)')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('p-value', rotation=270, labelpad=15)

    # Add text annotations
    for i in range(n):
        if pd.notna(pvalue_matrix[i, 0]):
            pval = pvalue_matrix[i, 0]
            text_color = 'white' if pval < 0.05 else 'black'
            sig_marker = '*' if pval < 0.05 else ''
            ax.text(0, i, f'{pval:.3f}{sig_marker}',
                   ha='center', va='center', color=text_color, fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pvalue_heatmap.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'pvalue_heatmap.png'), bbox_inches='tight')
    plt.close()


def plot_confidence_intervals(overall_stats, output_dir):
    """Plot with confidence intervals."""
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    strategies = overall_stats['Strategy']
    display_names = [get_display_name(s) for s in strategies]
    means = overall_stats['Mean']
    ci_lower = overall_stats['CI_Lower']
    ci_upper = overall_stats['CI_Upper']

    # Calculate error bars
    yerr_lower = means - ci_lower
    yerr_upper = ci_upper - means

    # Use consistent colors for all strategies
    colors = [get_color(s) for s in strategies]

    for i, (x, y, c, yl, yu) in enumerate(zip(range(len(strategies)), means, colors, yerr_lower, yerr_upper)):
        ax.errorbar(x, y, yerr=[[yl], [yu]], fmt='o', markersize=12, capsize=7, capthick=2,
                    linewidth=2, color=c, alpha=0.9)

    ax.set_xlabel('Strategy')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Mean Rewards with 95% Bootstrap Confidence Intervals')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(display_names, rotation=20, ha='right')
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
    fig, ax = plt.subplots(figsize=(10.5, 6.3))

    for strategy, data in progression_data.items():
        display_name = get_display_name(strategy)
        color = get_color(strategy)

        if strategy == 'PPO':
            linestyle = '-'
            linewidth = 2.8
            alpha = 1.0
            marker = 'o'
            markersize = 8
        else:
            linestyle = '--'
            linewidth = 2.1
            alpha = 0.8
            marker = 's'
            markersize = 6

        ax.plot(data['windows'], data['means'], label=display_name,
               linestyle=linestyle, linewidth=linewidth, alpha=alpha,
               marker=marker, markersize=markersize, color=color)

    ax.set_xlabel('Test Window Index')
    ax.set_ylabel('Mean Cumulative Reward')
    ax.set_title('Strategy Performance Across Rolling Test Windows')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='best', framealpha=0.9)
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

    print("Generating publication-quality visualizations...")

    refreshed_stats_path = os.path.join(output_dir, "overall_statistics.csv")
    overall_stats.to_csv(refreshed_stats_path, index=False)
    print(f"  Refreshed statistics written to: {refreshed_stats_path}")

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

    print(f"\n All visualizations saved to: {output_dir}")
    print("  Generated files:")
    print("    - mean_comparison.pdf/png")
    print("    - boxplot_comparison.pdf/png")
    print("    - pvalue_heatmap.pdf/png")
    print("    - confidence_intervals.pdf/png")
    print("    - window_progression.pdf/png")


if __name__ == "__main__":
    main()
