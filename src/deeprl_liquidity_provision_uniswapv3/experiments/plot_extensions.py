"""Publication-oriented figures for the geometry, retail, and headroom extensions."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from ..data.pools import CORE

POOL_LABELS = {
    "usdc_weth_005": "USDC/WETH 0.05%", "usdc_weth_030": "USDC/WETH 0.30%",
    "wbtc_weth_005": "WBTC/WETH 0.05%", "wbtc_weth_030": "WBTC/WETH 0.30%",
    "weth_usdt_005": "WETH/USDT 0.05%", "weth_usdt_030": "WETH/USDT 0.30%",
}
STRATEGY_LABELS = {
    "Passive": "Passive", "PassiveWidthSweep": "Fixed width",
    "VolProportionalWidth": "Vol-proportional", "ILMinimizer": "IL minimizer",
    "ReactiveRecentering": "Reactive", "RecentreWhenOut": "Recenter out",
    "PPO": "PPO", "A2C": "A2C", "DQN": "DQN", "QRDQN": "QR-DQN",
    "RECURRENTPPO": "Recurrent PPO",
}


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Keep the repository's hyphenated names and emit the exact underscore aliases
    # used by the paper source. Both are regenerated from the same figure object.
    for name in {stem, stem.replace("-", "_")}:
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_retail(path: Path, out: Path) -> None:
    df = pd.read_csv(path)
    df = df[df.regime == "conversion"]
    capitals = sorted(df.capital.unique())
    gas = sorted(df.gas.unique())
    limit = float(np.abs(df.difference).max())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(3, 2, figsize=(7.15, 5.35), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, pool in zip(axes.flat, CORE):
        p = df[df.pool == pool].pivot(index="capital", columns="gas", values="difference")
        p = p.reindex(index=capitals, columns=gas)
        image = ax.imshow(p.values, cmap="RdBu", norm=norm, aspect="auto", origin="lower")
        for y in range(len(capitals)):
            for x in range(len(gas)):
                value = p.iloc[y, x]
                color = "white" if abs(value) > 0.52 * limit else "black"
                ax.text(x, y, f"{value/1000:+.1f}k" if abs(value) >= 1000 else f"{value:+.0f}",
                        ha="center", va="center", fontsize=6.3, color=color)
        ax.set_title(POOL_LABELS[pool], fontsize=7.5)
        ax.set_xticks(range(len(gas)), [f"${x}" for x in gas])
        ax.set_yticks(range(len(capitals)), [f"${x/1000:g}k" for x in capitals])
        ax.tick_params(labelsize=6.8)
        ax.axvline(-0.5, color="none")
    for ax in axes[-1]:
        ax.set_xlabel("Gas per recenter", fontsize=7.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Position capital", fontsize=7.2)
    cbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.015)
    cbar.ax.tick_params(labelsize=6.8)
    cbar.set_label("Recenter minus passive reward (USD per 1,500-hour window)",
                   fontsize=7.2)
    _save(fig, out, "retail-viability-frontier")


def plot_geometry(path: Path, out: Path) -> None:
    df = pd.read_csv(path)
    order = list(STRATEGY_LABELS)
    limit = float(max(abs(df.ci_low.min()), abs(df.ci_high.max())))
    fig, axes = plt.subplots(3, 2, figsize=(7.15, 5.8), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, pool in zip(axes.flat, CORE):
        p = df[df.pool == pool].set_index("strategy").reindex(order)
        y = np.arange(len(order))
        x = p.difference.to_numpy()
        lo = x - p.ci_low.to_numpy()
        hi = p.ci_high.to_numpy() - x
        learned = np.array([s in {"PPO", "A2C", "DQN", "QRDQN", "RECURRENTPPO"}
                            for s in order])
        ax.errorbar(x[~learned], y[~learned], xerr=[lo[~learned], hi[~learned]],
                    fmt="o", color="#5b5b5b", ecolor="#9a9a9a", capsize=2,
                    markersize=4, label="Heuristic")
        ax.errorbar(x[learned], y[learned], xerr=[lo[learned], hi[learned]],
                    fmt="s", color="#277da1", ecolor="#7ca9bd", capsize=2,
                    markersize=4, label="Learned")
        sig = p.p_holm.to_numpy() < 0.05
        ax.scatter(x[sig], y[sig], facecolors="none", edgecolors="#d1495b",
                   s=58, linewidths=1.2, zorder=4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(POOL_LABELS[pool], fontsize=7.5)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlim(-limit * 1.04, limit * 1.04)
        ax.set_yticks(y, [STRATEGY_LABELS[s] for s in order])
        ax.tick_params(labelsize=6.7)
        ax.invert_yaxis()
    for ax in axes[-1]:
        ax.set_xlabel("Matched minus nominal geometry (USD/window)", fontsize=7.2)
    axes[0, 0].legend(frameon=False, loc="lower left", fontsize=6.7)
    _save(fig, out, "action-geometry-effects")


def plot_headroom(path: Path, out: Path) -> None:
    """Two scoped headlines: local action disagreement and deployable policy lift."""
    df = pd.read_csv(path).set_index("pool").reindex(CORE)
    y = np.arange(len(CORE))
    labels = [POOL_LABELS[pool] for pool in CORE]
    blue, orange = "#0072B2", "#D55E00"
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.15, 2.6), constrained_layout=True, sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )

    for offset, horizon, marker, color, label in (
        (-0.10, "168h", "o", blue, "168h primary"),
        (+0.10, "24h", "D", orange, "24h robustness"),
    ):
        estimate = df[f"disagreement_{horizon}"].to_numpy(float)
        lo = df[f"disagreement_{horizon}_ci_low"].to_numpy(float)
        hi = df[f"disagreement_{horizon}_ci_high"].to_numpy(float)
        left.errorbar(
            estimate,
            y + offset,
            xerr=[estimate - lo, hi - estimate],
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2,
            markersize=4.5,
            linewidth=1,
            label=label,
        )
    left.axvline(0.10, color="#555555", linestyle="--", linewidth=0.9)
    left.set_xlim(0, 1)
    left.set_xticks(np.linspace(0, 1, 6),
                    [f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    left.set_xlabel("Oracle differs from fixed width", fontsize=7.5)
    left.set_yticks(y, labels)
    left.tick_params(labelsize=7)
    left.invert_yaxis()
    left.grid(axis="x", alpha=0.2)
    left.legend(frameon=False, fontsize=6.8, loc="upper left")

    estimate = df["contextual_minus_fixed"].to_numpy(float)
    lo = df["contextual_ci_low"].to_numpy(float)
    hi = df["contextual_ci_high"].to_numpy(float)
    significant = df["contextual_p_holm"].to_numpy(float) < 0.05
    right.errorbar(
        estimate,
        y,
        xerr=[estimate - lo, hi - estimate],
        fmt="s",
        color=blue,
        ecolor="#6BAED6",
        capsize=2,
        markersize=4.5,
        linewidth=1,
    )
    right.scatter(
        estimate[significant],
        y[significant],
        facecolors="none",
        edgecolors=orange,
        s=56,
        linewidths=1.1,
        zorder=4,
    )
    right.axvline(0, color="black", linewidth=0.9)
    right.grid(axis="x", alpha=0.2)
    right.set_xlabel("Decision stump minus fixed width (USD/window)", fontsize=7.5)
    right.tick_params(labelsize=7)
    right.tick_params(axis="y", labelleft=False)
    _save(fig, out, "state-contingent-headroom")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retail", type=Path)
    ap.add_argument("--geometry", type=Path)
    ap.add_argument("--headroom", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/figures"))
    args = ap.parse_args()
    if not args.retail and not args.geometry and not args.headroom:
        ap.error("pass --retail, --geometry, and/or --headroom")
    if args.retail:
        plot_retail(args.retail, args.out)
    if args.geometry:
        plot_geometry(args.geometry, args.out)
    if args.headroom:
        plot_headroom(args.headroom, args.out)


if __name__ == "__main__":
    main()
