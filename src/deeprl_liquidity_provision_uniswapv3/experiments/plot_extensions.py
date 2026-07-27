"""Publication-oriented figures for the action-geometry and retail extensions."""
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
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_retail(path: Path, out: Path) -> None:
    df = pd.read_csv(path)
    df = df[df.regime == "conversion"]
    capitals = sorted(df.capital.unique())
    gas = sorted(df.gas.unique())
    limit = float(np.abs(df.difference).max())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.2), constrained_layout=True,
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
                        ha="center", va="center", fontsize=7.5, color=color)
        ax.set_title(POOL_LABELS[pool], fontsize=10)
        ax.set_xticks(range(len(gas)), [f"${x}" for x in gas])
        ax.set_yticks(range(len(capitals)), [f"${x/1000:g}k" for x in capitals])
        ax.axvline(-0.5, color="none")
    for ax in axes[-1]:
        ax.set_xlabel("Gas per recenter")
    for ax in axes[:, 0]:
        ax.set_ylabel("Position capital")
    cbar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.02)
    cbar.set_label("Recenter minus passive reward (USD per 1,500-hour window)")
    fig.suptitle("Retail viability with pool fee + 5 bp conversion friction", fontsize=13)
    _save(fig, out, "retail-viability-frontier")


def plot_geometry(path: Path, out: Path) -> None:
    df = pd.read_csv(path)
    order = list(STRATEGY_LABELS)
    limit = float(max(abs(df.ci_low.min()), abs(df.ci_high.max())))
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True,
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
        ax.set_title(POOL_LABELS[pool], fontsize=10)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlim(-limit * 1.04, limit * 1.04)
        ax.set_yticks(y, [STRATEGY_LABELS[s] for s in order])
        ax.invert_yaxis()
    for ax in axes[-1]:
        ax.set_xlabel("Matched geometry minus paper geometry (USD/window)")
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.suptitle("Effect of matching executable band geometry across fee tiers", fontsize=13)
    _save(fig, out, "action-geometry-effects")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retail", type=Path)
    ap.add_argument("--geometry", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/figures"))
    args = ap.parse_args()
    if not args.retail and not args.geometry:
        ap.error("pass --retail and/or --geometry")
    if args.retail:
        plot_retail(args.retail, args.out)
    if args.geometry:
        plot_geometry(args.geometry, args.out)


if __name__ == "__main__":
    main()
