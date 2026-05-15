from __future__ import annotations

import math
import shutil

import matplotlib.pyplot as plt


COLUMNWIDTH = 418.25


def apply_paper_style(fsize: int = 16) -> None:
    """Apply a paper-friendly Matplotlib style with LaTeX fallback."""
    use_tex = shutil.which("latex") is not None
    params = {
        "text.usetex": use_tex,
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": fsize,
        "legend.fontsize": fsize,
        "xtick.labelsize": fsize,
        "ytick.labelsize": fsize,
        "axes.titlesize": fsize,
        "axes.labelsize": fsize,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
    }
    plt.rcParams.update(params)


def set_size(width: float, fraction: float = 1.0, subplots: tuple[int, int] = (1, 1)) -> tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX."""
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def create_figure(width: float = COLUMNWIDTH, fraction: float = 1.0):
    return plt.subplots(figsize=set_size(width=width, fraction=fraction))


def create_figures(
    nrows: int,
    ncols: int,
    width: float = COLUMNWIDTH,
    fraction: float = 1.0,
    tupsize: tuple[float, float] | None = None,
):
    if tupsize is not None:
        return plt.subplots(nrows=nrows, ncols=ncols, figsize=tupsize)
    return plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=set_size(width=width, fraction=fraction, subplots=(nrows, ncols)),
    )
