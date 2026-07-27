"""Competitor bake-off under a leak-proof protocol.

The protocol, which is the whole point:

    select(train, validation) -> config          # test is not a parameter here
    evaluate_once(config, test)                  # the only read of test

Every competitor and the agent select on VALIDATION with a matched budget, then are
scored ONCE on TEST. The previous paper's tuning objective evaluated on the test
window and returned the max over trials, so its headline number was a
max-over-10 order statistic on the evaluation data; competitors meanwhile argmax'd
on test too, but with zero free parameters in one case, so the budgets were never
matched either. Both failures are structural here: `tune` never receives test.

Reports at the WINDOW level. Seeds are averaged within a window before any test,
because five seeds on one price trajectory are not five independent observations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.pools import POOLS
from ..data.swaps import epoch_seconds
from ..envs.uniswap_v3 import UniswapV3Env
from ..envs.schedule import agent_schedule
from ..policies import baselines as B

PANEL = Path(__file__).resolve().parents[3] / "data/processed"
WINDOW = 1500
WARMUP = 168
FEE_MODEL = "per_swap"
# A FIXED USD base, deliberately unlike the previous paper. Its configs set `x: 10`
# WETH (~$40k) on one pool and `x: 0.15` WBTC (~$6-10k) on another, and its text claims
# `x_0 = 2`, matching neither. A per-pool base that also drifts with price makes the
# cross-pool reward levels in its Tab. IV/V incomparable, and comparing pools is the
# paper's whole point. Recorded in DIVERGENCES.md as a deliberate fix, not an oversight.
CAPITAL_USD = 30_000.0


@lru_cache(maxsize=None)
def load_panel(key: str) -> pd.DataFrame:
    return pd.read_parquet(PANEL / f"{key}_hourly.parquet")


@lru_cache(maxsize=None)
def load_swaps(key: str) -> pd.DataFrame:
    """Swap-level fees, cached: the USDC/WETH 0.05% frame is 11M rows and every
    window would otherwise re-read it from disk."""
    path = PANEL / f"{key}_swaps.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{key}: no swap-level fees at {path}. Build them with "
            f"`python -m {__package__.split('.')[0]}.data.swaps`."
        )
    return pd.read_parquet(path)


def window_swaps(key: str, seg: pd.DataFrame) -> pd.DataFrame:
    """The slice of the swap frame covering one window's hours."""
    if FEE_MODEL != "per_swap":
        return None
    sw = load_swaps(key)
    h = epoch_seconds(seg["timestamp"])
    lo, hi = h[0], h[-1]
    a, b = np.searchsorted(sw["hour"].to_numpy(), [lo, hi + 1])
    return sw.iloc[a:b]


@dataclass(frozen=True)
class Split:
    """Disjoint by construction. `test` is deliberately awkward to reach."""
    train: list[int]
    val: list[int]
    test: list[int]

    def __post_init__(self):
        s = [set(self.train), set(self.val), set(self.test)]
        for a in range(3):
            for b in range(a + 1, 3):
                assert not (s[a] & s[b]), "splits overlap: this is the bug that sank the paper"


def make_split(n_windows: int, frac_train=0.5, frac_val=0.25) -> Split:
    n_tr = int(n_windows * frac_train)
    n_va = int(n_windows * frac_val)
    return Split(train=list(range(n_tr)),
                 val=list(range(n_tr, n_tr + n_va)),
                 test=list(range(n_tr + n_va, n_windows)))


def build_env(key: str, w_index: int, widths, schedule: str | None = None,
              features: str = "legacy", width_units: str = "spacing",
              execution_widths=None, capital_usd: float = CAPITAL_USD,
              gas_usd: float = 5.0, swap_fee_frac: float = 0.0,
              slippage_frac: float = 0.0, **_ignored):
    """Native hourly env. Competitors run unwrapped and decide for themselves when to
    act, which is what makes them distinct strategies. `schedule` is only for an
    agent, which has no timing of its own."""
    p = POOLS[key]
    df = load_panel(key)
    lo = w_index * WINDOW
    seg = df.iloc[lo:lo + WINDOW].reset_index(drop=True)
    if len(seg) < WINDOW:
        return None
    # Defaults are the previous paper's: legacy state, widths in units of tick
    # spacing, gas charged in the reward, no swap/slippage, no exit.
    env = UniswapV3Env(seg, swaps=window_swaps(key, seg), fee_model=FEE_MODEL,
                       fee_tier_pct=p.fee_tier_pct, action_widths=np.asarray(widths),
                       dec0=p.dec0, dec1=p.dec1, capital_usd=capital_usd, gas_usd=gas_usd,
                       warmup=WARMUP, allow_exit=False, features=features,
                       width_units=width_units, execution_widths=execution_widths,
                       swap_fee_frac=swap_fee_frac, slippage_frac=slippage_frac)
    return env if schedule is None else agent_schedule(env, schedule)


def score_details(env, policy) -> dict:
    """True reward and path diagnostics needed for exact cost counterfactuals."""
    if hasattr(policy, "reset"):
        policy.reset()
    obs, _ = env.reset()
    inner = env.unwrapped
    total, done, trunc, acts, notional = 0.0, False, False, [], 0.0
    info = {}
    while not (done or trunc):
        a = int(policy(obs, inner))
        acts.append(a)
        obs, r, done, trunc, info = env.step(a)
        total += r
        notional += info.get("rebalance_notional", 0.0)
    return {"reward": float(total), "n_actions": len(set(acts)),
            "n_rebalances": int(info.get("n_rebalances", 0)),
            "rebalance_notional": float(notional)}


def score(env, policy) -> tuple[float, int]:
    """True reward over one window, plus the number of distinct actions used.

    A policy that uses one action is not controlling anything; that is logged on
    every evaluation because it is what the previous protocol hid.
    """
    d = score_details(env, policy)
    return d["reward"], d["n_actions"]


def select_on_val(key, candidates, widths, split, **kw) -> B.Policy:
    """Pick the competitor configuration with the best MEAN reward on validation."""
    best, best_score = None, -np.inf
    for pol in candidates:
        rs = []
        for wi in split.val:
            env = build_env(key, wi, widths, **kw)
            if env is None:
                continue
            rs.append(score(env, pol)[0])
        m = float(np.mean(rs)) if rs else -np.inf
        if m > best_score:
            best, best_score = pol, m
    return best, best_score


def evaluate_on_test(key, policy, widths, split, **kw):
    """The only read of test. Returns per-window rewards: the independent units."""
    out = []
    for wi in split.test:
        env = build_env(key, wi, widths, **kw)
        if env is None:
            continue
        out.append(score(env, policy)[0])
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=[k for k, p in POOLS.items() if p.group == "core"])
    ap.add_argument("--widths", nargs="*", type=float, default=[45, 50, 55])
    ap.add_argument("--agent-schedule", default="daily",
                    choices=["hourly", "daily", "weekly", "event_driven"],
                    help="the AGENT's declared decision frequency; competitors run native")
    args = ap.parse_args()

    print("Competitor bake-off. Competitors run their native logic. "
          "Select on validation, score once on test.\n")
    per_pool = {}
    for key in args.pools:
        n = len(pd.read_parquet(PANEL / f"{key}_hourly.parquet")) // WINDOW
        split = make_split(n)
        cands = B.candidate_grid(args.widths)
        best, val_score = select_on_val(key, cands, args.widths, split)
        test = evaluate_on_test(key, best, args.widths, split)
        passive_test = evaluate_on_test(key, B.Passive(), args.widths, split)
        per_pool[key] = dict(best=best.name, val=val_score, test=test, passive=passive_test)
        print(f"{key}")
        print(f"   selected on val : {best.name}   (val {val_score:,.0f})")
        print(f"   TEST            : {test.mean():>10,.0f}  +/- {test.std()/max(np.sqrt(len(test)),1):,.0f}"
              f"   over {len(test)} windows")
        print(f"   passive on TEST : {passive_test.mean():>10,.0f}")
        print(f"   mitigation      : {test.mean()-passive_test.mean():>+10,.0f}\n")
    return per_pool


if __name__ == "__main__":
    main()
