"""H2: does a learned policy mitigate the loss by more than the best competitor?

Same protocol the competitors faced:

    train on TRAIN windows -> select config on VALIDATION -> score ONCE on TEST

`tune()` never receives test. The agent's decision schedule is declared (daily by
default, per the pre-registration) rather than tuned, because it was chosen by
inspecting validation results and re-picking it after a test read would rebuild the
leak this rebuild exists to remove.

Seeds are averaged within a window before anything is reported: several seeds on one
price trajectory are not several observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor

from ..agents.registry import make_agent
from ..data.pools import POOLS
from ..envs.schedule import agent_schedule
from ..policies import baselines as B
from .bakeoff import (CAPITAL_USD, FEE_MODEL, PANEL, WINDOW, build_env,
                      evaluate_on_test, load_panel, make_split, score, select_on_val,
                      window_swaps)

# Deliberately small and declared up front. The competitors are selected from ~60
# configurations; a sprawling agent grid would make "matched budget" a fiction in
# the other direction from the original paper, which gave PPO 10 Optuna trials plus
# its choice of action space while ILMinimizer had no free parameters at all.
PPO_GRID = [
    dict(learning_rate=3e-4, ent_coef=0.01),
    dict(learning_rate=3e-4, ent_coef=0.0),
    dict(learning_rate=1e-3, ent_coef=0.01),
]
SEEDS = (42, 123, 256)


def train_env(key, widths, split, schedule, features="legacy", n_windows=None,
              reward_shaping="none", width_units="spacing", execution_widths=None):
    """One long episode over the TRAIN windows. Test is unreachable from here.

    Shaping belongs HERE and only here. Evaluation always runs an unshaped env, so
    `score` reads the true reward and a control variate can never touch a reported
    number.
    """
    p = POOLS[key]
    df = load_panel(key)
    lo = split.train[0] * WINDOW
    hi = (split.train[-1] + 1) * WINDOW
    seg = df.iloc[lo:hi].reset_index(drop=True)
    from ..envs.uniswap_v3 import UniswapV3Env
    env = UniswapV3Env(seg, swaps=window_swaps(key, seg), fee_model=FEE_MODEL,
                       fee_tier_pct=p.fee_tier_pct,
                       action_widths=np.asarray(widths), dec0=p.dec0, dec1=p.dec1,
                       capital_usd=CAPITAL_USD, gas_usd=5.0, warmup=168, allow_exit=False,
                       features=features, reward_shaping=reward_shaping,
                       width_units=width_units, execution_widths=execution_widths)
    return Monitor(agent_schedule(env, schedule))


class AgentPolicy:
    """Wraps a trained model so it plugs into the same scoring path as a heuristic.

    Threads the recurrent state. `predict()` RETURNS the next hidden state, and
    dropping it re-zeros the LSTM on every step, so a RecurrentPPO policy evaluated
    that way is not the policy that was trained: measured, 13 of 30 actions differed.
    Feed-forward algorithms return `None` here and are unaffected, so this costs
    nothing and is not conditional on the algorithm.
    """

    def __init__(self, model, name):
        self.model = model
        self.name = name
        self._state = None
        self._start = True

    def reset(self) -> None:
        self._state = None
        self._start = True

    def __call__(self, obs, env):
        a, self._state = self.model.predict(
            obs, state=self._state, episode_start=np.array([self._start]),
            deterministic=True)
        self._start = False
        return int(a)


def score_agent(key, model, widths, w_indices, schedule, features="legacy",
                width_units="spacing", execution_widths=None):
    out = []
    for wi in w_indices:
        env = build_env(key, wi, widths, schedule=schedule, features=features,
                        width_units=width_units, execution_widths=execution_widths)
        if env is None:
            continue
        out.append(score(env, AgentPolicy(model, "agent"))[0])
    return np.asarray(out)


def run_pool(key, algo, widths, schedule, steps):
    n = len(pd.read_parquet(PANEL / f"{key}_hourly.parquet")) // WINDOW
    split = make_split(n)

    # --- competitors: select on validation, score once on test ---------------
    best_comp, comp_val = select_on_val(key, B.candidate_grid(widths), widths, split)
    comp_test = evaluate_on_test(key, best_comp, widths, split)
    passive_test = evaluate_on_test(key, B.Passive(), widths, split)

    # --- agent: select config on validation, score once on test --------------
    best_cfg, best_val, best_models = None, -np.inf, None
    for cfg in PPO_GRID:
        models, vals = [], []
        for s in SEEDS:
            m = make_agent(algo, train_env(key, widths, split, schedule), seed=s, **cfg)
            m.learn(total_timesteps=steps)
            models.append(m)
            vals.append(score_agent(key, m, widths, split.val, schedule).mean())
        v = float(np.mean(vals))
        if v > best_val:
            best_cfg, best_val, best_models = cfg, v, models

    # seeds averaged WITHIN each window before anything is reported
    per_seed = np.vstack([score_agent(key, m, widths, split.test, schedule) for m in best_models])
    agent_test = per_seed.mean(axis=0)

    return dict(pool=key, comp_name=best_comp.name, comp_val=comp_val,
                comp_test=comp_test, passive_test=passive_test,
                agent_cfg=best_cfg, agent_val=best_val, agent_test=agent_test,
                agent_seed_spread=per_seed.mean(axis=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=[k for k, p in POOLS.items() if p.group == "core"])
    ap.add_argument("--widths", nargs="*", type=float, default=[45, 50, 55])
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--schedule", default="daily",
                    choices=["hourly", "daily", "weekly", "event_driven"])
    ap.add_argument("--steps", type=int, default=20_000)
    args = ap.parse_args()

    print(f"H2: {args.algo.upper()} ({args.schedule}) vs the best competitor.")
    print("Both selected on validation, both scored once on test.\n")
    rows = []
    for key in args.pools:
        r = run_pool(key, args.algo, args.widths, args.schedule, args.steps)
        rows.append(r)
        a, c, p = r["agent_test"].mean(), r["comp_test"].mean(), r["passive_test"].mean()
        print(f"{key}")
        print(f"   passive                 {p:>10,.0f}")
        print(f"   best competitor         {c:>10,.0f}   ({r['comp_name']})")
        print(f"   {args.algo.upper():<22} {a:>10,.0f}   (cfg {r['agent_cfg']}, "
              f"seed spread {r['agent_seed_spread'].std():,.0f})")
        print(f"   agent - competitor      {a-c:>+10,.0f}")
        print(f"   agent - passive         {a-p:>+10,.0f}\n")

    A = np.concatenate([r["agent_test"] for r in rows])
    C = np.concatenate([r["comp_test"] for r in rows])
    P = np.concatenate([r["passive_test"] for r in rows])
    print(f"{'POOLED':<24} {'agent':>10} {'competitor':>12} {'passive':>10}")
    print(f"{'':<24} {A.mean():>10,.0f} {C.mean():>12,.0f} {P.mean():>10,.0f}")
    print(f"\nagent - competitor: {A.mean()-C.mean():+,.0f}   "
          f"agent - passive: {A.mean()-P.mean():+,.0f}   (n={len(A)} test windows)")
    return rows


if __name__ == "__main__":
    main()
