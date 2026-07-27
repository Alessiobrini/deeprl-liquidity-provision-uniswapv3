import sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
import numpy as np, pandas as pd
from src.deeprl_liquidity_provision_uniswapv3.experiments.agent_arm import train_env, score_agent, PPO_GRID, SEEDS
from src.deeprl_liquidity_provision_uniswapv3.experiments.bakeoff import make_split, PANEL, WINDOW
from src.deeprl_liquidity_provision_uniswapv3.agents.registry import make_agent
from src.deeprl_liquidity_provision_uniswapv3.data.pools import POOLS

widths=[100.,200.,500.,2000.]; sched="event_driven"; steps=20_000
core=[k for k,p in POOLS.items() if p.group=="core"]
splits={k: make_split(len(pd.read_parquet(PANEL/f"{k}_hourly.parquet"))//WINDOW) for k in core}

print("Does the rejected paper's 13-feature state rescue RL?")
print("Same protocol, same schedule, same budget. Only the observation changes.\n")
print(f"{'features':<10} {'pooled val':>11} {'pooled TEST':>12} {'vs passive':>11}")
print("-"*50)
for feats in ("compact","legacy"):
    best_i,best_v,store=None,-np.inf,{}
    for i,cfg in enumerate(PPO_GRID):
        store[i]={k:[make_agent("ppo", train_env(k,widths,splits[k],sched,features=feats), seed=s, **cfg)
                     for s in SEEDS] for k in core}
        for k in core:
            for m in store[i][k]: m.learn(total_timesteps=steps)
        v=np.mean([score_agent(k,m,widths,splits[k].val,sched,features=feats).mean()
                   for k in core for m in store[i][k]])
        if v>best_v: best_i,best_v=i,v
    te=np.concatenate([np.vstack([score_agent(k,m,widths,splits[k].test,sched,features=feats)
                                  for m in store[best_i][k]]).mean(axis=0) for k in core])
    print(f"{feats:<10} {best_v:>11,.0f} {te.mean():>12,.0f} {te.mean()-(-3151):>+11,.0f}")
print("\nreference: global rule RecentreWhenOut(w=2000) TEST -829 (+2,322 vs passive)")
