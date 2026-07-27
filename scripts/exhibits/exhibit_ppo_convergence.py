import sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
import numpy as np, pandas as pd
from src.deeprl_liquidity_provision_uniswapv3.experiments.agent_arm import train_env, score_agent, PPO_GRID
from src.deeprl_liquidity_provision_uniswapv3.experiments.bakeoff import make_split, PANEL, WINDOW
from src.deeprl_liquidity_provision_uniswapv3.agents.registry import make_agent

key="usdc_weth_005"; widths=[100.,200.,500.,2000.]; sched="daily"
n=len(pd.read_parquet(PANEL/f"{key}_hourly.parquet"))//WINDOW
split=make_split(n)
print(f"Has PPO converged by 20k? VALIDATION only; test untouched.")
print(f"train span: {len(split.train)} windows -> ~{len(split.train)*1500//24} daily decisions/episode\n")
print(f"{'steps':>9} {'episodes':>9} {'val mean':>10} {'seed sd':>9} {'actions':>9}")
print("-"*52)
ckpts=[10_000,20_000,50_000,100_000,200_000]
models={s: make_agent("ppo", train_env(key,widths,split,sched), seed=s, learning_rate=3e-4, ent_coef=0.01)
        for s in (42,123,256)}
prev=0
for ck in ckpts:
    vals=[]
    for s,m in models.items():
        m.learn(total_timesteps=ck-prev, reset_num_timesteps=False)
        vals.append(score_agent(key,m,widths,split.val,sched).mean())
    prev=ck
    print(f"{ck:>9,} {ck//875:>9} {np.mean(vals):>10,.0f} {np.std(vals):>9,.0f}")
