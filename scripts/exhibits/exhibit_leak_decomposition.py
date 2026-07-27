import sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
import numpy as np, pandas as pd
from src.deeprl_liquidity_provision_uniswapv3.experiments.agent_arm import train_env, score_agent, PPO_GRID, SEEDS
from src.deeprl_liquidity_provision_uniswapv3.experiments.bakeoff import make_split, PANEL, WINDOW, evaluate_on_test, select_on_val
from src.deeprl_liquidity_provision_uniswapv3.agents.registry import make_agent
from src.deeprl_liquidity_provision_uniswapv3.policies import baselines as B

key="usdc_weth_005"; widths=[100.,200.,500.,2000.]; sched="daily"; steps=20_000
n=len(pd.read_parquet(PANEL/f"{key}_hourly.parquet"))//WINDOW
split=make_split(n)

# Train the grid once; reuse for both selection rules so the ONLY difference is
# which split the config is chosen on.
runs=[]
for cfg in PPO_GRID:
    for s in SEEDS:
        m=make_agent("ppo", train_env(key,widths,split,sched), seed=s, **cfg)
        m.learn(total_timesteps=steps)
        runs.append(dict(cfg=cfg, seed=s, model=m,
                         val=score_agent(key,m,widths,split.val,sched).mean(),
                         test=score_agent(key,m,widths,split.test,sched).mean()))

# (a) honest: pick the config by VALIDATION, report its mean test
best_cfg=max(PPO_GRID, key=lambda c: np.mean([r["val"] for r in runs if r["cfg"]==c]))
honest=np.mean([r["test"] for r in runs if r["cfg"]==best_cfg])

# (b) the previous paper's rule: maximise on TEST, report that maximum
leaked_max=max(r["test"] for r in runs)
# (c) same rule but per-seed max, which is literally what train_optimized_ppo returned
leaked_per_seed=np.mean([max(r["test"] for r in runs if r["seed"]==s) for s in SEEDS])

# competitors, honest vs crippled to the old action grid
comp,_=select_on_val(key, B.candidate_grid(widths), widths, split)
comp_test=evaluate_on_test(key, comp, widths, split).mean()
old_widths=[45.,50.,55.]                    # the previous paper's grid, minus hold
comp_old,_=select_on_val(key, B.candidate_grid(old_widths), old_widths, split)
comp_old_test=evaluate_on_test(key, comp_old, old_widths, split).mean()

print(f"{key}: what each ingredient of the previous protocol was worth\n")
print(f"{'PPO, config chosen on VALIDATION (honest)':<48} {honest:>10,.0f}")
print(f"{'PPO, best-of-10 chosen on TEST (their rule)':<48} {leaked_max:>10,.0f}")
print(f"{'PPO, per-seed max on TEST (their exact code)':<48} {leaked_per_seed:>10,.0f}")
print(f"{'  -> inflation bought by the leak':<48} {leaked_per_seed-honest:>+10,.0f}\n")
print(f"{'best competitor, full action grid':<48} {comp_test:>10,.0f}   ({comp.name})")
print(f"{'best competitor, THEIR action grid {45,50,55}':<48} {comp_old_test:>10,.0f}   ({comp_old.name})")
print(f"{'  -> handicap imposed on the competitors':<48} {comp_old_test-comp_test:>+10,.0f}\n")
print(f"Honest gap   (PPO - competitor): {honest-comp_test:>+10,.0f}")
print(f"Their gap    (leaked PPO - crippled competitor): {leaked_per_seed-comp_old_test:>+10,.0f}")
