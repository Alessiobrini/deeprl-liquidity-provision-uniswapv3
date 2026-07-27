import sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
import numpy as np
from src.deeprl_liquidity_provision_uniswapv3.experiments.bakeoff import build_env, score, make_split
from src.deeprl_liquidity_provision_uniswapv3.policies import baselines as B

W=[100.,200.,500.,2000.]
key="usdc_weth_005"
pols=[B.Passive(), B.VolProportionalWidth(3), B.ILMinimizer(24), B.ILMinimizer(24, only_when_out=True),
      B.ReactiveRecentering(200,0.01,0.01), B.RecentreWhenOut(2000)]
print("Does the schedule actually bind? Same window, both schedules.\n")
print(f"{'policy':<42} {'hourly':>10} {'event':>10} {'differs':>8}")
print("-"*74)
for p in pols:
    row=[]
    for sch in ("hourly","event_driven"):
        e=build_env(key, 20, W, schedule=sch)
        row.append(score(e,p)[0])
    d = "yes" if abs(row[0]-row[1])>1e-6 else "no"
    print(f"{p.name:<42} {row[0]:>10,.0f} {row[1]:>10,.0f} {d:>8}")
