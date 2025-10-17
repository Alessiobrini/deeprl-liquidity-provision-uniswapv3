import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

sys.path.append(str(Path(__file__).resolve().parents[1])) 

from custom_env import Uniswapv3Env
from baseline_strategies import PassiveWidthSweep

params = {
    "delta": 0.05,
    "action_values": [0, 10, 20, 30, 40, 50],  # example widths
    "filename": "rl-code/data_price_uni_h_time.csv",
    "x": 2,
    "gas": 5.0
}

df = pd.read_csv(params["filename"])

env = Uniswapv3Env(
    delta=params["delta"],
    action_values=np.array(params["action_values"]),
    market_data=df[["price"]],
    x=params["x"],
    gas=params["gas"]
)

pws = PassiveWidthSweep(width_candidates=[10, 20, 30, 40])
best_w, best_r = pws.evaluate(env)
print("Best width and raw reward:", best_w, best_r)