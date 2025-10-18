"""
Test PPO vs baselines on rolling windows.
Computes dynamic n_steps and uses clip_range from YAML.
"""

import numpy as np
import pandas as pd
import os
import yaml
from pathlib import Path
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from custom_env_folder.custom_env import Uniswapv3Env, CustomMLPFeatureExtractor
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from baseline_strategies import (
    PassiveWidthSweep,
    VolProportionalWidth,
    ILMinimizer,
    ReactiveRecentering,
)

def evaluate_baselines(test_env, baseline_configs):
    rewards = {}
    for name, BaselineClass, kwargs in baseline_configs:
        baseline = BaselineClass(**kwargs)
        # fresh copy of the environment for each baseline
        env_copy = Uniswapv3Env(
            delta=test_env.delta,
            action_values=test_env.action_values,
            market_data=pd.DataFrame(test_env.market_data, columns=["price"]),
            x=test_env.initial_x,
            gas=test_env.gas,
        )
        _, reward = baseline.evaluate(env_copy)
        rewards[name] = reward
    return rewards

def evaluate_model(model, env):
    obs, _info = env.reset()
    total_reward = 0.0
    done, truncated = False, False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config")
    # load hyperparameters
    with open(os.path.join(config_dir, "uniswap_rl_param_1108.yaml"), "r") as f:
        params = yaml.safe_load(f)

    # load price data (two columns: timestamp, price)
    df = pd.read_csv(os.path.join(base_dir, params['filename']))
    prices = df[["price"]].reset_index(drop=True)

    # split into windows of length 1500 rows; drop last incomplete if necessary
    window_size = 1500
    windows = [prices.iloc[i:i+window_size].reset_index(drop=True)
               for i in range(0, len(prices), window_size)]
    if windows and len(windows[-1]) < window_size:
        windows.pop()

    ppo_rewards, baseline_rewards = [], {
        "PassiveWidthSweep": [],
        "VolProportionalWidth": [],
        "ILMinimizer": [],
        "ReactiveRecentering": [],
    }

    # iterate windows: train on 5 windows, test on the 6th
    for i in range(len(windows) - 5):
        print(f"Running rolling window {i}")
        train_df = pd.concat(windows[i:i+5], ignore_index=True)
        test_df = windows[i+5].reset_index(drop=True)

        # create environments
        train_env = Uniswapv3Env(
            delta=params["delta"],
            action_values=np.array(params["action_values"], dtype=float),
            market_data=train_df,
            x=params["x"],
            gas=params["gas_fee"],
        )
        test_env = Uniswapv3Env(
            delta=params["delta"],
            action_values=np.array(params["action_values"], dtype=float),
            market_data=test_df,
            x=params["x"],
            gas=params["gas_fee"],
        )

        # compute n_steps dynamically (like the original script)
        n_steps = max(1, len(train_env.market_data) // 3)

        # build PPO agent
        policy_kwargs = dict(
            features_extractor_class=CustomMLPFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                activation=params["activation"],
                hidden_dim=params["dim_hidden_layers"],
            ),
        )
        model = PPO(
            "MlpPolicy",
            Monitor(train_env),
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            vf_coef=params["vf_coef"],
            gae_lambda=params["gae_lambda"],
            n_steps=n_steps,
            batch_size=params["batch_size"],
            target_kl=params["target_kl"],
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        # train
        model.learn(total_timesteps=params["total_timesteps_1"])

        # evaluate PPO on test window
        ppo_reward = evaluate_model(model, Monitor(test_env))
        ppo_rewards.append(ppo_reward)

        # evaluate baselines on test window
        baseline_configs = [
            ("PassiveWidthSweep",
             PassiveWidthSweep,
             {"width_candidates": [w for w in params["action_values"] if w != 0],
              "deposit_action_idx": 1}),
            ("VolProportionalWidth",
             VolProportionalWidth,
             {"k_values": [5, 10, 15], "base_factor": 100.0, "deposit_action_idx": 1}),
            ("ILMinimizer",
             ILMinimizer,
             {"horizon_hours": 24, "base_factor": 100.0, "deposit_action_idx": 1}),
            ("ReactiveRecentering",
             ReactiveRecentering,
             {"vol_thresholds": [0.005, 0.01],
              "price_jump_thresholds": [0.005, 0.01],
              "width_ticks": params["action_values"][1],
              "deposit_action_idx": 1}),
        ]
        base_rews = evaluate_baselines(test_env, baseline_configs)
        for name, val in base_rews.items():
            baseline_rewards[name].append(val)

    # summarize results
    print("\n========= Summary across windows =========")
    ppo_mean = np.mean(ppo_rewards)
    ppo_std  = np.std(ppo_rewards)
    print(f"PPO: mean = {ppo_mean}, std = {ppo_std}")
    for key, vals in baseline_rewards.items():
        print(f"{key}: mean = {np.mean(vals):}, std = {np.std(vals):}")
