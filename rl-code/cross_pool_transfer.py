"""
Cross-Pool Transfer Learning Experiments

Tests two types of transfer:
1. Cross-Asset Transfer (same time period, different asset)
2. Temporal Transfer (future time period, different asset)

Hypothesis: If BTC/ETH are correlated, model should perform better on same time period
even with different asset, compared to future period.
"""

import numpy as np
import pandas as pd
import os
import yaml
from pathlib import Path
import sys
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from custom_env_folder.custom_env import Uniswapv3Env, CustomMLPFeatureExtractor


def load_pool_data(pool_name):
    """Load data and config for a specific pool."""
    base_dir = Path(__file__).parent

    if pool_name == 'weth':
        config_file = "uniswap_rl_param_1108.yaml"
        data_file = "data_price_uni_h_time.csv"
    elif pool_name == 'wbtc':
        config_file = "pool_wbtc_usdc_030.yaml"
        data_file = "data/data_price_wbtc_usdc_030_h_time.csv"
    else:
        raise ValueError(f"Unknown pool: {pool_name}")

    # Load config
    with open(base_dir / "config" / config_file, "r") as f:
        config = yaml.safe_load(f)

    # Load data
    df = pd.read_csv(base_dir / data_file)

    return config, df[['price']]


def train_ppo(train_df, config, seed=42):
    """Train PPO model on training data."""
    print(f"    Training PPO (seed={seed})...")

    # Create environment
    env = Uniswapv3Env(
        delta=config["delta"],
        action_values=np.array(config["action_values"], dtype=float),
        market_data=train_df,
        x=config["x"],
        gas=config["gas_fee"],
        reward_type='IL',
    )

    # Policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomMLPFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            activation=config["activation"],
            hidden_dim=config["dim_hidden_layers"],
        ),
    )

    # Create model
    model = PPO(
        "MlpPolicy",
        Monitor(env),
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        gae_lambda=config["gae_lambda"],
        n_steps=max(1, len(train_df) // 3),
        batch_size=config["batch_size"],
        target_kl=config["target_kl"],
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
    )

    # Train
    model.learn(total_timesteps=config["total_timesteps_1"], progress_bar=False)

    print(f"    Training completed")
    return model


def evaluate_model(model, test_df, target_config, seed=42):
    """Evaluate model on test data using target pool configuration."""
    # Create test environment with target pool config
    test_env = Uniswapv3Env(
        delta=target_config["delta"],
        action_values=np.array(target_config["action_values"], dtype=float),
        market_data=test_df,
        x=target_config["x"],
        gas=target_config["gas_fee"],
        reward_type='IL',
    )

    obs, _info = test_env.reset(seed=seed)
    total_reward = 0.0
    done, truncated = False, False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        total_reward += reward

    return total_reward


def run_transfer_experiments(n_seeds=3):
    """
    Run comprehensive transfer learning experiments.

    Experiments:
    1. Train WETH -> Test WETH (same period) - Baseline
    2. Train WETH -> Test WETH (future period) - Temporal generalization
    3. Train WETH -> Test WBTC (same period) - Cross-asset transfer
    4. Train WETH -> Test WBTC (future period) - Cross-asset + temporal
    5-8. Reverse: Train WBTC -> Test WETH (same 4 scenarios)
    """
    print("="*70)
    print("Cross-Pool Transfer Learning Experiments")
    print("="*70)

    # Load data
    print("\nLoading pool data...")
    weth_config, weth_data = load_pool_data('weth')
    wbtc_config, wbtc_data = load_pool_data('wbtc')

    print(f"  WETH data: {len(weth_data):,} hours")
    print(f"  WBTC data: {len(wbtc_data):,} hours")

    # Find overlapping time period (WETH has less data)
    # Use WETH's full range and find corresponding WBTC period
    max_hours = min(len(weth_data), len(wbtc_data))

    # Split into periods
    # 70% training, 15% same-period test, 15% future test
    train_split = int(max_hours * 0.70)
    same_period_split = int(max_hours * 0.85)

    print(f"\nTime period splits:")
    print(f"  Training period: 0 - {train_split:,} hours")
    print(f"  Same-period test: 0 - {same_period_split:,} hours (overlaps with training)")
    print(f"  Future-period test: {train_split:,} - {same_period_split:,} hours")

    # Prepare data splits
    weth_train = weth_data.iloc[:train_split].reset_index(drop=True)
    weth_same_test = weth_data.iloc[:same_period_split].reset_index(drop=True)
    weth_future_test = weth_data.iloc[train_split:same_period_split].reset_index(drop=True)

    wbtc_train = wbtc_data.iloc[:train_split].reset_index(drop=True)
    wbtc_same_test = wbtc_data.iloc[:same_period_split].reset_index(drop=True)
    wbtc_future_test = wbtc_data.iloc[train_split:same_period_split].reset_index(drop=True)

    # Storage for results
    results = {
        'WETH->WETH_same': [],
        'WETH->WETH_future': [],
        'WETH->WBTC_same': [],
        'WETH->WBTC_future': [],
        'WBTC->WBTC_same': [],
        'WBTC->WBTC_future': [],
        'WBTC->WETH_same': [],
        'WBTC->WETH_future': [],
    }

    seeds = [42, 123, 256, 512, 1024][:n_seeds]

    # Run experiments for each seed
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"Seed {seed_idx + 1}/{n_seeds} (seed={seed})")
        print(f"{'='*70}")

        # Experiment 1-4: Train on WETH
        print("\n[1/2] Training on WETH...")
        weth_model = train_ppo(weth_train, weth_config, seed=seed)

        print("  Testing on:")

        # Test on WETH same period
        reward = evaluate_model(weth_model, weth_same_test, weth_config, seed=seed)
        results['WETH->WETH_same'].append(reward)
        print(f"    WETH (same period): {reward:.2f}")

        # Test on WETH future period
        reward = evaluate_model(weth_model, weth_future_test, weth_config, seed=seed)
        results['WETH->WETH_future'].append(reward)
        print(f"    WETH (future period): {reward:.2f}")

        # Test on WBTC same period (cross-asset transfer)
        reward = evaluate_model(weth_model, wbtc_same_test, wbtc_config, seed=seed)
        results['WETH->WBTC_same'].append(reward)
        print(f"    WBTC (same period): {reward:.2f}")

        # Test on WBTC future period (cross-asset + temporal)
        reward = evaluate_model(weth_model, wbtc_future_test, wbtc_config, seed=seed)
        results['WETH->WBTC_future'].append(reward)
        print(f"    WBTC (future period): {reward:.2f}")

        # Experiment 5-8: Train on WBTC
        print("\n[2/2] Training on WBTC...")
        wbtc_model = train_ppo(wbtc_train, wbtc_config, seed=seed)

        print("  Testing on:")

        # Test on WBTC same period
        reward = evaluate_model(wbtc_model, wbtc_same_test, wbtc_config, seed=seed)
        results['WBTC->WBTC_same'].append(reward)
        print(f"    WBTC (same period): {reward:.2f}")

        # Test on WBTC future period
        reward = evaluate_model(wbtc_model, wbtc_future_test, wbtc_config, seed=seed)
        results['WBTC->WBTC_future'].append(reward)
        print(f"    WBTC (future period): {reward:.2f}")

        # Test on WETH same period (cross-asset transfer)
        reward = evaluate_model(wbtc_model, weth_same_test, weth_config, seed=seed)
        results['WBTC->WETH_same'].append(reward)
        print(f"    WETH (same period): {reward:.2f}")

        # Test on WETH future period (cross-asset + temporal)
        reward = evaluate_model(wbtc_model, weth_future_test, weth_config, seed=seed)
        results['WBTC->WETH_future'].append(reward)
        print(f"    WETH (future period): {reward:.2f}")

    # Analyze results
    print(f"\n\n{'='*70}")
    print("TRANSFER LEARNING RESULTS")
    print(f"{'='*70}\n")

    # Create results DataFrame
    summary_data = []

    for experiment, rewards in results.items():
        train_pool, rest = experiment.split('->')
        test_pool, period = rest.split('_')

        rewards_arr = np.array(rewards)

        summary_data.append({
            'Experiment': experiment,
            'Train_Pool': train_pool,
            'Test_Pool': test_pool,
            'Period': period,
            'Mean': np.mean(rewards_arr),
            'Std': np.std(rewards_arr, ddof=1),
            'Min': np.min(rewards_arr),
            'Max': np.max(rewards_arr),
        })

    summary_df = pd.DataFrame(summary_data)

    # Print results
    print(summary_df.to_string(index=False))

    # Hypothesis testing
    print(f"\n{'='*70}")
    print("HYPOTHESIS TESTS")
    print(f"{'='*70}\n")

    # Test 1: WETH->WBTC same period vs future period
    same = np.array(results['WETH->WBTC_same'])
    future = np.array(results['WETH->WBTC_future'])
    t_stat, p_value = stats.ttest_ind(same, future)

    print("H1: WETH->WBTC performs better on same period than future")
    print(f"  Same period mean: {np.mean(same):.2f} ± {np.std(same, ddof=1):.2f}")
    print(f"  Future period mean: {np.mean(future):.2f} ± {np.std(future, ddof=1):.2f}")
    print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    print(f"  Result: {'✓ CONFIRMED' if (np.mean(same) > np.mean(future) and p_value < 0.05) else '✗ NOT SIGNIFICANT'}\n")

    # Test 2: WBTC->WETH same period vs future period
    same = np.array(results['WBTC->WETH_same'])
    future = np.array(results['WBTC->WETH_future'])
    t_stat, p_value = stats.ttest_ind(same, future)

    print("H2: WBTC->WETH performs better on same period than future")
    print(f"  Same period mean: {np.mean(same):.2f} ± {np.std(same, ddof=1):.2f}")
    print(f"  Future period mean: {np.mean(future):.2f} ± {np.std(future, ddof=1):.2f}")
    print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    print(f"  Result: {'✓ CONFIRMED' if (np.mean(same) > np.mean(future) and p_value < 0.05) else '✗ NOT SIGNIFICANT'}\n")

    # Test 3: Cross-asset transfer effectiveness
    weth_weth_same = np.array(results['WETH->WETH_same'])
    weth_wbtc_same = np.array(results['WETH->WBTC_same'])
    transfer_ratio = np.mean(weth_wbtc_same) / np.mean(weth_weth_same) * 100

    print("H3: Cross-asset transfer preserves performance")
    print(f"  WETH->WETH (same): {np.mean(weth_weth_same):.2f}")
    print(f"  WETH->WBTC (same): {np.mean(weth_wbtc_same):.2f}")
    print(f"  Transfer efficiency: {transfer_ratio:.1f}%\n")

    # Save results
    output_dir = Path(__file__).parent / "output" / "transfer_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "transfer_learning_results.csv", index=False)

    # Save detailed results
    detailed_df = pd.DataFrame(results)
    detailed_df.to_csv(output_dir / "transfer_learning_detailed.csv", index=False)

    print(f"Results saved to: {output_dir}")

    return summary_df, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Cross-pool transfer learning experiments')
    parser.add_argument('--seeds', type=int, default=3, help='Number of random seeds')
    args = parser.parse_args()

    summary_df, results = run_transfer_experiments(n_seeds=args.seeds)
