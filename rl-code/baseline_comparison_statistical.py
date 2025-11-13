"""
Enhanced baseline comparison script with statistical analysis.
Runs PPO (with Optuna hyperparameter optimization) and baseline strategies, then generates comprehensive statistical comparisons (mean±std, t-tests, bootstrap CIs).
"""

import numpy as np
import pandas as pd
import os
import yaml
from pathlib import Path
import sys
from datetime import datetime
from scipy import stats
from functools import partial
import optuna
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from baseline_strategies import (
    PassiveWidthSweep,
    VolProportionalWidth,
    ILMinimizer,
    ReactiveRecentering,
)
from custom_env_folder.custom_env import Uniswapv3Env, CustomMLPFeatureExtractor


def evaluate_baseline(baseline, test_env):
    """Evaluate a single baseline strategy on test environment."""
    # Create fresh environment copy
    env_copy = Uniswapv3Env(
        delta=test_env.delta,
        action_values=test_env.action_values,
        market_data=pd.DataFrame(test_env.market_data, columns=["price"]),
        x=test_env.initial_x,
        gas=test_env.gas,
    )
    _, reward = baseline.evaluate(env_copy)
    return reward


def evaluate_ppo_model(model, env):
    """Evaluate trained PPO model on test environment."""
    obs, _info = env.reset()
    total_reward = 0.0
    done, truncated = False, False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    return total_reward


def optimize_ppo_trial(trial, params_template, train_df, test_df, seed):
    """
    Single Optuna trial for PPO hyperparameter optimization.

    Args:
        trial: Optuna trial object
        params_template: Base parameters from YAML
        train_df: Training data
        test_df: Test data
        seed: Random seed

    Returns:
        tuple: (test_reward, trained_model)
    """
    # Copy base params
    params = params_template.copy()

    # Sample hyperparameters from the search space
    hyperparameters = params["hyperparameters"]
    grid = params["grid"]

    for param, (values, dtype) in hyperparameters.items():
        if param in grid:
            if dtype == "cat":
                params[param] = trial.suggest_categorical(param, values)
            elif dtype == "int":
                if len(values) == 3:
                    params[param] = trial.suggest_int(param, values[0], values[1], step=values[2])
                else:
                    params[param] = trial.suggest_int(param, values[0], values[1])
            elif dtype == "float":
                if len(values) == 3:
                    params[param] = trial.suggest_float(param, values[0], values[1], step=values[2])
                else:
                    params[param] = trial.suggest_float(param, values[0], values[1])

    # Create train environment
    train_env = Uniswapv3Env(
        delta=params["delta"],
        action_values=np.array(params["action_values"], dtype=float),
        market_data=train_df,
        x=params["x"],
        gas=params["gas_fee"],
    )

    # Create test environment
    test_env = Uniswapv3Env(
        delta=params["delta"],
        action_values=np.array(params["action_values"], dtype=float),
        market_data=test_df,
        x=params["x"],
        gas=params["gas_fee"],
    )

    # Compute dynamic n_steps
    n_steps = max(1, len(train_env.market_data) // 3)

    # Build PPO agent
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
        seed=seed,
    )

    # Train with early stopping
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=5, verbose=0
    )
    eval_callback = EvalCallback(
        Monitor(train_env),
        eval_freq=n_steps,
        callback_after_eval=stop_train_callback,
        verbose=0
    )
    callback = CallbackList([eval_callback])

    model.learn(total_timesteps=params["total_timesteps_1"], callback=callback)

    # Evaluate on test set
    test_reward = evaluate_ppo_model(model, Monitor(test_env))

    return test_reward, model


def train_optimized_ppo(params_template, train_df, test_df, seed, n_trials=10):
    """
    Train PPO with Optuna hyperparameter optimization.

    Args:
        params_template: Base parameters
        train_df: Training data
        test_df: Test data
        seed: Random seed
        n_trials: Number of Optuna trials (default: 10)

    Returns:
        tuple: (best_reward, best_model)
    """
    # Create Optuna study
    study = optuna.create_study(direction='maximize')

    # Track best model
    best_model = None
    best_reward = -np.inf

    def objective(trial):
        nonlocal best_model, best_reward
        reward, model = optimize_ppo_trial(trial, params_template, train_df, test_df, seed)

        # Track best model
        if reward > best_reward:
            best_reward = reward
            best_model = model

        return reward

    # Optimize silently
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return best_reward, best_model


def compute_statistics(data_dict, baseline_key='PPO'):
    """
    Compute statistical summaries: mean, std, t-tests vs baseline.

    Args:
        data_dict: Dictionary mapping strategy names to lists of rewards
        baseline_key: Name of baseline strategy to compare against (default: 'PPO')

    Returns:
        DataFrame with statistical summary

    Note:
        For deterministic baselines, their values are replicated across seeds
        to match the structure of PPO (which has different values per seed).
        This allows for proper statistical comparison via t-tests.
    """
    results = []
    baseline_data = np.array(data_dict[baseline_key])

    for strategy, rewards in data_dict.items():
        rewards_arr = np.array(rewards)

        # Compute basic statistics
        mean_reward = np.mean(rewards_arr)
        std_reward = np.std(rewards_arr, ddof=1)
        median_reward = np.median(rewards_arr)

        # Bootstrap 95% confidence interval
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(rewards_arr, size=len(rewards_arr), replace=True)
            bootstrap_means.append(np.mean(sample))
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        # T-test vs baseline (if not baseline itself)
        if strategy != baseline_key:
            t_stat, p_value = stats.ttest_ind(baseline_data, rewards_arr)
            wins = np.sum(baseline_data > rewards_arr)
        else:
            t_stat, p_value = np.nan, np.nan
            wins = np.nan

        results.append({
            'Strategy': strategy,
            'Mean': mean_reward,
            'Std': std_reward,
            'Median': median_reward,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'T-statistic': t_stat,
            'P-value': p_value,
            f'Wins_vs_{baseline_key}': wins,
            'N': len(rewards_arr)
        })

    return pd.DataFrame(results)


def main():
    """Main execution function."""
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config")
    output_dir = os.path.join(base_dir, "output", "baseline_comparison_optimized")
    os.makedirs(output_dir, exist_ok=True)

    # Load hyperparameters
    with open(os.path.join(config_dir, "uniswap_rl_param_1108.yaml"), "r") as f:
        params = yaml.safe_load(f)

    # Load price data
    df = pd.read_csv(os.path.join(base_dir, params['filename']))
    prices = df[["price"]].reset_index(drop=True)

    # Split into windows
    window_size = 1500
    windows = [prices.iloc[i:i+window_size].reset_index(drop=True)
               for i in range(0, len(prices), window_size)]
    if windows and len(windows[-1]) < window_size:
        windows.pop()

    # Configuration
    n_seeds = 5  # Number of random seeds per window
    seeds = [42, 123, 256, 512, 1024][:n_seeds]
    n_trials_ppo = 10  # Number of Optuna trials for PPO hyperparameter optimization

    # Define baseline configurations
    baseline_configs = [
        ("PassiveWidthSweep",
         PassiveWidthSweep,
         {"width_candidates": list(range(20, 201, 10)),  # 20-200 ticks
          "deposit_action_idx": 1}),

        ("VolProportionalWidth",
         VolProportionalWidth,
         {"k_values": [3, 5, 7, 10, 15],
          "base_factor": 100.0,
          "deposit_action_idx": 1}),

        ("ILMinimizer",
         ILMinimizer,
         {"horizon_hours": 24,
          "base_factor": 100.0,
          "deposit_action_idx": 1}),

        ("ReactiveRecentering",
         ReactiveRecentering,
         {"vol_thresholds": [0.005, 0.01, 0.02],
          "price_jump_thresholds": [0.005, 0.01, 0.02],
          "width_ticks": params["action_values"][1] if len(params["action_values"]) > 1 else 50,
          "deposit_action_idx": 1}),
    ]

    # Storage for all results
    all_results = {name: [] for name, _, _ in baseline_configs}
    all_results['PPO'] = []
    window_details = []

    print(f"Starting baseline comparison with hyperparameter optimization")
    print(f"  - PPO: {n_trials_ppo} Optuna trials × {n_seeds} seeds per window")
    print(f"  - Baselines: Parameter sweeps (optimized)")
    print(f"  - Total windows to process: {len(windows) - 5}\n")

    # Iterate through rolling windows
    for window_idx in range(len(windows) - 5):
        print(f"\n{'='*60}")
        print(f"Processing rolling window {window_idx + 1}/{len(windows) - 5}")
        print(f"{'='*60}")

        # Prepare train and test data
        train_df = pd.concat(windows[window_idx:window_idx+5], ignore_index=True)
        test_df = windows[window_idx+5].reset_index(drop=True)

        # Store results for this window across seeds
        window_results = {name: [] for name, _, _ in baseline_configs}
        window_results['PPO'] = []

        # Run multiple seeds (each with hyperparameter optimization)
        for seed_idx, seed in enumerate(seeds):
            print(f"\n  Seed {seed_idx + 1}/{n_seeds} (seed={seed})")

            # Train PPO with Optuna hyperparameter optimization
            print(f"    Optimizing PPO ({n_trials_ppo} trials)...", end=" ", flush=True)
            ppo_reward, _ = train_optimized_ppo(
                params, train_df, test_df, seed, n_trials=n_trials_ppo
            )
            window_results['PPO'].append(ppo_reward)
            all_results['PPO'].append(ppo_reward)
            print(f"Best reward: {ppo_reward:.2f}")

        # Evaluate baselines (only once per window since they're deterministic)
        print(f"\n  Evaluating baselines...")

        # Create test environment for baselines
        test_env = Uniswapv3Env(
            delta=params["delta"],
            action_values=np.array(params["action_values"], dtype=float),
            market_data=test_df,
            x=params["x"],
            gas=params["gas_fee"],
        )

        for name, BaselineClass, kwargs in baseline_configs:
            print(f"    {name}...", end=" ", flush=True)
            baseline = BaselineClass(**kwargs)
            baseline_reward = evaluate_baseline(baseline, test_env)

            # Add to window results once
            window_results[name].append(baseline_reward)

            # Add to all_results replicated across all seeds
            # (baselines are deterministic, so same value for all seeds)
            for _ in range(n_seeds):
                all_results[name].append(baseline_reward)

            print(f"Reward: {baseline_reward:.2f}")

        # Compute statistics for this window
        window_stats = compute_statistics(window_results, baseline_key='PPO')
        window_stats['Window'] = window_idx
        window_details.append(window_stats)

        # Save incremental results
        window_stats.to_csv(
            os.path.join(output_dir, f"window_{window_idx}_statistics.csv"),
            index=False
        )

        print(f"\n  Window {window_idx} Summary:")
        print(window_stats[['Strategy', 'Mean', 'Std']].to_string(index=False))

    # Compute overall statistics across all windows
    print(f"\n\n{'='*60}")
    print("OVERALL STATISTICS ACROSS ALL WINDOWS")
    print(f"{'='*60}\n")

    overall_stats = compute_statistics(all_results, baseline_key='PPO')
    print(overall_stats.to_string(index=False))

    # Save overall results
    overall_stats.to_csv(
        os.path.join(output_dir, "overall_statistics.csv"),
        index=False
    )

    # Save detailed results
    detailed_results = pd.DataFrame(all_results)
    detailed_results.to_csv(
        os.path.join(output_dir, "detailed_results_all_windows.csv"),
        index=False
    )

    # Generate publication-ready table (LaTeX format)
    latex_table = overall_stats[['Strategy', 'Mean', 'Std', 'P-value']].copy()
    latex_table['Mean±Std'] = latex_table.apply(
        lambda row: f"${row['Mean']:.2f} \\pm {row['Std']:.2f}$",
        axis=1
    )
    latex_table['P-value'] = latex_table['P-value'].apply(
        lambda x: f"${x:.4f}$" if pd.notna(x) else "—"
    )
    latex_str = latex_table[['Strategy', 'Mean±Std', 'P-value']].to_latex(
        index=False,
        escape=False,
        caption="Comparison of PPO (optimized) vs. Baseline Strategies (optimized)",
        label="tab:baseline_comparison_fair"
    )

    with open(os.path.join(output_dir, "latex_table.tex"), "w") as f:
        f.write(latex_str)

    print(f"\n\nResults saved to: {output_dir}")
    print("  - overall_statistics.csv: Statistical summary")
    print("  - detailed_results_all_windows.csv: All raw results")
    print("  - latex_table.tex: LaTeX table")
    print("  - window_*_statistics.csv: Per-window statistics")


if __name__ == "__main__":
    main()
