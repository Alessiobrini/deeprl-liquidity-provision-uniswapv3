"""
Quick test script to verify WBTC config and environment initialization
"""
import pandas as pd
import numpy as np
import yaml
import os
import sys
from pathlib import Path

# Add custom_env_folder to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "custom_env_folder"))

from custom_env import Uniswapv3Env

def test_wbtc_config():
    """Test WBTC config file and environment initialization"""

    # Setup paths
    base_dir = Path(__file__).resolve().parents[1]  # rl-code directory
    config_dir = base_dir / "config"
    config_file = "pool_wbtc_usdc_030.yaml"

    print("="*70)
    print("WBTC/USDC Environment Configuration Test")
    print("="*70)

    # Load config
    print(f"\n[1/4] Loading config from {config_file}...")
    with open(config_dir / config_file, "r") as f:
        params = yaml.safe_load(f)

    print(f"  Config loaded successfully!")
    print(f"  Pool fee tier: {params['delta']}%")
    print(f"  Action values (tick offsets): {params['action_values']}")
    print(f"  Initial liquidity (BTC): {params['x']}")
    print(f"  Data file: {params['filename']}")

    # Load market data
    print(f"\n[2/4] Loading market data...")
    data_path = base_dir / params['filename']

    if not data_path.exists():
        print(f"  [ERROR] Data file not found: {data_path}")
        print(f"  Please run data collection first:")
        print(f"    cd data && python data_collection_multipool.py --pool wbtc_usdc_030")
        return False

    df = pd.read_csv(data_path)
    market_data = df[['price']]

    print(f"  Data loaded: {len(market_data):,} hours")
    print(f"  Price range: ${market_data['price'].min():,.2f} - ${market_data['price'].max():,.2f}")
    print(f"  Mean price: ${market_data['price'].mean():,.2f}")

    # Initialize environment
    print(f"\n[3/4] Initializing Uniswap v3 environment...")
    try:
        env = Uniswapv3Env(
            delta=params['delta'],
            action_values=np.array(params['action_values']),
            market_data=market_data,
            x=params['x'],
            gas=params['gas_fee']
        )
        print(f"  Environment initialized successfully!")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space} (n={env.action_space.n} actions)")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize environment: {e}")
        return False

    # Test a few steps
    print(f"\n[4/4] Testing environment with 10 random steps...")
    try:
        obs, info = env.reset()
        print(f"  Initial observation shape: {obs.shape}")
        print(f"  Initial price: ${info.get('price', 'N/A')}")

        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if done or truncated:
                print(f"  Episode ended at step {step+1}")
                break

        print(f"  Completed {step+1} steps successfully")
        print(f"  Total reward: {total_reward:.4f}")

    except Exception as e:
        print(f"  [ERROR] Environment step failed: {e}")
        return False

    print("\n" + "="*70)
    print("TEST PASSED: WBTC environment is ready for training!")
    print("="*70)

    return True

if __name__ == "__main__":
    success = test_wbtc_config()
    sys.exit(0 if success else 1)
