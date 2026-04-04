#!/bin/bash
#SBATCH --job-name=transfer_parallel
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/transfer_parallel_%j.out
#SBATCH --error=logs/transfer_parallel_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=am1015@duke.edu

# =============================================================================
# Cross-Pool Transfer Learning - Parallel Execution
# 4 GPUs process different seeds simultaneously for 4x speedup
# Each GPU runs all 8 transfer scenarios for its assigned seeds
# =============================================================================

echo "=============================================================="
echo "Parallel transfer learning started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "=============================================================="

nvidia-smi

# Create directories
mkdir -p logs
cd "${HOME}/deeprl-liquidity-provision-uniswapv3/rl-code" || exit 1

# Pre-create output directories for each GPU to avoid race conditions
mkdir -p output/transfer_learning_gpu0
mkdir -p output/transfer_learning_gpu1
mkdir -p output/transfer_learning_gpu2
mkdir -p output/transfer_learning_gpu3
echo "Pre-created GPU output directories"

# Function to run transfer learning on a specific GPU with specific seeds
run_seed_subset() {
    local GPU_ID=$1
    local SEED_START=$2
    local SEED_END=$3

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    echo "GPU $GPU_ID: Processing seeds from index $SEED_START to $SEED_END"

    # Create a temporary Python script that runs specific seeds
    cat > "transfer_gpu${GPU_ID}_temp.py" << 'EOFPYTHON'
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main function
from cross_pool_transfer import (
    load_pool_data, train_ppo, evaluate_model
)

def run_transfer_for_seeds(seed_indices, gpu_id):
    """Run transfer learning for specific seed indices."""
    print(f"GPU {gpu_id}: Running seeds {seed_indices}")

    # All possible seeds (matching baseline comparison)
    all_seeds = [42, 123, 256, 512, 1024]
    selected_seeds = [all_seeds[i] for i in seed_indices]

    print(f"Selected seeds: {selected_seeds}")

    # Load data (same as main script)
    print("\nLoading pool data...")
    weth_config, weth_data = load_pool_data('weth')
    wbtc_config, wbtc_data = load_pool_data('wbtc')

    print(f"  WETH data: {len(weth_data):,} hours")
    print(f"  WBTC data: {len(wbtc_data):,} hours")

    # Data splits
    max_hours = min(len(weth_data), len(wbtc_data))
    train_split = int(max_hours * 0.70)
    same_period_split = int(max_hours * 0.85)

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

    # Run experiments for each seed
    for seed_idx, seed in enumerate(selected_seeds):
        print(f"\n{'='*70}")
        print(f"GPU {gpu_id} - Seed {seed_idx + 1}/{len(selected_seeds)} (seed={seed})")
        print(f"{'='*70}")

        # Train on WETH
        print("\n[1/2] Training on WETH...")
        weth_model = train_ppo(weth_train, weth_config, seed=seed)

        print("  Testing on:")
        reward = evaluate_model(weth_model, weth_same_test, weth_config, seed=seed)
        results['WETH->WETH_same'].append(reward)
        print(f"    WETH (same period): {reward:.2f}")

        reward = evaluate_model(weth_model, weth_future_test, weth_config, seed=seed)
        results['WETH->WETH_future'].append(reward)
        print(f"    WETH (future period): {reward:.2f}")

        reward = evaluate_model(weth_model, wbtc_same_test, wbtc_config, seed=seed)
        results['WETH->WBTC_same'].append(reward)
        print(f"    WBTC (same period): {reward:.2f}")

        reward = evaluate_model(weth_model, wbtc_future_test, wbtc_config, seed=seed)
        results['WETH->WBTC_future'].append(reward)
        print(f"    WBTC (future period): {reward:.2f}")

        # Train on WBTC
        print("\n[2/2] Training on WBTC...")
        wbtc_model = train_ppo(wbtc_train, wbtc_config, seed=seed)

        print("  Testing on:")
        reward = evaluate_model(wbtc_model, wbtc_same_test, wbtc_config, seed=seed)
        results['WBTC->WBTC_same'].append(reward)
        print(f"    WBTC (same period): {reward:.2f}")

        reward = evaluate_model(wbtc_model, wbtc_future_test, wbtc_config, seed=seed)
        results['WBTC->WBTC_future'].append(reward)
        print(f"    WBTC (future period): {reward:.2f}")

        reward = evaluate_model(wbtc_model, weth_same_test, weth_config, seed=seed)
        results['WBTC->WETH_same'].append(reward)
        print(f"    WETH (same period): {reward:.2f}")

        reward = evaluate_model(wbtc_model, weth_future_test, weth_config, seed=seed)
        results['WBTC->WETH_future'].append(reward)
        print(f"    WETH (future period): {reward:.2f}")

    # Save GPU-specific results
    output_dir = Path(__file__).parent / "output" / f"transfer_learning_gpu{gpu_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_df = pd.DataFrame(results)
    detailed_df.to_csv(output_dir / f"gpu{gpu_id}_detailed_results.csv", index=False)

    print(f"\nGPU {gpu_id} results saved to: {output_dir}")

    return results

if __name__ == "__main__":
    import sys
    seed_start = int(sys.argv[1])
    seed_end = int(sys.argv[2])
    gpu_id = int(sys.argv[3])

    seed_indices = list(range(seed_start, seed_end + 1))
    run_transfer_for_seeds(seed_indices, gpu_id)
EOFPYTHON

    python -u "transfer_gpu${GPU_ID}_temp.py" $SEED_START $SEED_END $GPU_ID \
        2>&1 | tee "logs/transfer_gpu${GPU_ID}_${SLURM_JOB_ID}.log" &
}

# Split 5 seeds across 4 GPUs:
# GPU 0: seed indices 0-1 (seeds 42, 123)
# GPU 1: seed indices 2-3 (seeds 256, 512)
# GPU 2: seed index 4 (seed 1024)
# GPU 3: additional seeds if needed (can add seeds 2048, 4096 for 7 total)

echo "Launching parallel transfer learning across 4 GPUs..."
echo "GPU 0: seed indices 0-1"
echo "GPU 1: seed indices 2-3"
echo "GPU 2: seed index 4"
echo ""

# Launch processes in parallel
run_seed_subset 0 0 1
run_seed_subset 1 2 3
run_seed_subset 2 4 4

# Wait for all background jobs to finish
wait

echo ""
echo "=============================================================="
echo "All parallel training completed at: $(date)"
echo "=============================================================="

# List results from each GPU
echo "Results saved to separate GPU directories:"
for gpu_id in 0 1 2; do
    if [ -d "output/transfer_learning_gpu${gpu_id}" ]; then
        echo "  GPU ${gpu_id}: output/transfer_learning_gpu${gpu_id}/"
        ls -lh "output/transfer_learning_gpu${gpu_id}/" | grep -E "\.csv$" || true
    fi
done

# Clean up temporary scripts
rm -f transfer_gpu*_temp.py

echo ""
echo "To merge results, run: python merge_transfer_results.py"
echo "Job finished at: $(date)"
