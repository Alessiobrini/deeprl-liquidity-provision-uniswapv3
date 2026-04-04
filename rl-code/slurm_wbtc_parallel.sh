#!/bin/bash
#SBATCH --job-name=wbtc_parallel
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/wbtc_parallel_%j.out
#SBATCH --error=logs/wbtc_parallel_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=am1015@duke.edu

# =============================================================================
# WBTC Parallel Training - 4 GPUs for 4x speedup
# Each GPU processes a subset of rolling windows
# =============================================================================

echo "=============================================================="
echo "Parallel job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "=============================================================="

nvidia-smi

# Create directories
mkdir -p logs
cd "${HOME}/deeprl-liquidity-provision-uniswapv3/rl-code" || exit 1

# Pre-create output directories for each GPU to avoid race conditions
mkdir -p output/baseline_comparison_wbtc_gpu0
mkdir -p output/baseline_comparison_wbtc_gpu1
mkdir -p output/baseline_comparison_wbtc_gpu2
mkdir -p output/baseline_comparison_wbtc_gpu3
echo "Pre-created GPU output directories"

# Function to run training on a specific GPU for specific windows
run_window_subset() {
    local GPU_ID=$1
    local START_WINDOW=$2
    local END_WINDOW=$3
    local TOTAL_WINDOWS=$4

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    echo "GPU $GPU_ID: Processing windows $START_WINDOW to $END_WINDOW"

    python -u baseline_comparison_wbtc.py \
        --start-window $START_WINDOW \
        --end-window $END_WINDOW \
        --gpu-id $GPU_ID \
        2>&1 | tee "logs/wbtc_gpu${GPU_ID}_${SLURM_JOB_ID}.log" &
}

# Calculate total windows (based on data size)
# WBTC has ~40,801 hours, window_size=1500, so ~27 windows, minus 5 for rolling = ~22 windows
TOTAL_WINDOWS=22

# Split work across 4 GPUs
GPU0_START=0
GPU0_END=5

GPU1_START=6
GPU1_END=11

GPU2_START=12
GPU2_END=16

GPU3_START=17
GPU3_END=$((TOTAL_WINDOWS - 1))

echo "Launching parallel training across 4 GPUs..."
echo "GPU 0: windows $GPU0_START-$GPU0_END"
echo "GPU 1: windows $GPU1_START-$GPU1_END"
echo "GPU 2: windows $GPU2_START-$GPU2_END"
echo "GPU 3: windows $GPU3_START-$GPU3_END"
echo ""

# Launch all 4 processes in parallel
run_window_subset 0 $GPU0_START $GPU0_END $TOTAL_WINDOWS
run_window_subset 1 $GPU1_START $GPU1_END $TOTAL_WINDOWS
run_window_subset 2 $GPU2_START $GPU2_END $TOTAL_WINDOWS
run_window_subset 3 $GPU3_START $GPU3_END $TOTAL_WINDOWS

# Wait for all background jobs to finish
wait

echo ""
echo "=============================================================="
echo "All parallel training completed at: $(date)"
echo "=============================================================="

# List results from each GPU
echo "Results saved to separate GPU directories:"
for gpu_id in 0 1 2 3; do
    if [ -d "output/baseline_comparison_wbtc_gpu${gpu_id}" ]; then
        echo "  GPU ${gpu_id}: output/baseline_comparison_wbtc_gpu${gpu_id}/"
        ls -lh "output/baseline_comparison_wbtc_gpu${gpu_id}/" | grep -E "\.csv$" || true
    fi
done

echo ""
echo "To merge results, run: python merge_parallel_results.py"
