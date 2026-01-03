#!/bin/bash
#SBATCH --job-name=transfer_learning
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --constraint="a6000|a5000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/transfer_learning_%j.out
#SBATCH --error=logs/transfer_learning_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=am1015@duke.edu

# =============================================================================
# Cross-Pool Transfer Learning Experiments
# Tests transfer between WETH/USDC and WBTC/USDC pools
# =============================================================================

echo "=============================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "=============================================================="

# Check GPU assignment
nvidia-smi
echo "=============================================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=$SLURM_STEP_GPUS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate virtual environment if needed
# source ~/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Set output directory to persistent storage (NFS home directory)
export OUTPUT_BASE_DIR="${HOME}/deeprl-liquidity-provision-uniswapv3/rl-code/output"
mkdir -p "$OUTPUT_BASE_DIR/transfer_learning"

echo "Output will be saved to: $OUTPUT_BASE_DIR/transfer_learning"
echo "=============================================================="

# Change to working directory
cd "${HOME}/deeprl-liquidity-provision-uniswapv3/rl-code" || exit 1

# Run transfer learning with 5 seeds (matching baseline comparison)
echo "Starting cross-pool transfer learning experiments..."
echo "Running with 5 seeds across 8 transfer scenarios..."
echo "Start time: $(date)"
echo ""

python -u cross_pool_transfer.py --seeds 5 2>&1 | tee "logs/transfer_learning_${SLURM_JOB_ID}_detailed.log"

EXIT_CODE=$?

echo ""
echo "=============================================================="
echo "Transfer learning completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================================="

# Summary of output files
if [ -d "$OUTPUT_BASE_DIR/transfer_learning" ]; then
    echo "Output files created:"
    ls -lh "$OUTPUT_BASE_DIR/transfer_learning/"
    echo ""
    echo "Results saved to: $OUTPUT_BASE_DIR/transfer_learning/"

    # Show quick preview of results
    if [ -f "$OUTPUT_BASE_DIR/transfer_learning/transfer_learning_results.csv" ]; then
        echo ""
        echo "Preview of results:"
        head -20 "$OUTPUT_BASE_DIR/transfer_learning/transfer_learning_results.csv"
    fi
else
    echo "WARNING: Output directory not found!"
fi

echo ""
echo "=============================================================="
echo "To generate visualizations, run: python visualize_transfer_learning.py"
echo "=============================================================="
echo "Job finished at: $(date)"
exit $EXIT_CODE
