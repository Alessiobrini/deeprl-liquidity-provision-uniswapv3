#!/bin/bash
#SBATCH --job-name=wbtc_baseline
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --constraint="a6000|a5000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/wbtc_training_%j.out
#SBATCH --error=logs/wbtc_training_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=am1015@duke.edu

# =============================================================================
# WBTC/USDC Baseline Comparison Training - Optimized for Duke CS Cluster
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
mkdir -p "$OUTPUT_BASE_DIR"

echo "Output will be saved to: $OUTPUT_BASE_DIR"
echo "=============================================================="

# Change to working directory
cd "${HOME}/deeprl-liquidity-provision-uniswapv3/rl-code" || exit 1

# Run training with detailed output
echo "Starting WBTC baseline comparison training..."
echo "Start time: $(date)"
echo ""

python -u baseline_comparison_wbtc.py 2>&1 | tee "logs/wbtc_training_${SLURM_JOB_ID}_detailed.log"

EXIT_CODE=$?

echo ""
echo "=============================================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================================="

# Summary of output files
if [ -d "$OUTPUT_BASE_DIR/baseline_comparison_wbtc" ]; then
    echo "Output files created:"
    ls -lh "$OUTPUT_BASE_DIR/baseline_comparison_wbtc/"
    echo ""
    echo "Results saved to: $OUTPUT_BASE_DIR/baseline_comparison_wbtc/"
else
    echo "WARNING: Output directory not found!"
fi

echo "Job finished at: $(date)"
exit $EXIT_CODE
