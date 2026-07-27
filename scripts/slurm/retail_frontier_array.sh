#!/bin/bash
#SBATCH --job-name=uniswap-retail
#SBATCH --requeue
#SBATCH --array=0-35
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/retail_%A_%a.out
#SBATCH --error=slurm_logs/retail_%A_%a.err

set -euo pipefail
export PYTHONNOUSERSITE=1
PROJECT=${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}
PYTHON=${CONDA_PREFIX:?activate the deeprl-uniswap env before sbatch}/bin/python3
OUT=${1:-outputs/retail_frontier_v1}
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-36}
cd "$PROJECT"

srun "$PYTHON" -m src.deeprl_liquidity_provision_uniswapv3.experiments.retail_frontier \
  --out "$OUT" --shard "$SLURM_ARRAY_TASK_ID" --of "$N_SHARDS"
