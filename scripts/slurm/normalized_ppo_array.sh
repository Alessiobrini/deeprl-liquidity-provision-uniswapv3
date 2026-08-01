#!/bin/bash
#SBATCH --job-name=uniswap-normppo
#SBATCH --requeue
#SBATCH --array=0-11
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/normppo_%A_%a.out
#SBATCH --error=slurm_logs/normppo_%A_%a.err
#
# Gated reward-normalization diagnostic. Six shards per USDC/WETH fee tier keep the
# independent rolling units parallel while preserving their deterministic names.

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT=${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}
PYTHON=${CONDA_PREFIX:?activate the deeprl-uniswap env before sbatch}/bin/python3
OUT=${1:-outputs/normalized_ppo_v1}
POOLS=(usdc_weth_005 usdc_weth_030)
N_SHARDS=6
POOL_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
SHARD=$((SLURM_ARRAY_TASK_ID / 2))
POOL=${POOLS[$POOL_INDEX]}

cd "$PROJECT"
if [ ! -x "$PYTHON" ]; then
  echo "no interpreter at $PYTHON; activate the project environment" >&2
  exit 1
fi
if [ ! -d "$PROJECT/data/processed" ]; then
  echo "no panel at $PROJECT/data/processed" >&2
  exit 1
fi

echo "host $(hostname) pool $POOL shard $SHARD/$N_SHARDS out $OUT task $SLURM_ARRAY_TASK_ID"
srun "$PYTHON" \
  -m src.deeprl_liquidity_provision_uniswapv3.experiments.normalized_ppo \
  --out "$OUT" \
  --pools "$POOL" \
  --shard "$SHARD" \
  --of "$N_SHARDS"
