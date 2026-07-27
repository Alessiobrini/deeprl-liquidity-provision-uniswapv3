#!/bin/bash
#SBATCH --job-name=uniswap-transfer
#SBATCH --requeue
#SBATCH --array=0-17
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/transfer_%A_%a.out
#SBATCH --error=slurm_logs/transfer_%A_%a.err
set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT=${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}
PYTHON=${CONDA_PREFIX:?activate the deeprl-uniswap env before sbatch}/bin/python3
OUT=${1:-outputs/transfer_v1}
ALGO=${2:-ppo}
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-18}

cd "$PROJECT"
srun "$PYTHON" -m src.deeprl_liquidity_provision_uniswapv3.experiments.transfer_rolling \
  --out "$OUT" --algo "$ALGO" --shard "$SLURM_ARRAY_TASK_ID" --of "$N_SHARDS" \
  --steps 20000 --seeds 42 123 --schedule event_driven --shaping none
