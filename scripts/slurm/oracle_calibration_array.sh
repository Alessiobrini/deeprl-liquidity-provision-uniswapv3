#!/bin/bash
#SBATCH --job-name=uniswap-oracle-null
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/oracle_null_%A_%a.out
#SBATCH --error=slurm_logs/oracle_null_%A_%a.err
#
# One task owns one pool and its 24 held-out rolling windows. The runner performs
# counterfactual environment replay but no RL training.

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT=${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}
PYTHON=${CONDA_PREFIX:?activate the project environment before sbatch}/bin/python3
OUT=${1:-outputs/oracle_calibration_v1}
POOLS=(
  usdc_weth_005
  usdc_weth_030
  wbtc_weth_005
  wbtc_weth_030
  weth_usdt_005
  weth_usdt_030
)
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}

cd "$PROJECT"
if [ ! -x "$PYTHON" ]; then
  echo "no interpreter at $PYTHON; activate the project environment" >&2
  exit 1
fi
if [ ! -d "$PROJECT/data/processed" ]; then
  echo "no panel at $PROJECT/data/processed" >&2
  exit 1
fi

echo "host $(hostname) pool $POOL out $OUT task $SLURM_ARRAY_TASK_ID"
srun "$PYTHON" \
  -m src.deeprl_liquidity_provision_uniswapv3.experiments.oracle_calibration \
  --out "$OUT" \
  --pools "$POOL"
