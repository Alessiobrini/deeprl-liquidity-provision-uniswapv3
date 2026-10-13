#!/bin/bash
#SBATCH --job-name=uniswap-oracle-pool-null
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/oracle_pool_null_%A_%a.out
#SBATCH --error=slurm_logs/oracle_pool_null_%A_%a.err

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT=${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}
PYTHON=${CONDA_PREFIX:?activate the project environment before sbatch}/bin/python3
ORACLE_ROOT=${1:-outputs/oracle_calibration_v1}
OUT=${2:-outputs/oracle_pool_null_v1}
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
srun "$PYTHON" \
  -m src.deeprl_liquidity_provision_uniswapv3.experiments.oracle_pool_null \
  --oracle-root "$ORACLE_ROOT" \
  --out "$OUT" \
  --pools "$POOL"
