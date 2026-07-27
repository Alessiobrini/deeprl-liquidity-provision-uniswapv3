#!/bin/bash
# Run the walk-forward protocol for several algorithms, then print one table each.
#
#   ./scripts/run_algos.sh                          # all five, all six pools
#   ./scripts/run_algos.sh "ppo a2c"                # a subset
#   ./scripts/run_algos.sh "ppo" outputs/my_run     # a named run directory
#   SHARD=0 OF=8 ./scripts/run_algos.sh             # one eighth of the work
#
# Each (algorithm, pool, rolling step) is an independent work unit that writes its own
# JSON and is skipped if already done, so this is resumable and shardable: stop it with
# ctrl-C, rerun, and it picks up. The config keys the filename, so several algorithms
# can share one output directory without colliding.
#
# On DCC use the array wrapper instead, which maps $SLURM_ARRAY_TASK_ID onto the same
# --shard index:  sbatch scripts/slurm/rolling_array.sh
set -euo pipefail

ALGOS=${1:-"ppo a2c dqn qrdqn recurrentppo"}
OUT=${2:-outputs/algos_v1}
SHARD=${SHARD:-0}
OF=${OF:-1}
STEPS=${STEPS:-20000}
SEEDS=${SEEDS:-"42 123"}
POOLS=${POOLS:-""}          # empty = the six core pools
WIDTH_UNITS=${WIDTH_UNITS:-spacing}
WIDTHS_WAS_SET=${WIDTHS+x}
WIDTHS=${WIDTHS:-"45 50 55"}
EXECUTION_WIDTHS=${EXECUTION_WIDTHS:-""}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# `python` is not on a bare PATH. Activate the env first (`conda activate
# deeprl-uniswap`), or pass PYTHON=/path/to/python explicitly.
PY=${PYTHON:-python}
command -v "$PY" >/dev/null || { echo "no interpreter '$PY' on PATH. Run 'conda activate deeprl-uniswap' first, or set PYTHON=..." >&2; exit 1; }
MOD=src.deeprl_liquidity_provision_uniswapv3.experiments.rolling

if [ ! -d data/processed ] || [ -z "$(ls -A data/processed 2>/dev/null)" ]; then
  echo "No panel in data/processed/." >&2
  echo "Unzip the data bundle here, or build it: see README.md 'Getting the data'." >&2
  exit 1
fi

pool_args=""
[ -n "$POOLS" ] && pool_args="--pools $POOLS"

# EVERY invocation must carry the SAME config args, run and aggregate alike. The
# config is hashed into each unit's filename, and the aggregate filters on that hash,
# so an aggregate call that omits --steps/--seeds computes a different hash and matches
# nothing. Keep this one list; do not spell the flags out twice.
# shellcheck disable=SC2086
# Do not spell out the default widths.  argparse's declared default is an integer
# list, whereas an explicit `--widths 45 50 55` is parsed as floats; JSON preserves
# that distinction in the resume hash.  Omitting an unset default therefore keeps the
# completed paper-compatible collection addressable.  Exploratory treatments set
# WIDTHS explicitly and retain their own already-frozen hash.
CFG=(--algo "" --steps "$STEPS" --seeds $SEEDS --width-units "$WIDTH_UNITS" $pool_args)
[ -n "$WIDTHS_WAS_SET" ] && CFG+=(--widths $WIDTHS)
[ -n "$EXECUTION_WIDTHS" ] && CFG+=(--execution-widths $EXECUTION_WIDTHS)

for algo in $ALGOS; do
  echo ""
  echo "=============================================================="
  echo " $algo   shard $SHARD/$OF   $STEPS steps   seeds $SEEDS"
  echo "=============================================================="
  CFG[1]="$algo"
  $PY -m $MOD "${CFG[@]}" --out "$OUT" --shard "$SHARD" --of "$OF"
done

# Only the last shard to finish should report, so a sharded run does not print six
# partial tables. With --of 1 there is only one shard and this always runs.
if [ "$OF" = "1" ]; then
  for algo in $ALGOS; do
    echo ""
    echo "### $algo"
    CFG[1]="$algo"
    $PY -m $MOD "${CFG[@]}" --out "$OUT" --aggregate
  done
fi
