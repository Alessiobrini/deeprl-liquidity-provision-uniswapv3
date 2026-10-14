# Reproduce the paper results

Install the environment from the root README.
Obtain the processed panel and saved experiment collections from the authors.
Place the panel files in `data/processed/`.
Place each experiment collection in its named directory under `outputs/`.

If the panel is elsewhere, set `CLEB_DATA_DIR` to its directory.
This variable changes file location only.
The calculation defaults remain unchanged.

From the repository root, run:

```bash
python scripts/reproduce_paper.py --input-root outputs --out outputs/paper_reproduced --figures
```

The script audits all collections before it writes aggregate results.
It regenerates inference at block lengths 2, 4, 6, and 8.
It also regenerates the retail counts, transfer counts, and pool-level oracle-null table.
Figure generation uses the block-4 geometry and retail tables.
The script writes a manifest with hashes for the generated artifacts.

The null collection needs 144 `.npz` files in `outputs/oracle_pool_null_v1/`.
A saved CSV alone cannot regenerate its p-values.
If any required collection is incomplete, the script stops.
Do not label a partial reproduction as complete.

## Regenerate the null draws

This step needs the processed panel and the completed oracle-calibration collection.
It replays local action returns but does not train a policy.

```bash
python -m src.deeprl_liquidity_provision_uniswapv3.experiments.oracle_pool_null \
  --oracle-root outputs/oracle_calibration_v1 --out outputs/oracle_pool_null_v1
python -m src.deeprl_liquidity_provision_uniswapv3.experiments.oracle_pool_null \
  --oracle-root outputs/oracle_calibration_v1 --out outputs/oracle_pool_null_v1 --aggregate
```

On a cluster, activate the environment and set `PROJECT_ROOT` before submission:

```bash
mkdir -p slurm_logs
sbatch scripts/slurm/oracle_pool_null_array.sh outputs/oracle_calibration_v1 outputs/oracle_pool_null_v1
```

The runner rejects cached files with a different seed or draw count.
Use a separate output directory for a changed configuration.

## Release contents

The [experiment record](protocols.md) describes the implemented designs and their limitations.
The `artifacts/` directory contains aggregate CSVs and their manifest.
Large datasets, raw result JSONs, model checkpoints, and private machine paths stay outside this directory.
An artifact-only review checks reported numbers but does not reproduce the underlying simulation.
The full reproduction command needs the panel and complete experiment collections.

The pool-null table includes both exceedance fractions and corrected Monte Carlo p-values.
Use the column that corresponds to the paper's declared definition.
The null-fraction quantiles are conditional randomization intervals, not window-bootstrap confidence intervals.
