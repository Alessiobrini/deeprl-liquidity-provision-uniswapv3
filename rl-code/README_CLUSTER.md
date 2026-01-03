# Running WBTC Training on Duke Cluster

## Submit Job

```bash
ssh [netID]@login.cs.duke.edu
cd ~/deeprl-liquidity-provision-uniswapv3/rl-code
mkdir -p logs
sbatch slurm_wbtc_training.sh
```

## Check Status

```bash
squeue -u am1015                      # Job status
tail -f logs/wbtc_training_*.out      # Watch output
```

## Fast Parallel (4 GPUs, 4x faster)

```bash
sbatch slurm_wbtc_parallel.sh        
```

## Results

Saved to `output/baseline_comparison_wbtc/` in home directory:
- `overall_statistics.csv` - Summary stats
- `detailed_results_all_windows.csv` - Raw data
- `latex_table.tex` - Paper table

## Cancel Job

```bash
scancel JOBID
```

---

# Transfer Learning Experiments

## Submit Job

```bash
sbatch slurm_transfer_learning.sh     # Single GPU, 5 seeds, ~24 hours
```

## Fast Parallel (3 GPUs, 3x faster)

```bash
sbatch slurm_transfer_parallel.sh     # ~8 hours
python merge_transfer_results.py      # After completion
```

## Visualize Results

```bash
python visualize_transfer_learning.py
```

## Results

Saved to `output/transfer_learning/`:
- `transfer_learning_results.csv` - Summary stats (8 scenarios)
- `transfer_learning_detailed.csv` - Per-seed results
- `hypothesis_tests.csv` - Statistical tests
- `*.png` - 4 publication-quality plots
