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
