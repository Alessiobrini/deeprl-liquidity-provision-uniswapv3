# Execution-Aware Benchmarking of Reinforcement Learning for Concentrated Liquidity Provision

**Arjun Mahesh** and **Alessio Brini**, Pratt School of Engineering, Duke University

Accepted at the 7th ACM International Conference on AI in Finance (ICAIF 2026), Milan, Italy, November 16 and 17, 2026.

This repository holds the code for the paper: the Uniswap v3 environment, the five learned and six heuristic strategies, the walk-forward protocol, and the scripts that produce every result.

## Abstract

We introduce the Concentrated Liquidity Execution Benchmark (CLEB), an execution-aware testbed for reinforcement learning (RL) in liquidity provision on Uniswap v3. We control four evaluation axes at once, fee accounting, model selection without data leakage, executable action geometry, and operating costs. We build CLEB from 22.6 million Ethereum mainnet swaps across six pools (three token pairs at two fee tiers) and evaluate five RL algorithms and six heuristics on held-out walk-forward windows. We average across seeds within windows and report paired bootstrap intervals with Holm correction. When rebalancing conserves value and fees are attributed swap by swap, every strategy's mean sits below holding and no algorithm dominates across pools. Evaluated deterministically, each learned policy executes one action per pool in its first seed's recorded trace. To test whether this collapse is economically appropriate, we evaluate every action at each decision state and fit a decision stump over the same state variables. The stump produces no reliable test advantage over a fixed range width selected on validation, and in no pool does a learned policy improve on the best validation heuristic at every block length. The hindsight best action, the maximum over the four realized outcomes, disagrees with any fixed choice 75% of the time when the actions carry no information, so we report the hindsight comparison as descriptive, not as evidence of capturable state contingency. Matching executable action geometry across fee tiers changes which strategy has the highest mean in four of six pools and eliminates one learned policy's apparent advantage at the 0.30% tier. Across capital and gas levels, recentering an out-of-range position is worse than leaving it unchanged in every 0.05% cell and only descriptively favorable in most 0.30% cells. With this state and action set, CLEB therefore characterizes active liquidity provision on this panel as heterogeneous loss mitigation rather than premium capture.

## Citation

```bibtex
@inproceedings{mahesh2026execution,
  title     = {Execution-Aware Benchmarking of Reinforcement Learning for Concentrated Liquidity Provision},
  author    = {Mahesh, Arjun and Brini, Alessio},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance (ICAIF '26)},
  address   = {Milan, Italy},
  publisher = {ACM},
  year      = {2026}
}
```

The DOI and page numbers will be added once the proceedings are published.

The benchmark extends Xu and Brini, *Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning* ([arXiv 2501.07508](https://arxiv.org/abs/2501.07508), AAAI 2025 Workshop on AI for Social Impact), whose environment, state, action semantics and reward it keeps, and it attributes fees from each of the 22.6 million swaps.

## Setup

```bash
conda env create -f environment.yml
conda activate deeprl-uniswap
pip install -e ".[dev]"           # editable install; no PYTHONPATH needed afterwards
pytest -q                          # 200 tests, a few seconds
```

TA-Lib needs its C library, which pip cannot install on its own. It comes from
conda-forge in `environment.yml`; without conda, `brew install ta-lib` first.

## Data

The benchmark is built from public Uniswap v3 Swap events on Ethereum mainnet for six pools, USDC/WETH, WBTC/WETH and WETH/USDT at the 0.05% and 0.30% fee tiers, from May 2021 to June 2026, about 44,900 hours per pool. The processed panel, an hourly table and a per-swap fee table for each pool in Parquet format (about 460 MB), is too large for GitHub, so we share it on request. Please write to Alessio Brini (alessio.brini@duke.edu) or Arjun Mahesh (arjun.mahesh@duke.edu).

Place the files in `data/processed/`, which is all the code reads. To rebuild them from raw Swap events instead, set the raw-data path at the top of `aggregate.py` to a folder holding one Parquet file of raw events per pool, with the event amounts, `sqrtPriceX96`, `liquidity` and `tick`, and run:

```bash
python -m src.deeprl_liquidity_provision_uniswapv3.data.aggregate   # hourly panel
python -m src.deeprl_liquidity_provision_uniswapv3.data.swaps       # per-swap fees
```

As a check, `aggregate.py` recovers each pool's fee tier from the raw data and asserts that the median disagreement between a swap's two token legs equals the declared tier.

## Running experiments

**Several algorithms, the easy way:**

```bash
./scripts/run_algos.sh                        # ppo a2c dqn qrdqn recurrentppo, 6 pools
./scripts/run_algos.sh "ppo a2c"              # a subset
STEPS=50000 ./scripts/run_algos.sh "ppo"      # a longer budget
POOLS="usdc_weth_005" ./scripts/run_algos.sh "ppo"     # one pool, to try it out
```

Resumable and shardable. One work unit is one (algorithm, pool, rolling step). It
writes its own JSON and is skipped if already done. Stop with ctrl-C, rerun, it picks
up where it stopped. The config keys the filename, so several algorithms share one
output directory without colliding.

**Splitting across machines:**

```bash
SHARD=0 OF=8 ./scripts/run_algos.sh           # this machine takes one eighth
```

Shard *i* names the same units on every machine, so shards merge with no
reconciliation. Then read whatever has landed:

```bash
python -m src.deeprl_liquidity_provision_uniswapv3.experiments.rolling \
       --out outputs/algos_v1 --algo ppo --aggregate
```

The aggregate reports how many units are missing and refuses to pass a partial table
off as a finished run.

**On a SLURM cluster:** `scripts/slurm/rolling_array.sh` is a thin array wrapper that
maps `$SLURM_ARRAY_TASK_ID` onto the same `--shard` index, so there is no
cluster-only code path. Export `PROJECT_ROOT` and activate the conda environment
before `sbatch`, and set `REMOTE_HOST`, `REMOTE_PROJECT`, and `REMOTE_CONDA` when using
`scripts/sync_to_cluster.sh`.

This is a **CPU** workload. It runs at about 6,100 steps/s on CPU. The nets are far too small
for a GPU to beat kernel-launch overhead, and the env emits float64, which Apple MPS
cannot take at all. Request CPU nodes.

### Practitioner-grounded extensions

The retail cost frontier replays deterministic policies at five capital and five gas
levels, then adds a transparent token-conversion counterfactual:

```bash
python -m src.deeprl_liquidity_provision_uniswapv3.experiments.retail_frontier \
  --out outputs/retail_frontier_v1
python -m src.deeprl_liquidity_provision_uniswapv3.experiments.retail_frontier \
  --out outputs/retail_frontier_v1 --aggregate \
  --report-dir outputs/retail_frontier_v1/aggregate_block4
```

The matched-action-geometry treatment preserves the observable action labels but
executes the same raw-tick half-widths on both fee tiers:

```bash
WIDTHS="45 50 55" EXECUTION_WIDTHS="480 540 600" \
  ./scripts/run_algos.sh \
  "ppo a2c dqn qrdqn recurrentppo" outputs/action_geometry_v1
python -m src.deeprl_liquidity_provision_uniswapv3.experiments.action_geometry \
  --paper outputs/algos_v1 --matched outputs/action_geometry_v1 \
  --report-dir outputs/action_geometry_v1/aggregate_block4
```

Both aggregators refuse partial collections.

## The protocol

Walk-forward, with selection carved out of the training block so the test window is
never a tuning parameter:

```
train  windows i..i+3    6,000h   fit
val    window  i+4       1,500h   select the config; test is not reachable from here
test   window  i+5       1,500h   read once
                                  refit on train+val, then roll forward one window
```

Every window is tested exactly once, by a policy fit only on windows preceding it.
144 window-tests on the core panel. Both arms, learned and heuristic, select on
validation and read test once.

## Layout

```
src/deeprl_liquidity_provision_uniswapv3/
  data/       pools.py (registry), aggregate.py (hourly panel), swaps.py (per-swap fees)
  envs/       uniswap_v3.py (the MDP), features.py (the 13-feature state), schedule.py
  agents/     registry.py (5 algorithms), extractors.py (the BatchNorm policy net)
  policies/   baselines.py (the heuristic strategies)
  experiments/
    rolling.py      the walk-forward protocol. Start here.
    bakeoff.py      splits, env construction, scoring
    agent_arm.py    training envs
    combined.py, sensitivity.py, action_geometry.py, transfer_rolling.py
    state_headroom.py, oracle_calibration.py, normalized_ppo.py, ppo_budget_ablation.py
    retail_frontier.py, retail_frontier_summary.py, transfer_robustness_summary.py
    plot_extensions.py   the paper's figures
    validate_*.py        audits of the raw outputs
scripts/
  run_algos.sh          multi-algorithm runner
  sync_to_cluster.sh    push / pull / status against a SLURM cluster
  slurm/                array wrappers, one per experiment
tests/              200 tests; the protocol and the accounting are both covered
```

## Where each result comes from

Every experiment module writes one JSON per work unit with `--out <dir>` (sharded with `--shard i --of n`), and the same module with `--aggregate --report-dir <dir> --bootstrap-block <b>` produces the paired bootstrap intervals with Holm correction. The paper reports block lengths 2, 4, 6 and 8, with 4 as the primary length. Each module lists its options under `--help`.

| Result in the paper | Module in `experiments/` | Cluster wrapper in `scripts/slurm/` |
|---|---|---|
| Multi-pool benchmark (Table 2) | `rolling.py`, aggregated by `combined.py` | `rolling_array.sh` |
| Matched executable action geometry (Figure 1) | `rolling.py` with matched widths, then `action_geometry.py` | `rolling_array.sh` |
| Hindsight action diagnostic and decision stump (Table 3) | `state_headroom.py` | `state_headroom_array.sh` |
| Sign-flip decomposition of the hindsight gap | `oracle_calibration.py` | `oracle_calibration_array.sh` |
| Reward-normalization follow-up for PPO | `normalized_ppo.py` | `normalized_ppo_array.sh` |
| Retail frontier (Figure 2) | `retail_frontier.py`, `retail_frontier_summary.py` | `retail_frontier_array.sh` |
| Cross-pool transfer | `transfer_rolling.py`, `transfer_robustness_summary.py` | `transfer_array.sh` |
| Decision-schedule, reward and state variants | `rolling.py` variants, compared by `sensitivity.py` | `rolling_array.sh` |
| Longer training budgets for PPO | `ppo_budget_ablation.py` | `ppo_budget_ablation_array.sh` |
| Figures 1 and 2 | `plot_extensions.py` | |

## Algorithms

`registry.py` carries five, all on the same `Discrete` action space so the comparison
is about the algorithm and not the action space:

| key | notes |
|---|---|
| `ppo` | the incumbent |
| `a2c` | on-policy contrast |
| `dqn` | off-policy contrast (vanilla DQN with a target network, not Double DQN) |
| `qrdqn` | distributional (sb3-contrib) |
| `recurrentppo` | LSTM policy (sb3-contrib) |

SAC and TD3 are absent by design: they need a `Box` action space, which is a
different formulation of the problem.

`--paper-extractor` uses the earlier policy network (BatchNorm -> [4,2] -> 128, tanh)
instead of the SB3 default MLP. The BatchNorm is load-bearing, because the 13-feature
state carries raw prices (~2,000) next to raw liquidity (~1e18).
