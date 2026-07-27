# RL for active liquidity provision in Uniswap v3

Multi-pool benchmarking and transfer learning.

Successor to *"Improving DeFi Accessibility through Efficient Liquidity Provisioning
with Deep Reinforcement Learning"* ([arXiv 2501.07508](https://arxiv.org/abs/2501.07508),
AAAI 2025 Workshop on AI for Social Impact).

The environment, state, action semantics, reward, and walk-forward training follow the
earlier work. Fees are the exception: they now come from every individual swap
(22.6M of them) rather than from a price-displacement formula.

**New here? Read [`HANDOVER.md`](HANDOVER.md)** — what changed, what did not, and how
to run it. [`SPEC.md`](SPEC.md) is the append-only pre-registration and forensic
experiment log; its historical result sections include explicitly void reads. The
audited current results are in [`reports/COMPLETE_EXPERIMENT_RESULTS.md`](reports/COMPLETE_EXPERIMENT_RESULTS.md).
[`DIVERGENCES.md`](DIVERGENCES.md) is the full audit with file:line evidence.

---

## Setup

```bash
conda env create -f environment.yml
conda activate deeprl-uniswap
pip install -e ".[dev]"           # editable install; no PYTHONPATH needed afterwards
pytest -q                          # 162 tests, ~3s
```

TA-Lib needs its C library, which pip cannot install on its own. It comes from
conda-forge in `environment.yml`; without conda, `brew install ta-lib` first.

## Getting the data

The panel is not in git (458MB). Two options.

**Use the bundle.** Unzip `uniswap_panel.zip` into `data/processed/`. That is all the
code reads.

```bash
unzip dist/uniswap_panel.zip -d data/processed/
```

**Or rebuild it** from raw swap events. Needs the sibling `defi-rv` checkout
(~4.9GB of raw parquet), which is a prerequisite this repo cannot fetch for you:

```bash
python -m src.deeprl_liquidity_provision_uniswapv3.data.aggregate   # hourly panel
python -m src.deeprl_liquidity_provision_uniswapv3.data.swaps       # per-swap fees
```

Six core pools, 3 asset pairs x 2 fee tiers, 2021-05 to 2026-06, ~44,900 hours each.
The fee tier is independently recovered from the raw data as a check: `aggregate.py`
asserts that the median disagreement between a swap's two token legs equals the
declared tier, on all 12 pools.

## Running experiments

**Several algorithms, the easy way:**

```bash
./scripts/run_algos.sh                        # ppo a2c dqn qrdqn recurrentppo, 6 pools
./scripts/run_algos.sh "ppo a2c"              # a subset
STEPS=50000 ./scripts/run_algos.sh "ppo"      # a longer budget
POOLS="usdc_weth_005" ./scripts/run_algos.sh "ppo"     # one pool, to try it out
```

Resumable and shardable. One work unit is one (algorithm, pool, rolling step); it
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
before `sbatch`; set `REMOTE_HOST`, `REMOTE_PROJECT`, and `REMOTE_CONDA` when using
`scripts/sync_to_cluster.sh`.

This is a **CPU** workload. Measured ~6,100 steps/s on CPU; the nets are far too small
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

Both aggregators refuse partial collections. The frozen protocols, practitioner
motivation, audit status, and final results are under `reports/`.

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
    rolling.py      THE protocol: walk-forward. Start here.
    transfer_rolling.py, sensitivity.py, combined.py
    retail_frontier.py, action_geometry.py, validate_*.py
    bakeoff.py      splits, env construction, scoring
    agent_arm.py    training envs
    leaderboard.py, global_rule.py, transfer_strategies.py
scripts/
  run_algos.sh      multi-algorithm runner
  sync_to_cluster.sh    push / pull / status against a SLURM cluster
  slurm/            array wrappers
  exhibits/         one script per cited claim in SPEC.md
reports/            frozen extension protocols, audits, results, and figures
tests/              162 tests; the protocol and the accounting are both covered
rl-code/            the earlier codebase, kept for reference only. Superseded; do not
                    build on it or on the outputs under rl-code/output/.
```

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

SAC and TD3 are deliberately absent: they need a `Box` action space, which is a
different formulation of the problem.

`--paper-extractor` uses the earlier policy network (BatchNorm -> [4,2] -> 128, tanh)
instead of the SB3 default MLP. The BatchNorm is load-bearing, because the 13-feature
state carries raw prices (~2,000) next to raw liquidity (~1e18).
