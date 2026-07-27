# Handover: what changed, and how to run it

For Arjun. Read the first two sections before running anything; the rest is reference.

The short version: **the setup is still the paper's setup.** Same 13-feature state, same
action semantics, same reward, same walk-forward training. What changed is a list of
bugs, plus one thing that is genuinely new: fees now come from every individual swap
instead of from a formula.

Everything below is verified against `rl-code/` rather than remembered.
`DIVERGENCES.md` has the full audit with file:line evidence; `SPEC.md` preserves the
append-only pre-registration and forensic log. The audited final rerun is now in
`reports/COMPLETE_EXPERIMENT_RESULTS.md`.

---

## 1. The one big change: fees come from real swaps now

**Before.** `custom_env.py::_calculate_fee` derived the fee from price displacement:

```python
fee = (self.delta / (1 - self.delta)) * self.l * (sqrt(p_prime) - sqrt(p))
```

Three things were wrong with that, all verified:

- **The rate was 105x too high.** `delta/(1-delta)` = **5.26%**, where the 0.05% tier is
  0.0005. The pools charged 5% and 30%.
- **Down-moves earned nothing.** Both `p` and `p_prime` were keyed on `pt`, never
  `pt_1`, so for any in-range falling price the function returns **exactly 0.0**. Fees
  accrued only when the price went up.
- **It only ever saw arbitrage flow**, because the dataset was `timestamp,price` with
  no volume at all. Measured on the real panel, that model gives fee income of ~1.5% of
  impermanent loss, so the reward was essentially `-IL` and there was nothing to trade
  off.

**Now.** Fees are attributed **per swap**, from swap-level Ethereum data:

| pool | swaps | swaps/hour |
|---|---|---|
| usdc_weth_005 | 11,160,555 | 248.6 |
| weth_usdt_005 | 6,671,309 | 148.6 |
| wbtc_weth_005 | 2,223,526 | 49.5 |
| weth_usdt_030 | 1,238,156 | 27.6 |
| usdc_weth_030 | 939,022 | 20.9 |
| wbtc_weth_030 | 394,412 | 8.8 |
| **total** | **22,626,980** | |

Each swap carries its own price interval, its own active liquidity, and its own fee:

```
fee = SUM over swaps in the hour of
        fee_usd(swap)                     the fee THAT swap paid, off its gross input leg
      * overlap(swap interval, our band)  the share of the swap that executed in our range
      * L / (liquidity(swap) + L)         our share of the liquidity active AT that swap
```

Why each piece matters:

- **A swap has extent in price, not a location.** A Swap event reports the price *after*
  the swap, so the swap traversed `[previous swap's price, this swap's price]`. A band
  covering part of that interval earns part of the fee. Uniswap v3 credits fee growth
  per unit of *active* liquidity as the price crosses each tick range, so this is what
  the protocol actually does.
- **The proration measure follows the input leg.** At constant L, a token1 input is
  `L*d(sqrtP)` and a token0 input is `L*d(1/sqrtP)`. Prorating everything in sqrt-price
  is exact for one direction and wrong for the other, and about half of all swaps are
  token0-in. Each swap is apportioned in its own measure.
- **The fee is on the GROSS input.** v3's `computeSwapStep` scales the remaining input by
  `(1e6 - feePips)/1e6` and keeps the difference, so `feeAmount / gross_input == fee`
  exactly. The input leg is the one with a positive amount, and the event's figure for
  it is gross because the fee stays in the pool.
- **`L / (pool_L + L)` retires the sole-provider assumption.** The chain-reported
  liquidity excludes our hypothetical position, so adding our own L dilutes us.

**Checks that this is right:** all 22.6M swaps re-aggregate to the hourly panel's
`fees_usd` within **0.10%** on every pool. And the fee tier is recovered independently
from the raw data: the median disagreement between a swap's two token legs equals the
declared tier on all 12 pools (`aggregate.py::_assert_sides_agree`), which is a free
check that we did not mislabel a pool's tier.

**Honest note on what this bought us:** the agent still *decides* hourly. Only the
accounting is finer. Measured against the hourly approximation, per-swap fees change
fee income by **-7% to +11%** with no consistent sign, and move the optimal width on
**no pool**. It is the correct accounting and it is worth a paragraph in the paper as a
methodological point (we found no prior work comparing per-swap against hourly fee
attribution), but it is not what changed the results. Do not oversell it.

---

## 2. Quick start

```bash
conda env create -f environment.yml
conda activate deeprl-uniswap
pip install -e ".[dev]"
pytest -q                                  # 162 tests, ~3s. If these fail, stop.

unzip dist/uniswap_panel.zip -d data/processed/ # 458MB, sent separately, not in git

./scripts/run_algos.sh "ppo" outputs/try   # one algorithm, all 6 pools
```

Then the real thing:

```bash
./scripts/run_algos.sh                     # ppo a2c dqn qrdqn recurrentppo
```

**It is resumable.** One work unit is one (algorithm, pool, rolling step). Each writes
its own JSON and is skipped if already done. Ctrl-C and rerun and it picks up. You can
stop it any time.

**It is shardable.** `SHARD=0 OF=8 ./scripts/run_algos.sh` takes an eighth. Shard *i*
names the same units on every machine, so your shard and a cluster shard merge into one
directory with no reconciliation.

**Sizing.** The full grid is 5 algorithms x 6 pools x 24 steps x 8 configs x 2 seeds,
plus a refit each. That is a lot for a laptop: start with `POOLS="usdc_weth_005"` or one
algorithm, see what it costs, and shard the rest. It is **CPU-bound** (~6,100 steps/s);
do not reach for a GPU, the nets are far too small and the env emits float64 which MPS
cannot take at all.

Useful knobs:

```bash
STEPS=50000 ./scripts/run_algos.sh "ppo"            # longer budget
POOLS="usdc_weth_005 wbtc_weth_030" ./scripts/run_algos.sh "ppo"
SEEDS="42 123 256" ./scripts/run_algos.sh "ppo"
```

Read results:

```bash
python -m src.deeprl_liquidity_provision_uniswapv3.experiments.rolling \
       --out outputs/algos_v1 --algo ppo --aggregate
```

The aggregate says how many units are missing and refuses to present a partial table as
a finished run.

---

## 3. What is the same as the paper (deliberately)

These are defaults now, not options. The job is to fix bugs, not to change the problem.

| | |
|---|---|
| **State** | the same 13 features: price, tick, w, L, EWM sigma, ma24, ma168, 3 Bollinger bands, ADXR, BOP, DX |
| **Actions** | `{0, 45, 50, 55}`, and a width is a **multiple of the pool's tick spacing** |
| **Reward** | `fees - dIL - gas`, gas charged in the reward, position value left gross |
| **Costs** | flat $5 gas, no swap or slippage cost |
| **Training** | walk-forward: fit on 5 windows of 1,500h, test on the next, roll, refit |
| **Exit** | none. The LP must quote, as before |

### The action grid is NOT sub-1%, and this matters

`custom_env.py:292` does `tl, tu = m - self.d*self.w, m + self.d*self.w` with
`d = _fee_to_tickspacing(0.05) = 10`. So:

| action | ticks | band (0.05% pool, d=10) | band (0.30% pool, d=60) |
|---|---|---|---|
| 45 | 450 | **+/-4.60%** | **+/-31.0%** |
| 50 | 500 | +/-5.13% | +/-35.0% |
| 55 | 550 | +/-5.65% | +/-39.1% |

I got this wrong for a while: I read `{45,50,55}` as raw ticks, got +/-0.5% bands,
concluded the paper's grid was degenerate, and priced a "competitor handicap" at -2,140.
That was my unit error, not a defect of the paper. **It is retracted.** The paper's grid
is fine and is essentially the +/-5% band that any real WETH/USDC LP would quote.

The right-hand column is worth your attention for the paper: the **same action is a
different strategy on a 0.30% pool** (+/-31% vs +/-4.6%). The paper described this as a
"misalignment with WBTC's tick spacing". It is not a misalignment, and it could explain
the cross-pool difference the paper attributed to fee regime. Untested.

---

## 4. Bugs fixed (these change the numbers)

Each is in `DIVERGENCES.md` with evidence.

1. **Fee rate 105x too high; zero fees on down-moves.** See section 1.
2. **Rebalancing teleported +/-44% of position value.**
   `self.l = xt / (1/sqrt(pt) - 1/sqrt(pu))` sized new liquidity from the **x leg alone**
   and the y leg was re-derived next step:

   | price drift at rebalance | value change |
   |---|---|
   | -2.0% | **+44.4%** |
   | 0.0% | -0.0% |
   | +2.0% | **-43.6%** |

   On a ~$40k position that is +/-$17,000 per rebalance against a $5 stated gas cost.
   **This is the single largest term in the published rewards.** Rebalancing is
   value-conserving now.
3. **The tuning objective was the test set.** `optimize_ppo` returned
   `evaluate_model(rl_model, Monitor(test_env))` as its Optuna objective, inside every
   rolling step, and checkpointed on test improvement. Worth **+675**. Also,
   hyperparameters leaked *across* rolling steps: study 1 wrote `_r1`, study 2 wrote back
   to the base YAML, so window *i*'s tuned values became window *i+1*'s defaults.
   Selection now happens on a validation window carved out of the training block; test is
   not reachable from the tuning path.
4. **Costs were charged twice** (a bug I introduced and then removed): the cost was
   deducted from position value AND subtracted in the reward. It scaled with how often a
   strategy acted, so it taxed every active arm and left passive, the benchmark,
   untouched. The suite now holds the env to the episode identity across four cost
   regimes.
5. **The agent searched 1 config while the heuristics searched 48** (also mine). The
   config was hardcoded and the validation score was computed and thrown away.
   **Every RL number produced before this is void.** Now an 8-config grid selected on
   validation, with tests pinning the agent's budget inside the heuristics' range.
6. **The competitors were not the paper's competitors.** Three separate faults, all
   mine, all now fixed and mutation-tested:
   - The width-to-action mapping dropped the paper's `0` ("do nothing") from the
     candidate list, so every computed width, however small, was forced onto a real
     band.
   - `base_factor` was raised 100 -> 1e4, which overshoots the `{45,50,55}` grid, so
     every config saturated at the widest band. `VolProportionalWidth(k=3)` and
     `(k=15)` were then **the same policy**, identical reward, one distinct action. Ten
     configs that are one policy is not a grid, and selecting over it is not selection.
   - They sized off `env.vol` (24h rolling std of simple returns) where the paper sizes
     off `env.ew_sigma` (EWM std of LOG returns, alpha=0.05). Different numbers.

   **What the paper's competitors actually are**, now that they are faithful: at
   `base_factor=100` against an hourly sigma of ~0.006, `VolProportionalWidth` computes
   `int(3*0.006*100) = 1`, whose nearest action is 0. So it **holds throughout** and
   scores exactly Passive. Same for `ILMinimizer(H=24)`. Measured on one window:

   | policy | reward | distinct actions |
   |---|---|---|
   | Passive (never act) | -7,540 | 1 |
   | VolProportionalWidth(k=3) | **-7,540** | **1** |
   | VolProportionalWidth(k=15) | **-7,540** | **1** |
   | ILMinimizer(H=24) | **-7,540** | **1** |
   | ILMinimizer(H=168) | -6,673 | 2 |
   | ReactiveRecentering | -5,501 | 2 |
   | RecentreWhenOut(50) | -3,141 | 2 |

   This is NOT something to repair. Raising `base_factor` so the strategy "works" would
   invent a competitor the paper never ran. But it is worth knowing when reading the
   paper's tables: of its four competitors, `PassiveWidthSweep` swept a single width on
   WETH, and two of the other three are passive. `base_factor` is exposed so a
   non-degenerate version can be swept deliberately and reported as an addition.

7. **`features="legacy"` did not replicate the paper's state.** It fed rolling means of
   *returns* where the paper fed rolling means of *price* (~3 orders of magnitude
   apart), a rolling std of simple returns where the paper used an EWM std of *log*
   returns (alpha=0.05), and the tick span where the paper carried the raw action
   integer. Fixed and tested.
8. **The policy network was gone.** The paper used a `CustomMLPFeatureExtractor` whose
   first layer is `BatchNorm1d(obs_dim, affine=False)`. That BatchNorm is load-bearing:
   the state carries raw prices (~2,000) next to raw liquidity (~1e18). Ported as
   `agents/extractors.py`, reachable via `--paper-extractor`.
9. **Swap chain order.** Swaps were sorted on `block_timestamp` with a non-stable
   quicksort while 65% share a timestamp, leaving 41% out of true chain order. Matters
   because the per-swap model chains each interval off its predecessor. Now sorted on
   `(block_number, log_index)`, stable. Effect on fees: under 0.1%. Real bug, immaterial
   outcome.
10. **Tick bounds were not mintable** at the 0.30% tier. The paper's own `d*w` convention
   fixes this for free, since `d*w` is always a multiple of `d`.

---

## 5. Things in the paper that do not match the paper's code

Not our bugs. You need to know them because the paper cannot be reproduced as written.

- **The state vector in the text is largely fiction.** It describes a one-step return, an
  EWMA of returns, the interval bounds, the relative position within it, and the fee
  tier. **None are in the code.** Nine of the thirteen features that *are* in the code
  are undisclosed in the text.
- **Eq. (4) is sign-inverted as printed.** As written it pays the agent to lose money.
  The code is right.
- **Timesteps: text says 1,000,000, the runner passes 100,000.** `total_timesteps:
  1000000` sits in the YAML and is never read. The paper's explanation for WBTC
  underperformance rests on a number that was never used.
- **"5 PPO agents under different random initializations" do not exist** in
  `uniswap_test.py`: `seed=256`, fixed, every trial and every window.
- **Initial capital matches no config.** Text says `x_0 = 2`; the configs say `x: 10`
  WETH (~$40k) and `x: 0.15` WBTC (~$6-10k). The cross-pool reward levels in Tab. IV/V
  are not comparable to each other, independent of everything else. We use a fixed
  $30,000 base instead, which is a deliberate change, recorded as such.
- **A second Optuna stage is hidden.** Tab. II calls gamma, lambda, clip "fixed"; study 2
  tunes `ent_coef, gamma, clip_range`, and the YAML the final test env loads carries
  `clip_range: 0.05`, `ent_coef: 1e-5`, contradicting both stated values.
- **Tab. III's action grid does not describe the WBTC runs at all** (`[0,60,120,180]`,
  sharing no entry with the stated search space).
- **The transfer result did not depend on the transfer.** `train` is a subset of the
  same-period test set; the reward is an episode sum over segments with a 5.67x length
  ratio; 40 result cells hold 23 distinct values, with different source pools producing
  bit-identical rewards on the same target. The script's own printed efficiency is 714%,
  not the paper's 88.5%.

---

## 6. Panel

Six pools, balanced 3 pairs x 2 fee tiers, 2021-05 to 2026-06, ~44,900 hours each,
sharing one clock.

```
usdc_weth_005  usdc_weth_030
wbtc_weth_005  wbtc_weth_030
weth_usdt_005  weth_usdt_030
```

Tier varies within pair, which is what identifies a fee-tier effect free of the asset
confound. The paper's two-pool design varied asset and tier together and could not
separate them.

**The paper's WBTC/USDC pool (`0x99ac8ca7...`) is not here.** The sibling `defi-rv`
project never fetched it, so it has no substrate. That is a data-availability accident,
not a design choice, and it means half the paper's evidence base can be replaced but
not reproduced or refuted. The transfer axis needs it. WBTC/USDT is excluded for
thinness (52.9% and 22.1% of hours have no swaps at all).

---

## 7. Where to look

```
experiments/rolling.py     THE protocol. Start here.
envs/uniswap_v3.py         the MDP: fees, IL, rebalancing, costs
data/swaps.py              per-swap fee attribution
agents/registry.py         5 algorithms
policies/baselines.py      the heuristics
tests/                     128 tests. The protocol and the accounting are both covered.
SPEC.md                    pre-registration + current results, including retractions
DIVERGENCES.md             the full audit, with file:line evidence
rl-code/                   the old code. Reference only. Every number under
                           rl-code/output/ came from the leaked, unit-bugged pipeline.
```

## 8. Open, and worth your time

**Status update, 2026-07-22:** the first two items below are complete. The corrected
walk-forward collection contains all five algorithms, and inference now averages seeds
within windows, reports pools separately, uses a seeded circular moving-block bootstrap,
and applies Holm correction within each pool. See `reports/PI_DIRECTIVE_AUDIT.md` and
`reports/COMPLETE_EXPERIMENT_RESULTS.md`. The final three sensitivity items remain open
as stated.

- **The walk-forward re-run under the fixed agent budget.** Every RL number predates it.
  This is the one that matters and it is exactly what `run_algos.sh` does.
- **Window-level paired statistics.** `aggregate` pools 6 heterogeneous pools into one n
  and never clusters by pool: 24 adjacent windows from one pool are not independent, and
  Wilcoxon assumes they are. No per-pool breakdown either, so the largest-scale pool
  dominates the pooled mean. `SPEC.md` pre-registers bootstrap CIs; none are computed.
  Needs no compute and is valid regardless of how the RL lands.
- **Early stopping** (`StopTrainingOnNoModelImprovement(5,5)` + `EvalCallback`) is not
  ported, so the paper's budget was data-dependent and ours is flat.
- **The paper's PPO coefficients** (gamma=0.999, lambda=0.9999, target_kl=0.3,
  n_steps=len/3, batch=256) are not ported; we run SB3 defaults.
- **Gas is a flat $5 across 2021-2026**, when it ran 100+ gwei early and collapsed after
  Dencun. Directionally favours rebalancing in the early sample.

## 9. Known, and not fixed

- **`features="legacy"` breaks if `ma_windows` is not length 2** (it indexes
  `price_mas[0]` and `[1]`, and `n_feat` is hardcoded to 13). Nothing in the CLI passes
  `ma_windows`, so it is a latent trap rather than a live bug.
- **`w_action` is stale after an EXIT**: the legacy state reports a live band width
  while holding no position. Unreachable under `allow_exit=False`, which is the
  default.
- **DQN and QR-DQN search 4 configs where PPO searches 8**, because they have no
  entropy coefficient. Real asymmetry, now visible rather than padded with duplicate
  runs.
- **On the paper's own `{45,50,55}` grid, several heuristic configs coincide.** The
  "configs offered" counts overstate the competitors' real search: `VolProportionalWidth`
  is passive at every k. See section 4.

## 10. Do not

- Trust any number in `rl-code/output/`.
- Trust an RL result produced before the agent-budget fix.
- Repeat "the paper's action grid is sub-1%" or the -2,140 competitor handicap. Both are
  retracted; they were my unit error.
- Commit anything you would not publish. Objects pushed to a repo in a fork network stay
  reachable by SHA from every other repo in that network, including forks that predate
  the push and never synced, and making the repo private does not retract them.
