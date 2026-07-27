# Active liquidity provision in Uniswap v3

> **Authoritative-results update, 2026-07-22.** This file is an append-only
> pre-registration and forensic experiment log; it deliberately preserves void and
> retracted reads below. The completed post-fix collections, independent hashes, and
> dependence-aware results are in `reports/COMPLETE_EXPERIMENT_RESULTS.md` and
> `outputs/final_audit.json`. Do not cite a numerical result from this file merely
> because it appears later than another section.

## Research objective

Given that a liquidity provider must quote, does active range management reduce the
loss relative to passive provision, and can a learned policy do it better than the
heuristics practitioners use?

The benchmark is passive LP, not holding. A market maker's business is providing
liquidity, so "withdraw and hold" is not an available strategy. Without that
constraint the problem is degenerate: within a window, fees and impermanent loss
both scale with liquidity at a near-constant ratio, so reward is linear in exposure
with a negative coefficient and the optimum is to provide nothing. Measured in our
own numbers, not imported from a theorem.

## Background

An LP cannot eliminate impermanent loss. The claim under test is mitigation.

**The loss in this paper is impermanent loss against holding the initial basket, and
nothing else.** That is what the previous paper measured and what its environment
computed. Loss-versus-rebalancing is a DIFFERENT quantity against a DIFFERENT
benchmark (a continuously rebalanced portfolio, not a held basket), and it is not
used here. Two corrections to earlier drafts of this file, recorded because the
confusion cost real work:

- Calling the degenerate wide-band optimum "the loss-versus-rebalancing result" was
  wrong. LVR does not say fees fail to cover the loss; it says arbitrage loss is
  positive and grows with volatility. Whether fees exceed it is a separate question
  with its own literature. Our degenerate optimum is an empirical fact about this
  panel, established by our own width sweep, and it needs no theorem to license it.
- `reward_shaping="lvr"` was built on that mistaken framing and is REJECTED by
  measurement (see below). It is retained only as a documented negative ablation.

The old repository's `custom_env_new_lvr.py` is not a counterexample: no script
imports it, only a commented-out notebook line refers to it, and it subtracts a
dimensionless ratio (`lvr = ll / vp`) from dollar fees, so its units do not agree.
It never produced a number in the paper.

## Method

Uniswap v3 concentrated liquidity as a Markov Decision Process. The LP holds a
position over a tick range and chooses, at each decision point, whether to hold or
redeploy at one of several widths. Reward is realized profit and loss against a
hold benchmark: the position's share of the pool's actual fee income while in
range, less the change in impermanent loss, less gas and swap costs.

Fee income is the LP's share of real hourly pool fees, `fees_usd * L / (pool_L + L)`,
earned only while in range. This replaces a price-displacement fee model that
captured arbitrage flow only and understated fee income to ~1.5% of impermanent
loss, and it retires the sole-provider assumption in the same expression.

## Data

- Sources: swap-level Uniswap v3 events, Ethereum mainnet, aggregated hourly.
  See `data/raw/PROVENANCE.md`.
- Time period: 2021-05-05 to 2026-06-19. ~44,900 hours per pool.
- Frequency: hourly.
- Panel: 3 asset pairs x 2 fee tiers, balanced. USDC/WETH, WETH/USDT, WBTC/WETH,
  each at 0.05% and 0.30%. Tier varies within pair, which identifies a fee-tier
  effect free of the asset confound.
- Excluded: WBTC/USDT, both tiers (52.9% and 22.1% of hours have no swaps; the
  0.05% pool's median hourly volume is $0). Recorded in the pool registry.
- Variables: see `data/DATA_DICTIONARY.md`.

## Pre-registration (frozen before touching test data)

- **Date frozen:** 2026-07-16
- **Hypotheses:**
  - H1. Active range management mitigates the LP loss relative to passive
    provision, on the core panel.
  - H2. A learned policy (PPO) mitigates the loss by more than the best competitor
    strategy selected under the same protocol.
  - H3. Mitigation varies with the fee tier, identified within asset pair.
  - H4. A policy transfers across pools, and transfer degrades less across asset
    pairs at a fixed fee tier than across fee tiers within a fixed pair. The 3x2
    panel identifies this: USDC/WETH 0.05% -> 0.30% holds the asset fixed and varies
    the tier; USDC/WETH 0.05% -> WBTC/WETH 0.05% holds the tier fixed and varies the
    asset. The previous paper claimed the target pool's fee regime sets the ceiling
    rather than asset similarity, but its two-pool design confounded the two and its
    transfer numbers did not depend on the source policy.
- **Train / out-of-sample split (exact):** windows of 1,500 hours, taken in
  chronological order per pool, non-overlapping. First 50% train, next 25%
  validation, final 25% test. Disjoint by construction and asserted in code
  (`experiments/bakeoff.py::Split`). Test is read once, after selection.
- **Primary metric (single scalar + direction):** mean realized reward per test
  window, in USD, on a $30,000 position. Higher is better. Reported per pool and
  pooled.
- **Independent unit:** the window. Seeds are averaged within a window before any
  test. Five seeds on one price trajectory are not five observations.
- **Statistical test:** Wilcoxon signed-rank, paired by window, given heavy tails.
  Seeded bootstrap confidence intervals. Holm correction across the competitor
  family within each pool.
- **Stopping rule:** one test read per configuration. If a configuration is changed
  after a test read, the change is dated and justified here, and the result is
  reported as exploratory.

### Choices fixed before the test read

Each of these was selected by inspecting training or validation results, so each is
frozen here to keep the test read honest.

- **Decision frequency is a property of the STRATEGY, not the environment.** "I
  rebalance hourly" is a choice a strategy makes. Forcing one schedule on everything
  destroys what distinguishes the competitors, since `ReactiveRecentering` fires on
  a volatility or jump signal, `VolProportionalWidth` recomputes a width, and
  `Passive` never acts. Each competitor is a complete specification including its
  own timing, so **competitors run their native logic on the hourly environment and
  decide for themselves whether to act.**
  A learned agent has no native timing, so one is declared: **daily (every 24h) is
  the primary**, with hourly, weekly, and event-driven as sensitivity arms.
  Daily is not the most favorable choice. On validation, PPO scores -6,502 hourly,
  -3,791 daily, -1,509 weekly, -1,965 event-driven, so daily concedes roughly 2,300
  against weekly. It is chosen because it is the realistic operating frequency for
  an LP: a claim that holds at a realistic cadence is worth more than one that needs
  a weekly cadence to survive. The sensitivity arms are reported whatever they say.
- **Action grid: widths {100, 200, 500, 2000} ticks, plus hold.** The previous
  grid `{0, 45, 50, 55}` was itself selected by maximizing test reward and offers
  only sub-1% bands, in range ~1.2% of the time.
- **Algorithms:** PPO, A2C, DQN. All on the same `Discrete` space and the same
  budget.
- **Training budget: 20,000 agent steps.** Fixed from a convergence check on
  VALIDATION, not chosen to flatter a result: PPO plateaus from 20k to 100k at
  about -1,650 (10k -2,019; 20k -1,671; 50k -1,640; 100k -1,693; 200k -1,839), so
  more training does not help and slightly hurts. 20k is ~22 passes over the train
  span. Recorded because an unstated budget would leave every agent result open to
  "it was under-trained".
- **Agent config grid:** learning rate in {3e-4, 1e-3} x entropy coefficient in
  {0, 0.01}, selected on validation. Deliberately small. The previous paper gave PPO
  10 Optuna trials plus its choice of action space while `ILMinimizer` had no free
  parameters at all; a sprawling agent grid would make "matched budget" a fiction in
  the opposite direction.
- **Competitor set:** the paper's four (PassiveWidthSweep, VolProportionalWidth,
  ILMinimizer, ReactiveRecentering) plus RecentreWhenOut and Passive. Original
  functional forms and parameter grids. Matched selection budget.
- **Capital: $30,000. Gas: flat $5.** Both load-bearing and neither yet defended;
  see LIMITATIONS.md. Sensitivity to both is a robustness exhibit, not a headline.

## Status

### READ THIS FIRST: which numbers in this file are alive

This file is append-only and its sections are NOT in date order, so reading it top to
bottom will hand you void numbers before it hands you live ones. Nothing below is
trustworthy without checking it against this table.

| section | status |
|---|---|
| "Verified (tests actually run)" immediately below | **VOID.** Produced before the cost double-charge fix, the competitor fixes, and the agent-budget fix. Its "+1,754 pooled / 56% of the loss" and its H2 rejection are superseded by the CLEAN READ. |
| H3 "sign REVERSED" table | **VOID as evidence for the paper's claim.** It measures mitigation; the paper's sentence is about a transfer target's absolute ceiling. Different quantity, and on a panel missing the paper's own WBTC/USDC pool. |
| "CLEAN READ" | **The most recent full read**, but it still predates the competitor fixes and the agent-budget fix, so its RL arms are not trustworthy either. |
| "Results at REALISTIC widths" and everything under "Superseded" | **VOID.** Double-charged costs. |
| "The rejected paper's result, decomposed" | **RETRACTED in part.** Only the leak (+675) survives; the -2,140 action-grid handicap was our own 10x unit error. |
| "LVR reward shaping" | Live as a negative result. |
| "RESOLVED" sections (fee discretization, double charge) | Live. |

**Every RL number in this file predates the fix that gave the agent a real search
budget, and is therefore void.** The agent's config was hardcoded and its validation
score discarded, so it searched ONE configuration while the competitors searched up to
27. Re-running is the top open item.

**Every competitor number in this file predates the competitor fixes**, which found
that `VolProportionalWidth` and `ILMinimizer` were saturating at the widest band with
every configuration collapsed onto one policy, and that all four read the wrong sigma
series.

### Verified (tests actually run) -- SEE THE TABLE ABOVE, the results here are VOID
- NOTE: the H1 numbers below were produced with the competitors consulted hourly
  while PPO was event-driven. They stand as competitor-vs-passive results, but the
  competitor-vs-PPO comparison they imply is void and is being re-run with the
  frequency applied uniformly.
- Fee tier recovered independently from the raw swap data: the median disagreement
  between a swap's two token legs equals the fee tier, on all 12 pools.
- Environment: 17 tests, three verified by mutation (reintroducing the 100x fee
  bug, the sole-provider fee, and the value-manufacturing rebalance each fail the
  test written for them).
- H1 on the competitor set, clean test read: mitigation +2,104 to +3,231 on the four
  WETH-against-stablecoin pools; +335 and +366 on the two WBTC/WETH pools.

- **H1 SUPPORTED.** Clean test read, 48 windows, 6 pools. Active management mitigates
  **+1,754 pooled** (passive -3,151, best competitor -1,397), 56% of the loss.
- **H2 REJECTED for PPO.** Clean test read, one read, budget fixed from a validation
  convergence check. Pooled: PPO -3,137 against competitor -1,397 and passive -3,151.
  **PPO is indistinguishable from doing nothing (+15 over 48 windows)** and loses to
  the best competitor by 1,740. Failure mode differs by pool: on `wbtc_weth_030` it
  learns to never act (seed spread 0, identical to passive); on `weth_usdt_030` it
  acts and is 852 WORSE than passive. Not a budget artifact (converged at 20k), not
  a frequency artifact (daily beats hourly), not a leak artifact (both sides selected
  on validation).
- **H3 SUPPORTED, with the sign REVERSED against the previous paper.** Mitigation by
  pair and tier:

  | pair | 0.05% | 0.30% | tier effect |
  |---|---|---|---|
  | USDC/WETH | +2,104 | +3,231 | +1,127 |
  | WETH/USDT | +2,191 | +2,302 | +111 |
  | WBTC/WETH | +366 | +335 | -31 |

  The **asset pair dominates and the fee tier barely matters**: a 6x gap across
  pairs against a noisy, sign-flipping effect across tiers. The previous abstract
  claimed "the target pool's economic regime, rather than asset similarity, sets the
  absolute reward ceiling". Its two-pool design varied asset and tier together and
  could not have separated them. Caveat: 3 pairs, so the tier effect is weakly
  powered; the pair effect is not.

### RESOLVED: fees are now attributed per swap, and the discretization was worth ~5%

The environment used to check whether we were in range at the END of each hour and,
if so, credit the WHOLE hour's aggregated fees. `usdc_weth_005` sees ~250 swaps an
hour, so a narrow band crosses in and out repeatedly inside one hour, and any hour
ending out of range earned zero. The prediction recorded here was that this
**systematically penalised concentration**, that the penalty grew as the band
narrowed, and that it was large enough to overturn the results, because scaling fee
income by two moves the best passive width from +/-170% to +/-5%.

**Fees are now attributed per swap** (`data/swaps.py`, `envs/uniswap_v3.py`). Each
swap carries its own price interval (a Swap event reports the price AFTER the swap,
so the swap traversed from the previous swap's price to its own), its own active
liquidity, and its own fee, taken off the gross input leg. A band covering part of a
swap's interval earns that part of the fee, apportioned linearly in sqrt-price. The
22.6M swaps of the core panel re-aggregate to the hourly panel's `fees_usd` within
0.10% on every pool. The agent still decides hourly, or at whatever cadence its
schedule imposes; only the accounting changed.

**The prediction was wrong.** Passive LP at a fixed width, medians over the 8 test
windows, per-swap fees against hourly fees, all six core pools:

| pool | swaps/h | fee lift at +/-0.5% | fee lift at +/-65% | best width, hourly | best width, per-swap |
|---|---|---|---|---|---|
| usdc_weth_005 | 249 | 1.03x | 0.94x | 5000 | 5000 |
| usdc_weth_030 | 21 | 1.10x | 1.00x | 5000 | 5000 |
| wbtc_weth_005 | 50 | 1.09x | 0.99x | 5000 | 5000 |
| wbtc_weth_030 | 9 | 0.94x | 1.00x | 5000 | 5000 |
| weth_usdt_005 | 149 | 1.01x | 0.98x | 5000 | 5000 |
| weth_usdt_030 | 28 | 1.00x | 1.03x | 5000 | 5000 |

The discretization was worth between -7% and +11% of fee income, with no consistent
sign, and **the optimal width does not move on any pool**. It is nowhere near the 2x
the knife-edge needs. The busiest pool (249 swaps/h) shows 1.04x and the quietest
(9 swaps/h) shows 0.94x, so the effect does not even scale with swap frequency the
way the argument required.

The reason the argument failed: a passive +/-0.5% band is in range only 3.4% to 5.1%
of the time, and that is not an artifact. Over a 1,500-hour window the price leaves a
+/-0.5% band and does not come back. The band is dead for real reasons, not
accounting ones. Within-hour crossing is rare precisely because the price is almost
never near a narrow band's edge.

Two things follow. The per-swap model is the correct accounting and stays, and the
knife-edge in fee scale is still real: **the results below rest on our fee income
being right within a factor of two, and the discretization was not the thing that
could have moved it.** If fee income is wrong by 2x the cause is elsewhere (the
liquidity-share denominator, or the USD valuation of a leg), and that is where an
independent check is worth its cost.

### RESOLVED, and it does move the numbers: gas and swap were charged TWICE

Found by an independent audit of the fee and value accounting, not by the tests,
which could not see it. Acting deducted its costs from the position's value:

    self._set_range(i, width, max(v - gas_cost - swap_cost, 0.0))

so `_value_usd(j)` was already net of them, and `il_total = hold - value` carried
them straight back into `d_il` on the same step. The reward then subtracted them
again:

    reward = fee - d_il - gas_cost - swap_cost      # gas and swap, twice

Fees are paid out rather than reinvested and costs come out of the position, so the
episode identity is `sum(reward) == cum_fees - (IL_final - IL_at_entry)`. Measured
against it, on a 300-hour synthetic path:

| scenario | sum(reward) | fees - IL | costs | gap |
|---|---|---|---|---|
| never acts, no costs | -2,070.5 | -2,070.5 | 0.0 | -0.0 |
| acts daily, gas $5 | -341.0 | -189.1 | 151.8 | **-151.8** |
| acts every 6h, gas $5 | -757.5 | -149.4 | 608.1 | **-608.1** |
| acts daily, gas $50 | -1,441.5 | -750.4 | 691.1 | **-691.1** |

The gap equals the costs to the decimal in every scenario. The audit measures the
double charge at **~$1,199 per 1,500h window** at a daily cadence on the real panel,
against a pooled mitigation headline of +1,754 and a PPO-vs-competitor gap of 1,740.
It is the same order as every number this paper reports.

**It is not neutral across arms, and it biased toward the reported conclusions.**
The double charge is proportional to how often a strategy acts, so it taxed active
management (H1), taxed PPO relative to passive (H2), and inflated
`ReactiveRecentering`'s -2,883. Passive never acts and never paid it, so the
benchmark every claim is measured against was the one arm the defect could not
touch.

Fixed by keeping one channel: costs are deducted from the position's value, and
`reward = fee - d_il`. That is preferable to the alternative (gross value, costs in
the reward) because it makes a cost compound against the capital that remains.

**Why the tests missed it.** `test_reward_decomposes_exactly` asserted the reward
formula against its own terms, which is true by construction whatever the formula
says, and every other test either never acted or never checked a sum. The suite now
holds the environment to the episode identity across four cost regimes
(`test_reward_sums_to_fees_minus_il`), verified by mutation.

### Also fixed from the same audit

- **Swap chain order.** `_read_swaps` sorted on `block_timestamp` with pandas'
  default quicksort, which is not stable, while 65% of swaps share a timestamp with
  another swap (tie groups up to 251). That left 41% of swaps out of true chain
  order. Invisible to the hourly panel, which takes only the hour's last row, but
  the swap-level model chains each swap's price interval off its predecessor, so the
  order IS the data. Now sorted on `(block_number, log_index)`, stable. Measured
  effect on fees: under 0.1%. Real bug, immaterial outcome.
- **Proration measure.** The fee is levied on the input leg, so the share of the fee
  a sub-interval carries is the share of the INPUT it absorbed. At constant L the
  token1 input is `L*d(sqrtP)` and the token0 input is `L*d(1/sqrtP)`, so a single
  sqrt-price proration is exact for one direction and wrong for the other, and about
  half of all swaps are token0-in. Each swap is now apportioned in its own measure.
- **Tick bounds were not mintable.** The half-width was not snapped to the tick
  spacing, so at the 0.30% tier (spacing 60) a 100-tick half-width described a
  position that cannot be opened on-chain. Affected all three 0.30% pools. Now
  snapped.

### Rejected: Loesch et al. is NOT an external calibration of our fee model

An audit proposed that Loesch et al. (arXiv 2111.09192) validate our fee income:
they report total fees of $199.3m against total impermanent loss of $260.1m across
17 pools, a ratio of 0.766, and this environment produces 0.75 to 0.82. Checked
against the paper itself rather than taken second-hand:

- **The numbers are real** and their definition is ours. They define IL against the
  HODL value of the originally contributed basket, and their "fee adjusted" position
  strips fees and gas OUT so that IL is a pure divergence term. Same benchmark, and
  gross of fees, which is the comparison's precondition.
- **The ratio is theirs**, reported as "the IL was $260m or 130% of the fees earned".
- **It still does not validate anything.** Their 0.766 is a dollar-weighted aggregate
  over every position in 17 pools across 4.5 months, dominated by the largest
  positions and driven by the realized ETH path and the empirical width distribution.
  Ours is one hypothetical $30,000 position at one width over one window. Their own
  dispersion settles it: by duration slice their fees/IL runs from about 0.91 to
  about 0.56, and two pools exceed 1.0. A band that wide cannot discriminate our
  0.75 to 0.82 from anything. **A match here is a coincidence of two different
  aggregations, not evidence, and claiming otherwise would put a false validation in
  the paper.**

Usable only for the qualitative ordering: IL exceeded fees in aggregate on v3 over
their window, so the sign of our reward is consistent with observed behavior. It
would become a real calibration only by simulating their pool set, widths, position
sizes, and price path, and aggregating the same way.

Two traps recorded for whoever cites this. The paper says **49.5% of wallets** had
negative returns, so "a majority of positions were net negative" is wrong twice
over: it is not a majority, and it is wallets rather than positions. And the paper
contradicts itself on how many pools earned fees exceeding IL, saying two on p. 25
and three in the conclusion. It is also a non-peer-reviewed industry preprint
(Topaze Blue), which is worth a word in the text if it carries any weight. Correct
entry: `@misc`, Stefan Loesch, Nate Hindman, Mark B. Richardson, Nicholas Welch,
2021, arXiv 2111.09192, DOI 10.48550/arXiv.2111.09192. No peer-reviewed version
exists.

**The CLEAN READ below postdates the per-swap and double-charge fixes. It PREDATES the
competitor fixes and the agent-budget fix, so its RL arms and its competitor arms are
both superseded. See the status table at the top of this section.**

### CLEAN READ: per-swap fees, costs charged once, stable chain order, mintable ticks

Realistic widths {100, 200, 500} = +/-1%, +/-2%, +/-5%. Agent event-driven, 20k steps.
48 test windows, 6 pools. Everything selected on validation, one test read.

| strategy | kind | val | TEST | mitigation | vs best | p (Holm) |
|---|---|---|---|---|---|---|
| RecentreWhenOut | heuristic | -1,442 | **-1,787** | **+1,386** | - | - |
| ILMinimizer | heuristic | -1,364 | -1,833 | +1,339 | -46 | 1.000 |
| VolProportionalWidth | heuristic | -1,412 | -1,859 | +1,314 | -72 | 0.882 |
| **PPO (event-driven)** | rl | -2,200 | -2,681 | +492 | **-894** | **0.0019** |
| A2C (event-driven) | rl | -799 | -2,820 | +352 | -1,033 | **0.0063** |
| DQN (event-driven) | rl | -1,552 | -3,140 | +32 | -1,353 | **0.0170** |
| PassiveWidthSweep | heuristic | -1,021 | -3,155 | +18 | -1,368 | **0.0337** |
| Passive (never act) | heuristic | -1,110 | -3,173 | 0 | -1,386 | **0.0221** |
| ReactiveRecentering | heuristic | -3,204 | -3,440 | -267 | -1,653 | **0.0000** |

**Charging costs once changed both conclusions from "underpowered" to significant,
and it did NOT rescue RL.** Before the fix nothing survived Holm except
`ReactiveRecentering` being bad. Now:

- **H1 SUPPORTED, and significant.** Active range management mitigates **+1,386** of a
  -3,173 passive loss, i.e. **44%**, and passive loses to `RecentreWhenOut` at
  **p=0.022** paired over 48 windows. The three active heuristics cluster within 72 of
  each other at the top, which is a stronger statement than one rule winning: several
  independent ways of managing the range all recover roughly the same amount.
- **H2 REJECTED, and significant.** PPO trails the best rule by **894 at p=0.0019**,
  winning **25%** of 48 windows. Before the fix the gap was 429 at p=0.43, i.e. not
  established. It is now established, against us. Every RL algorithm loses to every
  active heuristic, and DQN is statistically indistinguishable from passive.

The double charge had been taxing the heuristics and PPO alike, compressing
everything toward passive and destroying the power to separate anything. Removing it
raised the whole active field and left RL behind.

Note PPO's validation (-2,200) is now WORSE than its test rank suggests, while A2C
has the best validation of any RL arm (-799) and the second-worst RL test (-2,820).
Simple rules move -400 to +50 from validation to test; the RL arms move -480 to
-2,000. Fourteen training windows is not enough for a neural policy.

### Superseded: results at realistic widths under the double-charged reward

A +/-2000-tick band is -18% to +22%, i.e. approximately a v2 position, and it wins by
avoiding IL rather than earning fees (fees fall 4,177 -> 829 across widths while IL
falls 6,223 -> 1,299). No LP would deploy it, so it is a flawed benchmark. Capping
widths to {100, 200, 500} ticks = +/-1%, +/-2%, +/-5%, which is what real WETH/USDC v3
LPs quote:

| strategy | val | TEST | mitigation | vs best | p (Holm) |
|---|---|---|---|---|---|
| RecentreWhenOut(500) | -1,972 | -2,314 | +837 | - | - |
| ILMinimizer | -1,855 | -2,394 | +758 | -79 | 0.885 |
| VolProportionalWidth | -1,842 | -2,469 | +682 | -155 | 1.000 |
| **PPO (event-driven)** | **-1,623** | -2,743 | +408 | -429 | **0.426** |
| Passive | -1,140 | -3,151 | 0 | -837 | 0.304 |
| PassiveWidthSweep | -1,028 | -3,175 | -23 | -860 | 0.343 |
| DQN | -1,270 | -3,181 | -29 | -866 | 0.293 |
| A2C | -1,074 | -3,360 | -209 | -1,046 | 0.069 |
| ReactiveRecentering | -5,607 | -6,034 | -2,883 | -3,720 | **0.000** |

**After Holm correction NOTHING is distinguishable except ReactiveRecentering being
bad.** PPO trails the best rule by 429 at p=0.43, winning 42% of 48 windows. So "RL
loses" is NOT established at realistic widths; we are underpowered to rank anything.
Mitigation also drops from +2,322 (wide) to +837 (realistic), so the 74% headline was
mostly the degenerate band.

Note PPO has the BEST validation score of all nine (-1,623) and finishes 4th on test.
Heuristics move -40 to +203 from validation to test; every RL algorithm drops 1,100
to 1,850. Simple rules cannot overfit 14 training windows; a neural policy can.

### LVR reward shaping: REJECTED (and it was never the paper's quantity anyway)

This arm exists because an earlier draft of this file mislabelled the degenerate
optimum "the loss-versus-rebalancing result", an audit took that framing at face
value and proposed the reward should BE loss-versus-rebalancing, and it was built.
The paper measures impermanent loss against holding. LVR is a different benchmark
and was not in the previous paper at any point. The arm is kept only as a documented
negative ablation: shaping the reward does not fix the learning problem.

The argument was that `fee - dIL` is unlearnable because one step of dIL against the
hold basket is

    dIL = (hold0 - a0) * dP  -  (1/2) * a0'(P) * dP^2

whose first term is a first-order martingale no action influences, and that
benchmarking against a portfolio holding the position's CURRENT amounts for one step
cancels it exactly, because that benchmark's exposure IS a0*dP. That is
loss-versus-rebalancing. Implemented as `reward_shaping="lvr"`, trained on, evaluated
on the true reward.

The mechanism works: LVR is non-negative, zero out of position, never touches the
reported reward, and cuts the reward's standard deviation by 2.8x to 18x. **It does
not make the problem learnable.** Pooled validation, same widths, comparable arms:

| arm | pooled val | distinct actions |
|---|---|---|
| Passive (never act) | **-1,110** | 1.0 |
| ILMinimizer(24,out) | -1,364 | 2.7 |
| RecentreWhenOut(500) | -1,442 | 2.0 |
| PPO, lvr shaping | -1,304 | **1.1** |
| PPO, no shaping | -2,345 | 1.2 |
| PPO, shadow shaping | -2,674 | 1.4 |

LVR is worth +1,041 over no shaping, and the gain is **PPO learning to stop churning
and converge toward doing nothing**. It still does not reach passive, and at 1.1
distinct actions it is not controlling anything. The pre-registered guard (hard-fail
a run using one action) fails it. The shadow variate is worse than no shaping at all.

**The reason is a regime shift between validation and test, and it explains the whole
RL result.** Passive is the BEST arm on validation (-1,110) and the WORST on test
(-3,173); `RecentreWhenOut` is the worst on validation (-1,442) and the best on test
(-1,787). The ordering inverts. Mean absolute price move per window is 23.2% on
train, 26.4% on validation, 30.7% on test, and on the four WETH-against-stablecoin
pools the test windows move 38% against 26% on validation.

So an agent that selects on validation is selecting in a regime where doing nothing
is optimal, learns exactly that, and is then tested in a regime where recentring
pays. **The heuristics are immune because their functional form is conditional**:
"recentre when out of range" fires only when needed, costing little in a calm regime
and paying in a volatile one. A learned policy instead absorbs the marginal frequency
of acting from its training regime and carries it forward. That is why every RL arm
loses 480 to 2,000 from validation to test while the rules move 50 or less.

This is a genuine finding rather than a defect, and it is the honest reading of H2:
**the failure is not the algorithm, the reward, the budget, the features, or the leak.
It is that a chronological split across a non-stationary market asks a learned policy
to generalize across a regime change, and a two-line conditional rule does not have to.**

**Open, and the one thing that could still make RL work:** the current design trains
once on the first 50% and tests on the last 25%, a gap of years. No practitioner would
do that. Walk-forward retraining (fit on a rolling recent window, test on the next
one, roll) is the standard answer to non-stationarity and is what an LP would actually
run. It is untested here and is the highest-value remaining experiment.

### Features: the previous paper's state is BETTER, and it does not matter

The legacy 13-feature state (price, tick, width, liquidity, sigma, ma24, ma168,
Bollinger x3, ADXR, BOP, DX, on 12h rolling candles) replicated exactly in
`envs/features.py`, selectable via `features="legacy"`:

| features | pooled val | pooled TEST | vs passive |
|---|---|---|---|
| compact (rebuilt) | -1,489 | -2,689 | +462 |
| legacy (the paper's) | -1,485 | -2,308 | +843 |

Legacy is worth +381 to PPO, so the rebuilt state was a mild handicap. Validation is
identical (-1,489 vs -1,485), so legacy does not fit better, it GENERALISES better.
Does not close the gap to the rules. Note the legacy state never contained a fee
signal, because the old fee model made fees a deterministic function of price.

### The rejected paper's result, decomposed and reproduced

On `usdc_weth_005`, isolating each ingredient of the old protocol:

| | |
|---|---|
| PPO, config chosen on validation (honest) | -2,913 |
| PPO, per-seed max on TEST (their exact code) | -2,238 |
| -> inflation bought by the leak | **+675** |
| Best competitor, full action grid | -1,457 (ILMinimizer(H=24,out)) |
| Best competitor, THEIR grid {45,50,55} | -3,596 (PassiveWidthSweep(w=50)) |
| -> handicap imposed on the competitors | **-2,140** |
| **Honest gap** (PPO - competitor) | **-1,456**, PPO loses |
| **Their gap** (leaked PPO - crippled competitor) | **+1,358**, PPO wins |

**RETRACTED: the action-grid half of this decomposition is an artifact of a 10x unit
error of OUR OWN, not a defect of the previous paper.**

The claim was that the dominant term is the action grid, that `{45,50,55}` are
+/-0.5% bands in range ~1% of the time, and that this handicapped the competitors by
-2,140. It rested on feeding `{45,50,55}` into THIS env, which reads action widths as
raw ticks. The previous env does not. It multiplies by the tick spacing:

    self.d = self._fee_to_tickspacing(self.delta)          # 0.05 -> 10
    tl, tu = m - self.d * self.w, m + self.d * self.w      # custom_env.py:292, 314

So action 45 is **450 ticks = +/-4.60%**, 50 is +/-5.13%, 55 is +/-5.65%. Verified
against the source. The previous paper's grid was never sub-1%; it is essentially
this rebuild's own `500` (+/-5.13%), i.e. the width we adopted as the realistic fix
after calling theirs degenerate. **Their action grid was fine. We misread it, then
priced our misreading at -2,140 and called it the dominant term.**

The leak itself (+675, tuning on test) is the one part of this decomposition that
survives; it was measured on our own env with our own grid and does not depend on the
unit error. Everything else in the table above is void and must be re-run with widths
`{450, 500, 550}` before any of it is repeated.

Two consequences that follow from the same fact:
- The same grid on the 0.30% pools has `d=60`, so action 45 is 2,700 ticks, a **+/-31%
  band**. The previous paper called this a "misalignment" with WBTC's tick spacing. It
  is not a misalignment, it is a different strategy: +/-4.6% on WETH against +/-31% on
  WBTC, from the identical action. That alone can explain the cross-pool differences
  the paper attributed to fee regime.
- Any statement here derived from "the old grid is sub-1%", including "in range ~1.2%
  of the time", was computed in the wrong env and is withdrawn.
- H4. Not yet tested. The previous transfer result is void: its "same-period" test
  set was a superset of its training set, its reward was an episode sum compared
  across segments of different length, and its saved outputs repeat bit-for-bit
  across different source pools, so the reported 88.5% did not depend on the
  transfer.
- The WBTC/WETH validation-to-test collapse (+302 validation, -1,852 test) is
  unexplained. Those two rows are not trustworthy until it is.
- Gas is a flat constant across 2021 to 2026, when it ran 100+ gwei early and
  collapsed after Dencun. Directionally favors rebalancing in the early sample.

## Milestones

- [x] Panel built and validated
- [x] Environment rebuilt with unit tests
- [x] Pre-registration frozen
- [x] Competitors under the leak-proof protocol (H1)
- [x] PPO under the same protocol (H2): rejected
- [ ] A2C and DQN under the same protocol (H2)
- [ ] Window-level paired statistics
- [x] Fee-tier effect within pair (H3): supported, sign reversed vs the previous paper
- [ ] Cross-pool transfer on the 3x2 design (H4)
- [ ] Paper draft
