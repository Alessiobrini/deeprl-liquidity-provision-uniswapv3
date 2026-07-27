# Execution-aware active liquidity provision

`FINAL — AUDITED`, 2026-07-22.

## Corrected baseline

The rejected paper's numerical conclusions do not survive the audited successor
protocol. Correct per-swap fee attribution, value-conserving rebalancing, leak-free
configuration selection, and window-level inference produce three central facts:

1. every strategy has negative mean reward against the hold benchmark in every pool;
2. no RL algorithm dominates across the balanced six-pool panel; and
3. cross-pool transfer is mostly statistically unresolved, with the one robust
   favorable PPO direction still loss-making.

The expanded comparison covers PPO, A2C, DQN, QR-DQN, and RecurrentPPO. The rejected
manuscript itself used only PPO as a learned method; its other four strategies were
PassiveWidthSweep, VolProportionalWidth, ILMinimizer, and ReactiveRecentering. The
successor adds Passive and RecentreWhenOut as transparent references.

The independent audit verifies 720 primary, 720 rolling-transfer, and 864 sensitivity
units. Seeds 42 and 123 are averaged within each window; adjacent windows and repeated
seeds are never treated as independent. Inference uses a seeded circular moving-block
bootstrap and Holm correction within each pool and hypothesis family.

## Why execution constraints are the natural extension

Practitioner evidence repeatedly identifies a fixed-cost viability problem. Uniswap
governance reported that 30% of positions held no more than $2,500 and estimated that
the profitable share fell from 53% before gas to 39% after gas. The same discussion
documents a contributor who deployed more than 200 small positions and was profitable
before gas but negative after it. Uniswap's official documentation confirms that an
out-of-range position earns no fees and ends entirely in one asset. Forum LPs describe
the resulting loop: swap, remove, and remint to resume earning, only to have gas,
slippage, and attention costs consume the yield.

The BIS evidence supplies a population-level complement to those anecdotes:
sophisticated providers capture a larger share of trades and substantially higher
absolute and relative profits than retail providers. Together, these sources motivate
an identifiable question that fits the current simulator: when do realistic operating
costs make active recentering worse than leaving a position alone?

Sources and a ranked list of adjacent research questions are in
`PRACTITIONER_RESEARCH.md`.

## Experiment 1: retail recentering viability

The frozen frontier varies position capital over $1,500, $5,000, $10,000, $30,000,
and $100,000 and gas over $1, $5, $25, $100, and $250. It compares
RecentreWhenOut—whose width is selected on the preceding validation window—with an
unchanged Passive position. The conversion treatment charges half the position at the
pool fee tier plus 5 basis points whenever recentering fires.

The collection is complete at 720/720 pool/window/capital units, with SHA-256
`a529455c1b5223807de4abd2aeeb5b00818eaa864f0c452895441291e410b9fc`.

With conversion friction, active recentering is worse in all 25 capital/gas cells in
each 0.05% pool. It is descriptively better in 22/25 USDC/WETH 0.30% cells, 22/25
WBTC/WETH 0.30% cells, and 23/25 WETH/USDT 0.30% cells. No favorable 0.30% cell survives
within-pool Holm correction at moving-block lengths 4, 6, or 8. At block length 4,
28 adverse cells survive: 26 in 0.05% pools and two small-capital/high-cost cells in
0.30% pools. These counts are regenerated from the four `frontier.csv` files by
`experiments.retail_frontier_summary`, rather than transcribed by hand.

Adding conversion friction changes magnitudes but changes the sign of none of the 150
pool/capital/gas cells relative to gas alone. The visible tier split therefore is not
created by omitting conversion costs. It still cannot be attributed to the fee tier,
because the manuscript action convention makes the same label a roughly +/-5% band at
0.05% but a +/-31--39% band at 0.30%.

## Experiment 2: matched executable action geometry

This experiment changes only the executed band. Observable labels and the 13-feature
state remain `{0,45,50,55}`, but the three entry actions execute raw half-widths
`{480,540,600}` ticks on every tier, corresponding to approximately
0.953/1.049, 0.947/1.055, and 0.942/1.062 lower/upper price ratios. It reruns all five
learned algorithms and six heuristic arms over six pools and 24 windows, then pairs
each matched result with the corresponding completed paper-geometry window.

The collection is complete at 720/720 units, with no audit problems and SHA-256
`5a78de91c0d0feedc0a6d6a8fa88dd31b2722c7c4a2dc8251e6522e71790cd43`.
All 66 matched strategy/pool means remain negative, but the highest-mean strategy
changes in four of six pools.

Two within-pool geometry effects survive Holm correction at block lengths 2, 4, 6,
and 8: ReactiveRecentering improves by $46/window on USDC/WETH 0.05%, while QR-DQN
falls by $1,115/window on USDC/WETH 0.30%. For QR-DQN, the high-tier-minus-low-tier
geometry difference is -$1,100 (primary 95% block CI -$1,739 to -$408; Holm p=0.0081)
and is robust to all four block lengths. Its USDC/WETH 0.30% reward falls from -$335
under paper geometry to -$1,450 under matched geometry, reversing its apparent
high-tier strength.

Every logged first-seed learned policy uses one distinct action per pool under both
geometries. The treatment changes the economics and ranking of a selected constant
width; it does not reveal state-dependent learned control. By contrast, the
RecentreWhenOut heuristic's action diversity rises from 1.21--1.50 to 2.00 in the
0.30% pools, confirming that the paper's very wide high-tier bands suppress its
trigger.

The conclusion is specific rather than universal: fee-tier claims are not invariant
to executable action geometry, but there is no single geometry effect shared by every
strategy. Full results are in `ACTION_GEOMETRY_RESULTS.md`.

## ICAIF-facing contribution

The defensible contribution is no longer “PPO beats common LP strategies.” It is:

- an audit showing how fee, rebalance-value, and selection leakage can reverse an RL
  market-making result;
- a 22.6-million-swap, balanced multi-pool evaluation with dependence-aware inference;
- evidence that no tested RL architecture is a universal remedy after correction;
- a practitioner-grounded fixed-cost frontier for small LPs; and
- an identification test separating fee tier from executable action geometry.

This reframes active LP as an execution-aware loss-mitigation problem. That negative
and heterogeneous result is more credible—and more useful—than another architecture
sweep claiming profitability.

## Boundaries and next work

- The fresh panel does not contain the manuscript's WBTC/USDC pool, so the completed
  successor design cannot reproduce that pool-specific table.
- Historical, time-varying Ethereum gas remains the highest-priority robustness item;
  the frontier uses transparent counterfactual levels rather than claiming a gas path.
- Paper-style early stopping and manuscript PPO coefficients remain declared
  replication sensitivities.
- A genuine Layer-2 comparison, asymmetric ranges, withdrawal, or hedging would need
  new data or a different action/benchmark design and should not be smuggled into this
  experiment.

## Reproducibility

Exact local and Duke cluster commands are in `REPRODUCIBILITY_COMMANDS.md`. The
complete corrected results, PI checklist, retail protocol/results, practitioner
sources, and action-geometry protocol are in the adjacent reports. Every final result
directory carries an independent audit JSON and the aggregation is repeated at moving
block lengths 2, 4, 6, and 8.
