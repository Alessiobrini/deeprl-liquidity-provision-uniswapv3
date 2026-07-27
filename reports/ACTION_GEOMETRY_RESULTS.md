# Economically matched action-geometry results

Finalized: 2026-07-22. Protocol frozen in `ACTION_GEOMETRY_PROTOCOL.md` before
reading any matched-geometry test result.

## Completeness and validation

- 720/720 `(algorithm, pool, window)` units: five algorithms, six pools, and
  24 held-out windows.
- Exact walk-forward split, 20,000-step budget, seeds 42 and 123, 13-feature state,
  reward, $30,000 capital, $5 gas, configuration grids, and action labels from the
  corrected primary collection.
- The only treatment is execution of raw half-widths `{480,540,600}` ticks instead
  of multiplying labels `{45,50,55}` by pool tick spacing.
- The independent audit reports `COMPLETE`, no problems, and collection SHA-256
  `5a78de91c0d0feedc0a6d6a8fa88dd31b2722c7c4a2dc8251e6522e71790cd43`.
- The audit passes on consecutive runs and rejects unexpected result JSON.

## Main result

Economically matching widths does not rescue profitability: all 66 strategy/pool
matched means remain negative. It does materially change rankings. The highest-mean
strategy changes in four of six pools:

| pool | paper-geometry best | matched-geometry best |
|---|---:|---:|
| USDC/WETH 0.05% | Passive (-$1,252) | PPO (-$1,096) |
| USDC/WETH 0.30% | RecentreWhenOut (-$319) | RecentreWhenOut (-$459) |
| WBTC/WETH 0.05% | QR-DQN (-$464) | A2C (-$209) |
| WBTC/WETH 0.30% | PPO (-$87) | DQN (-$186) |
| WETH/USDT 0.05% | RecurrentPPO (-$698) | DQN (-$765) |
| WETH/USDT 0.30% | RecentreWhenOut (-$111) | RecentreWhenOut (-$220) |

At the primary moving-block length 4, two of 66 within-pool geometry contrasts survive
Holm correction across the eleven strategies in that pool:

| pool / strategy | matched minus paper | 95% block CI | Holm p |
|---|---:|---:|---:|
| USDC/WETH 0.05% / ReactiveRecentering | +$46 | +$24 to +$66 | 0.0011 |
| USDC/WETH 0.30% / QR-DQN | -$1,115 | -$1,845 to -$481 | 0.0121 |

Both remain significant at block lengths 2, 4, 6, and 8. Other apparent effects are
block-sensitive: PPO on USDC/WETH 0.05% (+$751) appears only at blocks 6 and 8, while
RecurrentPPO on WETH/USDT 0.05% (-$1,215) appears only at blocks 6 and 8.

## Fee tier versus geometry

Descriptively, matched geometry helps 18 of 33 strategy/pool cells in the 0.05% tier
but only 8 of 33 in the 0.30% tier. The mean geometry effect across those cells is
-$23 per window at 0.05% and -$181 at 0.30%. These counts are not independent tests:
the eleven strategies share the same market paths.

The paired within-pair difference-in-differences gives one robust specific result.
For QR-DQN on USDC/WETH, the high-tier geometry effect minus the low-tier effect is
-$1,100 (primary 95% CI -$1,739 to -$408; Holm p=0.0081 across the three pairs). It
survives all four block lengths. QR-DQN changes only -$15 on USDC/WETH 0.05% but falls
-$1,115 on USDC/WETH 0.30%, moving from the best learned high-tier mean under paper
geometry (-$335) to the worst learned matched mean (-$1,450).

No other strategy has a tier difference-in-differences that survives every block
length. The evidence therefore rejects a geometry-invariant reading of the QR-DQN
high-tier result, but it does not identify one universal causal fee-tier effect across
all policies.

## Policy behavior diagnostic

The logged first-seed deterministic policy for every learned algorithm uses one
distinct action in every pool under both geometries. Matching widths changes reward
and rank, but does not make the learned policies state-dependent by this diagnostic.
This suggests that much of the learned comparison is selecting a constant width, not
learning active conditional control.

The transparent out-of-range heuristic does react to the treatment. In the three
0.30% pools, RecentreWhenOut's mean number of distinct observed actions rises from
1.21--1.50 under the very wide paper bands to 2.00 under matched bands. This is direct
evidence that the paper convention suppresses the recenter trigger at the high tier.

## Interpretation

The rejected paper's claim that the target fee regime sets the reward ceiling cannot
be recovered from this successor design. The old action convention coupled fee tier
to economically different bands, and removing that coupling changes four pool winners
and reverses QR-DQN's apparent high-tier strength. Yet matching geometry does not make
any strategy profitable or produce a universal algorithm winner.

For an ICAIF revision, the useful claim is methodological: cross-tier RL comparisons
must match executable position geometry before attributing performance to fee tier.
Nominally identical discrete actions are not comparable treatments when pool tick
spacing changes their percentage width by roughly an order of magnitude.

## Artifacts

- Primary results: `outputs/action_geometry_v1/aggregate_block4/`
- Dependence checks: adjacent `aggregate_block2`, `aggregate_block6`, and
  `aggregate_block8` directories
- Independent audit: `outputs/action_geometry_v1/audit.json`
- Static figure: `reports/figures/action-geometry-effects.{pdf,png}`
