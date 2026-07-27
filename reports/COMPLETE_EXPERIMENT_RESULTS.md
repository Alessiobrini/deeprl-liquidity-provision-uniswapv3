# Complete corrected experiment results

Finalized: 2026-07-22. These results supersede every numerical result under the
legacy `rl-code/output/` directory and the corresponding claims in the rejected
paper.

## Reproducibility and completeness

The final collection contains exactly:

- 720 primary walk-forward units: 5 algorithms x 6 pools x 24 test windows;
- 720 rolling transfer units: 5 algorithms x 6 source pools x 24 source windows,
  each evaluated on all six target pools; and
- 864 sensitivity units: 6 variants x 6 pools x 24 windows.

Every learned result averages seeds 42 and 123 within its window. Statistical
inference treats a test window—not a seed—as the observation. The primary analysis
uses a seeded circular moving-block bootstrap (10,000 resamples, seed 20260716,
block length 4), paired window-level differences, and Holm correction separately
within each pool/target and hypothesis family. Block lengths 2, 6, and 8 provide a
dependence-sensitivity analysis.

`outputs/final_audit.json` independently verifies exact keys and counts, valid
walk-forward splits, two-seed means, and equality between native transfer cells and
the primary run. It reports `COMPLETE` with no problems and these collection hashes:

| collection | units | SHA-256 |
|---|---:|---|
| primary | 720 | `312e01711fd0f010d854960f683039d2dc541e868dc4befb1639d44edb3e516b` |
| transfer | 720 | `f9173983df822e34f3104c08420f894115de2cef16e96866a304d7eeca7034bc` |
| sensitivity | 864 | `4bb5bcd5237b5894fa81bbe59509790ad42c00f4d00df7b99cdde9c22bd23d28` |

## Primary walk-forward results

No strategy has a positive mean reward in any pool, and no learned algorithm
dominates across pools. Results therefore measure loss mitigation relative to a
passive LP, not profitability or outperformance of holding.

| pool | highest-mean strategy | mean test reward (USD/window) |
|---|---|---:|
| USDC/WETH 0.05% | Passive | -1,252 |
| USDC/WETH 0.30% | RecentreWhenOut | -319 |
| WBTC/WETH 0.05% | QR-DQN | -464 |
| WBTC/WETH 0.30% | PPO | -87 |
| WETH/USDT 0.05% | RecurrentPPO | -698 |
| WETH/USDT 0.30% | RecentreWhenOut | -111 |

At the primary block length, two comparisons survive within-pool Holm correction:

- RecentreWhenOut versus Passive on WETH/USDT 0.30%: +$1,029
  (95% block-bootstrap CI +$297 to +$1,858; Holm p=0.042).
- PPO versus validation-selected ILMinimizer on WETH/USDT 0.05%: +$841
  (95% CI +$264 to +$1,454; Holm p=0.023).

Both are sensitive to the bootstrap block length and must be reported as suggestive,
not robust discoveries. All other prespecified primary comparisons are unresolved
after Holm correction.

## Rolling cross-pool transfer

Transfer uses the same leak-free rolling split as the native experiments. For each
source step, configuration selection occurs on the validation window, refitting uses
the five preceding windows, and evaluation occurs once on the aligned held-out test
window of every target. Unlike the rejected paper, no test interval overlaps the
training interval.

All 180 algorithm/source/target mean rewards are negative. Consequently, the old
"transfer efficiency" ratio is not reported: dividing one negative or near-zero
reward by another has no useful performance interpretation.

At block length 4, PPO has four Holm-significant paired effects, all of which remain
significant at block lengths 2, 4, 6, and 8:

| target <- source | mean difference vs native (USD/window) | 95% CI | Holm p | windows improved |
|---|---:|---:|---:|---:|
| USDC/WETH 0.05% <- WETH/USDT 0.05% | +986 | [+319, +1,676] | 0.0084 | 54% |
| WETH/USDT 0.05% <- USDC/WETH 0.05% | -869 | [-1,385, -299] | 0.0057 | 17% |
| WETH/USDT 0.05% <- WBTC/WETH 0.05% | -879 | [-1,586, -164] | 0.0100 | 17% |
| WETH/USDT 0.05% <- WETH/USDT 0.30% | -673 | [-1,078, -229] | 0.0057 | 25% |

The first result is a directional loss reduction, not profitability: transferred
PPO still has a negative mean reward. A2C, DQN, and RecurrentPPO have no significant
transfer contrast at block length 4. QR-DQN has one negative contrast at block
length 4 (USDC/WETH 0.30% <- USDC/WETH 0.05%, -$539, Holm p=0.0156), but it does not
survive every block-length specification. Thus there is no general evidence of
cross-pool transferability, symmetry, or a stable fee-tier-driven reward ceiling.

## Sensitivity experiments

The sensitivity collection compares the primary PPO specification with hourly,
daily, and weekly forced schedules; shadow and LVR shaping; and the paper-compatible
extractor. At the primary block length, the only Holm-significant effect is:

- hourly scheduling on WETH/USDT 0.05%: -$2,448 versus event-driven PPO
  (95% CI -$3,348 to -$1,420; Holm p=0.0006).

This adverse hourly-scheduling effect remains significant at block lengths 2, 4, 6,
and 8. No other sensitivity contrast is significant at block lengths 2, 4, or 6.
Two extra contrasts appear only at block length 8 and are therefore labeled
block-sensitive rather than robust: paper extractor on WETH/USDT 0.30% (-$152) and
LVR shaping on that pool (+$265).

## Defensible conclusions

1. Correct per-swap fees, value-conserving rebalancing, and leak-free model
   selection overturn the rejected paper's headline performance claims.
2. Active control is not profitable on average under this reward/accounting setup;
   at best, particular policies mitigate losses in particular pools.
3. There is no universal algorithm winner. Pool-specific ranking is substantial,
   while most paired advantages are statistically unresolved after dependence-aware
   inference and multiplicity correction.
4. Transfer is asymmetric and mostly unresolved. The one robust favorable PPO
   direction remains loss-making, so the former 88.5% efficiency claim must be
   removed.
5. Event-driven operation is materially safer than forcing hourly decisions on at
   least WETH/USDT 0.05%; reward shaping and extractor changes show no broadly robust
   improvement.

## Machine-readable results

- Primary: `outputs/algos_v1/combined_block4/{summary,comparisons}.csv`
- Transfer: `outputs/transfer_v1/aggregate_<algorithm>_block4/{summary,comparisons}.csv`
- Sensitivity: `outputs/sensitivity_v1/aggregate_block4/{summary,comparisons}.csv`
- Dependence checks: the corresponding `block2`, `block6`, and `block8` directories
- Independent audit: `outputs/final_audit.json`

No experiment collection described in this report is partial. Earlier malformed
cluster submissions were identified before inference, cancelled, and excluded; the
auditor checks only the exact final design above.

The updated panel does not include the manuscript's WBTC/USDC pool. These are final
results for the PI's balanced six-pool successor design, not an exact rerun of the
paper's missing WBTC/USDC substrate.
