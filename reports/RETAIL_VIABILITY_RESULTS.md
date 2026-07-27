# Retail recentering viability frontier

Finalized: 2026-07-22. Protocol frozen in `RETAIL_VIABILITY_PROTOCOL.md` before
reading these results.

## Completeness and validation

- 720/720 `(pool, window, capital)` units.
- Exact replay across five capital levels; gas and conversion costs are deterministic
  accounting transformations of cost-free paths.
- The independent audit reports `COMPLETE`, no problems, and collection SHA-256
  `a529455c1b5223807de4abd2aeeb5b00818eaa864f0c452895441291e410b9fc`.
- At the paper's `$30,000`, `$5`, zero-conversion-cost setting, every one of the 144
  active and passive window rewards reproduces the completed primary collection to
  numerical tolerance.

## Main result

With pool-fee-plus-5-bp conversion friction, recentering only when out of range is:

- worse than leaving the original position alone in all 25 capital/gas cells of each
  0.05% pool; and
- descriptively better in 22/25 cells for USDC/WETH 0.30%, 22/25 for WBTC/WETH
  0.30%, and 23/25 for WETH/USDT 0.30%.

The favorable 0.30% region excludes mainly `$1,500` positions at `$100`--`$250` gas
and `$5,000` positions at `$250` gas. Examples of mean reward differences per
1,500-hour test window under conversion friction are:

| pool | capital | gas | recenter minus passive |
|---|---:|---:|---:|
| USDC/WETH 0.05% | $30,000 | $25 | -$871 |
| WBTC/WETH 0.05% | $30,000 | $25 | -$611 |
| WETH/USDT 0.05% | $30,000 | $25 | -$1,300 |
| USDC/WETH 0.30% | $30,000 | $25 | +$798 |
| WBTC/WETH 0.30% | $30,000 | $25 | +$165 |
| WETH/USDT 0.30% | $30,000 | $25 | +$983 |

Adding conversion friction changes the magnitude but not the descriptive frontier:
none of the 150 pool/capital/gas cells changes sign relative to the gas-only
counterfactual. Averaged over the 25 cells within a pool, conversion reduces the
active-minus-passive difference by $10--$224 per window (the exact amount varies with
the validation-selected width). Thus the paper's omitted conversion cost matters to
PnL, but it is not what creates the visible tier split in this grid.

This descriptive tier split is not itself confirmatory evidence. Holm correction was
applied over all 50 frontier cells within a pool. At block length 4, no favorable
0.30% cell survives that correction. Twenty-eight adverse cells survive: 26 are in
0.05% pools and two are small-capital/high-cost cells in 0.30% pools. The total number
of Holm-significant adverse conversion cells is 28 at block lengths 2, 4, and 6, and
36 at block length 8; no favorable cell is significant at blocks 4, 6, or 8.

## Interpretation

The result should not be summarized as "high fees make active management work." Under
the paper convention, the action grid's lower/upper price ratios run from roughly
0.956/1.046 to 0.946/1.057 at 0.05%, but from 0.763/1.310 to 0.719/1.391 at 0.30%.
The latter leave the range much less frequently, so the active rule pays fewer
repositioning costs. Capital magnifies both the economic gain or loss and conversion
friction; gas mainly determines whether small positions cross zero.

This frontier therefore does two useful things:

1. it directly answers the practitioner question of when a simple out-of-range rule
   fails after operational costs; and
2. it supplies independent motivation for the matched-action-geometry experiment,
   which tests whether the apparent tier split remains after both tiers execute the
   same percentage-width bands.

## Artifacts

- Machine-readable results: `outputs/retail_frontier_v1/aggregate_block4/frontier.csv`
- Regenerated headline counts:
  `outputs/retail_frontier_v1/significance_summary.csv`, produced by
  `python -m src.deeprl_liquidity_provision_uniswapv3.experiments.retail_frontier_summary`
- Dependence checks: adjacent `aggregate_block2`, `block6`, and `block8` directories
- Audit: `outputs/retail_frontier_v1/audit.json`
- Figure: `reports/figures/retail-viability-frontier.{pdf,png}`
