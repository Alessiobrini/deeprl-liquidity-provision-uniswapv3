# Practitioner-grounded extension: retail recentering viability

Frozen before generating any frontier result: 2026-07-22.

## Practitioner motivation

Across Uniswap and DeFi forums, LPs repeatedly describe the same operational problem:
a narrow position exits its range and stops earning fees, but repositioning it incurs
gas, a token-conversion trade, slippage, and monitoring effort. Small LPs report that
mainnet costs can consume the extra fees, while wider or less frequently managed
positions are more viable. A Uniswap governance contributor described deploying more
than 200 small positions and being profitable before gas but unprofitable after it;
the same discussion reports roughly 0.5 million gas for a vault rebalance and a
community analysis in which the fraction of profitable positions falls from 53% to
39% after gas. The personal account is anecdotal and the community estimates are not
a causal population study, but they align with empirical work finding that retail LPs
update less actively than sophisticated providers.

Sources motivating the design:

- https://gov.uniswap.org/t/temperature-check-upgrade-the-nonfungiblepositionmanager-smart-contract-to-reduce-gas-consumption/16517
- https://www.reddit.com/r/defi/comments/1s84fa6/how_do_you_guys_reduce_rebalancing_costs_when/
- https://www.reddit.com/r/defi/comments/12wh5hc/question_about_uniswap_gas_fee_on_eth_blockchain/
- https://www.bis.org/publ/work1227.htm
- https://arxiv.org/abs/2309.08431
- https://developers.uniswap.org/docs/get-started/concepts/liquidity-providers/concentrated-liquidity

## Frozen design

- Same six-pool panel, per-swap fee accounting, 24 rolling test windows, reward, and
  paper action grid as the corrected primary experiment.
- Position capital: `$1,500`, `$5,000`, `$10,000`, `$30,000`, and `$100,000`.
- Flat gas per recenter: `$1`, `$5`, `$25`, `$100`, and `$250`. These are transparent
  counterfactual levels, not claims about the gas price in a particular hour.
- Two conversion-cost regimes:
  - `gas_only`: the paper convention, no swap or slippage charge;
  - `conversion`: half the position is converted at the pool fee tier plus 5 basis
    points of slippage whenever it is recentered.
- Active rule: `RecentreWhenOut` with width in `{45,50,55}` selected on the immediately
  preceding validation window separately for every capital/cost condition.
- Reference: `Passive`, which leaves the initial width-45 position in place. The
  initial position is already deployed; the experiment measures subsequent management
  costs, not wallet onboarding or final withdrawal.
- A work unit is one `(pool, rolling step, capital)` and caches cost-free reward,
  recenter count, and recenter notional. Every gas/conversion counterfactual is then an
  exact accounting transformation of the same path, not a new stochastic evaluation.
- Primary output is a descriptive capital-by-gas viability map per pool. Paired
  moving-block confidence intervals and Holm-adjusted p-values are provided, but the
  large frontier is exploratory rather than a family of confirmatory claims.

## Question

For a realistic out-of-range trigger, how large must a position be—and how cheap must
repositioning be—for active recentering to mitigate loss relative to leaving the
position alone? Does omitting the token-conversion cost materially move that frontier?
