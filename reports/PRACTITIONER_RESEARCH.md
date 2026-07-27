# Practitioner research and project fit

Research pass: 2026-07-22. Forum statements below are treated as qualitative
motivation, not representative survey evidence.

## Recurrent LP problems

### 1. Repositioning is a fixed-cost problem that disadvantages small LPs

The strongest recurring complaint is not merely impermanent loss; it is paying gas
and conversion costs repeatedly to keep a concentrated position useful. One Uniswap
governance contributor reported that more than 200 small 1--2 ETH positions were net
positive before gas but net negative after it, and documented roughly 532,577 gas for
a vault rebalance. The same community analysis estimated that 53% of positions were
profitable before gas but only 39% after gas, and that 30% of positions held at most
$2,500. Another governance discussion explicitly models gas and operational overhead
as fixed costs independent of LP capital. Reddit questions give the same retail
framing: whether a $1,500 or $5,000 position can ever amortize several mainnet
transactions.

Sources:

- [Uniswap governance: reducing position-manager gas](https://gov.uniswap.org/t/temperature-check-upgrade-the-nonfungiblepositionmanager-smart-contract-to-reduce-gas-consumption/16517)
- [Uniswap governance: LP objective and operational overhead](https://gov.uniswap.org/t/uniswap-objective-function/21005)
- [Reddit: reducing rebalancing costs](https://www.reddit.com/r/defi/comments/1s84fa6/how_do_you_guys_reduce_rebalancing_costs_when/)
- [Reddit: a $5,000 LP asks whether gas consumes the fees](https://www.reddit.com/r/defi/comments/12wh5hc/question_about_uniswap_gas_fee_on_eth_blockchain/)

Project response: the completed retail viability frontier varies capital, gas, and
the omitted token-conversion friction while preserving the corrected per-swap panel.

### 2. Out-of-range capital earns no fees, but chasing price can be worse

Uniswap's own documentation states that liquidity stops earning fees outside its
interval and becomes entirely one asset. Forum LPs repeatedly ask whether to remove,
swap, and remint after leaving the range; the obstacle is that doing so locks in the
new inventory and pays gas/slippage. Practical advice clusters around wider ranges,
rebalancing only after a threshold, or accepting periods out of range.

Sources:

- [Uniswap developer documentation: concentrated liquidity](https://developers.uniswap.org/docs/get-started/concepts/liquidity-providers/concentrated-liquidity)
- [Uniswap launch explanation: active liquidity](https://blog.uniswap.org/uniswap-v3)
- [Reddit: range choice, slippage, gas, and operational overhead](https://www.reddit.com/r/defi/comments/u32812/choosing_a_uniswap_v3_price_range_the_fundamentals/)
- [Reddit: why not constantly adjust ranges?](https://www.reddit.com/r/UniSwap/comments/rtdw0b/what_i_dont_get_about_uniswap_v3_polygon/)

Project response: `RecentreWhenOut`, `ReactiveRecentering`, and the learned
event-driven schedule already model three versions of this decision. The new
action-geometry experiment tests whether the paper made the threshold problem
artificially different across fee tiers.

### 3. Retail and sophisticated LPs operate differently

The BIS study of on-chain Uniswap v3 positions finds that retail providers manage
positions less actively, capture a smaller share of fees, and earn lower returns on
capital than sophisticated providers, even though retail positions suffer less adverse
selection. Earlier empirical work similarly argues that v3's additional decisions and
active-management demands disadvantage retail participants. Research on Layer 2 finds
that cheaper position updates are associated with more concentrated, capital-efficient
liquidity.

Sources:

- [BIS: Decentralised dealers?](https://www.bis.org/publ/work1227.htm)
- [Heimbach, Schertenleib, and Wattenhofer](https://arxiv.org/abs/2205.08904)
- [Layer 2 be or Layer not 2 be](https://arxiv.org/abs/2403.09494)

Project response: the paper's fixed $30,000/$5 setup cannot speak for retail. The
frontier explicitly reports where a simple active rule fails by position size and
cost. A future cross-chain study would require actual L2 swap and gas data and is
outside the current Ethereum-only panel.

### 4. Reported fee yield is not true performance versus holding

Forum users describe collecting visible fees while nevertheless losing to a simple
hold portfolio. This is consistent with empirical and theoretical research showing
that fee income, concentration risk, inventory loss, and repositioning costs must be
evaluated together. It also explains why interface APR alone is not the right target.

Sources:

- [Cartea, Drissi, and Monga](https://arxiv.org/abs/2309.08431)
- [Reddit: fees earned while losing to holding](https://www.reddit.com/r/defi/comments/1qbk5y8/i_was_making_fees_as_a_uniswap_lp_and_still/)

Project response: this is already central to the corrected reward. Every current
mean is negative against the hold benchmark, so the paper now describes mitigation
rather than profitability.

### 5. Automation solves attention, not necessarily economics

Forums frequently recommend vaults and alerting tools, but users also warn that
automatic swap-based repositioning can multiply fees. Uniswap governance catalogues
automated managers chiefly around rebalancing, fee collection, and gas savings. The
research opportunity is therefore not "automation versus manual" in the abstract;
it is whether a transparent trigger covers its complete operational cost.

Sources:

- [Uniswap governance overview of LP managers](https://gov.uniswap.org/t/a-deep-dive-of-uniswaps-governance/19792)
- [Reddit: manual versus managed concentrated liquidity](https://www.reddit.com/r/defi/comments/1ru83rn/serious_question_do_you_manage_your_lp_positions/)

Project response: event-driven, daily, weekly, and hourly schedules are already tested.
The robust adverse result for forced hourly PPO on WETH/USDT 0.05% is consistent with
the practitioner concern about over-management.

## Natural experiments ranked by fit

| idea | real-world relevance | implementation fit | decision |
|---|---|---|---|
| Capital/gas/conversion-cost viability frontier | High | Existing costs and deterministic rules | Completed in this extension |
| Match executable band geometry across fee tiers | High; range width is the core choice | Existing action mapper | Completed for all five algorithms and six heuristics |
| Historical hourly Ethereum gas | High | Needs a trustworthy external gas series and transaction gas model | Next robustness item |
| Asymmetric/skewed ranges | High for inventory-aware LPs | Requires a new action space and retuning every baseline | Valuable but not low-complexity |
| Optional exit/withdrawal | High in volatility regimes | Changes the quoting mandate and benchmark | Separate paper question |
| Hedged/delta-neutral LP | Relevant to professional desks | Requires funding, hedge execution, and a different benchmark | Out of current scope |
| L2 comparison | Direct retail relevance | Requires new pools, chain clocks, gas, and swap panels | Future data collection |
| Fee compounding/manager fees | Operationally relevant | Requires custody and harvesting assumptions | Secondary sensitivity |

## Research direction

The strongest ICAIF narrative is not that a more complex RL architecture finally
wins. It is that correct accounting and realistic execution constraints reveal when
active liquidity management is economically meaningful, when it degenerates to a
static rule, and why apparent fee-tier effects can actually be action-geometry or
cost-scale effects. That question is both practitioner-grounded and identifiable in
the balanced 3x2 panel.
