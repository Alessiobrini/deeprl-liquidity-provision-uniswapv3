# State-contingent control headroom

Finalized: 2026-07-28. The prospective design was frozen in
`STATE_HEADROOM_PROTOCOL.md` before generating a test result.

## Completeness and audit

- 144/144 `(pool, rolling step)` units: six pools and 24 held-out windows.
- Same corrected per-swap accounting, value-conserving rebalancing, 13 features,
  reward, $30,000 capital, $5 gas, event-driven schedule, and matched executable
  action geometry as CLEB.
- The independent audit reports `COMPLETE`, no problems, and collection SHA-256
  `adf86f0d0ee3108f5e35519fa2bd01755088e88d0f2da521a0662fa7b54271fc`.
- One inferential observation is one held-out window. Results use a seeded circular
  moving-block bootstrap and the prospectively declared four-contrast H3 Holm family
  within each pool.

## Main result

There is substantial local state-contingent opportunity, but the deployable
reward-maximizing decision stump does not capture it reliably.

| Pool | 168h oracle disagreement (95% block CI) | 24h disagreement | Decision stump minus fixed width (95% block CI) | Holm p |
|---|---:|---:|---:|---:|
| USDC/WETH 0.05% | 71.1% (63.4%, 77.9%) | 69.5% | +$90 (-$785, +$951) | 1.000 |
| USDC/WETH 0.30% | 73.2% (64.2%, 81.7%) | 73.2% | +$291 (-$736, +$1,172) | .804 |
| WBTC/WETH 0.05% | 69.8% (58.3%, 79.1%) | 67.4% | +$154 (-$331, +$685) | .814 |
| WBTC/WETH 0.30% | 67.0% (53.4%, 80.1%) | 68.8% | -$224 (-$649, +$225) | .635 |
| WETH/USDT 0.05% | 68.4% (59.1%, 77.1%) | 69.5% | +$194 (-$465, +$915) | .945 |
| WETH/USDT 0.30% | 72.8% (64.4%, 80.8%) | 72.4% | -$506 (-$1,127, +$206) | .134 |

The disagreement estimand is the mean across windows of the within-window fraction
of full-horizon decision states at which the validation-selected fixed width is not
among the realized reward-maximizing actions. It is not a pooled state-level
replicate count. The estimates are stable under block lengths 2, 4, 6, and 8.

The pre-specified "collapse is rational because there is little headroom" criterion
fails in every pool: its upper confidence limit had to lie below 10% at both 168 and
24 hours, whereas every point estimate exceeds 67%.

## Dollar headroom and decision-stump regret

Mean 168-hour local headroom ranges from $188 to $419 per decision state across the
six pools. Every local-headroom contrast survives the four-member within-pool Holm
correction at the primary block length (`p_Holm=0.0004`), as does local decision-stump
regret. These dollar quantities are averages of local fixed-horizon
counterfactuals, not sums that can be interpreted as a deployable episode return.

The decision stump is fitted to realized 168-hour action rewards on training states,
its minimum leaf size is selected by deployed validation reward, and it is refit
before one test read. Its held-out lift over the validation-selected fixed width
ranges from -$506 to +$291 per 1,500-hour window. No pool's confidence interval
excludes zero, and none survives Holm correction at any tested block length.

This distinguishes the two claims the earlier action-diversity result could not:

1. the action set contains substantial ex-post state-contingent opportunity; but
2. the existing 13 features and a deliberately simple causal policy do not turn
   that opportunity into reliable out-of-sample value.

The result therefore rejects "the agents collapse because one action is almost
always optimal." It does not establish that the oracle choices are predictable.

## Learned-policy comparison

One learned comparator is selected at each rolling step from PPO, A2C, DQN, QR-DQN,
and RecurrentPPO using validation reward only. Against the decision stump, it has
no significant advantage in five pools. On WETH/USDT 0.30%, the
validation-selected learned arm improves by $804 per window (95% block CI $294 to
$1,346, `p_Holm=.0048`), surviving block lengths 2, 4, 6, and 8. Because the
completed learned policies use one observed action, this is evidence for better
validation-based static-action selection on that pool, not evidence of
state-dependent RL control.

## Interpretation and gate decision

The informative outcome is between the two simple stories:

- **Not no-opportunity:** the full-horizon oracle frequently changes the action and
  has positive local headroom at both horizons.
- **Not demonstrated predictability:** the deployable decision stump does not
  significantly beat a fixed width in any pool.

The prospectively gated reward-scale diagnostic was therefore activated. It tested
reward-only `VecNormalize` with the existing PPO entropy grid on the matched
USDC/WETH 0.05% and 0.30% pair. Its complete 48-window collection finds no
increase in action diversity, a significant held-out loss at 0.05%, and no
detectable reward change at 0.30%. Thus raw reward scale is not a sufficient
explanation under this standard remedy. See `NORMALIZED_PPO_PROTOCOL.md` and
`NORMALIZED_PPO_RESULTS.md`.

## Artifacts

- Primary analysis: `outputs/state_headroom_v1/aggregate_block4/`
- Dependence checks: adjacent `aggregate_block2`, `aggregate_block6`, and
  `aggregate_block8` directories
- Independent audit: `outputs/state_headroom_v1/audit.json`
- Paper-oriented figure: `reports/figures/state-contingent-headroom.{pdf,png}`
- Gated follow-up: `outputs/normalized_ppo_v1/aggregate_block4/`
