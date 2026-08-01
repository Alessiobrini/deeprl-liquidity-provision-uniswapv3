# Prospective extension: state-contingent control headroom

Frozen before generating or reading any result from this experiment: 2026-07-28.

## Question

The completed CLEB collection shows that every learned policy uses one distinct
action per pool. That observation alone cannot distinguish two explanations:

1. the corrected environment and action set contain little economically useful
   state-contingent control, so a constant action is rational; or
2. useful state-contingent control exists, but the five learned algorithms fail to
   recover it.

This experiment measures local full-horizon headroom and evaluates one deliberately
simple, deployable state-contingent policy. It is a prospective exploratory
extension. Every complete result will be reported regardless of direction.

## Frozen economic and evaluation design

- Six core pools and the same 24 leak-free walk-forward test windows per pool.
- The same 13 legacy features, corrected per-swap fee accounting, reward,
  $30,000 capital, $5 gas, no exit, and event-driven decision schedule as CLEB.
- Economically matched executable half-widths `{480,540,600}` raw ticks on every
  pool, while preserving observable action labels `{0,45,50,55}`.
- Four train windows, one validation window, and one test window at every rolling
  step. Test is read exactly once.
- The primary counterfactual horizon is 168 hours, the schedule's declared maximum
  inter-decision interval. A 24-hour horizon is a pre-specified robustness check.
  A "little headroom" interpretation is permitted only if it holds at both
  horizons.
- Counterfactual action rewards are evaluated on clones of the same environment
  state. The action is applied once and then held for the remainder of the fixed
  horizon. Only states with the complete horizon remaining are included.
- The local clairvoyant diagnostic is not a deployable policy and is not described
  as a global dynamic-programming upper bound.

## Deployable policies

### Fixed action

For each rolling step, evaluate the four constant event-driven policies on the
validation window and select the highest-reward action. Ties choose the lowest
action index. Evaluate the selected action once on test.

### Reward-maximizing policy stump

The contextual arm is a deterministic one-split policy tree over the same 13
features. It is not trained to imitate the clairvoyant argmax label.

For each training window, visit decision states under each of the four constant
policies. To avoid allowing an hourly out-of-range path to dominate the sample,
retain at most one training state per 24 elapsed hours for each
`(window, reference action)` trajectory. At every retained state, use the simulator
to calculate the realized 168-hour reward of all four actions.

Fit the stump by directly maximizing the weighted mean of these realized action
rewards. Each `(window, reference action)` trajectory receives equal total weight.
The split feature, split threshold, and both leaf actions are optimized exactly.
If no split improves over a constant action, the fitted policy is constant.

The only tuning parameter is minimum leaf size in `{20,10,5}`, ordered from simpler
to more flexible. Fit all three candidates on the four training windows and select
minimum leaf size by the policy's deployed cumulative reward on validation. Refit
the selected specification on train plus validation, then evaluate once on test.
Thus the policy itself is causal and deployable; future test outcomes are used only
by the separately labeled clairvoyant diagnostic.

## Local diagnostics

At every full-horizon decision state visited by the validation-selected fixed
policy on test, evaluate all four actions. For each horizon record:

- **action disagreement:** whether the fixed action is absent from the set of
  reward-maximizing actions; exact numerical ties count as agreement;
- **local headroom:** best local action reward minus fixed-action reward; and
- **contextual regret:** best local action reward minus the reward of the
  contextual tree's action at that same fixed-policy state.

The action-disagreement fraction is the primary headroom summary because it is less
sensitive than the dollar gap to reward scale and window variance. A strong
"collapse is rational" statement requires the upper 95% block-bootstrap confidence
limit to be below 10% at both 168 and 24 hours. Otherwise the paper will report the
estimated headroom without a no-headroom conclusion.

## RL comparator and multiplicity

At each rolling step, select one of PPO, A2C, DQN, QR-DQN, and RecurrentPPO using
only that step's matched-geometry validation scores from the completed collection.
Ties follow that fixed algorithm order. Its already-held-out test reward is the
validation-selected RL comparator.

The new H3 state-contingency family contains exactly four two-sided contrasts
within each pool:

1. 168-hour local clairvoyant headroom: oracle minus fixed;
2. deployed contextual lift: contextual minus fixed;
3. deployed RL increment: validation-selected RL minus contextual; and
4. 168-hour local contextual regret: oracle minus contextual action.

Holm correction is applied across these four contrasts separately within each
pool. One observation is one test-window quantity. No seed is unpacked as an
independent observation.

The action-disagreement fractions and the 24-hour horizon results are
pre-specified descriptive diagnostics rather than additional null-hypothesis
families. They receive seeded circular moving-block bootstrap confidence intervals.

## Uncertainty, robustness, and reporting

- Primary circular moving-block bootstrap: 10,000 resamples, seed 20260728, block
  length four.
- Dependence checks: block lengths two, six, and eight.
- Report pools separately; do not pool heterogeneous markets.
- A collection is final only at 144/144 `(pool, rolling step)` units.
- The main paper table/figure will contain only two quantities per pool:
  168-hour action disagreement (with the 24-hour estimate shown as a robustness
  marker) and deployed contextual-minus-fixed reward.
- Remaining diagnostics are retained in machine-readable CSV files and summarized
  in prose to respect ICAIF's eight-page limit.

## Conditional follow-up

Reward-normalized, entropy-regularized PPO remains gated. It will be run only if
the completed headroom experiment finds non-negligible local opportunity. If both
horizons support little headroom, optimization diagnostics cannot change the
economic conclusion and no additional PPO sweep will be run.
