# Exploratory extension: economically matched action geometry

Frozen before generating any matched-geometry test result: 2026-07-22.

## Motivation

The paper grid `{45,50,55}` is measured in multiples of pool tick spacing. In 0.05%
pools its three lower/upper price ratios run from approximately 0.956/1.046 to
0.946/1.057; in 0.30% pools they run from 0.763/1.310 to 0.719/1.391. The published
interpretation attributed cross-pool behavior to fee tier, although fee tier and
feasible position geometry changed together. Tick-symmetric bands are symmetric in
log price, so their ordinary percentage moves are not literally symmetric.

This exploratory experiment changes only the executed action geometry. The observable
action labels and legacy state remain `{0,45,50,55}`, while the three enter actions
execute raw half-widths `{480,540,600}` ticks, all mintable at both 10- and 60-tick
spacing. Their lower/upper price ratios are approximately 0.953/1.049, 0.947/1.055,
and 0.942/1.062 on every pool. Hold remains implicit. Keeping the labels fixed
prevents the treatment from also changing the legacy state's width feature or the
heuristic parameter grid.

## Frozen design

- Six balanced pools, 24 rolling test windows per pool.
- Five learned algorithms: PPO, A2C, DQN, QR-DQN, and RecurrentPPO.
- All six existing heuristic arms, including the four manuscript competitors.
- Same 13 features, reward, $30,000 capital, $5 gas, event-driven learned-agent
  schedule, 20,000 training steps, seeds 42 and 123, configuration grids, and
  leak-free 4-train/1-validation/1-test walk-forward selection as the corrected
  primary collection.
- Seeds are averaged within each window. Test windows are never treated as seed-level
  replicates.
- Primary contrast: matched raw-tick reward minus paper spacing-unit reward, paired
  by strategy, pool, and test window.
- Uncertainty: seeded circular moving-block bootstrap, 10,000 resamples, seed
  20260722, primary block length 4; block lengths 2, 6, and 8 are dependence checks.
- Multiplicity: Holm correction across the eleven strategy contrasts within each
  pool.
- Secondary diagnostics: change in number of distinct actions and the within-pair
  geometry-by-tier difference-in-differences. The latter is explicitly exploratory.

## Questions

1. Does economically matching band widths materially change performance in 0.30%
   pools relative to the paper convention?
2. Are changes substantially smaller in 0.05% pools, where the two grids are already
   similar?
3. Does matching geometry alter the ranking or action diversity of the five learned
   algorithms?
4. After removing the geometry confound, is there still evidence supporting a general
   fee-tier interpretation?

This extension is not a preregistered confirmatory test of the rejected manuscript.
It is a prospectively specified exploratory experiment motivated by the completed
audit, and its full collection will be reported regardless of direction.
