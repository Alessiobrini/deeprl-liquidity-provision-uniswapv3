# Open items after the third section review: all closed, 2026-08-01

The 116 confirmed findings of `section-review-2026-07-31-third.md` were applied across
three judge-gated commits. Five items could not be settled from the manuscript and were
raised as blue in-text queries. All five are now answered and the queries are deleted.

## The five, and how each closed

1. **The fourth prespecified contrast.** Reported. Oracle minus stump action, mean \$214
   to \$444 per decision state, CI lower limits \$150 to \$335 and upper limits \$267 to
   \$583.
2. **The entropy clause on the gated follow-up.** Entropy *was* selected on validation.
   `STATE_HEADROOM_PROTOCOL.md` registers the follow-up as "reward-normalized,
   entropy-regularized PPO". Both the method and the results now say so, and the results
   report the selection counts (0.01 chosen in 11/24 low-tier and 8/24 high-tier windows).
3. **The follow-up's Holm family.** One member per pool, so the adjusted and raw
   p-values are equal. This is why $p=.0001$ reconciles with the $.0004$ floor of the
   four-member family.
4. **The denominator for the 28 adverse cells.** 28 of 150 per treatment, with Holm
   applied within each pool over one 50-contrast family.
5. **The transfer counts.** PPO has one favorable and three adverse directions surviving
   all four block lengths, while A2C, DQN, QR-DQN, and RecurrentPPO each have zero.

## The correction that came out of it

Answering these surfaced a defect in the diagnostic itself. The action-disagreement
statistic is uncalibrated: with four actions and exchangeable values, the ex-post argmax
differs from any fixed action 75% of the time by construction. Every observed rate (67.0%
to 73.2%) sits below that null and **every bootstrap interval in the run record contains
it**. The dollar headroom carries the same selection bias, which Holm correction does not
address.

The paper therefore retracts the state-contingent-headroom claim and reports the oracle
comparison as descriptive. Full detail in the correction at the top of
`STATE_HEADROOM_RESULTS.md`. Every deployable result stands unchanged: the stump gives no
reliable held-out gain, the learned policies each execute one action in the logged
first-seed trace, and reward normalization does not restore action diversity.

## Two protocol notes worth keeping

- **The headroom gate is the upper-limit form.** `STATE_HEADROOM_PROTOCOL.md`, frozen
  2026-07-28, registers the "little headroom" criterion as the *upper* 95% confidence
  limit below 10% at both horizons. An earlier edit reverted the paper to a lower-limit
  form. That was wrong and has been undone.
- **The H-label collision is resolved.** `SPEC.md` (2026-07-16) keeps H1 to H4, since it
  is the earlier freeze, carries the most references, and is asserted on in
  `tests/test_combined.py`. `STATE_HEADROOM_PROTOCOL.md` (2026-07-28) called its family
  "the new H3", which collided; its families are renumbered **H5** (state contingency)
  and **H6** (the gated reward-normalization follow-up) as a dated amendment recorded in
  both the protocol and the results, labels only with no design or result change. The
  manuscript carries no H-labels at all, since it defines none of them.

## Resolved earlier from the code, for the record

Tick spacings 10 and 60; heuristics consulted hourly while only learned agents carry a
declared schedule (`envs/schedule.py`); ILMinimizer scales volatility by twice the square
root of a 24- or 168-hour period against the volatility-proportional arm's selected
multiplier (`policies/baselines.py`); training discount is the Stable-Baselines3 default
0.99; the hourly grid forward-fills state and zero-fills flows (`data/aggregate.py`).

Three references were added and verified through OpenAlex, with Crossref where a Crossref
DOI exists: `raffin2021sb3`, `kunsch1989jackknife`, `holm1979simple`.
