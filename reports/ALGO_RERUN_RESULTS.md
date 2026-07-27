# Corrected walk-forward algorithm rerun

Date read: 2026-07-19. This memo records the first complete read after the fee,
rebalancing, leakage, competitor, and agent-selection fixes documented in
`HANDOVER.md`.

Status note (2026-07-22): this is a chronological checkpoint. The rolling transfer
collection mentioned as unfinished below has since completed and is reported in
`COMPLETE_EXPERIMENT_RESULTS.md`; no table in this memo is replaced by partial data.

## Collection and protocol

- 720 JSON work units: 5 algorithms x 6 pools x 24 test windows.
- Algorithms: PPO, A2C, DQN, QR-DQN, RecurrentPPO.
- Two seeds per learned arm (42, 123), averaged inside each window before inference.
- 20,000 training steps; event-driven schedule; no reward shaping.
- Each test window is read once after configuration selection on the preceding
  validation window and refitting on the five preceding windows.
- One inferential observation is one seed-averaged test window, never one seed.
- Primary uncertainty: circular moving-block bootstrap, block length 4, 10,000
  resamples, seed 20260716. Holm correction is applied separately within each pool
  and hypothesis family.
- No pooled significance test is reported because pools are heterogeneous and
  adjacent windows are serially dependent.

The machine-readable artifacts are `outputs/algos_v1/combined_block4/summary.csv`
and `comparisons.csv`; the complete human-readable table is `report.txt` in the same
directory.

## Main findings

No learned algorithm dominates across pools. The highest mean strategy is:

| pool | highest mean strategy | mean test reward (USD) |
|---|---|---:|
| USDC/WETH 0.05% | Passive | -1,252 |
| USDC/WETH 0.30% | RecentreWhenOut | -319 |
| WBTC/WETH 0.05% | QR-DQN | -464 |
| WBTC/WETH 0.30% | PPO | -87 |
| WETH/USDT 0.05% | RecurrentPPO | -698 |
| WETH/USDT 0.30% | RecentreWhenOut | -111 |

Every strategy has negative mean absolute reward on every pool. Results therefore
concern mitigation of LP loss relative to Passive, not positive profitability or
outperformance of holding.

Under the primary block-length-4 analysis, two within-pool comparisons survive Holm:

- H1, WETH/USDT 0.30%: RecentreWhenOut improves on Passive by $1,029 per window
  (block 95% CI +$297 to +$1,858; Holm p=0.042).
- H2, WETH/USDT 0.05%: PPO improves on the validation-selected ILMinimizer by $841
  per window (block 95% CI +$264 to +$1,454; Holm p=0.023).

All other H1 and H2 comparisons are unresolved after within-pool Holm correction.
In particular, a high rank in a pool is not evidence of a reliable advantage when
the paired interval crosses zero.

## Dependence sensitivity

Block length was not fixed in the original pre-registration, so block lengths 2, 4,
6, and 8 were checked without changing any reward or retraining any agent.

| comparison | block 2 | block 4 | block 6 | block 8 |
|---|---:|---:|---:|---:|
| RecentreWhenOut vs Passive, WETH/USDT 0.30% | significant | significant | not significant | not significant |
| PPO vs ILMinimizer, WETH/USDT 0.05% | not significant | significant | significant | significant |
| RecurrentPPO vs ILMinimizer, WETH/USDT 0.05% | not significant | not significant | significant | significant |

The two primary significant results are therefore block-length-sensitive. They
should be described as suggestive evidence and accompanied by this sensitivity, not
presented as robust discoveries.

## Consequences for the paper

The rejected paper's claims that PPO broadly outperforms competitors and that the
target fee regime determines the reward ceiling are not supported by this corrected
rerun. A defensible revised claim is narrower: learned and heuristic policies have
pool-specific performance, with no universal winner, and conclusions depend strongly
on correct accounting, leak-free selection, and dependence-aware inference.

At the time of this checkpoint, cross-pool transfer on the balanced 3x2 design was
unfinished and could not reuse the rejected paper's overlapping same-period protocol.
That leak-free collection is now complete; see `COMPLETE_EXPERIMENT_RESULTS.md`.
