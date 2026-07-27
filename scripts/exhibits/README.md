# Exhibit scripts

Each produced a result cited in SPEC.md. Per the figures-need-a-committed-source
rule, they live here rather than in /tmp, so any number in the paper can be
regenerated end to end.

| script | what it establishes |
|---|---|
| `exhibit_leak_decomposition.py` | **PARTLY RETRACTED.** The leak (tuning on test) bought +675, and that part stands. The -2,140 "action-grid handicap" and the -1,456 -> +1,358 sign flip were OUR OWN 10x unit error: the previous env multiplies a width by tick spacing, so its `{45,50,55}` is +/-4.6% to +/-5.7%, not sub-1%. Re-run with widths in the right units before citing anything here. |
| `exhibit_fee_scaling_knife_edge.py` | Doubling fee income flips the optimum from a degenerate +/-170% band to a realistic +/-5% one. Everything rests on fee income being right within 2x. |
| `exhibit_fee_discretization.py` | Refutes the hypothesis that hourly fee accounting was what could move it. Per-swap attribution changes fee income by -7% to +11% across the six pools, with no consistent sign, and moves the optimal width on none of them. The per-swap model is still the correct accounting and is what the environment now runs. |
| `exhibit_legacy_vs_compact_features.py` | **WITHDRAWN.** `features="legacy"` did not replicate the paper's state when this ran (means of returns where the paper used means of price, wrong sigma series, tick span where the paper carried the raw action). The +381 measured neither state. Re-run. |
| `exhibit_ppo_convergence.py` | PPO plateaus by 20k agent steps; the budget is not a strawman. |
| `exhibit_schedule_binds.py` | The decision schedule binds on hourly-acting policies and is a no-op on act-when-out ones. |
| `exhibit_timing_unpredictable.py` | Pre-window features do not predict a window's realized reward: sign accuracy at or below chance on 5 of 6 pools. |

Run from the repo root with the project interpreter:

    python scripts/exhibits/<name>.py
