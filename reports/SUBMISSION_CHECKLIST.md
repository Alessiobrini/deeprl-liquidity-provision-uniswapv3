# ICAIF 2026 submission checklist (as of 2026-08-01)

Paper state: `paper/RL_for_liquidity_provisioning/icaif_main.tex`, commit `8c095f5`.
Compiles clean at **8 pages**, zero undefined references or citations, **zero overfull
boxes**, all pages visually checked. `\Comments=0` and every draft comment macro is
deleted, so no query text remains in the source.

The state-contingency claim was retracted on 2026-08-01. The disagreement statistic is
uncalibrated against a no-information null, so the paper reports the comparison as
descriptive. The quantity is named the hindsight best action throughout, no longer the
oracle, since that name implied a correctness the retraction denies. See the correction at the top of
`STATE_HEADROOM_RESULTS.md`. Every deployable result is unchanged.

## Must do before submitting

1. ~~**Anonymize.**~~ Done. Line 16 carries `anonymous`, the title block prints
   "Anonymous Author(s)", and the `acks` block is suppressed. `nonacm` is still set, so
   drop it and fill in `\acmConference` (line 56) only at camera-ready.
2. ~~**Code link.**~~ Decided: no URL while the submission is blind. The note promises
   release and explains the omission, and it now hangs off the first sentence of the
   Results rather than a page-1 contribution bullet. Add the real link at camera-ready.
3. ~~**Push the state-contingency code.**~~ Done. PR #10 merged 2026-08-01, adding
   `experiments/state_headroom.py`, `experiments/normalized_ppo.py`, their validators,
   three test files, and the frozen `STATE_HEADROOM_PROTOCOL.md`. All 179 tests pass.
   Every number in Subsec. 4.3 is now regenerable.

## Known, accepted, no action needed

- Page 8 holds only four reference entries. Normal, and it keeps the paper at the limit.
- Two all-float pages (5 and 6) carry the two large figures and the benchmark table.
- Prose reports $841 for the PPO-versus-ILMinimizer contrast where the rounded table
  cells imply $840. This is rounding of the unrounded means. Numbers are locked to the
  runs, so it was left alone. Confirm against the run output if you want it exact.
- The bibliography cleared the pre-submission gate. See `bib-verification-2026-07-31.md`:
  every entry is real, correctly attributed, and chronologically sound. The checker's
  14 MISS verdicts are an artifact of its author comparison, adjudicated by hand.

## Worth a look if you have time

- Four floats land one to two pages after the text discussing them. Fixable by moving
  the `table*`/`figure*` blocks earlier in the source, but each move perturbs the whole
  float queue, so it was left stable rather than risked the night before the deadline.
- Figure 1's axis labels abbreviate the arm names; the caption now glosses all of them,
  but regenerating the figure with the canonical names would be cleaner.
