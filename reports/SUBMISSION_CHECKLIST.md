# ICAIF 2026 submission checklist (as of 2026-07-31)

Paper state: `paper/RL_for_liquidity_provisioning/icaif_main.tex`, commit `f049420`.
Compiles clean at **8 pages**, zero undefined references or citations, no overfull box
above 1.5pt, all pages visually checked.

## Must do before submitting

1. **Anonymize.** ICAIF 2026 is double-blind and the CFP asks for the `anonymous`
   parameter. Change line 16 to:
   `\documentclass[sigconf,review,anonymous]{acmart}`
   This was test-compiled and still fits in 8 pages. It replaces the author block with
   "Anonymous Author(s)" and automatically suppresses the `acks` block, so the DAREC
   funder and the donor name disappear without editing them. Dropping `nonacm` at the
   same time prints the ACM reference block, so fill in `\acmConference` (line 53) if
   you drop it.
2. **Code link.** The footnote on the first page promises "an anonymized repository for
   peer review" but carries no URL. Either add an anonymous4open / blinded Zenodo link
   or drop the sentence. 39% of ICAIF papers ship replication code and reviewers look.
3. **Push the state-contingency code.** The oracle, decision stump, and reward-normalized
   PPO experiments are not in the shared repo (its head is still the PR #9 merge), so the
   67 to 73%, the $188 to $419 headroom, the $804, and the $893 cannot currently be
   regenerated. Ask Arjun to push those scripts.

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
