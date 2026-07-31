# Prose tightening pass, `paper/RL_for_liquidity_provisioning/icaif_main.tex`, 2026-07-31

We ran a section-by-section tightening pass in reading order: two concision editors per section (one for redundancy and length, one for phrasing and sharpness) proposed before/after trims, an adversarial verifier kept only the trims that strictly preserve meaning, voice, hedges, numbers, and cross-references, and each section handed its canonical terms, protected spans, and open threads forward to the next. **This is REPORT-ONLY. Nothing has been applied to the manuscript.**

## Summary

We confirmed **45 trims** across seven sections, saving **roughly 232 words**. By risk: **41 low, 4 medium, 0 high**. The editors proposed 75 candidate trims in total (many overlapping on the same span); 9 were rejected outright and the rest were merged, repaired, or subsumed.

The sections with the most to gain are **Results (81 words, 12 trims)**, **Introduction (42 words, 10 trims)**, and **Environment and Evaluation Design (29 words, 9 trims)**. Results alone accounts for about a third of the harvest, and three of its cuts remove a whole sentence each.

Overall verdict: the paper is already reasonably tight at the sentence level, and there is no fat in the numbers, hedges, or evidence. The real slack is structural repetition. The four-axis enumeration (fee accounting, leak-free model selection, executable action geometry, operating costs) was spelled out six times in the body, and the paper carried several standalone pointer sentences ("Table~X reports ..."), announcement sentences that describe the structure of what follows instead of delivering it, and one closer whose purpose clause was duplicated almost verbatim two sections apart. Removing those accounts for most of the 232 words. At acmart sigconf widths the harvest is on the order of 18 to 22 lines, so this is a meaningful but not transformative contribution to the page budget.

## Trims by section

### Abstract

~16 words saved. Proposed 7, kept 4, rejected 1 (the remainder merged into kept trims).

- [ ] **[low] restatement** (L71) - saves ~7 words. REVISED by the verifier. Why: the four axis names are already enumerated verbatim in sentence two, so the closer re-lists them for no gain; the contribution claim survives as a back-reference.
    - before: `and supplies the fee accounting, leak-free model selection, executable action geometry, and operating-cost treatment needed to evaluate it.`
    - after:  `and supplies the control over all four axes needed to evaluate it.`
- [ ] **[low] filler** (L71) - saves ~4 words. Why: removes the anthropomorphic reporting frame "the benchmark reveals"; the conditional scope is still carried by the leading "Under ..." clause.
    - before: `Under value-conserving rebalancing and per-swap fee attribution, the benchmark reveals that every strategy has negative mean reward relative to holding`
    - after:  `Under value-conserving rebalancing and per-swap fee attribution, every strategy has negative mean reward relative to holding`
- [ ] **[low] long-sentence** (L71) - saves ~4 words. REVISED by the verifier (repaired a dangling object in the editor's version). Why: splits a 60-word sentence in two and names CLEB, defined one sentence earlier; every number survives verbatim.
    - before: `We build the benchmark from 22.6 million Ethereum mainnet swaps across six Uniswap v3 pools arranged as three token pairs at two fee tiers, and we evaluate five RL algorithms and six heuristic policies on 24 held-out walk-forward windows per pool, averaging performance across random seeds within each window and reporting paired circular moving-block bootstrap intervals with within-pool Holm correction.`
    - after:  `We build CLEB from 22.6 million Ethereum mainnet swaps across six Uniswap v3 pools spanning three token pairs at two fee tiers. We evaluate five RL algorithms and six heuristic policies on 24 held-out walk-forward windows per pool, averaging performance across seeds within each window and reporting paired circular moving-block bootstrap intervals with within-pool Holm correction.`
- [ ] **[low] roundabout** (L71) - saves ~1 word. Why: "test" carries the same purpose as "determine" and is the sanctioned verb; the participial clause makes the same assertion.
    - before: `To determine whether this collapse is economically rational, we run a diagnostic that evaluates every action at the same decision state.`
    - after:  `To test whether this collapse is economically rational, we run a diagnostic evaluating every action at the same decision state.`

### Introduction

~42 words saved plus one freed line. Proposed 15, kept 10, rejected 1.

- [ ] **[med] restatement** (L115) - saves ~15 words. Why: fifth full listing of the four axes in the body; the standing claim survives in full and "four controls" matches L122's own wording. Risk is med because the back-reference now depends on the L102 enumeration across an intervening contribution list.
    - before: `Together these results establish CLEB as a reusable standard for evaluating concentrated-liquidity RL, one that holds learned policies to per-swap fee accounting, leak-free model selection, executable action geometry, and explicit operating costs, so that a reported gain reflects skill rather than an evaluation artifact.`
    - after:  `Together these results establish CLEB as a reusable standard for evaluating concentrated-liquidity RL under all four controls, so that a reported gain reflects skill rather than an evaluation artifact.`
- [ ] **[low] restatement** (L104) - saves ~9 words. Why: the immediately preceding paragraph names all four in order and calls them "four axes", so the antecedent is adjacent and canonical.
    - before: `Once fee accounting, leak-free model selection, executable action geometry, and operating costs are controlled, do learned policies provide state-dependent loss mitigation for concentrated-liquidity providers?`
    - after:  `Once these four axes are controlled, do learned policies provide state-dependent loss mitigation for concentrated-liquidity providers?`
- [ ] **[med] restatement** (L110) - saves ~8 words. REVISED by the verifier (merged both editors). Why: the preceding sentence already describes the validation-only walk-forward protocol, its seed averaging, bootstrap, and multiplicity correction. Risk is med because attribution moves from the two named components to the protocol as a whole.
    - before: `Leak-free model selection and dependence-aware inference identify which apparent advantages survive and which are artifacts of the evaluation.`
    - after:  `This protocol separates apparent advantages that survive from evaluation artifacts.`
- [ ] **[low] nominalization** (L102) - saves ~3 words. Why: the nominalized subject becomes the gerund naming the same act; the count of four axes and the canonical term are untouched.
    - before: `The measured answer to whether a learned policy helps is nevertheless fragile along four axes.`
    - after:  `Measuring whether a learned policy helps is nevertheless fragile along four axes.`
- [ ] **[low] roundabout** (L104) - saves ~3 words. Why: both editors proposed this identically; the CLEB definition site stays intact and the subject sits next to its verb.
    - before: `The Concentrated Liquidity Execution Benchmark (CLEB) that we introduce controls all four at once.`
    - after:  `Our Concentrated Liquidity Execution Benchmark (CLEB) controls all four at once.`
- [ ] **[low] filler** (L106) - saves ~0 words but frees a full line. Why: folds a one-line lead-in into the end of the preceding paragraph, removing the line and its paragraph spacing.
    - before: `the contribution.\n\nOur contributions are:`
    - after:  `the contribution. Our contributions are:`
- [ ] **[low] roundabout** (L111) - saves ~1 word. Why: both editors proposed this identically; matches the abstract's own phrasing "a retail frontier over capital and operating costs".
    - before: `We map a retail frontier that varies capital, gas, and conversion cost`
    - after:  `We map a retail frontier over capital, gas, and conversion cost`
- [ ] **[low] filler** (L109) - saves ~1 word. Why: the realized-versus-proxy contrast is carried by the rest of the same sentence, so the intensifier adds nothing.
    - before: `measures the fee income a position would actually have earned`
    - after:  `measures the fee income a position would have earned`
- [ ] **[low] weak-verb** (L112) - saves ~1 word. Why: collapses "shows that X is" into one verb of equal evidential force; the "at least one apparent" hedge survives verbatim.
    - before: `and shows that at least one apparent high-tier RL advantage is an artifact of tier geometry rather than of learned skill`
    - after:  `and identifies at least one apparent high-tier RL advantage as an artifact of tier geometry rather than of learned skill`
- [ ] **[low] weak-verb** (L100) - saves ~1 word. REVISED by the verifier (the editor's "matters most" asserted an untested superlative; repaired to the comparative).
    - before: `This trade-off is particularly consequential for small positions.`
    - after:  `This trade-off matters more for small positions.`

### Related Work

~22 words saved. Proposed 9, kept 4, rejected 2.

- [ ] **[low] restatement** (L122) - saves ~13 words. Why: near-verbatim duplicate of the L115 closer, which the introduction pass kept intact; the enumeration of the four controls survives in full.
    - before: `operating costs, so that a measured difference reflects policy behavior rather than an evaluation artifact.`
    - after:  `operating costs.`
- [ ] **[low] filler** (L120) - saves ~5 words. Why: the sentence announces the relation instead of delivering it, and the relation is already asserted two sentences earlier and demonstrated by the sentence that follows.
    - before: `The two views are complementary. LVR bounds the cost of provision`
    - after:  `LVR bounds the cost of provision`
- [ ] **[low] roundabout** (L120) - saves ~3 words. Why: passive to active with the same causal direction; every term the cited work uses is retained, satisfying claim-grounding.
    - before: `the adverse selection cost that LPs incur when stale prices are picked off by better-informed arbitrageurs`
    - after:  `the adverse selection cost LPs incur when better-informed arbitrageurs pick off stale prices`
- [ ] **[low] filler** (L122) - saves ~1 word. Why: both verbs coordinate cleanly under the single subject "We"; compounds with the 13-word cut in the same sentence block.
    - before: `under one leak-free protocol, and we test cross-pool policy reuse`
    - after:  `under one leak-free protocol and test cross-pool policy reuse`

### Environment and Evaluation Design

~29 words saved. Proposed 16, kept 9, rejected 2 (plus two variants discarded inside merges).

- [ ] **[low] restatement** (L193) - saves ~5 words. REVISED by the verifier (merged both editors). Why: removes a bare pointer sentence and asserts the identity the sentence actually makes; the cross-reference is preserved. Best line-freeing trim in this section.
    - before: `Table~\ref{tab:setup} summarizes the design. The panel contains a balanced $3\times2$ pair-by-tier design on a shared hourly clock.`
    - after:  `The panel is a balanced $3\times2$ pair-by-tier design on a shared hourly clock (Table~\ref{tab:setup}).`
- [ ] **[low] restatement** (L164) - saves ~4 words. Why: the horizon is fixed three times in adjacent text (the equation's summation limit, the $H=168$ statement, and the completeness requirement).
    - before: `applying candidate action $a$ once and holding thereafter over a fixed horizon.`
    - after:  `applying candidate action $a$ once and holding thereafter.`
- [ ] **[low] roundabout** (L168) - saves ~4 words. Why: the 10\% threshold, the strict inequality, the statistic, and the prespecification defense all survive word for word.
    - before: `We treat the diagnostic as finding headroom when the lower confidence limit on the disagreement fraction exceeds 10\%`
    - after:  `We declare headroom when the lower confidence limit on the disagreement fraction exceeds 10\%`
- [ ] **[low] filler** (L152) - saves ~3 words. Why: the whole subsection specifies CLEB's environment, so "of the benchmark" adds no referent a reader lacks; the defensibility content survives.
    - before: `The label grid is a design choice of the benchmark, and`
    - after:  `The label grid is a design choice, and`
- [ ] **[low] roundabout** (L159) - saves ~3 words. Why: "turn out to" is a discovery frame with no propositional content, and it narrates our path to the finding.
    - before: `On test, the learned deterministic policies turn out to execute a single width per pool`
    - after:  `On test, the learned deterministic policies execute a single width per pool`
- [ ] **[low] roundabout** (L199) - saves ~3 words. Why: the sentence's subject is already the comparison being run, so "add" carries the same relation in one word.
    - before: `the retail frontier settings extend the comparison to the operating costs a practitioner faces`
    - after:  `the retail frontier settings add the operating costs a practitioner faces`
- [ ] **[low] long-sentence** (L199) - saves ~3 words. Why: names the event the check actually detects; the "visible rather than hidden inside a mean" contrast stays verbatim.
    - before: `so a policy that collapses to a single width is visible rather than hidden inside a mean`
    - after:  `so a collapse to a single width is visible rather than hidden inside a mean`
- [ ] **[low] weak-verb** (L148) - saves ~2 words. REVISED by the verifier (the larger 11-word deletion was declined because it would drop the only prose statement of per-swap fee attribution).
    - before: `The overlap is measured in $\sqrt{p}$ for token1 input and $1/\sqrt{p}$ for token0 input. This uses each swap's actual direction, range traversal, and active liquidity.`
    - after:  `We measure the overlap in $\sqrt{p}$ for token1 input and $1/\sqrt{p}$ for token0 input, using each swap's actual direction, range traversal, and active liquidity.`
- [ ] **[low] nominalization** (L159) - saves ~2 words. Why: the gerund subject compresses to a plain noun phrase without touching the disjunction that follows.
    - before: `Observing one action alone cannot show whether a single width is economically appropriate`
    - after:  `One observed action cannot show whether a single width is economically appropriate`

### Results

~81 words saved. Proposed 15, kept 12, rejected 1. Largest single-section harvest.

- [ ] **[low] restatement** (L209) - saves ~16 words. Why: the deleted sentence asserts an appearance the paragraph itself refutes with evidence at L219, and the following clause delivers the mechanism.
    - before: `Read at its nominal labels, the discrete action set looks like a comparable treatment across fee tiers. The primary labels`
    - after:  `The primary labels`
- [ ] **[low] restatement** (L287) - saves ~16 words. Why: restates in negated form the claim now carried at the head of the subsection; its content is asserted directly by the surviving sentence. Frees a full line.
    - before: `The combined evidence therefore rejects the explanation that collapse is rational because conditional opportunity is absent. Headroom exists ex post, but neither the decision stump nor the learned agents demonstrate reliable state-dependent exploitation.`
    - after:  `Headroom exists ex post, but neither the decision stump nor the learned agents demonstrate reliable state-dependent exploitation.`
- [ ] **[low] restatement** (L303) - saves ~11 words. Why: the deleted sentence announces the structure of the three sentences that follow instead of delivering anything; the descriptive-versus-statistical split survives with its evidence intact.
    - before: `The frontier separates a clear descriptive tier pattern from statistical evidence. RecentreWhenOut is worse in all 25 cells`
    - after:  `RecentreWhenOut is worse in all 25 cells`
- [ ] **[med] restatement** (L265) - saves ~9 words. Why: both percentages, the horizon, and the cross-reference are preserved. Risk is med because the rewrite drops "mean within-window", though the caption two lines below defines the quantity verbatim and the abstract already uses this compressed form.
    - before: `Table~\ref{tab:headroom} reports the mean within-window fraction of decision states at which the oracle action differs from the validation-selected fixed width. At 168 hours, disagreement ranges from 67.0\% to 73.2\% across pools.`
    - after:  `At 168 hours, the oracle action differs from the validation-selected fixed width in 67.0\% to 73.2\% of decision states across pools (Table~\ref{tab:headroom}).`
- [ ] **[low] roundabout** (L258) - saves ~6 words. Why: merged both editors; the table caption already states it reports mean held-out reward per pool and that every mean remains below holding.
    - before: `The leak-free cross-section shows that the effective policy is pool- and tier-specific and that every mean sits below holding. Table~\ref{tab:benchmark} gives the per-pool means.`
    - after:  `In the leak-free cross-section, the effective policy is pool- and tier-specific and every mean sits below holding (Table~\ref{tab:benchmark}).`
- [ ] **[low] redundancy** (L312) - saves ~6 words. Why: L152 already states the range-exit consultation rule, so the appositive only restates it; removing it reunites the comparison with its dollar magnitude.
    - before: `worse than event-driven PPO, the primary range-exit consultation schedule, by \$2,448`
    - after:  `worse than event-driven PPO by \$2,448`
- [ ] **[low] redundancy** (L294) - saves ~5 words. Why: the figure caption already states each cell is the mean paired difference; the cross-reference moves into the opening sentence rather than being dropped.
    - before: `We next compare RecentreWhenOut with Passive over five capital levels and five gas levels, mapping who active management helps once operating costs vary. We select the active width on the preceding validation window. In the conversion treatment, each recenter additionally converts half the position at the pool fee tier plus five basis points. Figure~\ref{fig:retail-frontier} reports the mean paired difference.`
    - after:  `We compare RecentreWhenOut with Passive over five capital levels and five gas levels (Figure~\ref{fig:retail-frontier}), mapping who active management helps once operating costs vary. We select the active width on the preceding validation window, and in the conversion treatment each recenter additionally converts half the position at the pool fee tier plus five basis points.`
- [ ] **[low] roundabout** (L204) - saves ~4 words. REVISED by the verifier (repaired a nominalized clause into "testing whether"). Why: the sequencing verbs only restate the order the subsections already appear in; all four axis names and all four cross-references survive.
    - before: `We organize the results around the four axes. Matched executable action geometry comes first (Sec.~\ref{sec:geometry}), followed by the multi-pool benchmark under per-swap fee accounting and leak-free model selection (Sec.~\ref{sec:benchmark}). The state-contingency diagnostic (Sec.~\ref{sec:headroom}) tests whether the learned policies leave usable conditional signal unexploited, and the retail frontier (Sec.~\ref{sec:retail}) evaluates operating costs.`
    - after:  `We organize the results around the four axes: matched executable action geometry (Sec.~\ref{sec:geometry}), the multi-pool benchmark under per-swap fee accounting and leak-free model selection (Sec.~\ref{sec:benchmark}), the state-contingency diagnostic testing whether the learned policies leave usable conditional signal unexploited (Sec.~\ref{sec:headroom}), and the retail frontier under operating costs (Sec.~\ref{sec:retail}).`
- [ ] **[low] roundabout** (L312) - saves ~4 words. Why: replaces a trailing relative clause built on the weak verb "leaves" with a direct consequence clause; the claim is not strengthened.
    - before: `which leaves event-driven recentering as the schedule these checks favor`
    - after:  `so these checks favor event-driven recentering`
- [ ] **[low] nominalization** (L289) - saves ~2 words. Why: converts the nominalization into a verb; the sufficiency qualifier and the scope clause survive verbatim.
    - before: `Thus unnormalized reward scale is not a sufficient explanation for collapse under this common adjustment.`
    - after:  `Thus unnormalized reward scale does not sufficiently explain collapse under this common adjustment.`
- [ ] **[low] weak-verb** (L260) - saves ~1 word. REVISED by the verifier (restored the definite article the editor dropped, which would have made the claim generic).
    - before: `The added algorithmic complexity does not buy state-dependent control on this panel`
    - after:  `The added algorithmic complexity buys no state-dependent control on this panel`
- [ ] **[low] weak-verb** (L265) - saves ~1 word. Why: turns a passive with the agent trailing in a by-phrase into an active sentence; the force of the negation is unchanged.
    - before: `The collapse is not explained by one action being almost always optimal.`
    - after:  `One action being almost always optimal does not explain the collapse.`

### Discussion and Limitations

~17 words saved. Proposed 7, kept 2, rejected 2.

- [ ] **[low] restatement** (L336) - saves ~15 words. REVISED by the verifier (merged three overlapping proposals and repaired a preposition). Why: folds the sixth full enumeration and a standalone pointer sentence into the subject; the table's Axis column keeps the names visible on the same page.
    - before: `Fee accounting, leak-free model selection, executable action geometry, and operating costs change the conclusions more than swapping one RL architecture for another does. Which agent a provider would prefer follows from these four choices before it follows from the choice of algorithm. Table~\ref{tab:axes} summarizes the four axes.`
    - after:  `The four axes in Table~\ref{tab:axes} change the conclusions more than the choice of RL architecture does, so a provider's preferred agent follows from these four choices before it follows from the algorithm.`
- [ ] **[low] roundabout** (L338) - saves ~2 words. Why: both editors proposed this identically; the rejected reading is still named and still rejected, so the hedge keeps its full force.
    - before: `under the present features and training budget, rather than as proof that a single width is optimal`
    - after:  `under the present features and training budget, not as proof that a single width is optimal`

### Conclusion

~25 words saved. Proposed 6, kept 4, rejected 0 (two were repaired inside merges).

- [ ] **[low] restatement** (L345) - saves ~14 words. REVISED by the verifier (merged both editors). Why: the methodological-priority claim is stated twice elsewhere, at L336 and in the abstract's closer, and the Conclusion keeps its own contribution claim in the L347 closer. Marked medium in the verifier's own assessment because a whole claim clause goes; both surviving statements were checked in the file first.
    - before: `the identity of the pool winners, so the benchmark establishes what credible evaluation of learned liquidity provision requires first.`
    - after:  `which arm wins each pool.`
- [ ] **[low] roundabout** (L347) - saves ~7 words. REVISED by the verifier (the editor's version demoted "the benchmark" from subject, which would leave the paper's last sentence attributing the contribution to the choices rather than to CLEB). Why: cuts a topic restatement the preceding sentence already covers, plus a gerund compression.
    - before: `By pinning down these four evaluation choices, the benchmark gives future work on learned liquidity provision a common ground on which to measure new architectures and reward designs.`
    - after:  `By pinning down these four evaluation choices, the benchmark gives future work common ground for measuring new architectures and reward designs.`
- [ ] **[low] nominalization** (L345) - saves ~2 words. Why: unnominalizes and matches the parallel form of the preceding item; the axis name "operating costs" stays contiguous.
    - before: `and an explicit treatment of operating costs`
    - after:  `and operating costs treated explicitly`
- [ ] **[low] roundabout** (L347) - saves ~2 words. Why: a prepositional phrase carrying the identical condition, with no change to what the sentence asserts.
    - before: `When those elements are in place,`
    - after:  `With those elements in place,`

## Flagged (do not auto-apply)

Four trims carry medium risk and should be read in place before applying.

1. **L115 (Introduction), 15 words.** The collective back-reference "under all four controls" depends on the L102 enumeration, which sits across an intervening contribution list. Confirm on the compiled page that a reader meeting "all four controls" has the L102 listing in recent view. Note that if this is applied together with the L104 trim, L102 becomes load-bearing for the whole section and must not be trimmed in any later pass.
2. **L110 (Introduction), 8 words.** Attribution moves from the two named components (leak-free model selection, dependence-aware inference) to "this protocol" as a whole. Both are named one sentence earlier, but check that the referent reads unambiguously.
3. **L265 (Results), 9 words.** The rewrite drops "mean within-window" from the prose statement of the disagreement rate. The caption at L269 defines the quantity verbatim and the abstract uses the same compressed form, so the paper stays internally consistent, but this is the one place a precision qualifier leaves the body text.
4. **L345 (Conclusion), 14 words.** A whole claim clause is deleted. The verifier confirmed the methodological-priority claim survives at L336 and in the abstract's closer. If the L336 trim is applied at the same time, re-read both together to make sure the claim is still asserted once in the back matter.

Two protected spans came out of this pass and should not be touched by any later trim, since deleting them would create a contradiction or a naming break: **"under that correction" at L265** (the .0004 floor is the bootstrap resolution times the four-member Holm factor, and the paper reports .0001 at L289) and **the label "an action-diversity check" at L199** (it is the definition site for the definite reference at L260). Also protected from earlier passes: the L102 enumeration, the purpose clause at L115, the IL and LVR definition sites at L120, the CI definition at L219, and "Several limitations bound the result." at L340.

One flagged harvest target survives untouched because neither editor proposed it. The Conclusion's **L347 clause "The oracle disagrees with the validation-selected fixed width in most decision states"** is a qualitative duplicate of L338's "roughly two-thirds to three-quarters of decision states". It remains the best remaining deletion candidate in the back matter, but if it is taken, the ex-post-versus-deployable clause it carries ("yet the decision stump cannot convert that ex-post headroom into a reliable held-out gain") must survive verbatim in force. The two description columns of Table~\ref{tab:axes} are the only other sanctioned target; the Axis column entries must keep their exact wording, because the Discussion now leans on them for the axis names.

## How to apply

Mark `[x]` on the trims you want, then run `review-apply` on this file, applying one section at a time so the independent-judge gate reads a single section's diff at a time.
