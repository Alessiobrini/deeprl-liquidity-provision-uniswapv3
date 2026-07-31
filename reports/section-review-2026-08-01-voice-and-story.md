# Section-by-section review of `icaif_main.tex`: voice, story, and continuity

Method: we reviewed the paper in reading order, running `review-brini` (line level) and `review-zeropaper` (document level) on each section in parallel, verifying every proposal with an adversarial critic that rejects on doubt, and passing each section's canonical terms, locked quantities, and open threads forward to the next.

## Executive summary

We raised 192 candidate findings across the seven sections (92 from `review-brini`, 100 from `review-zeropaper`). After adversarial verification, 98 survive and 35 were rejected outright, with the remainder merged into consolidated items.

| Severity | Count |
| --- | --- |
| High | 1 |
| Medium | 44 |
| Low | 53 |
| **Total confirmed** | **98** |

The eight most consequential issues, major first.

1. **[HIGH] The action grid has no stated rationale.** The three nonzero labels span 45 to 55, and the sentence meant to defend them, "The label grid is a design choice", asserts nothing, so the paper's headline negative result reads as close to mechanical (Sec. 3, L161).
2. **[MED] The pass criterion is defined, promised twice, and never applied.** L207 defines passing CLEB, L214 and L365 echo it, and no sentence anywhere states the verdict for the eleven arms evaluated here (Sec. 3, Sec. 4, Sec. 6).
3. **[MED] The gated follow-up is reported with a treatment the prespecification never declared.** L176 declares reward-only normalization, L302 reports reward-only normalization plus validation-selected entropy (Sec. 3 against Sec. 4).
4. **[MED] Two different binding mechanisms are named one page apart.** L325 names the cost of over-trading, L352 names a failure to recover the conditional signal, and neither sentence acknowledges the other (Sec. 4 against Sec. 5).
5. **[MED] The single-action claim is stated unscoped four times against first-seed evidence.** The abstract, L119, L167, and L207 state it for all learned policies while L273 evidences it on the first-seed deterministic policy only, one clause away from a seed-averaging statement (Abstract, Sec. 1, Sec. 3).
6. **[MED] The four-axis sensitivity claim outruns the evidence.** The abstract (L82) and L350 assert that the measured answer follows all four axes, while the paper's own careful statement is that only the two axes we vary experimentally reorder anything (Abstract, Sec. 5).
7. **[MED] The conclusion omits the paper's most repeated finding and points future work at closed-off levers.** The collapse to a single observed action never appears in Sec. 6, and the final sentence nominates architectures and reward designs, the two levers L352 and L354 report as not binding, while L358 nominates state variables and action set (Sec. 6, L365).
8. **[MED] The retail frontier result is overstated in the Discussion.** L350 says operating costs decide "where recentering is favorable", while L316, L344, and L363 all record that no favorable cell survives Holm correction (Sec. 5, L350).

## Cross-section flow threads

Reading in order surfaced eight document-level threads that no single-section pass would catch.

**1. One word, several senses.** Four terms carry two or more referents across the paper, and the collisions concentrate in the sections least able to afford them. "Family" carries four senses (a heuristic arm's configuration set at L165, the Holm testing family at L196, L300, and L307, a loose kind-of-effect sense at L300, and an algorithm class at L354), three of them inside the subsections whose subject is inference. "Holding" names both the reference basket that defines the reward and the zero action (L143 against L161 and L165). "Width" names both a grid label and the executed half-width, which is exactly the distinction the geometry axis exists to separate. "The learned arms", "the learned policies", and "the learned agents" name one object in three places. We recommend reserving "family" for the Holm testing family, "holding" for the reference basket, "action" for the grid element, and "the learned policies" wherever the referent is deterministic test behavior, in one pass across L165, L167, L207, L300, L307, L352, and L354.

**2. Symbol collisions inside the method.** $H$ denotes both the hold value (L143) and the diagnostic horizon (L172), and $s$ denotes both the swap index (L152) and the decision state (L167). Both collisions meet in the same equation chain. Both are contained: $Q_H$ occurs at two lines and the swap index at five, so renaming to $K$ and $j$ touches Sec. 3 alone and propagates nowhere, since the Results reference the equations by label only.

**3. Canonical-name drift accumulates section by section.** The benchmark gains a second name at L129 ("our benchmark"), the fee accounting mechanism becomes "displacement-based fees" at L113 and "per-swap attribution" at L341 and "per-swap fee accounting" at L214 and L363, the nominal labels become "primary labels" at L222, the paired mean difference becomes a "lift" at L300, and the executable action geometry axis becomes "the discrete comparability question" at L131. Each is a one-word fix, and together they are the difference between a paper with four named axes and a paper with a dozen near-synonyms for them.

**4. Axis ordering has three variants.** L113 and Table `tab:axes` declare fee accounting, leak-free model selection, executable action geometry, operating costs. The contribution bullets deliver 1, 2, 4, 3, and the abstract's results sentence delivers the mirror-image break. The Results run geometry first. The abstract and the bullets must be reordered in one edit, since fixing either alone makes the two disagree.

**5. The pass criterion is the paper's one unpaid promise.** Sec. 3 builds the instrument at L207, Sec. 4 reports the evidence at L269 without turning the instrument on it, Sec. 5 passes the last natural opportunity, and Sec. 6 promises the test again at L365. One clause at L269, scoped to the learned arms, closes the loop, and the L365 echo then needs its verb split so "identifies" no longer takes both a measured quantity and an authored criterion.

**6. The mechanism story slips at the Sec. 4 to Sec. 5 seam.** The over-trading claim at L325 is the strongest mechanistic claim in the paper and its own paragraph does not support it, since daily and weekly schedules trade less often than the event trigger and also fail, and the representation check has nothing to do with frequency. The Discussion then names a different cause and never mentions the schedule evidence, which L356 does carry forward. This is one edit on both sides, not two independent findings.

**7. Two results sit outside the four-axis frame.** The transfer experiment at L323 is announced in Related Work, delivered in the Results, and picked up in neither the Discussion nor the Conclusion, even though it is the paper's most direct evidence for the pool specificity the provider paragraph asserts. The over-trading schedule result has the opposite problem, being asserted as a mechanism and then dropped.

**8. Every addition has to be paid for.** The paper is full, so the three substantive additions we recommend (the fee-attribution maintained assumption, the gate hypothesis at L176, and the hourly-grid construction rule) are funded by the deletions we confirm at L207, L231, L271, L300, and L363. The fee-attribution assumption in particular must land once, either at Eq. `eq:swap-fees` or in the L358 omissions list, never in both.

## Findings by section

### Abstract

Raw 21 (brini 11, zeropaper 10), confirmed 7, rejected 10.

- [ ] **[MED] narrative-overclaim + vague-noun** (L82, "where the measured answer is sensitive to four evaluation axes that CLEB controls at once") - "the measured answer" names no question, the "where" clause attaches to the venue rather than to the evaluation, and the body claims only two axes move the conclusions. Fix: assert the control directly and drop the sensitivity clause.
- [ ] **[MED] claim-scope** (L82, "the learned policies, evaluated deterministically, select only one observed action per pool") - stated for all learned policies while L273 evidences the first-seed deterministic policy, one clause after advertising seed averaging. Fix: scope to "every learned algorithm's first-seed deterministic policy".
- [ ] **[MED] redundancy + ambiguous-pronoun** (L82, "and it supplies the control over all four axes needed to evaluate it") - restates the opening sentence and the two pronouns point at different referents. Fix: delete the clause and end on the loss-mitigation finding.
- [ ] **[LOW] term-drift** (L82, "six heuristic policies") - the body says "heuristic arms" and "heuristics", and the abstract reserves "policies" for the learned arms one sentence later. Fix: write "six heuristic arms".
- [ ] **[LOW] explicit-over-shorthand** (L82, "reverses one learned policy's apparent high-tier strength") - anonymizes a reversal that Table `tab:axes` and L231 both attribute to QR-DQN. Fix: name QR-DQN, applied together with the axis-order fix below.
- [ ] **[LOW] acronym-single-use** (L82, "Proximal Policy Optimization (PPO)") - PPO appears once in the abstract, so the expansion is dead weight. Fix: drop the parenthetical.
- [ ] **[LOW] axis-order** (L82, retail frontier and geometry sentence) - the results are reported in the reverse of the declared axis order and joined by a "while" that signals a contrast between two unrelated axes. Fix: split and lead with geometry, matching the Results order.

### Introduction

Raw 26 (brini 10, zeropaper 16), confirmed 12, rejected 8.

- [ ] **[MED] vocabulary** (L115, "every strategy's mean trails holding") - a leaderboard verb, and a third wording for a claim the abstract and L265 both state as "sits below". Fix: write "sits below holding".
- [ ] **[MED] citation-subject** (L111, "Empirical work finds ... while practitioner evidence identifies") - two singular noun stand-ins in front of a three-key bundle, against the plural citation-as-subject pattern used throughout Related Work. Fix: restructure with an adverbial so the citation is the subject.
- [ ] **[MED] claim-grounding** (L113, "because displacement-based fees and non-conserving rebalances credit policies with value they did not earn") - an unattributed indicative accusation, plus a third name for the hourly price-displacement proxy. Fix: state it counterfactually with the canonical name, as L118 already does.
- [ ] **[MED] claim-scope** (L119, "Because every learned arm turns out to execute a single width") - unscoped against L273, and "turns out to" narrates our order of discovery. Fix: "every learned arm's deterministic policy executes a single width", in the same wording as the abstract fix.
- [ ] **[MED] enumeration-order** (L118 to L121, the four contribution bullets) - the bullets deliver axes 1, 2, 4, 3, a third ordering after L113 and the abstract. Fix: swap the retail-frontier and geometry bullets, applied together with the abstract reorder.
- [ ] **[MED] redundant-close** (L124, "Together these results offer CLEB as a reusable protocol ...") - the third statement of the four-axis claim, and the introduction never says why a negative result is interesting. Fix: replace with the payoff of the null result, consistent with L352 and L365.
- [ ] **[LOW] causal-chain** (L115, "which makes the corrected evaluation itself the contribution") - the third step of the paper's central chain is asserted in a trailing relative clause with its premise missing. Fix: split and state the link.
- [ ] **[LOW] ambiguity** (L120, "at every capital and gas level") - the phrase repeats within one sentence and the trailing modifier reads as a global claim where the result is tier specific. Fix: restore the tier scope and drop the repeat.
- [ ] **[LOW] anonymized-result** (L121, "one apparent high-tier RL advantage") - anonymous where L296 and L343 name QR-DQN. Fix: name QR-DQN, settled once with the abstract.
- [ ] **[LOW] shorthand** (L113, "dependence-aware inference") - a coined label for a procedure the paper names explicitly at L119 and L269. Fix: write "a moving-block bootstrap".
- [ ] **[LOW] coined-modifier** (L118, "a balanced three-pair by two-fee-tier design") - two on-the-spot compounds, and a third name for a design already called $3\times2$ at L201. Fix: "a balanced design over three token pairs and two fee tiers".
- [ ] **[LOW] readability** (L119, "selected on validation windows only does not reliably capture") - a garden path on the sentence carrying the headline negative result. Fix: comma off the modifier.

### Related Work

Raw 16 (brini 7, zeropaper 9), confirmed 8, rejected 2.

- [ ] **[MED] restatement-and-missing-bridge** (L131, "CLEB imposes all four axes at once.") - the third statement of the same claim inside seven lines, the verb drifts from "controls", and the section closes by asserting the gap rather than locating it. Fix: replace with a bridge naming the combination none of the reviewed benchmarks provides, without attributing an axis-specific failing to any named study.
- [ ] **[MED] name-drift** (L129, "Our benchmark ... It ... CLEB") - three names for the canonical object in three consecutive sentences, plus "logged flow" and "logged mainnet flow" within two sentences. Fix: use CLEB once and drop the duplicated destination.
- [ ] **[MED] coined-word** (L131, "algorithm un-trainability") - a hyphenated negation coined for a plain idea, inside a clause that stacks "coming from" twice. Fix: "separate return unpredictability from the algorithms' failure to train".
- [ ] **[MED] canonical-term-drift** (L131, "the discrete comparability question we ask") - a fourth name for executable action geometry, and no comparability question has been posed at this point in the reading order. Fix: "the continuous-action counterpart of the executable action geometry we match across arms and fee tiers".
- [ ] **[LOW] repeated-construction** (L131, "carry that separation to logged flow") - one transport phrase carries three sentences in a six-line section. Fix: vary the second and drop the now-redundant destination.
- [ ] **[LOW] restatement** (L131, "We benchmark learned agents against transparent heuristics under one leak-free protocol and test cross-pool policy reuse") - the first clause repeats contribution bullet 2, so the only new content sits behind a repeat. Fix: cut the repeated clause, applied together with the closer fix so the paragraph does not end on a bare citation.
- [ ] **[LOW] overclaim-attributed-to-source** (L129, "the main adverse selection cost LPs incur") - a ranking put in the cited authors' mouths. Fix: drop the superlative, after confirming the wording in the source.
- [ ] **[LOW] vague-noun** (L131, "under a rolling window approach") - padding on a procedure the paper names plainly at L201. Fix: "on a rolling window".

### Environment and Evaluation Design

Raw 33 (brini 16, zeropaper 17), confirmed 21, rejected 7.

- [ ] **[HIGH] defensibility** (L161, "The discrete action set is $\{0,45,50,55\}$ ... The label grid is a design choice.") - the paper's most load-bearing choice is defended by a sentence with no content, and three nonzero labels spanning 45 to 55 make the single-action finding look mechanical. Fix: state the basis on which the labels were placed and note that Subsec. `sec:geometry` varies what each label executes, taking the rationale from the design rather than inventing it.
- [ ] **[MED] scope-consistency** (L167 and L207, "the learned deterministic policies execute a single width per pool") - the third unscoped instance, and the one the diagnostic's motivation rests on. Fix: add the first-seed scope in the same wording as the abstract and L119.
- [ ] **[MED] notation-collision** (L143 against L172, $H$) - $H$ denotes both the hold value and the diagnostic horizon, and the two meet in the same equation chain. Fix: rename the horizon to $K$, touching L169 and L172 only.
- [ ] **[MED] notation-collision** (L152 against L167, $s$) - $s$ denotes both the swap index and the decision state, both introduced with the same "in hour $t$" framing. Fix: index swaps with $j$ and reserve $s$ for the decision state.
- [ ] **[MED] term-collision** (L165 against L196, "family") - three unrelated senses, with the Holm sense arriving thirty lines after the first. Fix: reserve "family" for the Holm testing family, rewrite L165, and use "algorithm classes" at L354.
- [ ] **[MED] term-collision** (L165, "holding") - names both the reference basket that defines the reward and the zero action, within thirty lines. Fix: reserve "holding" for the basket and name the zero action by its label.
- [ ] **[MED] method-results-mismatch** (L176 against L302) - the prespecified follow-up is declared as reward normalization alone and reported with validation-selected entropy. Fix: state the entropy handling where the follow-up is declared, or shorten L302, whichever matches the runs.
- [ ] **[MED] unpaid-promise** (L207, "We define passing CLEB in a pool as ...") - a named criterion set up for future work and never applied to the arms this paper evaluates. Fix: state the verdict once at L269, scoped to the learned arms.
- [ ] **[MED] narrative** (L176, "we prespecify a follow-up gated on that outcome") - the hypothesis the gate rests on arrives only in the Discussion at L352. Fix: name the candidate cause in the gate sentence.
- [ ] **[MED] claim-grounding** (L152 to L157, Eq. `eq:swap-fees`) - the attribution inserts our position into logged flow while reusing each swap's observed size, direction, and traversal fraction, and this maintained assumption is stated nowhere. Fix: add one clause stating the assumption and its defense, here or at L358, never both.
- [ ] **[LOW] term-ambiguity** (L167 and L207, "width") - carries the grid label in some sentences and the executed half-width in others, which is the distinction the geometry axis exists to separate. Fix: write "action" where the grid element is meant.
- [ ] **[LOW] undefined-quantity** (L176, "the lower confidence limit on the disagreement fraction") - a prespecified decision rule stated on a quantity defined only at L282. Fix: define it inline where the rule is stated.
- [ ] **[LOW] forward-dependence** (L165, L174, L176) - the selection discipline uses walk-forward terms before L201 defines them. Fix: point at Table `tab:setup` on first use, without reordering the subsections.
- [ ] **[LOW] passive-voice** (L150 and L161) - two passives where the agent is us, one of them the decision schedule on which the L325 result turns. Fix: write both in the first person, and say whether the heuristic arms face the same schedule.
- [ ] **[LOW] register** (L207, "CLEB packages ... plugs in by consuming") - product-documentation register in the subsection where the artifact contribution is judged. Fix: "provides" and "enters the benchmark by taking".
- [ ] **[LOW] grammar** (L150, "solves the new liquidity from that same USD value") - a wrong complement on the definition of value-conserving rebalancing. Fix: "solves for the new liquidity at that same USD value".
- [ ] **[LOW] caption** (L182, Table `tab:setup`) - the four-item list renames the table's own visible row labels. Fix: shorten to what the rows cannot carry.
- [ ] **[LOW] redundancy** (L207, "A comparable result reports all six pools ...") - every clause repeats L187, L193, L194, L195, and L203. Fix: delete and fold the one non-redundant element into the criterion.
- [ ] **[LOW] defensibility** (L176, "exceeds 10\%") - a prespecified threshold with no stated basis, on the gate that decides whether the one gated experiment runs. Fix: add the basis in a few words.
- [ ] **[LOW] clarity** (L161, "the average directional movement index ... and the directional movement index") - two list entries differing by one word, and the 13-dimensional count depends on their being distinct. Fix: distinguish them in place.
- [ ] **[LOW] defensibility** (L201, "aggregate them to a complete hourly grid per pool") - the rule for hours with no swap is unstated, and it feeds every state variable and every reward. Fix: name the carry-forward rule in a short clause.

### Results

Raw 43 (brini 20, zeropaper 23), confirmed 24, rejected 6.

- [ ] **[MED] unpaid-promise** (L269, "so both are suggestive rather than robust") - the natural home for the pass verdict, and it reports the leaderboard without ever saying that nothing clears the bar. Fix: append "and no arm passes CLEB in any pool".
- [ ] **[MED] overclaim-mechanism** (L325, "point to the cost of over-trading as the binding mechanism") - the paragraph's own nulls do not bear on trading frequency, and the Discussion names a different cause. Fix: scope the topic sentence to what the checks show.
- [ ] **[MED] scope-mismatch** (L273, the action-diversity check) - the central negative result rests on the first of two seeds while performance is seed averaged, and no sentence says why. Fix: state the scope and its basis in the same sentence, or report both seeds.
- [ ] **[MED] antecedent** (L269, "the best validation heuristic for that pool, ILMinimizer") - the nearest antecedent is the wrong pool. Fix: name the pool once, up front.
- [ ] **[MED] reporting-order** (L231) - each surviving effect is separated from its own interval by two sentences, with a third contrast reported in between, giving three p-values for two announced effects. Fix: report each effect once with its interval and p-value attached, then introduce the cross-tier contrast as such.
- [ ] **[MED] logic-order** (L278, "A single action is rarely optimal, so the collapse cannot be explained that way.") - the conclusion precedes its evidence and "that way" points at an explanation never stated. Fix: lead with the disagreement fractions and draw the inference.
- [ ] **[MED] missing-condition** (L307, the retail frontier setup) - the only experiment that names no executable action geometry, while the paragraph's closing mechanism argument holds only under the nominal labels. Fix: state the geometry in the setup sentence.
- [ ] **[MED] inference-overclaim** (L300, "so the design resolves a contrast of this family when the effect is present") - the oracle contrasts are an order of magnitude larger than any stump effect, so their significance does not establish power against a stump-sized effect, and the clause contradicts the preceding sentence. Fix: scope to magnitude and drop the second sense of "family".
- [ ] **[MED] loose-robust** (L323) - "robust" used twice as a bare adjective in a paragraph that never names the block lengths, against the paper's precise usage elsewhere. Fix: anchor both to the criterion.
- [ ] **[LOW] hedging** (L302, "does not sufficiently explain ... under a common adjustment") - two hedges stacked on one negation, drifting from the Discussion's plain statement at L352. Fix: match L352.
- [ ] **[LOW] prespecification-wording** (L302, "and validation-selected entropy") - entropy is already in the validation grid for every PPO run at L163, so naming it implies a treatment difference that does not exist. Fix: drop the clause, or declare it at L176.
- [ ] **[LOW] term-drift** (L222, "The primary labels") - "nominal" is the canonical name and "primary" is already reserved for the block length and the benchmark. Fix: "The nominal labels".
- [ ] **[LOW] redundancy** (L300, "Headroom exists ex post, but neither the decision stump nor the learned agents demonstrate reliable state-dependent exploitation.") - the third statement of the claim inside six lines, and it introduces a third name for the learned arms. Fix: delete the sentence.
- [ ] **[LOW] long-subject** (L325) - roughly forty words separate the subject from "produce". Fix: split at the verb.
- [ ] **[LOW] run-on** (L307) - the exactness justification and the Holm family statement are welded into one 60-word sentence. Fix: split.
- [ ] **[LOW] negative-first** (L231, "is not an economically comparable treatment") - one of the four axes stated as what the action is not. Fix: state it positively, matching Table `tab:axes`.
- [ ] **[LOW] coined-compound** (L278, "every oracle-versus-fixed-width contrast") - a coined modifier for a contrast named in plain words at L174. Fix: "every contrast of the oracle against the fixed width".
- [ ] **[LOW] term-drift** (L300, "held-out lift") - a fourth name for the paired mean difference, at the point where its null result is reported. Fix: "held-out difference from the fixed width".
- [ ] **[LOW] passive-voice** (L239 caption, "Learned-policy performance is averaged across seeds") - the only such passive in the section, stated actively at L203. Fix: write it in the first person and keep the "below holding" clause.
- [ ] **[LOW] filler** (L231, "The treatment also moves a heuristic that can respond to it.") - announces the next sentence, which then runs 49 words and chains three claims. Fix: delete the announcement and split what follows.
- [ ] **[LOW] claim-grounding** (L271, the LVR citation) - the cited work measures against a rebalancing reference portfolio while the sentence measures against holding, so the cite can be read as sourcing the magnitude. Fix: scope the cite to the mechanism, after checking the source's wording.
- [ ] **[LOW] missing-reading-rule** (L269, "the highest-mean arm is pool- and tier-specific") - every comparison is within pool but no sentence says so, one page after the subsection discrediting cross-tier reading of the labels. Fix: add the reading rule at the head of the paragraph.
- [ ] **[LOW] dangling-result** (L323, the transfer paragraph) - reports counts and stops, and transfer appears nowhere downstream. Fix: tie it back to the pool- and tier-specific ordering, if the deletions free the line.
- [ ] **[LOW] redundancy** (L271, "and the choice among these arms is a choice of how much loss to carry") - the framing conclusion stated inside a paragraph about the level, and restated at L356 and L365. Fix: end the sentence earlier and leave the framing to the Discussion.

### Discussion and Limitations

Raw 35 (brini 18, zeropaper 17), confirmed 18, rejected 1.

- [ ] **[MED] overclaim** (L350, "imposing operating costs decides where recentering is favorable") - no favorable cell survives Holm, as L316, L344, and L363 all record. Fix: "locates where recentering is adverse and where it is only descriptively favorable".
- [ ] **[MED] internal-tension** (L350, "Which arm a provider would prefer therefore follows from these four axes") - collapses the two-axis distinction drawn one sentence earlier, and answers the same question L356 answers differently six lines later. Fix: make the subject the measured ranking at L350 and leave L356 to name the preferred arm.
- [ ] **[MED] claim-precision** (L352, "not evidence that the state is economically irrelevant") - the oracle establishes that the best action varies across decision states, not that the 13 observed variables carry that information, and L358 concedes the opposite in the same section. Fix: "not evidence that the best action is constant across decision states".
- [ ] **[MED] mechanism-conflict** (L352 against L325) - two binding mechanisms named one page apart, with the Discussion never mentioning the schedule evidence. Fix: scope L325 and add one reconciling clause at L352, as a single edit.
- [ ] **[MED] term-collision** (L354, "four families at once") - a fourth sense of "family" inside the paragraph about what the statistical evidence supports, with the correct term used two lines later. Fix: "four algorithm classes", settled together with L165.
- [ ] **[MED] argument-structure** (L354, the undertraining paragraph) - the first leg does not carry, since a single shared budget being too small is one explanation rather than four, and the closer concedes with "at this budget". Fix: lead with the stump, demote the budget to a supporting qualifier, and end on what the two features rule out.
- [ ] **[MED] overclaim** (L356, "Paying a fixed schedule is one choice the panel argues against") - anthropomorphic framing, and a single-pool contrast plus two nulls generalized into a recommendation. Fix: name the agent and scope the claim to the pool where the schedule is varied.
- [ ] **[LOW] scope** (L356, "the do-nothing and event-driven references are hard to displace") - reads as recommending and warning against RecentreWhenOut inside two sentences, since the tier split is never supplied. Fix: scope Passive and event-driven recentering separately.
- [ ] **[LOW] claim-grounding** (L354, "maximizes realized reward over a single split") - dropping "on training states" leaves the sample unnamed, and "carries no ... at all" is the vague-magnitude construction. Fix: restore "on training states" and write "poses no optimization difficulty".
- [ ] **[LOW] term-drift** (L352, "the learned agents' static behavior") - the property described belongs to the deterministic policy. Fix: "the learned policies", applied at L300 too, leaving "the agents are undertrained" at L354 as is.
- [ ] **[LOW] voice-construction** (L354, "What this panel supports is that ...") - a pseudo-cleft that puts the supporting in the panel's mouth. Fix: state the conclusion directly, as the paragraph's last line.
- [ ] **[LOW] filler** (L356, "For a provider, the evidence supports a narrow set of conclusions.") - announces conclusions the paragraph then delivers, and its scoping is repeated by the paragraph's closer. Fix: delete and open on the substance.
- [ ] **[LOW] dangling-result** (L356, the provider paragraph) - the transfer experiment is the most direct evidence for the pool specificity asserted here and appears nowhere. Fix: add a clause pointing at Subsec. `sec:robustness`, worded as "no general benefit" rather than "no benefit".
- [ ] **[LOW] sentence-structure** (L358) - two unrelated limitations welded with "and", with a trailing "which" that can attach to either. Fix: split.
- [ ] **[LOW] filler** (L358, "may simply carry too little conditional signal") - a filler intensifier stacked on a hedge, in the sentence stating the paper's main open question. Fix: drop "simply".
- [ ] **[LOW] missing-limitation** (L358, the omissions list) - the per-swap attribution's maintained assumption about routed size and direction is stated nowhere. Fix: extend the existing sentence, here or at Eq. `eq:swap-fees`, never both.
- [ ] **[LOW] caption** (L333, Table `tab:axes`) - the first sentence restates the table's own column headers, pushing the load-bearing sentence last, and names four rows two different ways. Fix: shorten and name all four rows the same way.
- [ ] **[LOW] term-drift** (L341, "Per-swap attribution") - drops "fee" in the row whose subject is fee accounting. Fix: "Per-swap fee attribution".

### Conclusion

Raw 18 (brini 10, zeropaper 8), confirmed 8, rejected 1.

- [ ] **[MED] unpaid-promise** (L365, "so CLEB identifies both the ex-post headroom ... and the out-of-sample test") - the connector credits the stump's failure with a result the oracle establishes, and one verb takes both a measured quantity and an authored criterion, on the paper's third statement of a criterion whose verdict is never given. Fix: split the credit and state that no arm on this panel clears the test.
- [ ] **[MED] missing-headline-finding** (L365, the result list) - the collapse to a single observed action, the paper's most repeated finding, is absent from the entire conclusion. Fix: add it with the first-seed deterministic scope, in place of one of the four-axis restatements.
- [ ] **[MED] geometry-scope** (L365, the leader counts) - a ranking is handed to the reader with no geometry named, then said four clauses later to move in four of six pools. Fix: label the first clause "Under the nominal labels".
- [ ] **[MED] story-arc** (L365, "common ground for evaluating new architectures and reward designs") - the final sentence nominates the two levers L352 and L354 report as not binding and omits the two L358 nominates. Fix: inherit L358, or scope architectures to what the evidence leaves open.
- [ ] **[MED] redundancy** (L363, "Matching executable action geometry reorders the pool winners, and mapping operating costs locates ...") - a near-verbatim repeat of L350 and of bullet L120, with the geometry claim then restated at L365 and the verb drifting a third time. Fix: replace with the joint consequence neither L350 nor L120 states, restoring the name "retail frontier".
- [ ] **[MED] restatement** (L363 and L365) - the four-axis control claim is asserted three times in two paragraphs, on top of four earlier statements, and "pinning down" is a fifth verb and a colloquial register. Fix: keep the enumeration at L363 and delete the two bare restatements.
- [ ] **[LOW] term-drift** (L363, "per-swap fee accounting") - blends the axis label and the mechanism name into a third form, in the sentence whose job is to name the axes canonically. Fix: "per-swap fee attribution", with L214 fixed in the same pass.
- [ ] **[LOW] coined-shorthand** (L365, "rather than universal profitable control") - a coined compression, and the fourth undefined modifier on "control" after state-dependent, predictable, and conditional. Fix: use the paper's own wording from L356.

## Suggested sequencing

We recommend applying nothing until the three decisions in group (c) are settled, since two of them determine the wording of edits in groups (a) and (b).

### (a) Quick line fixes, contained to one line each

Vocabulary and grammar: L115 "trails", L150 "solves the new liquidity", L358 "simply", L354 "carries no ... at all", L131 "un-trainability" and "approach", L129 superlative "the main".

Canonical-name substitutions: L113 "displacement-based fees", L131 "the discrete comparability question", L129 "our benchmark", L222 "primary labels", L300 "lift", L341 "Per-swap attribution", L214 and L363 "per-swap fee accounting", L365 "universal profitable control".

Readability and structure inside one sentence: L119 comma, L231 reporting order, L269 antecedent, L278 logic order, L307 run-on, L325 long subject, L358 split, L231 negative-first, L278 coined compound.

Voice and register: L150 and L161 passives, L239 caption passive, L207 register, L354 pseudo-cleft, L356 "the panel argues against", L356 filler opener, L302 hedges.

Captions: L182 (`tab:setup`) and L333 (`tab:axes`).

Cuts that fund the additions: L207, L231, L271, L300, L363.

### (b) Flow and structure edits that touch several sections at once

Each of these must be applied as a single pass, since fixing one instance alone creates a disagreement.

1. **The first-seed deterministic scope**, in identical wording at the abstract, L119, L167, L207, L273, and the new clause at L365.
2. **The axis ordering**, reordering the abstract's results sentence and the contribution bullets at L120 and L121 together.
3. **The pass verdict**, one clause at L269 scoped to the learned arms, with the L365 echo rewritten so "identifies" no longer takes two objects.
4. **The mechanism reconciliation**, scoping L325 and adding one clause at L352.
5. **The "family" rename**, across L165, L300, L307, and L354.
6. **The "learned agents" rename**, at L300 and L352.
7. **The notation renames**, $H$ to $K$ at L169 and L172, and the swap index to $j$ across L152 to L157 and Eq. `eq:swap-fees`. Contained to Sec. 3.
8. **The four-axis claim discipline**, at the abstract's opening and at L350, so neither claims that all four axes move the ranking.
9. **The Related Work closer**, replacing the restatement with a bridge and trimming the "We benchmark ..." clause in the same edit.
10. **The undertraining paragraph reorder** at L354, with the duplicated qualifier at L352 cut in the same pass.
11. **The conclusion's four edits** at L363 and L365, which are one budget: the redundancy replacement and the two restatement deletions pay for the geometry scoping and the single-action clause. Re-read the section for length afterward.

### (c) Items needing a coauthor decision or a check we could not run

1. **The action grid rationale (HIGH, L161).** We cannot invent the basis on which 45, 50, and 55 were placed. This needs the design's own justification, and it is the finding a referee will reach first.
2. **The entropy clause (L176 against L302).** Whether the normalized rerun used a different entropy grid from standard PPO decides whether the fix is to declare it at L176 or to delete it at L302. This is a question about the runs, not about the wording.
3. **The QR-DQN naming decision.** The abstract and L121 anonymize a reversal that L231, L296, and L343 attribute to QR-DQN. One choice, applied to both places.
4. **The fee-attribution assumption.** The wording must be checked against the fee-attribution script, in particular what $\Lambda_s$ records, and the clause must land once, either at Eq. `eq:swap-fees` or in the L358 omissions list.
5. **The action-diversity seed question (L273).** If the second seed's deterministic policy was logged, reporting both is better than defending the first. If it was not, the paper needs the one-clause defense of the choice.
6. **The hourly-grid construction rule (L201).** The carry-forward rule for hours with no swap has to be read off the pipeline, not inferred.
7. **The two source checks.** The LVR characterization at L129 and its use at L271 both need the cited work's own wording before we change them.
8. **The headroom threshold (L176).** Whether 10 percent has a stated basis, or should be presented as an intentional design choice with its merits named.
