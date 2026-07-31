# Section-by-section review, second pass (2026-07-31)

Paper under review: `paper/RL_for_liquidity_provisioning/icaif_main.tex`.

Method: we reviewed the paper in reading order, one section at a time. For each section we ran `review-brini` (line level: voice, acronyms, captions, active voice, figures, citations) and `review-zeropaper` (document level: prose economy, consistency, referee, defensibility, claim grounding, page budget) in parallel, then an adversarial critic verified every proposed finding against the source file and rejected on doubt. Each section handed its canonical terms, symbols, numbers, and open threads forward to the next, so continuity defects that only appear on a top to bottom read were caught. This report is find only. Nothing has been applied.

Raw proposals: 221 (108 from `review-brini`, 113 from `review-zeropaper`). Confirmed after verification and merging: 98. Rejected: 61.

## Executive summary

We confirm **98 findings**: **3 high**, **39 medium**, **56 low**.

By section (raw brini + raw zeropaper, confirmed, rejected): Abstract 11+12, 6, 14. Introduction 17+13, 18, 2. Related Work 11+14, 6, 11. Environment and Evaluation Design 27+21, 23, 17. Results 22+28, 25, 7. Discussion and Limitations 12+15, 11, 7. Conclusion 8+10, 9, 3.

The most consequential issues, major first:

1. **The four-axis frame breaks at the opening of the Results** (L215). The sentence announces "the four axes" and then lists four items that do not map onto them: it folds fee accounting and leak-free model selection into one item and promotes the state-contingency diagnostic, which has no row in Table~\ref{tab:axes}, to a fourth. A reader who counts here and then reads the axes table gets two different sets.
2. **The Conclusion contradicts its own headline number** (L358 against L360). L358 states that matching executable action geometry "changes which arm wins each pool" while L360, the abstract, the contribution bullet, Subsec.~\ref{sec:geometry}, and Table~\ref{tab:axes} all scope the same result to four of six pools. The unscoped version is the one a referee reads first.
3. **The retail-frontier figure caption names a cost setting the text contradicts** (L311 against L315). The caption title says the cells are after gas and token-conversion friction, while the text treats conversion as a separate treatment layered on gas and counts 150 signs "relative to gas alone", which fits a single treatment. The exhibit cannot be read standalone, and the caption also omits the reward unit.
4. **The section that implements two of the four axes never names them** (L150 and L165). Value-conserving rebalancing and leak-free model selection appear only in the abstract, the introduction, Related Work, and the axes table. Sec.~3 builds both mechanisms and labels neither, so a reader cannot find the axes' definitions where they are constructed.
5. **The Discussion upgrades two design choices into measured effects** (L349 against the Table~\ref{tab:axes} caption on the same page). L349 credits all four axes with changing the conclusions more than the choice of architecture does, while the caption directly above concedes that only the geometry and operating-cost rows rest on measured contrasts. The abstract and the introduction were already scoped to that concession, so this is the one site that overturns it.
6. **The limitations paragraph blames a cause the paper's own diagnostic argues against** (L353 against L351). L351 uses the oracle to establish that within the same discrete action set the best action differs from the fixed width in roughly two thirds to three quarters of decision states, so coarseness of that set cannot explain why the agents never vary their action inside it.
7. **One Results sentence is literally false against Table~\ref{tab:benchmark}** (L272). "The learned arms operate as a single validation-selected width" reads as saying the learned arms reduce to the Fixed width arm, whose means differ from every learned mean in every pool. The intended claim is that each learned arm executes one constant width.
8. **The paper carries five nouns for its organizing frame.** Axes (L113, L115, L215, L349), controls (L124, L131), choices (L82, L349), elements and evaluation choices (L360). The drift is concentrated in the sentences that lock the contribution in, and it must be closed at all sites in one pass.

## Cross-section flow threads

Reading in order surfaced six document-level threads. Each spans sections and none of them can be fixed at a single site.

**Thread 1. The frame's noun.** The four-axis frame is the paper's organizing device and it is named five ways: axes (L113, L115, L215, L349, Table~\ref{tab:axes} caption), controls (L124, L131), choices (L82, L349), elements (L360), evaluation choices (L360). The canonical form is fixed at L113 and in the axes-table caption. The drift enters at the closing sentence of the introduction, recurs at the closing sentence of Related Work, and then compounds inside a single sentence at L349 and again inside the paper's final paragraph. We recommend deciding on **axes** once and applying it at L82, L124, L131, L349, and both L360 sites together, since fixing any one site leaves the paper with three nouns.

**Thread 2. The third axis has fragmented and the fourth loses its name at its own heading.** The canonical label is *matched executable action geometry* (L113, L215, the Subsec.~\ref{sec:geometry} title, Table~\ref{tab:axes}). Four sites shorten it: L175 and the L228 and L281 captions drop "action", and L232 drops both words. Separately, the fourth axis is *operating costs* everywhere except the L303 heading, which calls them execution costs while its own first sentence says operating costs. Two of the drift sites are captions, which are read standalone, and two are the hand-off points from the method into the Results.

**Thread 3. The modifier *validation-selected* attaches to three different objects.** It qualifies the fixed-width heuristic arm (L165, L175, L277), the pool-level best heuristic chosen on validation (L204, L208, L270), and the learned arm chosen on validation (L175, L299). The collision produces two live misreadings in the Results: at L270 it reads as a property of ILMinimizer rather than as ILMinimizer being the heuristic validation picked for that pool, and at L272 it makes the learned arms read as the Fixed width arm. We recommend a distinct name for the pool-level comparator, applied at L204, L208, and L270 together.

**Thread 4. The four axes are supported by two measured contrasts and two rationale rows, and the paper must hold that scoping everywhere.** The Table~\ref{tab:axes} caption concedes it. The abstract and the introduction respect it, framing fee accounting and leak-free model selection as accounting and selection rationale. The Results opening breaks the enumeration (thread 1's cousin) and the Discussion at L349 overturns the concession outright. The Conclusion's L358 "It rests on ..." framing is correct and must not be upgraded when L358 is rewritten for the geometry-count contradiction.

**Thread 5. The 13 inputs and the discrete action set each carry three names.** The method defines *the 13-dimensional state* and *13 state variables* (L161, L208, L272), then calls them *features* at L175 and at three Discussion sites (L351, L353), which is where they are blamed for the collapse. The action set is *the discrete action set* at L161, L208, and L351, then *the finite symmetric action grid* and *the coarse action space* at L353. Both drifts terminate in the limitations paragraph, so both should be closed in the same edit that resolves the L351 against L353 contradiction.

**Thread 6. Claims that must stay identical across sections.** The geometry result is *four of six pools* (abstract, L121, L232, L342, L360) and one apparent high-tier RL advantage reversed, QR-DQN, so L121's "at least one" and L358's "each pool" are the two outliers. The oracle is ex post and future aware, and L277 states that its headroom is not a deployable episode return, so the Conclusion's "opportunity available to future controllers" must be scoped. The reward-normalization experiment is PPO alone on the matched USDC/WETH pair, so the abstract's "gated reward normalization leaves the policy single-action", L301's unscoped "collapse", and L351's definite singular "the low-fee pool" all need the same scope. The retail frontier's split runs by fee tier and conversion friction reverses no sign, so the introduction's "where operating costs erase the difference" and the Conclusion's "operating costs decides whether recentering is viable" both credit the frontier with a reversal it does not produce. The action-diversity result is first-seed only (L272), which the unscoped restatements at L272 and L299 inherit.

On acronyms and symbols the paper is in good shape after the first pass. Every acronym is defined once at its first use in reading order and never re-expanded: RL, LP, CLEB in the abstract and again at their first body use, IL, LVR, PPO in Related Work, A2C, DQN, QR-DQN, RecurrentPPO, MEV in Sec.~3, CI in the Results. No forward dependence in the strict sense survives: every forward pointer is a signposted numbered cross-reference. The one symbol collision is $L_t$ (position liquidity) against $L_s$ (pool active liquidity) inside a single fraction at Eq.~\ref{eq:swap-fees}, and one display, $Q_H(s,a)$, omits its own second argument from the right-hand side.

## Findings by section

### Abstract (raw 23, confirmed 6, rejected 14)

- [ ] **[MED] term-drift** (L82, "and gated reward normalization leaves the policy single-action") - "gated reward normalization" fuses the prespecified gate with the treatment into a method name that appears nowhere else, and the singular "the policy" has only plural antecedents while the experiment ran on PPO alone. Fix: "and PPO retrained with reward normalization stays single-action".
- [ ] **[MED] referent** (L82, "at the same decision state") - reads as one shared state, while the body says every action at *each* decision state (L119, L172, L215). Fix: "at each decision state".
- [ ] **[LOW] canonical-name** (L82, "four choices" then "all four axes") - the abstract names one object two ways in one paragraph. Fix: "sensitive to four evaluation axes that CLEB controls at once".
- [ ] **[LOW] word-reuse** (L82, "retail frontier ... high-fee frontier") - "frontier" carries the named exhibit and then a region of cells. Fix: "only descriptively favorable in most high-fee cells".
- [ ] **[LOW] claim-scope** (L82, "uniformly harmful ... only descriptively favorable") - both halves rest on cell sign counts with no Holm survivor, but only the favorable half is hedged. Fix: "is worse in every low-fee cell and only descriptively favorable in most high-fee cells".
- [ ] **[LOW] claim-scope** (L82, "CLEB recasts learned liquidity management as ...") - generalizes past the panel, while L360 scopes the identical claim. Fix: prefix "On this panel,".

### Introduction (raw 30, confirmed 18, rejected 2)

- [ ] **[MED] garbled-sentence** (L119, "separates apparent advantages that survive from evaluation artifacts") - garden path, since "survive" takes a "from" complement. Fix: "separates advantages that survive correction from artifacts of the evaluation".
- [ ] **[MED] undefined-term** (L121, "common mintable raw-tick widths") - "mintable" occurs once in the paper and is never defined. Fix: "common raw-tick widths", matching L342.
- [ ] **[MED] claim-vs-evidence** (L120, "where operating costs erase the difference") - no result shows cost erasing the RecentreWhenOut against Passive difference, and L315 states that conversion friction reverses none of the 150 signs. Fix: "and where it is adverse at every capital and gas level".
- [ ] **[MED] missing-premise** (L119, "whether a single width is optimal or the agents fail to learn") - the disjunction presupposes the single-action collapse, which Sec.~1 never states. Fix: supply the premise in the same sentence and use "fail to recover conditional control".
- [ ] **[MED] citation-attribution** (L111) - two distinct claims share one trailing bundle of three keys with generic singular subjects, against the citation-as-subject convention of L129 and L131. Fix: split the bundle across the two claims after confirming which source carries which.
- [ ] **[LOW] claim-scope** (L115, "every strategy trails holding") - the evidence is a negative mean per window. Fix: "every strategy's mean trails holding".
- [ ] **[LOW] ambiguous-referent** (L119, "a decision stump selected the same way") - the procedures are not identical, since the stump's leaf size is chosen on validation reward and it is refit on train plus validation. Fix: "selected on validation windows only".
- [ ] **[LOW] term-drift** (L119, "at each state", twice) - collides with the 13-dimensional state vector of L161. Fix: "at each decision state" in both places.
- [ ] **[LOW] consistency** (L121, "at least one apparent high-tier RL advantage") - the abstract and L342 both say one. Fix: "one apparent high-tier RL advantage".
- [ ] **[LOW] terminology-drift** (L124, "under all four controls, so that a reported gain can be separated ...") - renames the frame and restates L119 in a suppressed-agent passive. Fix: "as a reusable protocol for evaluating concentrated-liquidity RL with all four axes controlled".
- [ ] **[LOW] enumeration-order** (L120, L121) - contribution bullets 3 and 4 invert the axis order L113 just fixed. Fix: swap the two bullets.
- [ ] **[LOW] scope** (L115, "no tested policy") - the diagnostic tested four arms, not all eleven. Fix: "neither the learned policies nor a decision stump", matching L351.
- [ ] **[LOW] description-accuracy** (L120, "over capital, gas, and conversion cost") - the exhibit is a capital-by-gas grid with conversion as a separate treatment. Fix: "over capital and gas levels, with and without token-conversion cost".
- [ ] **[LOW] referent** (L113, "prevents the held-out market path ... from inflating confidence") - a market path cannot inflate anything. Fix: "keeps a single held-out market path and correlated observations from inflating confidence".
- [ ] **[LOW] redundancy** (L118 against L113) - the displacement-proxy contrast is made twice. Fix: keep the mechanism in the bullet and trim the L113 causal clause.
- [ ] **[LOW] grammar** (L113, "operating costs, gas and token conversion, determine") - the appositive first reads as a three-item subject. Fix: parentheses.
- [ ] **[LOW] active-voice** (L115, "Once these four axes are controlled") - the paper's central question opens on a suppressed-agent passive inside a first-person paragraph. Fix: "Once we control these four axes".
- [ ] **[LOW] imprecise-claim** (L111, "eventually consists of one asset") - the conversion completes at the range boundary, so the two properties hold together. Fix: delete "eventually".

### Related Work (raw 25, confirmed 6, rejected 11)

- [ ] **[MED] terminology-drift** (L131, "CLEB adds the four controls at once, ...") - second site renaming the frame, and a third full listing of the four items. Fix: "CLEB imposes all four axes at once."
- [ ] **[MED] overclaim** (L129, "Our benchmark is the empirical counterpart of these analytical results.") - the paragraph then withdraws the correspondence by noting the different reference portfolio, and the paper estimates neither cited result. Fix: "Our benchmark takes these trade-offs to logged flow."
- [ ] **[MED] redundancy** (L129, "LVR bounds ... and CLEB measures ...") - the second half restates the preceding sentence, and its closing clause is a suppressed-agent passive. Fix: keep only the contrast in the reference portfolio, in the active voice.
- [ ] **[MED] claim-grounding** (L129, "LVR bounds the cost of provision") - contradicts the definition given two sentences earlier and misstates the cited result. Fix: "measures".
- [ ] **[MED] referent** (L131, "Each is evaluated against a benchmark of its own construction") - "each" has no stated antecedent and the clause is passive. Fix: "These three studies each build their own benchmark".
- [ ] **[LOW] terminology-drift** (L131, "small retail LP actors that do not systematically modify their positions") - a fourth label for the party the paper calls provider, doubling the defined LP, plus a relative pronoun that can attach to "heuristics". Fix: "small retail providers who do not systematically modify their positions".

### Environment and Evaluation Design (raw 48, confirmed 23, rejected 17)

- [ ] **[MED] notation-collision** (L152 and Eq.~\ref{eq:swap-fees}) - $L_t$ and $L_s$ are different quantities inside one fraction, and nothing states that the pool quantity excludes the simulated position. Fix: rename the pool quantity $\Lambda_s$ and state the exclusion once.
- [ ] **[MED] consistency-term** (L150 and L165) - value-conserving rebalancing and leak-free model selection are implemented here and named nowhere in the section. Fix: attach both labels at their implementation points.
- [ ] **[MED] canonical-name** (L175, L208, L228, L281) - two shortened variants of matched executable action geometry, two of them in captions. Fix: write the full axis name at all four sites.
- [ ] **[MED] claim-grounding** (L161, "the matched-geometry rerun ... tests how conclusions depend on it") - Subsec.~\ref{sec:geometry} holds the label grid fixed and varies only the executed width, so the promised robustness check is not the one run. Fix: state what the rerun varies.
- [ ] **[MED] ambiguity** (L204, L208, "the validation-selected heuristic") - collides with the validation-selected fixed width, which is itself one of the heuristic arms. Fix: rename the pool-level comparator, for example "the best validation heuristic", and apply at L204, L208, L270.
- [ ] **[MED] sentence-structure** (L163) - a relative clause carrying a training detail splits the first two items of the five-algorithm list. Fix: move the GAE clause into its own sentence.
- [ ] **[MED] referent** (L143, "the same initial token basket") - no basket has been introduced in the subsection, and $H_t$ is the reference the whole reward rests on. Fix: "the position's initial token holdings $(x_0,y_0)$ held outside the pool".
- [ ] **[MED] undefined-term** (L177, "the disagreement fraction") - the prespecified 10% gate is stated on a quantity first explained in a Results caption. Fix: define it where the oracle is introduced at L173.
- [ ] **[LOW] terminology** (L161, "Learned policies are consulted when price leaves the range or after at most 168 hours.") - the only statement of when a policy acts never names the decision state, and L173 then refers to an interval never declared. Fix: "We consult a policy at each decision state, ...".
- [ ] **[LOW] notation** (L170, $Q_H(s,a)=\sum_h r_{t+h}$) - the right side does not depend on $a$, which the next line maximizes over. Fix: index the rewards by the counterfactual.
- [ ] **[LOW] jargon** (L175, "deployed validation reward", "read test once") - one undefined compound and one piece of internal shorthand in the stump's selection rule. Fix: spell both out.
- [ ] **[LOW] consistency-methods-results** (L177 against L301) - the declared follow-up names one treatment ingredient, the executed run reports two. Fix: add the entropy clause at L177.
- [ ] **[LOW] consistency-inference** (L204) - the Holm enumeration reads as closed while further prespecified families exist at L175, L232, and L315. Fix: make it open, naming the primary-benchmark families as the instance.
- [ ] **[LOW] consistency-term** (L175, "the same 13 features") - third word for the state inside the section that defines it. Fix: "the same 13 state variables".
- [ ] **[LOW] table-shorthand** (L196, Table~\ref{tab:setup}, "block 4") - undecoded in a table that must read standalone. Fix: "block length 4".
- [ ] **[LOW] table-prose-tension** (L193, Windows row) - implies a 6,000 hour fit while L202 refits on all five pre-test windows, that is 7,500 hours. Fix: "6,000 h initial train".
- [ ] **[LOW] clarity** (L204, "count repeated heuristic results once") - never says what makes a result repeated. Fix: "count each deterministic heuristic once rather than once per seed".
- [ ] **[LOW] active-voice** (L204, "No pooled significance test is used.") - Fix: "We run no pooled significance test."
- [ ] **[LOW] terminology** (L150 against L353) - the same omission is an exit action here and optional withdrawal there. Fix: "exit action" at both.
- [ ] **[LOW] cross-reference** (L150, "specified in the next subsection") - structural pointer where every sibling is numbered, and the target has no label. Fix: label Subsec.~3.2 and point to it.
- [ ] **[LOW] cross-reference** (L208) - one clause carries a pointer and its sibling does not. Fix: add (Subsec.~\ref{sec:retail}).
- [ ] **[LOW] caption** (L183, Table~\ref{tab:setup}) - the caption's list omits the capital setting the table carries. Fix: "data panel, capital and cost settings, ...".
- [ ] **[LOW] structure** (L167, `\paragraph{State-contingency diagnostic.}`) - the only run-in bold heading in the paper. Fix: fold the label into the opening sentence.

### Results (raw 50, confirmed 25, rejected 7)

- [ ] **[HIGH] four-axis-frame** (L215) - the enumeration announced as the four axes bundles two of them and promotes the diagnostic, which has no row in Table~\ref{tab:axes}. Fix: separate the frame from the subsection map, keeping the settled geometry-first reading order.
- [ ] **[HIGH] caption-contradiction** (L311) - the caption says the cells are after gas and conversion friction while L315 counts 150 signs "relative to gas alone", and the reward unit is missing. Fix: name the displayed treatment and add the unit, or drop "relative to gas alone" if the panels do carry conversion.
- [ ] **[MED] term-drift** (L228, L281, L232) - two captions and one sentence shorten matched executable action geometry. Fix: write the full name, in the same pass as L175 and L208.
- [ ] **[MED] term-drift** (L303, subsection heading) - "execution costs" where the axis is operating costs, contradicting the subsection's own first sentence. Fix: retitle to "Retail viability under operating costs".
- [ ] **[MED] term-collision** (L277, "All confidence limits are far above the 10% headroom threshold") - the gate is one sided and on the disagreement fraction, and "headroom" then denotes a dollar amount two sentences later. Fix: "Every lower confidence limit on the disagreement fraction is far above the declared 10% gate at both horizons".
- [ ] **[MED] logic** (L277, "One action being almost always optimal does not explain the collapse.") - the topic sentence concedes a premise the following evidence refutes. Fix: "A single action is rarely optimal, so the collapse cannot be explained that way."
- [ ] **[MED] inference-reporting** (L232) - two announced survivors, magnitudes without p-values, then a corrected p-value for a third quantity. Fix: introduce the difference in differences as its own contrast.
- [ ] **[MED] misleading-crossref** (L232) - the headline geometry claim points at a figure of paired differences that cannot show which arm has the highest mean. Fix: attach the pointer to the half the figure shows.
- [ ] **[MED] term-collision** (L272, "the learned arms operate as a single validation-selected width") - literally says the learned arms are the Fixed width arm, which Table~\ref{tab:benchmark} contradicts. Fix: "behave as a single constant width despite access to 13 state variables".
- [ ] **[MED] internal-contradiction** (L324) - the topic sentence rules out the decision schedule using a result that shows the schedule moves the outcome by \$2,448. Fix: name the mechanism instead of excluding the schedule.
- [ ] **[MED] prose** (L324) - a 48 word closing sentence, "broadly robust" diluting the term the paper anchors to block lengths, and a set of null results said to favor event-driven recentering. Fix: split, anchor the term, and say the variants do not displace it.
- [ ] **[MED] claim-scope** (L301) - PPO on one fee-tier pair generalized to the panel, with a stacked hedge. Fix: adopt L351's wording, scoped to PPO and that pair.
- [ ] **[MED] unsupported-link** (L315, "therefore reads economically only in light of the geometry test") - the connective does not follow and the mechanism is never stated. Fix: state the executed-width link, using the ratios already at L223.
- [ ] **[LOW] term-drift** (L270, "the leak-free cross-section", "the effective policy") - a second noun for the panel and an undefined subject. Fix: "Under leak-free selection, the highest-mean arm is pool- and tier-specific ...".
- [ ] **[LOW] ambiguous-modifier** (L270, "the validation-selected ILMinimizer") - reads as a property of the arm rather than as the per-pool selection. Fix: "the validation-selected heuristic for that pool, ILMinimizer".
- [ ] **[LOW] claim-scope** (L272) - the first-seed diagnostic supports an unscoped conclusion, which L299 then inherits. Fix: carry the scope, or state that the second seed behaves the same.
- [ ] **[LOW] definition-mismatch** (L324, "which consults the policy at range exit") - half the protocol of L161, and it splits the predicate from its magnitude. Fix: delete the clause.
- [ ] **[LOW] grammar** (L306, "mapping who active management helps") - wrong case, dangling participle, and a fourth label for the provider. Fix: recast as a relative clause on the figure.
- [ ] **[LOW] referent** (L299, "A validation-selected learned arm beats the stump") - turns a named prespecified contrast into an unspecified member, in a register the paper avoids. Fix: "The validation-selected learned arm improves on the stump".
- [ ] **[LOW] term-drift** (L299, "validation-based selection") - second form of the paper's modifier inside the sentence that turns on it. Fix: "validation selection".
- [ ] **[LOW] caption** (L240, Table~\ref{tab:benchmark}) - the heuristic against learned block split carries the section's headline descriptive claim and is undocumented. Fix: add one clause.
- [ ] **[LOW] caption** (L228, "red rings mark effects surviving within-pool Holm correction") - does not say at which block length. Fix: state the criterion the figure actually applies.
- [ ] **[LOW] negative-framing** (L215, "not as an additional benchmark axis") - defines the checks by what they are not, and is needed only because the enumeration mislabels the diagnostic. Fix: cut once the enumeration is repaired.
- [ ] **[LOW] caption** (L311, "transparent counterfactuals") - undefined shorthand, stated negatively. Fix: "a declared grid of counterfactual costs rather than historical gas", reusing L353's wording.
- [ ] **[LOW] wording** (L322, "Our transfer experiment contains ...") - wrong verb, and the 180 means include the 30 same-pool baseline cells without saying so. Fix: "covers ... including the same-pool baseline".

### Discussion and Limitations (raw 27, confirmed 11, rejected 7)

- [ ] **[MED] frame-drift** (L349) - one sentence names the frame twice differently, names the learned arms both architecture and algorithm, and calls the winning arm an agent although heuristics lead three pools. Fix: one rewrite covering all three, applied with L360, L124, and L131.
- [ ] **[MED] overclaim** (L349) - credits all four axes with measured changes against the concession in the caption printed above it. Fix: scope the comparative claim to geometry and operating costs and keep the other two as rationale.
- [ ] **[MED] internal-contradiction** (L353 against L351) - blames the coarse action space, which L351 shows carries substantial state-contingent opportunity. Fix: keep the finite grid as a limit on attainable performance, not as an explanation of the collapse.
- [ ] **[MED] referent-scope** (L351, "the low-fee pool") - three low-fee pools exist, and the sentence names neither PPO nor the matched pair. Fix: name both and keep the closing clause verbatim as the model for L301.
- [ ] **[LOW] terminology-drift** (L351, L353) - "features" at three sites for the 13 state variables, plus L175. Fix: state variables throughout.
- [ ] **[LOW] terminology-drift** (L353) - three names for the action set in two sentences. Fix: action set at both L353 sites.
- [ ] **[LOW] terminology-drift** (L353, "optional withdrawal") - the same omission is an exit action at L150. Fix: choose once.
- [ ] **[LOW] caption-accuracy** (L332, "with the section carrying the evidence") - misdescribes the third column and contradicts the caption's own next sentence. Fix: "with a pointer to where each is developed".
- [ ] **[LOW] filler** (L353, "Several limitations bound the result.") - announces the paragraph instead of delivering it, under a heading that already says Limitations. Fix: delete and open on the substance.
- [ ] **[LOW] dangling-modifier** (L351, "..., separating ex-post opportunity from predictable control") - the participial clause attaches to the stump, while the separation comes from the pair of results. Fix: "so the two results separate ...".
- [ ] **[LOW] active-voice** (L353, "the retail frontier varies costs") - the exhibit becomes the agent of something we do. Fix: "we vary costs on a declared grid in the retail frontier".

### Conclusion (raw 18, confirmed 9, rejected 3)

- [ ] **[HIGH] overclaim-contradiction** (L358, "changes which arm wins each pool") - contradicts L360 two lines later and four prior statements of the same result. Fix: drop the count from L358 and let L360 carry "four of six pools". Never write "each pool".
- [ ] **[MED] claim-wider-than-evidence** (L358, "imposing operating costs decides whether recentering is viable at retail scale") - L315 states that varying cost reverses no sign, and the split runs by fee tier. Fix: fold into the L358 rewrite as "mapping operating costs locates where recentering is favorable and where it is adverse".
- [ ] **[MED] terminology-drift** (L360, "With those elements in place", "these four evaluation choices") - two more nouns for the frame in the closing paragraph. Fix: use axes at both sites, with the paper-wide pass.
- [ ] **[MED] overclaim** (L360, "the opportunity available to future controllers") - asserts availability that L277 declines to demonstrate, and adds a further noun for the learned policies. Fix: "the ex-post headroom in the action set and the out-of-sample test a future agent must pass".
- [ ] **[MED] garden-path** (L360, "A heuristic leads in three pools and three learned algorithms each lead one") - the first item of a three-item enumeration carries its own "and". Fix: repunctuate the headline sentence.
- [ ] **[LOW] overclaim** (L360, "The decision stump cannot convert ...") - turns non-significant contrasts into a capability claim, against L351's own verb. Fix: "does not convert ... into a reliable held-out gain".
- [ ] **[LOW] tense-consistency** (L358, "We presented CLEB") - the only past-tense first-person framing verb in the paper. Fix: "We present".
- [ ] **[LOW] terminology-drift** (L358, "an execution-aware benchmark of RL") - reads as benchmarking the algorithms rather than as the benchmark they are evaluated on. Fix: match L82's prepositions, "for RL in concentrated liquidity provision".
- [ ] **[LOW] word-choice** (L360, "measuring new architectures and reward designs") - Fix: "evaluating".

## Suggested sequencing

We propose three groups. Nothing here has been applied.

### (a) Quick line fixes, safe in isolation

These are self-contained, length-neutral or shorter, and do not depend on any decision elsewhere: Abstract L82 referent and word-reuse. Introduction L111 (twice), L113 (three items), L115 (twice), L118, L119 garbled sentence, L120 description accuracy, L121 both. Related Work L129 (the three findings on one sentence, rewritten once) and L131 referent and provider naming. Sec.~3 L143, L152, L163, L170, L175 jargon, L183, L193, L196, L204 (all four), L208, L150 cross-reference. Results L232 inference reporting and cross-reference, L270 both, L272 both, L277 both, L299 all three, L301, L306, L311 shorthand, L315, L322, L324 all three, L240 and L228 captions. Discussion L332, L351 dangling modifier, L353 filler and active voice. Conclusion L358 tense and prepositions, L360 repunctuation, modality, and word choice.

Two of these need a check against the run before wording: the L228 ring criterion (at which block length the figure marks survivors) and the L311 caption (which cost setting the panels display). Do not guess either.

### (b) Flow and structure edits touching multiple sections

Apply each as a single pass across all its sites, in this order.

1. **The frame's noun.** Decide on *axes* and apply at L82, L124, L131, L349, and both L360 sites together.
2. **The four-axis enumeration.** Repair L215 so the four axes and the four subsections are separate lists, and cut the L215 negative-framing clause in the same edit. This is the largest structural fix and it interacts with item 1.
3. **The axis names.** Write matched executable action geometry at L175, L208, L228, L232, L281, and retitle L303 to operating costs.
4. **The validation-selected collision.** Rename the pool-level comparator and apply at L204, L208, L270 together.
5. **State variables and the action set.** Apply state variables at L175, L351, L353, and action set at both L353 sites, inside the same edit that resolves the L351 against L353 contradiction.
6. **The exit action naming**, chosen once and applied at L150 or L353.
7. **The Conclusion's closing two paragraphs.** L358 findings 1 and 2 are one rewrite, and it depends on L360 retaining "four of six pools", so edit L358 and L360 in the same commit and re-read both paragraphs afterwards. Rewrite L360 once rather than in four passes, since the frame noun, the future-agent scoping, the repunctuation, and the modality fix all land in it.
8. **Introduction bullet order**, swapping bullets 3 and 4 so the enumeration matches the L113 axis order.
9. **Length.** The confirmed set is net shorter. The only additions are the missing-premise rewrite at L119, two short definitional clauses at L152 and L173, and two one-clause caption additions at L228 and L240. The offsets are the L113 causal-clause trim, the L124 trailing clause, the "mintable" and "at least one" deletions, the L167 heading fold, the L215 clause cut, the L324 clause deletion, the L332 caption clause, and the L353 opener. Apply each group whole rather than piecemeal so the page budget stays flat.

### (c) Items needing a coauthor or a decision

1. **The L111 citation split.** Which of `heimbach2022risks`, `aquilina2024decentralised`, `uniswap2022gas` carries the retail against sophisticated profit comparison and which carries the position-management cost claim must be confirmed against the sources before the bundle is split.
2. **The L311 caption.** Whether the plotted panels are gas only or gas plus conversion is a fact about the figure, not a wording choice. If they carry conversion, "relative to gas alone" at L315 has to go instead.
3. **The L228 ring criterion.** Whether the rings mark survival at the primary block length of 4 or at all four block lengths.
4. **L272 seed scoping.** Whether the second seed also collapses to one action. If it does, the conclusion can stay unscoped and the evidence sentence should say so. If it was not inspected, the prose must carry the first-seed scope through L272 and L299.
5. **The L161 claim about the label grid.** Confirming that Subsec.~\ref{sec:geometry} is correctly described as varying the label-to-width mapping rather than the grid, and that the grid's own limitation at L353 is where the design choice is defended.
6. **The L315 geometry link.** Whether the retail frontier runs under nominal labels, since the executed-width explanation of the tier split holds only in that case.
7. **The frame noun decision itself** (group b, item 1) is a naming call the coauthors should ratify once, since it changes the abstract.
