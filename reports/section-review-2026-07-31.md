# Section-by-section review of icaif_main.tex (2026-07-31)

Method note: we reviewed the paper section by section in reading order, running review-brini (line level) and review-zeropaper (document level) in parallel on each section, then an adversarial critic verified every finding (default reject on doubt) and passed canonical terms, locked numbers, and open threads forward to the next section.

## Executive summary

Across the seven sections the two reviewers raised 231 raw findings (109 brini, 122 zeropaper). After adversarial verification and merging of duplicates, 138 findings are confirmed and 15 were rejected. By severity: 5 high, 66 medium, 67 low. No finding challenges a number or a result. The defects are naming, wording, references, and structure.

The most consequential issues, major first:

1. **[HIGH]** Table `tab:setup` is an orphan float. Its label is never referenced anywhere in the paper (Environment, L172 to L175).
2. **[HIGH]** The benchmark table's heuristic row names ("Recenter when out", "IL minimizer", "Reactive") diverge from the canonical prose arm names used in the same section, blocking the row-to-result mapping in the paper's main exhibit (Results, L245 to L248).
3. **[HIGH]** The retail subsection points to "the geometry test below" although that subsection sits above it under the ratified order, a stale pointer from a prior ordering (Results, L304).
4. **[HIGH]** The heuristic roster never introduces the canonical arm names RecentreWhenOut, ILMinimizer, and ReactiveRecentering that Results and the tables use (Environment, L156).
5. **[HIGH]** The abstract writes the range "67.0--73.2\%" with an en dash, which the no-dashes rule bans for ranges (Abstract, L71).
6. **[MED, systemic]** The four canonical axes (fee accounting, leak-free model selection, executable action geometry, operating costs) drift off canon at roughly a dozen sites in every section, and both the abstract's closing list and the Conclusion's second list swap operating costs out of the frame the paper opens with (Abstract L71, Intro L102/L104/L115, Related Work L122, Results L204, Discussion tab:axes and L337, Conclusion L346/L348).
7. **[MED, systemic]** The oracle loses its canonical name in the very section that reports its results (Results writes "the ex-post best action" at L266/L270) and its canonical verb in the closing sections ("changes" instead of "disagrees with the validation-selected fixed width" at L339 and L348).
8. **[MED]** The 10\% headroom threshold is presented as "prespecified" but occurs only once in the paper, and the methods diagnostic prespecifies four contrasts and no threshold (Results, L266).

## Cross-section flow threads

These threads emerged only because the sections were read in order with context passed forward. Each requires one coordinated edit, not per-section patches.

**1. The four-axis frame.** The paper's central organizing device is a set of four evaluation axes, but only "operating costs" survives under its canonical name across sections. The drift spans every section: "realizable accounting" and "correct accounting" for fee accounting, "validation-only selection", "out-of-sample selection", and "selection discipline" for leak-free model selection, "executable range geometry", "executable percentage-width geometry", "matched executable geometry", "executable comparability", and "economically comparable actions" for executable action geometry. Worse, the frame itself breaks at both ends: the abstract's closing list swaps operating costs for dependence-aware inference, the intro's question and close list three axes against an explicit "all four" claim, and the Conclusion's second list swaps operating costs for "economically meaningful baselines". One coordinated edit must reach L102, L104, L115, L122, L204, the tab:axes Axis column (L329 to L332), L337, both L346 lists, and both L348 recaps, with exactly one enumeration surviving in the Conclusion.

**2. Canonical object names.** Several load-bearing objects carry two or three names across sections. The oracle: named only in methods, paraphrased as "the ex-post best action" in Results, given the wrong verb ("changes") in Discussion and Conclusion. The heuristic arms: the roster at L156 defines them descriptively, Results prose uses RecentreWhenOut, ILMinimizer, ReactiveRecentering, the benchmark table uses third variants, and L304 adds "Active recentering". The holding baseline: "initial hold basket" (L120, L229 caption) against the methods' canonical "the same initial token basket if held outside the pool". Also drifting: the bootstrap name (three variants), the collapse ("the one-action result" coined in Discussion), CI notation (the "to" form versus brackets, mixed dollar signs, leading plus and minus signs in prose), and block-length style (words versus digits). Each object needs one ratified form applied everywhere.

**3. Locked results need one phrasing each.** The matched-geometry result appears as "flips", "changes", "reorders", and "shifts" the pool winners, and the QR-DQN result as both "reverses ... strength" and "removes ... advantage". The two significant benchmark contrasts are correctly hedged as fragile to block length in Results and Discussion, and the headroom result is correctly kept as ex-post opportunity rather than achievable profit everywhere. The apply pass must conform the phrasing without touching any value, and must preserve the hedges.

**4. The seed-averaging wording defect.** "Learned seeds are averaged within windows" (the seeds are not averaged, performance across seeds is) appears at five sites: the abstract, L102 and L110 in the intro, L186 and L195 in methods, and the L229 table caption. One rewording, applied at all five.

**5. Acronym scope and debt.** The abstract is its own scope, and PPO and QR-DQN are undefined there. In the body, PPO, A2C, DQN, and QR-DQN reach first use at L110 undefined while their expansions sit at second use at L154, so exactly one site must keep the expansions. IL is used without definition (L156) though "impermanent loss" is spelled out at L120 without the acronym attached. CI is defined at its second use (L259) instead of its first (L220). ADXR, BOP, and DX are never expanded at all.

**6. Sequential-reading risks.** Three spots lean on later content: the diagnostic's motivation at L159 presupposes the policy collapse that only Results establishes, L210 presupposes QR-DQN's high-tier strength that only the benchmark table (which follows under the ratified order) shows, and the intro's "loss mitigation" question presupposes the every-strategy-trails-holding headline it never teases even though the intro is the one section licensed to tease it. All three are repaired with signposting clauses, not reordering.

**7. Redundant closers and the page budget.** The same wrap-up claim lands twice at L115 and L122, the Discussion paragraph at L337 restates two tab:axes rows and its own second sentence and nearly repeats the table caption, and the Conclusion enumerates the axes three times in two paragraphs. Cutting these duplicates is the natural harvest for the tighten pass against the roughly 0.35-page overage.

**8. Process-history leaks.** "The implemented legacy specification" (L152), "the original comparator families" (L156), and "the paper-compatible extractor" (L313) all narrate development history with referents no reader can resolve. Each must be rewritten as a final-state assertion anchored to existing cite keys or flagged with a % CHECK note.

## Findings by section

### Abstract

Raw 18 (6 brini, 12 zeropaper), 13 confirmed after merging, 1 rejected.

- [ ] **[HIGH] dash-punctuation** (67.0--73.2\%): en dash used for a range, banned. Fix: write "in 67.0\% to 73.2\% of decision states across pools".
- [ ] **[MED] acronym-undefined** (PPO, QR-DQN): both appear undefined in the abstract, which is its own acronym scope. Fix: "gated reward-normalized Proximal Policy Optimization (PPO)", and spell out or genericize the single QR-DQN use.
- [ ] **[MED] term-drift** (executable range geometry / executable percentage-width geometry): two non-canonical variants of the axis name in one paragraph. Fix: use "executable action geometry" in the axis list and tie the matched-geometry sentence back to it.
- [ ] **[MED] term-drift** (closing four-item list): the close swaps operating costs for dependence-aware inference, breaking the frame the abstract opens. Fix: mirror the four canonical axes, naming dependence-aware inference as an added element if it stays.
- [ ] **[MED] active-voice** ("a diagnostic evaluates every action"): dangling purpose clause, the diagnostic does not determine anything. Fix: "we run a diagnostic that evaluates every action at the same decision state".
- [ ] **[MED] clarity** ("the tested causal and learning policies"): "causal" has no referent in the abstract. Fix: "but neither the decision stump nor the learned policies capture it predictably".
- [ ] **[MED] clarity** ("averaging random seeds within windows"): literally says the seed values are averaged. Fix: "averaging across random seeds within each window".
- [ ] **[MED] sentence-length** (opening sentence): 61 words chaining definition, framing, sensitivity claim, and consequence. Fix: split after "automated market makers" and let the second sentence absorb "CLEB controls all four of these axes at once".
- [ ] **[LOW] redundancy** ("each of which can materially change the measured conclusion"): states the sensitivity twice. Fix: keep one formulation, drop "very" and the trailing clause, fold into the opening split.
- [ ] **[LOW] active-voice** ("The benchmark is built from"): passive with a clear agent. Fix: "We build the benchmark from 22.6 million Ethereum mainnet swaps ...".
- [ ] **[LOW] clarity** ("logged deterministic learned policies"): three stacked modifiers. Fix: "the learned policies, evaluated deterministically, select only one action per pool in the logs".
- [ ] **[LOW] term-drift** ("The 168-hour ex-post best action"): the canonical name "oracle" never appears. Fix: bridge once ("the 168-hour ex-post oracle action") or ensure the body's first "oracle" connects to this phrase.
- [ ] **[LOW] clarity** ("A retail frontier"): coined term with no gloss of what varies. Fix: "A frontier over retail-scale operating costs shows ...".

### Introduction

Raw 30 (13 brini, 17 zeropaper), 18 confirmed after merging, 2 rejected.

- [ ] **[MED] citations** (L100): singular filler subjects ("Empirical work finds", "practitioner evidence identifies") with three cite keys bundled over two claims. Fix: restructure with the citations as plural subjects and split the bundle by claim, verifying the key-to-claim mapping first.
- [ ] **[MED] consistency** (L104): the question conditions on three axes, the next sentence claims "all four". Fix: add operating costs or rewrite so the count matches, coordinated with the L104 redundancy cut.
- [ ] **[MED] consistency** (L115): the intro's close recaps only three axes. Fix: close on all four canonical axis names.
- [ ] **[MED] terminology** (L102/L104/L115 "realizable accounting" / "correct accounting"): the fee-accounting axis never appears under its canonical name. Fix: one canonical name at all three sites.
- [ ] **[MED] terminology** (L102/L104/L110/L115): four surface forms for leak-free model selection. Fix: canonical name at L102, L104, L115, eliminate "out-of-sample selection".
- [ ] **[MED] terminology** (L102/L104/L112/L115): three names for the geometry axis. Fix: "executable action geometry" everywhere, "economically comparable" at most once as a gloss.
- [ ] **[MED] acronyms** (L110): PPO, A2C, DQN, QR-DQN undefined at first body use. Fix: spell out here, or leave "five RL algorithms" and define at the methods roster, one site only.
- [ ] **[MED] voice** (L110 "averages seeds within windows"): misstates what is averaged. Fix: "averages performance across seeds within windows", align the L102 noun form.
- [ ] **[MED] referent** (L110 oracle sentence): the oracle arrives unglossed, "changes the action" states no baseline, "that headroom" has no antecedent. Fix: ground all three ("An ex-post oracle that picks the best action at each state frequently disagrees with the validation-selected fixed width, revealing state-contingent headroom ...").
- [ ] **[MED] redundancy** (L102/L104): "controls all four" stated twice with the reveal order inverted. Fix: state once, need before artifact.
- [ ] **[MED] referee-headline** (L104): the "loss mitigation" question presupposes the untold headline. Fix: add one teasing sentence (every strategy trails holding, no deployable policy captures the headroom).
- [ ] **[LOW] flow** (L102 "The measured answer"): refers to an answer whose question comes later. Fix: name the implicit question or move the L104 question earlier.
- [ ] **[LOW] clarity** (L102 safeguards sentence): three safeguards zipped against two threats, mechanism misstated. Fix: map safeguard to threat explicitly.
- [ ] **[LOW] voice** (L104 "intersection of benchmarking, robustness, evaluation, and RL"): avoid-word "robustness" plus vague placement. Fix: cut the clause or replace with "evaluation methodology for financial RL".
- [ ] **[LOW] redundancy** (L109): the displacement-proxy foil negated twice in one bullet. Fix: keep one contrast.
- [ ] **[LOW] restatement** (L115 vs L122): near-duplicate closing clauses across adjacent sections. Fix: keep L115, compress L122 to its transfer-testing content in the tighten pass.
- [ ] **[LOW] prose** (L100 "creating an operational decision"): decisions are made, not created. Fix: "creates an ongoing position-management problem".
- [ ] **[LOW] prose** (L111 retail-frontier bullet): three stacked prepositional phrases invite misparse. Fix: restructure ("a retail frontier that varies capital, gas, and conversion cost, comparing out-of-range recentering with leaving the original LP position unchanged").

### Related Work

Raw 26 (11 brini, 15 zeropaper), 15 confirmed after merging, 2 rejected.

- [ ] **[MED] citation-subject** (L120 opening): "Research ... characterizes" filler subject with a five-key bundle over three claims. Fix: citations as subject with plural verb, keys attached to claims.
- [ ] **[MED] citation-subject** (L122): passive "RL has been applied" plus "the literature documents" filler. Fix: "\cite{zhang2023adaptive,xu2025improving,anchuri2026rammstein} apply RL directly to Uniswap v3 range control, and \cite{brini2023deep,brini2026deep} document ...".
- [ ] **[MED] term-drift** (L122 contribution list): three of four axes off canon. Fix: canonical names, coordinated with the paper-wide axis edit.
- [ ] **[MED] referee-gap** (L122): the three nearest competitor papers get no characterization and no stated gap. Fix: add one verified clause naming what their evaluations omit (% CHECK if sources unavailable).
- [ ] **[MED] clarity** (L120 "who dynamically adjust ... and obtain"): misattachment assigns the closed-form width to the LPs rather than the authors. Fix: restructure with singular "adjusts" and a comma before "and obtain".
- [ ] **[LOW] restatement** (L122 closer): near-duplicates the L115 intro closer. Fix: coordinate so the four-axis frame survives in exactly one closer.
- [ ] **[LOW] restatement** (L122 roster sentence): restates the L110 contribution bullet. Fix: trim to the positioning clause, let L110 carry the roster.
- [ ] **[LOW] term-drift** (L120 "selection discipline"): another variant of the selection axis, plus a passive. Fix: "once CLEB imposes gas costs and leak-free model selection".
- [ ] **[LOW] scope-overclaim** (L122 "the broader financial-RL literature"): two same-group cites cannot carry a literature-wide claim. Fix: scope to the cited work (the citation-subject rewrite does this).
- [ ] **[LOW] terminology** (L120 "initial hold basket" / "holding"): two names for the baseline object. Fix: conform to the methods' canonical form, check the table caption in the same pass.
- [ ] **[LOW] citation-mapping** (L120 analytical bundle): three keys over two activities. Fix: attach keys to the activity each supports, verified first.
- [ ] **[LOW] sentence-structure** (L120 LVR sentence): a 20-word appositive strands the "and derive" coordination. Fix: split after the appositive.
- [ ] **[LOW] repetition** (L120 "under modeled price dynamics"): verbatim repeat within the paragraph. Fix: merge the recap into the pivot sentence.
- [ ] **[LOW] anticipatory-colon** (L120 "The two views are complementary:"): banned tease-then-reveal. Fix: replace the colon with a period.
- [ ] **[LOW] hyphenation** (L120 "better informed arbitrageurs"): compound modifier unhyphenated. Fix: "better-informed arbitrageurs".

### Environment and Evaluation Design

Raw 54 (29 brini, 25 zeropaper), 31 confirmed after merging, 5 rejected.

- [ ] **[HIGH] unreferenced-float** (L172 to L175): tab:setup is never referenced. Fix: add "Table~\ref{tab:setup} summarizes the design." at the walk-forward paragraph.
- [ ] **[HIGH] canonical-names** (L156): the roster never introduces RecentreWhenOut, ILMinimizer, ReactiveRecentering, which Results uses. Fix: attach the canonical name to each arm at definition and use only those names afterward.
- [ ] **[MED] acronyms** (L156 "IL-minimizing"): IL undefined at first use. Fix: "an impermanent-loss (IL) minimizing width", coordinated with the roster fix.
- [ ] **[MED] acronyms** (L152 ADXR, BOP, DX): never expanded anywhere. Fix: expand once at first use.
- [ ] **[MED] process-history** (L152 "the implemented legacy specification"): development-history narration with no resolvable referent. Fix: assert the final state or anchor to the comparator implementations via existing cite keys (% CHECK if unverifiable).
- [ ] **[MED] clarity** (L156 "the original comparator families" / "the first four active policies"): unresolvable referent, and "first four" wrongly sweeps in Passive. Fix: name the four active arms explicitly and state the source.
- [ ] **[MED] term-drift** (L166, L199): "matched executable geometry" and "nominal and matched action geometry". Fix: "matched executable action geometry", in the coordinated axis edit.
- [ ] **[MED] seed-averaging** (L186, L195): "seeds are averaged" wording again, plus a passive. Fix: "We average learned-policy performance across seeds within that window", conform the table row.
- [ ] **[MED] flow-sequential** (L159): the diagnostic's motivation presupposes the collapse result Results establishes. Fix: signposting clause ("Learned deterministic policies turn out to execute a single width per pool on test (Sec.~\ref{sec:results}) ...").
- [ ] **[MED] active-voice** (L166): three chained passives. Fix: "We select the minimum leaf size ..., refit the stump ..., and read test once."
- [ ] **[MED] ambiguity** (L141 "relative to holding and management costs"): misparse garbles the telescoping decomposition. Fix: "cumulative fees, minus the increase in loss relative to holding, minus management costs".
- [ ] **[MED] voice** (L141 semicolon): banned in prose. Fix: split into two sentences.
- [ ] **[MED] notation** (L128): Eq. (1) uses $p_t$ and $L_t$ before either is defined. Fix: one orientation sentence before the equation.
- [ ] **[MED] notation** (L143 bare $L$): one quantity, two symbols. Fix: write $L_t$ or state the relation.
- [ ] **[MED] captions** (L174 "Experimental design."): below the self-contained bar. Fix: expand to name scope, contents, and the retail-frontier variation.
- [ ] **[MED] defensibility** (L152 action set and trigger): load-bearing choices stated bare with only the "legacy" anchor. Fix: give a justification path (own contribution with the matched-geometry rerun as robustness, or standard practice anchored to the cited RL-for-Uniswap papers), jointly with the legacy rewrite.
- [ ] **[MED] clarity** (L168 "reward-only VecNormalize"): library class and flag in prose, unnamed $\gamma$, missing leading zero. Fix: describe the operation, name the class at most once in a footnote.
- [ ] **[LOW] clarity** (L168 "compare normalized against standard PPO"): elliptical noun. Fix: "compare normalized PPO against standard PPO, paired within each held-out window".
- [ ] **[LOW] voice** (L154 "randomly fixed reproducibility seeds"): self-contradictory parenthetical. Fix: "with two fixed seeds, 42 and 123".
- [ ] **[LOW] acronyms** (L154 expansions): sit at second body use given L110. Fix: settle one expansion site jointly with the intro fix.
- [ ] **[LOW] clarity** (L154 "networks $[4,2]$"): bracket notation undecoded. Fix: "hidden-layer widths".
- [ ] **[LOW] voice** (L156 "is selected on validation (and not on test)"): passive plus a says-what-it-is-not parenthetical. Fix: "We select every free configuration on validation windows only."
- [ ] **[LOW] consistency** (L164 "non-deployable diagnostic"): overloads the canonical term "diagnostic". Fix: "a non-deployable reference".
- [ ] **[LOW] active-voice** (L164 "are used"): passive. Fix: "We use only states with a complete horizon remaining."
- [ ] **[LOW] active-voice** (L193 "The panel is built"): passive opener. Fix: "We build the panel from ...".
- [ ] **[LOW] voice** (L193 "The same strict protocol is used"): passive plus self-praising "strict". Fix: "Cross-pool transfer uses the same protocol ...".
- [ ] **[LOW] consistency** (L195 block lengths words vs digits): mixed styles. Fix: digits throughout.
- [ ] **[LOW] term-drift** (L195 bootstrap name): three variants across prose, table, abstract. Fix: "paired circular moving-block bootstrap" everywhere, seeding mentioned once.
- [ ] **[LOW] notation** (L161 Eq. (4)): index $t$ never tied to state $s$. Fix: "At each decision state $s$, reached at hour $t$, ...".
- [ ] **[LOW] notation** (L137 Eq. (2) $a_t$): action symbol used one subsection early. Fix: signposting half-clause at the equation.
- [ ] **[LOW] clarity** (L134 "the usual one-sided inventory"): vague shorthand. Fix: state the mechanism (all token0 above, all token1 below).

### Results

Raw 51 (25 brini, 26 zeropaper), 30 confirmed after merging, 3 rejected.

- [ ] **[HIGH] stale-cross-reference** (L304 "the geometry test below"): the geometry subsection sits above. Fix: "the geometry test of Sec.~\ref{sec:geometry}", optionally "in light of".
- [ ] **[HIGH] terminology-drift** (L245 to L248 table rows): four arm names diverge from the section's own prose names, and "Recenter when out" spells the arm differently. Fix: rename rows to the canonical arm names in the same edit as the L156 roster.
- [ ] **[MED] axis-name-drift** (L204 to L205): "benchmark controls" plus three off-canon axis names, with the diagnostic interleaved as if it were an axis. Fix: open with the four canonical names mapped in order onto the subsections, diagnostic as an additional test.
- [ ] **[MED] terminology-drift** (L266, L270 "the ex-post best action"): "oracle" absent from the section reporting the oracle results. Fix: "the oracle (the ex-post best action of Sec.~3)" at first mention, then "oracle".
- [ ] **[MED] undefined-referent** (L266 "the prespecified 10\% headroom threshold"): only occurrence in the paper, prespecification unverifiable. Fix: state the threshold in Sec. 3 with the four contrasts, or drop "prespecified" and introduce it here as a design choice.
- [ ] **[MED] ambiguity** (L266 "Policy collapse does not occur because ..."): negation scope, the natural parse denies the collapse. Fix: "The collapse is not explained by one action being almost always optimal."
- [ ] **[MED] acronym-first-use** (L220 vs L259): CI expanded at second use. Fix: move the expansion to L220.
- [ ] **[MED] caption-wording** (L229 "Learned seeds are averaged"): the seed-averaging defect again. Fix: same rewording as L186 and L195.
- [ ] **[MED] terminology-drift** (L229 "the initial hold basket"): superseded baseline name. Fix: "Every mean remains below holding."
- [ ] **[MED] caption-incomplete** (L229): the caption omits which action geometry the means use, right after the subsection making geometry load-bearing. Fix: add one clause (verify nominal versus matched against the committed run first).
- [ ] **[MED] formulaic-opener** (L210, L259, L295, L311): four of five subsections open on the same foil-then-reversal template. Fix: keep the foil only at L210, open the others directly on the finding.
- [ ] **[MED] say-once** (L210 vs L220): "four of six pools" stated twice, the pitfall claim three times. Fix: keep the full claim once at L220, reduce L210 to the mechanism setup.
- [ ] **[MED] sequential-reading** (L210 QR-DQN): presupposes the high-tier strength the benchmark table establishes later. Fix: signpost "(Sec.~\ref{sec:benchmark})", no reorder.
- [ ] **[MED] undefined-referent** (L313 "shadow and LVR shaping, and the paper-compatible extractor"): three variants with no definition, one leaking development history. Fix: gloss each in a half-sentence with a source or % CHECK, or cut the unglossed items.
- [ ] **[MED] notation-consistency** (L220/L259/L288/L290/L311): two CI notations, inconsistent brackets, drifting interval name, leading signs and a USD postfix in prose. Fix: "to" form with \$ prefixes in prose, brackets only in tables, one interval name, no leading signs, values untouched.
- [ ] **[MED] undefined-referent** (L311 "ratio-based ``transfer efficiency''"): attacks a metric the paper never defines or computes. Fix: cut the foil and open on the experiment, or define whose metric it is.
- [ ] **[MED] vague-shorthand** (L313 "the execution interpretation"): undefined compressed label as a topic sentence. Fix: spell the thesis out once (execution costs from over-trading, not schedule or state, drive the losses).
- [ ] **[MED] defensibility** (L261 "first-seed deterministic policy"): the collapse evidence restricted to seed 42 without justification. Fix: verify the second seed against the run logs and report it, or state plainly that only seed 42 was logged. Never assert unchecked.
- [ ] **[LOW] grammar** (L311): comma splice on the PPO favorable direction. Fix: "In PPO's one robust favorable direction, training on WETH/USDT 0.05\% reduces loss ...".
- [ ] **[LOW] terminology-drift** (L304 "Active recentering", "block-four"): arm-name variant plus hyphenated block length. Fix: RecentreWhenOut, "at the primary block length of 4", conform L259's word style to digits.
- [ ] **[LOW] ambiguity** (L304 cell counts): tier qualifier attaches only to the last pool. Fix: "in the 0.30\% pools ... 22/25 (USDC/WETH), 22/25 (WBTC/WETH), and 23/25 (WETH/USDT)".
- [ ] **[LOW] ambiguity** (L266 $p_{\mathrm{Holm}}=.0004$): one p-value over many contrasts without stating it is shared. Fix: clarify scope in prose (verify which reading is true against the runs).
- [ ] **[LOW] clarity** (L288 unnamed winning arm): the one significant contrast whose winner goes unnamed. Fix: name the algorithm after verifying against the run logs.
- [ ] **[LOW] defensibility** (L290 "this standard remedy"): standard-practice claim without an anchor. Fix: soften to "this common adjustment" or leave a % JUSTIFY note.
- [ ] **[LOW] redundancy** (L311 transfer framing plus taylor2009transfer): repeats Related Work. Fix: drop the framing clause and the duplicate cite.
- [ ] **[LOW] terminology-drift** (L313 "event-driven"): new label for the Sec. 3 trigger, untied. Fix: tie the label to the trigger at first use or avoid it.
- [ ] **[LOW] register** (L288 "the convenient explanation", L210 "is the pitfall"): editorializing and a presupposing definite article. Fix: neutral phrasing, folded into the opener edits.
- [ ] **[LOW] passive-voice** (L295 "The active width is selected"): passive. Fix: "We select the active width on the preceding validation window."
- [ ] **[LOW] passive-voice** (L205 "are reported as robustness checks"): passive plus a second name for the L308 block. Fix: "We report schedule and transfer experiments as sensitivity checks", matching the heading.
- [ ] **[LOW] undefined-referent** (L311 "native PPO"): unglossed qualifier. Fix: "PPO trained on the target pool" at first use.

### Discussion and Limitations

Raw 29 (14 brini, 15 zeropaper), 18 confirmed after merging, 1 rejected.

- [ ] **[MED] naming-consistency** (tab:axes, L329 to L332): three of four Axis rows off canon. Fix: rename to the four canonical axis names, letting the table anchor the paper-wide edit.
- [ ] **[MED] naming-consistency** (L337 list): two of four names off canon, non-parallel fourth item. Fix: parallel list of the four canonical names.
- [ ] **[MED] redundancy** (L337): the sentence restates tab:axes rows 1 and 3 nearly verbatim next to the table pointer. Fix: cut the duplicating sentence, let the table carry the per-axis evidence.
- [ ] **[MED] naming-consistency** (L339/L341): three surface names for the collapse, "the one-action result" a new coinage. Fix: "the collapse" (or "policy collapse") in all three spots.
- [ ] **[MED] dangling-referent** (L339 "the corrected action set"): the paper corrects the evaluation, never the action set. Fix: "under the corrected evaluation, the discrete action set contains ...".
- [ ] **[LOW] claim-grounding** (L330 "two prespecified survivors"): the comparisons were prespecified, survival was not. Fix: "leaves two of the prespecified comparisons as survivors, each fragile to block length".
- [ ] **[LOW] consistency** (L331/L337 "removes ... advantage"): drifts from the locked "reverses ... apparent high-tier strength". Fix: conform to one form paper-wide.
- [ ] **[LOW] consistency** (L337 "reorders the pool winners"): second phrasing of the locked four-of-six result within the section. Fix: "changes the highest-mean strategy in four of six pools" (the L337 instance disappears with the redundancy cut).
- [ ] **[LOW] word-choice** (L339 "The oracle changes the fixed width"): wrong verb and dropped qualifier. Fix: "The oracle disagrees with the validation-selected fixed width in roughly two-thirds to three-quarters of decision states", conform L348 in the same edit.
- [ ] **[LOW] negative-first** (L339 opener): fifth instance of the negative-first template. Fix: lead with the positive claim, keep the guard as a consequence.
- [ ] **[LOW] active-voice** (L339 "is therefore best interpreted"): passive interpretation. Fix: "We therefore read the learned agents' static behavior as ...".
- [ ] **[LOW] word-choice** (L339 "worsens the low-fee pool"): object ellipsis. Fix: "worsens performance in the low-fee pool".
- [ ] **[LOW] active-voice** (L341 "Primary gas is fixed"): passive plus compressed qualifier. Fix: "We fix gas at \$5 across the sample in the primary benchmark."
- [ ] **[LOW] redundancy** (L341 "future diagnostics" twice): brackets the paragraph and collides with the canonical term "diagnostic". Fix: say it once, "remain for future work".
- [ ] **[LOW] word-choice** (L332 "The retail frontier is adverse"): the frontier is the grid, recentering is what is adverse. Fix: "Recentering is adverse in every low-fee cell of the retail frontier ...".
- [ ] **[LOW] redundancy** (L337 "Which agent a provider would prefer ..."): restates the axes-dominate claim. Fix: cut or merge.
- [ ] **[LOW] redundancy** (L337 table pointer): repeats the caption nearly verbatim. Fix: "Table~\ref{tab:axes} summarizes the four axes."
- [ ] **[LOW] filler** (L341 "transparently"): self-evaluating adverb. Fix: "varies costs on a declared grid".

### Conclusion

Raw 23 (11 brini, 12 zeropaper), 13 confirmed after merging, 1 rejected.

- [ ] **[MED] axis-name-drift** (L346 first list): three of four axes off canon. Fix: fold into the coordinated canonical-axis edit.
- [ ] **[MED] frame-mismatch** (L346 "namely" list): drops operating costs for "economically meaningful baselines". Fix: close on the same four canonical axes, or cut this second list entirely.
- [ ] **[MED] redundancy** (L346): the same four axes enumerated twice in adjacent sentences under two naming sets. Fix: keep one canonical list in the "It rests on" sentence.
- [ ] **[MED] locked-phrasing-drift** (L348 "The oracle changes the action frequently"): wrong verb, dropped comparator. Fix: "The oracle disagrees with the validation-selected fixed width in most decision states."
- [ ] **[MED] axis-name-drift** (L348): "executable geometry", bare "accounting" and "selection", plus a passive. Fix: canonical names or a backward pointer to the surviving L346 list, active voice.
- [ ] **[MED] restatement** (L348 closing recap): third full axis enumeration in the Conclusion, about 45 words. Fix: compress to "By pinning down these four evaluation choices, the benchmark gives future work ... a common ground ...".
- [ ] **[MED] overclaim** (L346 "Each of these choices flips a conclusion"): fee accounting sets levels, it does not flip winners. Fix: scope the claim, keep the flip verb only for geometry.
- [ ] **[LOW] locked-phrasing-drift** (L348 "cross-tier rankings shift"): third phrasing of the matched-geometry result. Fix: conform to "changes the highest-mean strategy in four of six pools".
- [ ] **[LOW] category-mismatch** (L348 "the effective policy ... is heterogeneous loss mitigation"): a policy is a controller, not a task. Fix: "the effective task across this panel is heterogeneous loss mitigation, not universal profitable control".
- [ ] **[LOW] negative-first-scaffolding** (L348 "not merely a negative result"): announcement sentence with a fuzzy "this". Fix: delete, the next sentence already carries the content.
- [ ] **[LOW] referent** (L348 "that ex-post headroom" / "a decision stump"): no antecedent for the headroom, indefinite article overgeneralizes the one tested stump. Fix: introduce the quantity and use "the decision stump".
- [ ] **[LOW] framing** (L346 opening): CLEB unnamed until midway through the second paragraph. Fix: "We presented CLEB, an execution-aware benchmark ..." up front.
- [ ] **[LOW] active-voice** (L348 "can be measured"): passive final clause. Fix: "on which future work can measure new architectures and reward designs".

## Suggested sequencing

We apply nothing here. The confirmed findings group into three passes.

### (a) Quick line fixes (single-site, mechanical)

The abstract en dash range, the L120 hyphenation, the L311 comma splice, the L141 semicolon, the anticipatory colon at L120, the single-site passives (L164, L193, L205, L295, L339, L341, L348), the fillers and register items ("strict", "transparently", "convenient", "randomly fixed"), the leading-zero and bracket-notation repairs (L154, L168), the ADXR/BOP/DX expansions, the CI expansion move to L220, the IL definition, the object-ellipsis and category fixes (L332, L339, L348), the "native PPO" gloss, the L304 cell-count qualifier, the tab:setup reference, and the caption expansion at L174.

### (b) Flow and structure edits touching multiple sections (one coordinated edit each)

1. The canonical-axis edit across L102, L104, L115, L122, L204, tab:axes (L329 to L332), L337, both L346 lists, and both L348 recaps, restoring the four-axis frame at both ends of the paper (abstract close, Conclusion second list).
2. The heuristic arm-name edit: introduce RecentreWhenOut, ILMinimizer, ReactiveRecentering at L156, conform the benchmark table rows (L245 to L248) and L304.
3. The oracle edit: name at L266/L270, verb "disagrees with the validation-selected fixed width" at L339 and L348, gloss at first body use (L110).
4. The seed-averaging rewording at its five sites (abstract, L102/L110, L186, L195, L229).
5. The holding-baseline conforming edit (L120, L229) to the methods' canonical form.
6. The locked-phrasing conforming edit for the matched-geometry and QR-DQN results (abstract, L210/L220, L331, L337, L348).
7. The CI-notation and block-length style edit ("to" form, \$ prefixes, no leading signs in prose, digits for block lengths, one bootstrap name).
8. The closer dedup: L115 versus L122, the L337 cuts, and the Conclusion's triple enumeration, harvested by the tighten pass against the 0.35-page overage.
9. The sequential-reading signposts (L159 collapse premise, L210 QR-DQN, the intro's headline tease) and the L304 stale-pointer repair.
10. The formulaic-opener rewrites (keep the foil at L210 only) and the abstract opening-sentence split.
11. The acronym-site decision: exactly one of L110 and L154 keeps the PPO/A2C/DQN/QR-DQN expansions, and the abstract defines PPO and handles QR-DQN in its own scope.

### (c) Items needing verification or a decision before editing

1. The 10\% headroom threshold: decide where it is stated (Sec. 3 with the four contrasts, or introduced at L266 as a design choice without "prespecified").
2. The first-seed action-diversity evidence (L261): check the second seed in the run logs before writing anything.
3. The unnamed winning learned arm at L288: read it off the logs, never guess.
4. The p-value scope at L266: verify against the committed runs whether .0004 is shared, a maximum, or per contrast.
5. The tab:benchmark geometry clause (L229): verify nominal versus matched against the committed run.
6. The referee-gap clause for zhang2023adaptive, xu2025improving, anchuri2026rammstein (L122): draft only after checking those sources, % CHECK if unavailable.
7. The "legacy specification" and "original comparator families" rewrites (L152, L156) and the "paper-compatible extractor" gloss (L313): each needs a source via existing cite keys or a % CHECK note, plus the action-set and trigger defensibility path.
8. The "standard remedy" claim (L290): soften or anchor, given the frozen bibliography.
9. Pre-submission note, no edit now: the acks and author blocks print by design under the current class options, and adding the anonymous option at submission hides both automatically.
