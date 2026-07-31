# Section-by-section review: icaif_main.tex

Method: we reviewed the paper one section at a time in reading order. For each section we ran review-brini (line-level: voice, acronyms, captions, active voice, figures, citations) and review-zeropaper (document-level: prose economy, consistency, referee, defensibility, claim-grounding, page budget) in parallel, passed every proposed finding through an adversarial critic that verified it against the text (default reject on doubt) and checked flow against the prior section, and carried each section's canonical terms, symbols, and open threads forward to the next. This synthesis consolidates the verified output.

## Executive summary

We confirmed 86 findings across the seven prose sections: 0 critical, 0 high, 32 medium, 54 low. No section produced a claim-grounding break against the locked numbers except one Conclusion overclaim. The dominant pattern is terminology and axis-count drift that only surfaces when the paper is read top to bottom.

The most consequential issues, major first:

- **[MED] The deployable contextual arm has no stable name.** It is called "policy stump" (abstract, intro), "exact one-split policy tree" / "contextual policy" (Methods L165), "shallow causal policy" (Results L274), "deployable policy stump" (Discussion L319), and "a simple deployable contextual policy" (Conclusion L328). One object, six labels.
- **[MED] The benchmark's axis count is inconsistent: four vs five.** Abstract L70 and intro L108-111 commit to four axes; Related Work L121 and Conclusion L326 list five (adding "dependence-aware inference"); Discussion L317 enumerates two different four-item sets in one paragraph so "these four choices" has no stable antecedent.
- **[MED] Two Methods passages depend on Results.** L167 states headroom "is present at both horizons" and L158 leans on the one-action-collapse result, both reported only at L240/L245. The sequential-reading rule permits result-teasing only in the intro.
- **[MED] Two figures are defined but never referenced.** fig:state-headroom (L271) and fig:geometry-effects (L302) carry no in-text \ref anywhere (grep-verified).
- **[MED] Conclusion overclaims cost treatment.** L328 says accounting, selection, action geometry, and cost treatment "each move the ranking," but Results L290 reports conversion friction reverses none of 150 signs and no favorable retail cell survives Holm; cost treatment does not reorder winners under the paper's own standard.
- **[MED] The static-width heuristic is named at least six ways.** "fixed action" (abstract), "static-policy" (intro L109), "fixed width" / "constant event-driven action" (Methods), "static widths" / "static-action selection" (Results), "static width" (Discussion L319).
- **[MED] Five algorithm acronyms are undefined at first body use.** PPO, A2C, DQN, QR-DQN, and RecurrentPPO all appear spelled only as acronyms at intro L109; none is on the pre-defined list and the abstract is a separate define-once scope.
- **[MED] The Discussion and Conclusion state the same argument twice.** L317-319 and L326-328 both give the methodological-flip, heterogeneous-loss-mitigation, and oracle-versus-deployable-policy points nearly verbatim, under a live 8-page budget.

## Cross-section flow threads

Reading in order pays off in threads no single-section pass would catch. The document-level threads:

1. **Five unresolved naming threads run the whole length of the paper and gain a fresh variant in every section instead of resolving.** The contextual arm (six names), the static-width heuristic (six-plus names), the oracle / argmax-over-Q_H quantity ("local ex-post-best action" / "local oracle" / "local full-horizon oracle" / "full-horizon oracle" / "the local best action"), the diagnostic ("full-horizon state-contingency" / "horizon-robust state-contingency" / "State-contingent headroom" / "horizon-robust suitability" / overloaded "one-action diagnostic"), and the frontier exhibit ("capital-cost" / "retail" / "practitioner" / "full"). Each needs one ratified name applied everywhere in a single pass.

2. **The contribution is enumerated with an unstable count and unstable labels.** The paper's spine is "we control four evaluation axes at once," yet the count flips to five in Related Work and the Conclusion, the fourth axis loses "range" ("matched executable range geometry" becomes "matched executable actions" / "executable actions matched across fee tiers"), and the Discussion gives two non-matching four-item sets. This is the framing a referee reads first, so the drift is load-bearing, not cosmetic.

3. **Two forward-reference risks in Methods break the top-to-bottom read.** The state-contingency diagnostic (L158) and the reward-normalized PPO retrain (L167) are both motivated by Results findings (one-action collapse, headroom at both horizons) that the reader has not yet met. Reframing them as design intent keeps the numbers in Results where they belong.

4. **Numbers are locked and consistent, with one exception.** Every headline claim the abstract and intro tease is satisfied and consistent in Results (67.0 to 73.2% disagreement, four of six winners flip, single-action collapse, all six means below holding, the QR-DQN tier-geometry reversal). The one claim that drifts from its evidence is the Conclusion's "cost treatment moves the ranking," which Results L290 does not support.

5. **A rhetorical frame and a mechanism description each repeat three-plus times under the page budget.** The "reflects skill rather than an evaluation artifact" frame appears at L111, L114, and L121; the per-swap fee-attribution mechanism is explained at L101, L108, L119, and again at L121. Both should be stated once and referenced, not re-explained.

## Findings by section

### Abstract (raw 15 -> confirmed 12 -> rejected 1)

- [ ] **[MED] dash-as-range** (67.0--73.2\%, L70) - En-dash violates the no-dashes rule and disagrees with body L245 ("ranges from 67.0\% to 73.2\%"). Fix: write "67.0 to 73.2\%".
- [ ] **[MED] term-drift** (reward-maximizing policy stump, L70) - Contextual arm named "policy stump" here but "one-split policy tree" (Methods) and "shallow causal policy" (Results). Fix: fix one canonical name, hold it through.
- [ ] **[MED] term-drift** (validation-selected fixed action, L70) - Static-width heuristic competes with "fixed width" / "constant event-driven action" / "static widths" later. Fix: one canonical term paper-wide.
- [ ] **[MED] acronym-undefined** (PPO, L70) - Used without expansion; abstract is its own define-once scope. Fix: "Proximal Policy Optimization (PPO)" or refer generically.
- [ ] **[MED] acronym-undefined** (QR-DQN, L70) - Undefined on first use. Fix: expand or refer generically.
- [ ] **[LOW] clarity** ("causal ... policies", L70) - "causal" is jargon here, clarified only at Results L274; abstract must stand alone. Fix: use "contextual" or drop the word.
- [ ] **[LOW] name-drift** (full-horizon state-contingency diagnostic, L70) - Diagnostic name drifts across abstract/intro/conclusion. Fix: one name, one modifier.
- [ ] **[LOW] wording-repeat** ("high-fee frontier", L70) - "frontier" appears twice in one sentence in two senses. Fix: rephrase the second, e.g. "high-fee pools".
- [ ] **[LOW] weak-intensifier** ("very sensitive", L70) - Vague filler. Fix: "is sensitive to" / "depends on".
- [ ] **[LOW] active-voice** ("The benchmark is built from ...", L70) - Passive, and inconsistent with active L108. Fix: "We build the benchmark from 22.6 million ... swaps".
- [ ] **[LOW] redundancy** ("Credible evaluation flips conclusions.", L70) - Restates the next sentence. Fix: delete; the next sentence carries it.
- [ ] **[LOW] coined-compound** ("The 168-hour ex-post-best action", L70) - Coined chain. Fix: unpack, e.g. "the best action ex post over the 168-hour horizon"; apply in body too.

### Introduction (raw 21 -> confirmed 11 -> rejected 3)

- [ ] **[MED] acronyms** (L109) - PPO, A2C, DQN, QR-DQN, RecurrentPPO appear at first body use with no expansion. Fix: expand each at L109, then use acronyms only.
- [ ] **[MED] terminology-drift** (reward-maximizing policy stump, L109) - Contextual arm under four names paper-wide. Fix: one canonical name; secondary descriptor once in parentheses at the Methods definition only.
- [ ] **[MED] terminology-drift** (horizon-robust state-contingency diagnostic, L109) - Three modifier+head-noun combinations across sections. Fix: one canonical name verbatim in abstract, intro, conclusion.
- [ ] **[LOW] terminology-drift** ("A local full-horizon oracle", L109) - Same object as abstract/body "local ex-post-best action". Fix: adopt one oracle name.
- [ ] **[LOW] terminology-drift** ("static-policy optimality", L109) - Fifth surface form of the static-width name. Fix: one canonical name.
- [ ] **[LOW] terminology-drift** ("a practitioner frontier", L110) - Same exhibit as abstract "capital-cost frontier". Fix: one exhibit name.
- [ ] **[LOW] redundancy** (L103) - Trailing "intersection of benchmarking, robustness, evaluation, and RL" is keyword-tagging filler and reaches for "robustness" outside its statistical sense. Fix: delete the clause.
- [ ] **[LOW] redundancy** (L114) - Closing sentence restates L103/L108 and echoes the L111/L121 skill-vs-artifact line. Fix: compress to one clause or cut.
- [ ] **[LOW] redundancy** (L108 bullet) - Per-swap fee mechanism restated after L101. Fix: compress the bullet to naming "per-swap fee attribution".
- [ ] **[LOW] citation-format** (underperformance\cite, L99) - \cite glued to the word with no tie. Fix: insert "~".
- [ ] **[LOW] voice** ("particularly consequential", L99) - Vague intensifier. Fix: drop or state the magnitude.

### Related Work (raw 16 -> confirmed 11 -> rejected 1)

- [ ] **[MED] axis-count-drift** (L121) - Lists FIVE contributions (adds "serial-dependence-aware inference", drops "leak-free", relabels the geometry axis) against the four committed elsewhere. Fix: restate exactly the four canonical axes with intro labels.
- [ ] **[MED] redundancy** (L121) - Re-enumerates the intro contribution bullets instead of contrasting with the literature. Fix: keep only the literature-contrast clause; cutting also resolves the count drift.
- [ ] **[MED] claim-grounding** (LVR, L119) - LVR is "measures" then "bounds" in one paragraph; "bounds" drifts from the first-mention verb and risks misstating Milionis et al. Fix: use "measures" throughout.
- [ ] **[LOW] redundancy** (L119 final sentence) - Restates the two preceding sentences. Fix: delete it (also removes the "bounds" occurrence).
- [ ] **[LOW] redundancy** (L119 mechanism phrase) - Third statement of the fee-attribution / rebalancing computation. Fix: keep only the LVR contrast.
- [ ] **[LOW] restatement** (L121) - Third use of "reflects X rather than an artifact" (L111, L114, L121). Fix: drop the clause; let L114 carry it once.
- [ ] **[LOW] active-voice** (L121) - "RL has been applied ... \cite{...}" is passive with cites trailing. Fix: cite as subject, e.g. "Recently, \cite{...} apply RL directly to ...".
- [ ] **[LOW] coined-compound** ("serial-dependence-aware inference", L121) - Coined chain, drifts from L109. Fix: "inference that accounts for serial dependence".
- [ ] **[LOW] ref-integrity** (L119) - Every \cite preceded by a plain space, not a tie. Fix: replace with "~".
- [ ] **[LOW] voice** ("financial-RL literature", L121) - Coined compound modifier. Fix: "literature on reinforcement learning in finance".
- [ ] **[LOW] framing** (L121) - The three RL-on-Uniswap-v3 works are cited but never characterized. Fix: add one clause naming their shared evaluation weakness; offset by cutting the redundant enumeration (page-neutral).

### Environment and Evaluation Design (raw 32 -> confirmed 17 -> rejected 3)

- [ ] **[MED] notation-collision** (H, L133 vs L160-163) - $H_t$ is the held-basket value; $H$ is the diagnostic horizon. Fix: rename the horizon (e.g. $\tau$ or $T_H$), keep $H_t$.
- [ ] **[MED] terminology-drift** (static-width heuristic, L155/L158/L165) - Four names within the section. Fix: one canonical name at L155, L158, and all four contrasts at L165.
- [ ] **[MED] terminology-drift** (contextual arm, L165) - "one-split policy tree" / "deployed contextual policy" / "contextual policy" here. Fix: adopt the ratified name.
- [ ] **[MED] forward-reference** (L167) - "Because headroom is present at both horizons ..." states a Results result (L245) as established. Fix: reframe as design intent, defer the headroom claim.
- [ ] **[MED] forward-dependence** (L158) - "The one-action result alone cannot show ..." presupposes the Results one-action collapse (L240). Fix: motivate the diagnostic on its own terms.
- [ ] **[MED] prose-redundancy** (L192, L194) - Restates Table 1 numbers (resamples, block length, window count, seed averaging). Fix: keep the conceptual protocol, drop the numeric restatements.
- [ ] **[LOW] notation-consistency** (L, L129 vs L142) - Position liquidity is $L_t$ then $L$; $L_s$ is active liquidity. Fix: one symbol for the position's LP liquidity, keep $L_s$.
- [ ] **[LOW] notation-collision** (s, L142 vs L158) - Swap index vs decision state. Fix: give the swap index a distinct letter (e.g. $j$).
- [ ] **[LOW] terminology-drift** (State-contingency diagnostic heading, L157) - Bare here vs three other combinations. Fix: one canonical name.
- [ ] **[LOW] terminology-drift** ("the local oracle", L163) - Drifts from abstract/intro. Fix: one oracle name.
- [ ] **[LOW] terminology-drift** ("matched executable geometry", L165) - Drops "range". Fix: "matched executable range geometry".
- [ ] **[LOW] active-voice** (L163, L167, L192, L194) - Several agentless passives. Fix: rewrite active with "we".
- [ ] **[LOW] voice** ("randomly fixed reproducibility seeds", L153) - Filler and internally odd. Fix: "two fixed seeds (42 and 123) for reproducibility".
- [ ] **[LOW] claim-grounding** (L151, L155) - "implemented legacy specification" / "original comparator families" leak development provenance. Fix: state the state design and heuristics as the paper's own choices.
- [ ] **[LOW] style** (\paragraph, L157) - Lone run-in heading amid \subsection structure. Fix: promote to \subsection or fold into the opening sentence.
- [ ] **[LOW] prose-redundancy** (L155) - Restates the leak-free selection rule that L192 specifies. Fix: drop the L155 parenthetical.
- [ ] **[LOW] caption** (L173) - Two-word caption "Experimental design." names nothing. Fix: name the design elements and units.

### Results (raw 25 -> confirmed 16 -> rejected 2)

- [ ] **[MED] terminology-drift** (static-width heuristic, L223/L240/L245/L274) - "Fixed width" (table) / "static widths" / "fixed action" / "static-action selection". Fix: one canonical name in table row, L240, L245, L274.
- [ ] **[MED] terminology-drift** (contextual arm, L249/L255/L274) - "policy stump" / "the stump" / "shallow causal policy" / "deployed contextual" / "Contextual"; "causal" is unexplained. Fix: one name across header, caption, prose; drop "shallow causal policy" and "contextual".
- [ ] **[MED] spelling-and-naming** (L238/L281/L286) - Prose CamelCase ("RecentreWhenOut", "ILMinimizer", "ReactiveRecentering") vs spaced table Title Case; British "Recentre" vs American "Recenter". Fix: one style, American "Recenter" everywhere.
- [ ] **[MED] unreferenced-float** (fig:state-headroom, L271) - Defined, never \ref'd. Fix: add a "Figure~\ref{fig:state-headroom}" call in the headroom prose.
- [ ] **[MED] unreferenced-float** (fig:geometry-effects, L302) - Defined, never \ref'd. Fix: add a "Figure~\ref{fig:geometry-effects}" call in the matched-geometry prose.
- [ ] **[MED] caption-states-results** (fig:retail-frontier, L286) - Caption narrates per-cell counts and Holm result that L290 repeats. Fix: trim to what the exhibit is; keep counts and Holm result in the body.
- [ ] **[LOW] caption-states-results** (L270, L301) - Captions state findings the body gives ("All right-panel intervals cross zero"; "two effects surviving Holm"). Fix: reduce to what the panels plot / the reading convention.
- [ ] **[LOW] terminology-drift** ("State-contingent headroom" title, L242) - Drifts from "state-contingency diagnostic". Fix: align to the canonical diagnostic name.
- [ ] **[LOW] terminology-drift** ("local ex-post-best action", L245) - Vs Methods "local oracle". Fix: one name across Methods and Results.
- [ ] **[LOW] terminology-drift** ("retail frontier" / "full frontier", L201/L281) - Vs abstract "capital-cost" / intro "practitioner". Fix: converge on one exhibit name.
- [ ] **[LOW] terminology-drift** ("standard PPO" vs "native PPO", L276/L310) - Same baseline, two labels. Fix: one term.
- [ ] **[LOW] coined-compound** (L245) - "ex-post-best action" and "local future-aware counterfactuals". Fix: unpack once, keep consistent with abstract/table.
- [ ] **[LOW] redundancy** (L238) - "every mean sits below holding" repeats the tab:benchmark caption (L208). Fix: state once; drop the interpretive caption echo.
- [ ] **[LOW] active-voice** (L281) - "The active width is selected ..." Fix: "We select the active width on the preceding validation window."
- [ ] **[LOW] undefined-shorthand** ("paper-compatible extractor", L312) - Provenance-leaking shorthand. Fix: name the extractor and what it is compatible with, or point to Methods.
- [ ] **[LOW] filler** (L201) - Roadmap sentence pre-announces a device each subsection opener performs. Fix: delete the second sentence of L201.

### Discussion and Limitations (raw 19 -> confirmed 10 -> rejected 2)

- [ ] **[MED] axis-list-inconsistency** (L317) - Two different four-item axis sets in one paragraph ("dependence-aware inference" substitutes for "value-conserving rebalancing"); "these four choices" has no stable antecedent, and Conclusion L326 lists five. Fix: state one canonical four-axis set in both sentences.
- [ ] **[MED] terminology-drift** ("deployable policy stump", L319) - Vs Conclusion "simple deployable contextual policy" and the Methods/Results names. Fix: one ratified name.
- [ ] **[MED] terminology-drift** ("fixed action" / "a static width", L319) - Two variants within one paragraph. Fix: the single ratified static-width name.
- [ ] **[MED] restatement** (L317-319 vs Conclusion L326-328) - Discussion and Conclusion restate nearly the same points under the page budget. Fix: keep the argument in one section, cut the other to a short wrap.
- [ ] **[LOW] terminology-drift** ("matched executable actions" / "matching executable geometry", L317) - Drops "range". Fix: "matched executable range geometry".
- [ ] **[LOW] terminology-drift** ("The full-horizon oracle", L319) - Drops "local". Fix: the ratified oracle name.
- [ ] **[LOW] terminology-overload** ("The one-action diagnostic", L319) - Overloads "diagnostic" onto the one-action collapse. Fix: reserve "diagnostic" for the state-contingency construct; call this "the one-action result".
- [ ] **[LOW] internal-redundancy** (L321) - "may reflect reward scale" re-opens what L319 downweighted. Fix: cut "reward scale or" or scope as residual scale effects.
- [ ] **[LOW] active-voice** (L319, L321) - Two agentless passives. Fix: rewrite active with "we".
- [ ] **[LOW] coined-compound** ("dependence-aware inference", L317) - Coined modifier, drifts from L121. Fix: "block-bootstrap inference that accounts for serial dependence" or fold into the inference protocol.

### Conclusion (raw 17 -> confirmed 9 -> rejected 1)

- [ ] **[MED] axis-count-drift** (L326) - Lists FIVE choices ("dependence-aware inference" as a fifth) against four committed at abstract/intro. Fix: reduce to the four canonical axes; fold serial-dependence inference into the selection/inference axis.
- [ ] **[MED] terminology-drift** ("A horizon-robust suitability diagnostic", L328) - Head noun "suitability" is a substantive drift. Fix: one canonical diagnostic name.
- [ ] **[MED] terminology-drift** ("a simple deployable contextual policy", L328) - Another label for the contextual arm. Fix: the ratified paper-wide name.
- [ ] **[MED] redundancy** (L326-328 vs Discussion L317-319) - Near-verbatim restatement of the Discussion. Fix: compress the Conclusion to a shorter wrap; do not state the loss-mitigation-plus-headroom finding twice.
- [ ] **[MED] overclaim** (L328) - "accounting, selection, action geometry, and cost treatment that each move the ranking" contradicts Results L290 (cost treatment reverses none of 150 signs, no Holm survivor). Fix: scope ranking flips to accounting, selection, and action geometry; describe cost treatment as characterized transparently.
- [ ] **[LOW] terminology-drift** ("executable actions matched across fee tiers", L326) - Drops "range". Fix: "matched executable range geometry".
- [ ] **[LOW] terminology-drift** ("The local best action", L328) - Further oracle variant. Fix: the ratified oracle name.
- [ ] **[LOW] internal-inconsistency** (L326-328) - Three non-matching enumerations of the benchmark's dimensions within two paragraphs. Fix: one canonical four-item set reused in all three.
- [ ] **[LOW] voice** ("not merely a negative result for RL", L328) - Double "result" in close range. Fix: e.g. "not merely a negative outcome for RL".

## Suggested sequencing

We group the confirmed findings by the kind of action they need. Nothing is applied here.

**(a) Quick line fixes (local, no cross-section decision).**
- Dash-to-"to" at abstract L70; citation ties "~" before every \cite at L99 and throughout Related Work L119/L121.
- Intensifier and filler cuts: "very sensitive" (L70), "particularly consequential" (L99), the "randomly fixed reproducibility seeds" parenthetical (L153), the L201 roadmap second sentence, the L328 double "result".
- Active-voice rewrites: abstract L70, Methods L163/L167/L192/L194, Results L281, Discussion L319/L321, Related Work L121.
- Coined-compound unpacks: "serial-dependence-aware inference" (L121), "financial-RL" (L121), "dependence-aware inference" (L317), "ex-post-best action" and "local future-aware counterfactuals" (L245).
- Style: promote or fold the lone \paragraph at L157; expand the two-word table caption at L173.
- Acronym expansions at intro L109 (PPO, A2C, DQN, QR-DQN, RecurrentPPO) and the abstract (PPO, QR-DQN).

**(b) Flow / structure edits that touch multiple sections (do these as coordinated passes).**
- Ratify one name each for the contextual arm, the static-width heuristic, the oracle, the diagnostic, and the frontier exhibit, then apply across abstract, intro, Methods, Results, Discussion, Conclusion, and every table header and caption in a single pass. This resolves the largest block of findings.
- Ratify four axes with one label set (keep "range" in "matched executable range geometry"), fix the count at Related Work L121, Discussion L317, and Conclusion L326, and align the three enumerations inside the Conclusion.
- Reframe the two Methods forward references (L158, L167) as design intent, leaving the numbers in Results.
- Rename the horizon symbol so $H$ stops colliding with $H_t$; reconcile $L$ vs $L_t$ and the $s$ swap-index/state reuse. If renamed, Results must adopt the new symbols.
- Cut the duplicated argument across Discussion (L317-319) and Conclusion (L326-328) to one full statement plus one short wrap.
- Remove the repeated fee-mechanism explanation (keep L101, trim L108/L119/L121) and the thrice-used "reflects skill rather than an artifact" frame (keep one).

**(c) Items needing a coauthor decision or a data/figure action.**
- Add the two missing \ref calls for fig:state-headroom and fig:geometry-effects (confirm which prose sentence each should anchor to).
- Decide whether "dependence-aware inference" is a genuine fifth axis (then commit it everywhere) or an inference-protocol detail (then demote it); this decision drives the axis-count fixes in (b).
- Scope the Conclusion "cost treatment moves the ranking" claim to match Results L290; confirm the intended reading of the retail-frontier result.
- Remove the provenance leaks ("legacy specification" L151, "original comparator families" L155, "paper-compatible extractor" L312) by stating the design directly, or cite a specific source if one exists.
- Trim the result-narrating captions (fig:retail-frontier L286, L270, L301) to descriptions, keeping the numbers in the body; confirm no number is lost that the exhibit itself cannot show.
- Add one clause characterizing the three RL-on-Uniswap-v3 prior works at L121 (framing gap a referee will notice), offset by the redundant-enumeration cut.
