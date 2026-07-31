# Open items after the third section review (2026-07-31)

The 116 confirmed findings of `section-review-2026-07-31-third.md` were applied across
three commits (`d46b39a`, `a48c354`, `37e5029`), each gated by an independent
fresh-context judge. The paper compiles at 8 pages with zero overfull boxes and zero
undefined references.

Everything below is what could NOT be settled from the manuscript, the code, or the run
record in this repository. Each has an inline `% CHECK:` note at the site in
`icaif_main.tex` so it cannot ship unnoticed.

## Needs Arjun or the run record

1. **The gated follow-up's treatment.** The Results previously said the reward-normalized
   PPO run also used validation-selected entropy, which the prespecification in
   Subsec. 3.2 does not carry. Per the decision taken during the pass, both sites now
   read reward-only normalization. If the run did select entropy on validation, add it
   back to Subsec. 3.2 as well, not to the Results alone.
   Site: the `% CHECK:` above "Because the diagnostic shows headroom".

2. **The gated follow-up's Holm family.** The reported $p_{\mathrm{Holm}}=.0001$
   reconciles with the $.0004$ resolution floor asserted for the four-member family only
   if this follow-up carries a one-member family. State the family in the sentence.
   Site: the `% CHECK:` after "Reward scale alone therefore does not explain".

3. **The denominator for the 28 surviving adverse cells.** The retail-frontier paragraph
   runs three cell populations (25 per pool, 150 signs, and the 50-cell Holm family per
   pool) and the count of 28 is attached to none of them.
   Site: the `% CHECK:` in Subsec. 4.4.

4. **The fourth prespecified contrast.** Subsec. 3.2 declares four Holm-corrected
   contrasts and the paper reports three, now stated explicitly ("We report the first
   three"). The oracle against the decision stump is still unreported. Either report it
   in one clause or leave it as a declared family member.

5. **Cross-pool transfer, the closing claim.** "we find no general transfer pattern for
   the other algorithms" is still qualitative. If the run record supports a count of
   surviving directions per algorithm, state it.

## Resolved from the code during this pass, for the record

These were group (c) items in the review and were settled by reading the repository, so
they are now in the paper and need no further action.

- **Tick spacings**: 10 at the 0.05% tier, 60 at the 0.30%
  (`src/.../envs/uniswap_v3.py`, `TICK_SPACING`). Verified consistent with the
  label-to-price ratios already quoted in Subsec. 4.1.
- **Heuristic decision schedule**: heuristics run on their native logic, consulted every
  hour, and decide for themselves whether to act. Only learned agents carry a declared
  schedule wrapper (`src/.../envs/schedule.py`, "Competitors are never wrapped").
- **ILMinimizer against the volatility-proportional strategy**: the first scales
  volatility by a selected multiplier $k \in \{3,5,7,10,15\}$, the second by twice the
  square root of a selected 24- or 168-hour period (`src/.../policies/baselines.py`).
- **Training discount**: never set in the grid, so it is the Stable-Baselines3 default
  $\gamma=0.99$.
- **Hourly grid fill rule**: state variables (price, liquidity, tick) forward-filled,
  flow variables zero-filled (`src/.../data/aggregate.py`).
- **Implementation**: Stable-Baselines3 plus its contrib package (QR-DQN, RecurrentPPO).
- **The seed question**: the run record states "every *logged first-seed* learned policy
  uses one distinct action per pool" (`reports/ICAIF_EXTENSION_RESULTS.md`). Only the
  first seed's action trace was logged, so the paper now scopes the claim in the Results
  and records the two-seed budget in Limitations. Do not strengthen it to both seeds
  unless the record is extended.

## Three references added, all verified

Cleared through OpenAlex and cross-checked (Crossref where a Crossref DOI exists,
Semantic Scholar and the Project Euclid landing page for the Künsch page range).

- `raffin2021sb3` — Raffin et al., JMLR 22(268), 2021. JMLR has no DOIs, which is why
  Crossref carries no record.
- `kunsch1989jackknife` — Künsch, Ann. Statist. 17(3):1217-1241, 1989.
- `holm1979simple` — Holm, Scand. J. Statist. 6:65-70, 1979. The JSTOR DOI is not
  registered with Crossref, which is normal for that era.

The circular variant of the block bootstrap is due to Politis and Romano (1992), a book
chapter that does not resolve in OpenAlex. Rather than attribute it to Künsch, the
sentence cites Künsch for block resampling itself. If the Politis-Romano chapter is
wanted, verify it outside the pipeline first.
