# Camera-ready experiment record

This record describes the implemented experiments and their saved collections.
It consolidates the release documentation on 13 October 2026.
It does not establish preregistration or change the historical experiment designs.
Family names below are code identifiers, not paper section numbers.

## Primary benchmark

- Six pools, five algorithms, and 24 rolling steps give 720 result files.
- Each step uses four training blocks, one validation block, and one test block.
- Blocks contain 1,500 hourly rows. Evaluation uses 168 warm-up rows and 1,330 hourly transitions.
- Pool grids start at different timestamps. Transfer matches block indices, with calendar offsets up to 28 hours.
- Selection and refitting precede the target scored period on the released panel.
- Capital is USD 30,000. Recenter gas is USD 5. Fees are paid out.
- Actions are hold and labels 45, 50, 55. Nominal widths multiply labels by tick spacing.
- PPO, A2C, DQN, QR-DQN, and RecurrentPPO use 20,000 steps and seeds 42 and 123.
- The grid uses learning rates 0.0003 and 0.001, networks [4, 2] and [64, 64], and entropy 0 or 0.01.
- DQN and QR-DQN omit entropy and therefore search four configurations.
- Rewards average seeds within each window. Deterministic heuristic copies count once.
- H1 compares five active heuristics with Passive. H2 compares five algorithms with the validation-selected heuristic.
- Holm correction applies within each pool and family.
- Inference uses 10,000 circular moving-block bootstrap resamples, with blocks 2, 4, 6, and 8.
- Block 4 is primary. The pass criterion requires a positive difference and Holm p below 0.05 at all four lengths.
- The primary bootstrap seed is 20260716, with deterministic pool/family offsets.

## Accounting and information limits

Stablecoins have one-dollar marks. WETH/stablecoin pools use their own hourly WETH price.
WBTC/WETH pools convert WETH through USDC/WETH 0.05% and WBTC through their own pool price.
Swap fees use input-leg price overlap and post-swap active liquidity.
Tick-crossing attribution is approximate because the records do not reconstruct every initialized liquidity region.
The hypothetical position does not alter the recorded price path.
The primary benchmark omits conversion costs, slippage, latency, compounding, and an exit action.
The 13-feature observation omits range boundaries and position age.
These restrictions limit the interpretation of policy collapse.

## Action geometry and retail frontier

Matched geometry preserves labels but executes half-widths 480, 540, and 600 raw ticks.
Geometry inference has eleven paired strategy contrasts within each pool.
The QR-DQN difference-in-differences has a separate exploratory three-pair family.
The bootstrap seed is 20260722.

The retail frontier compares RecentreWhenOut with Passive under nominal geometry.
Capital levels are 1,500, 5,000, 10,000, 30,000, and 100,000 dollars.
Gas levels are 1, 5, 25, 100, and 250 dollars.
Conversion charges the pool fee plus five basis points on half the position value.
Width selection uses cost-adjusted validation reward for each cell.
Each pool has one Holm family of 50 cells across both cost cases.
The bootstrap seed is 20260722.

## State headroom

The collection contains 144 pool-window files under matched geometry.
The diagnostic follows the validation-selected fixed action's test trajectory.
At each eligible decision state, it applies each action once and holds afterward.
The primary horizon is 168 hours. The robustness horizon is 24 hours.
Both diagnostics require a complete 168-hour horizon remaining.
The stump maximizes realized action reward on training states sampled every 24 hours.
Validation selects minimum leaf size from 20, 10, and 5 before a train-plus-validation refit.
H3 contains four contrasts per pool: oracle minus fixed, stump minus fixed,
validation-selected RL minus stump, and local oracle minus stump action.
The bootstrap seed is 20260728.
Oracle contrasts are descriptive because an ex-post maximum has selection bias.

## Reward normalization and training budget

Reward-normalized PPO uses matched geometry on both USDC/WETH fee tiers.
VecNormalize normalizes rewards only, with clip 10, gamma 0.99, and epsilon 1e-8.
Validation and test retain raw-dollar rewards.
The full PPO grid includes entropy 0 and 0.01.
H4 has one reward contrast per pool, so Holm equals the raw p-value.
The bootstrap seed is 20260728.

The post hoc PPO budget check uses matched geometry on the same two pools.
Budgets are 20,000, 100,000, and 500,000 steps.
Each budget independently selects the full PPO grid on validation.
Behavior records cover both seeds in every window.
H6 contains the two larger-budget contrasts against 20,000 steps within each pool.
The bootstrap seed is 20260802.
This targeted check does not establish convergence or exclude undertraining elsewhere.

## Oracle calibration

Each realized state-by-action matrix has four columns.
The additive projection separates grand, state, action, and residual interaction effects.
Each random block sign multiplies the complete four-action residual vector.
Column mean restoration preserves fitted state and action means.
This projection can change finite-sample residual covariance.
The block length is the ceiling of horizon divided by median decision spacing, with a minimum of one.
The null uses 2,000 draws and base seed 20260802, with pool/window offsets.
It assumes approximate blockwise sign symmetry and an adequate additive projection.
It is not an exchangeable-action null or a test of feature predictability.
Permuting whole rows cannot calibrate the selection premium because it leaves that statistic unchanged.
H5 contains two window-level calibrated-excess contrasts per pool at 168 hours.
The bootstrap seed is 20260802.

## Pool-level oracle null

This post hoc conditional comparison retains the full 168-hour null-premium draws for all 144 units.
Each pool draw averages one independently generated draw from each of its 24 windows.
The observed statistic also averages windows, not individual decision states.
The statistic is mean oracle reward minus the ex-post best constant action on test.
The paper's exceedance fraction counts null draws greater than or equal to the observed statistic.
It divides that count by 2,000.
The output also reports the Monte Carlo correction (count + 1) / (draws + 1).
A zero exceedance fraction does not imply a population p-value of zero.
Null quantiles describe a conditional randomization distribution, not a sampling confidence interval on economic value.
This diagnostic uses one contrast per pool without correction across pools.
No ratio is defined when the observed premium equals zero.

## Transfer and sensitivity

Transfer has 720 source-window files and six target outcomes in each file.
Each algorithm has 18 eligible off-diagonal directions.
Holm correction applies to three eligible source contrasts within each target pool.
The source validation history selects the configuration before refitting.
The sensitivity collection contains 864 PPO variant-window files.
Its six variants compare hourly, daily, weekly, shadow reward, LVR reward, and the input-normalized policy network.
Both analyses use bootstrap seed 20260716.

## Release evidence

Run `scripts/reproduce_paper.py` to audit collections and regenerate the aggregate tables.
The script records artifact hashes and collection audit digests.
It does not retrain policies or claim exact cross-hardware training reproducibility.
Keep historical result provenance unchanged, including its original commit and protocol path.
New result files reference this release record.
