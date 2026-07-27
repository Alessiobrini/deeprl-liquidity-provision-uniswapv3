"""The test window must be unreachable from anything except the single final read.

This is the defect that invalidated the previous paper: `optimize_ppo` was handed
`test_env` and returned the reward on it as its Optuna objective, inside every rolling
step, so the headline was a max-over-trials order statistic on the evaluation data.

The whole rebuild exists to remove that. Nothing tested it. Both of these mutations
passed the entire suite:

    vals.append(score_agent(key, m, widths, split.test, schedule).mean())   # select on test
    fit = Split(train=split.train + split.val + split.test, ...)            # fit on test

So the suite could not tell an honest protocol from the exact defect it was built to
prevent. These tests instrument the protocol and watch which windows it touches.

They stub out training entirely. What is under test is the PROTOCOL, i.e. which windows
reach which stage, not whether a policy learns anything.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deeprl_liquidity_provision_uniswapv3.experiments import rolling as R  # noqa: E402
from src.deeprl_liquidity_provision_uniswapv3.experiments.bakeoff import Split  # noqa: E402

SPLIT = Split(train=[0, 1, 2, 3], val=[4], test=[5])
TEST_W = 5


class DummyPolicy:
    """`heuristic_step` reads `.name` off the winner for reporting."""

    def __init__(self, name="dummy"):
        self.name = name


@pytest.fixture
def spy(monkeypatch):
    """Record every window index each stage touches, in order."""
    seen = []

    class DummyModel:
        def learn(self, total_timesteps):
            return self

        def predict(self, obs, deterministic=True):
            return 0, None

    def fake_train_env(key, widths, split, schedule, features="legacy",
                       n_windows=None, reward_shaping="none", width_units="spacing",
                       execution_widths=None):
        seen.append(("fit", sorted(split.train)))
        return object()

    def fake_make_agent(algo, env, seed, **kw):
        return DummyModel()

    def fake_score_agent(key, model, widths, w_indices, schedule, features="legacy",
                         width_units="spacing", execution_widths=None):
        seen.append(("score", sorted(w_indices)))
        return np.array([1.0])

    def fake_build_env(key, wi, widths, schedule=None, features="legacy", **kw):
        seen.append(("build", [wi]))
        return object()

    def fake_score(env, policy):
        return 0.0, 1

    monkeypatch.setattr(R, "train_env", fake_train_env)
    monkeypatch.setattr(R, "make_agent", fake_make_agent)
    monkeypatch.setattr(R, "score_agent", fake_score_agent)
    monkeypatch.setattr(R, "build_env", fake_build_env)
    monkeypatch.setattr(R, "score", fake_score)
    monkeypatch.setattr(R, "AGENT_GRID", [dict(learning_rate=1e-3), dict(learning_rate=3e-4)])
    return seen


def touches(seen, stage, window):
    return [i for i, (s, ws) in enumerate(seen) if s == stage and window in ws]


# ------------------------------------------------------------------ the agent

def test_agent_never_fits_on_test(spy):
    """`fit` must never include the test window. Catches: refit on train+val+test."""
    R.agent_step("p", SPLIT, [45], "ppo", "hourly", 10, [42], "none")
    bad = touches(spy, "fit", TEST_W)
    assert not bad, f"the agent was fitted on the test window at steps {bad}"


def test_agent_never_selects_on_test(spy):
    """Every scoring call before the final read must be on validation.

    Catches the previous paper's exact defect: scoring the candidate configs on test
    and keeping the best.
    """
    R.agent_step("p", SPLIT, [45], "ppo", "hourly", 10, [42], "none")
    scores = [i for i, (s, _) in enumerate(spy) if s == "score"]
    last = scores[-1]
    early = [i for i in touches(spy, "score", TEST_W) if i < last]
    assert not early, f"test window scored during selection at steps {early}"


def test_agent_reads_test_exactly_once_per_seed(spy):
    """One test read, after selection. More than one is a multiple test read."""
    R.agent_step("p", SPLIT, [45], "ppo", "hourly", 10, [42], "none")
    assert len(touches(spy, "score", TEST_W)) == 1


def test_selection_scores_only_validation(spy):
    """...and the selection loop must actually use validation, or 'selected on
    validation' is a fiction. This is the other half of the defect that had the agent
    computing a validation score and discarding it."""
    R.agent_step("p", SPLIT, [45], "ppo", "hourly", 10, [42], "none")
    assert touches(spy, "score", SPLIT.val[0]), "validation was never scored"


def test_every_config_is_tried_on_validation(spy):
    """A grid that is not searched is not a grid. Two configs, one seed => two
    validation scores before the test read."""
    R.agent_step("p", SPLIT, [45], "ppo", "hourly", 10, [42], "none")
    val_scores = touches(spy, "score", SPLIT.val[0])
    assert len(val_scores) == 2, f"expected one validation score per config, got {len(val_scores)}"


def test_refit_block_is_train_plus_val_only(spy):
    """The refit must use exactly the 5 windows preceding test, matching the previous
    paper's fitting block, and no more."""
    R.agent_step("p", SPLIT, [45], "ppo", "hourly", 10, [42], "none")
    fits = [ws for s, ws in spy if s == "fit"]
    assert fits[-1] == [0, 1, 2, 3, 4], f"refit block was {fits[-1]}"
    for f in fits[:-1]:
        assert f == [0, 1, 2, 3], f"selection fitted on {f}, expected train only"


# ------------------------------------------------------------- the heuristics

def test_heuristics_never_select_on_test(spy):
    """Both arms get the identical treatment, so the competitors are held to it too."""
    cands = [DummyPolicy(f'c{i}') for i in range(3)]
    R.heuristic_step("p", SPLIT, [45], cands)
    builds = [i for i, (s, _) in enumerate(spy) if s == "build"]
    early = [i for i in touches(spy, "build", TEST_W) if i < builds[-1]]
    assert not early, f"test window used during heuristic selection at {early}"


def test_heuristics_read_test_exactly_once(spy):
    R.heuristic_step("p", SPLIT, [45], [DummyPolicy(f'c{i}') for i in range(3)])
    assert len(touches(spy, "build", TEST_W)) == 1


def test_every_candidate_is_tried_on_validation(spy):
    R.heuristic_step("p", SPLIT, [45], [DummyPolicy(f'c{i}') for i in range(3)])
    assert len(touches(spy, "build", SPLIT.val[0])) == 3


# ---------------------------------------------------------- the split itself

def test_split_refuses_overlapping_train_and_test():
    """The last line of defence, and the one that caught a real refit bug."""
    with pytest.raises(AssertionError, match="splits overlap"):
        Split(train=[0, 1, 5], val=[4], test=[5])


def test_split_refuses_overlapping_train_and_val():
    with pytest.raises(AssertionError, match="splits overlap"):
        Split(train=[0, 1, 4], val=[4], test=[5])


def test_rolling_steps_never_test_on_fitted_data():
    """Across every step of a real panel-sized run."""
    for s in R.rolling_steps(29):
        fitted = set(s.train) | set(s.val)
        assert not (fitted & set(s.test)), f"step tests on fitted data: {s}"
