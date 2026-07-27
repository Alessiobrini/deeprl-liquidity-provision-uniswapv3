"""Protocol and aggregation tests for resumable cross-pool transfer."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.deeprl_liquidity_provision_uniswapv3.data.pools import CORE
from src.deeprl_liquidity_provision_uniswapv3.experiments import transfer_rolling as T


def args(**kw):
    base = dict(algo="ppo", widths=[45, 50, 55], schedule="event_driven",
                steps=20_000, seeds=[42, 123], shaping="none",
                paper_extractor=False, targets=CORE, sources=CORE, out="x",
                report_dir=None, shard=0, of=1, aggregate=False, force=False,
                bootstrap_seed=1, bootstrap_samples=100, bootstrap_block=4)
    base.update(kw)
    return argparse.Namespace(**base)


def test_transfer_config_tag_changes_with_every_computational_setting():
    base = T.config_tag(args())
    for changed in (dict(algo="a2c"), dict(widths=[45]), dict(schedule="daily"),
                    dict(steps=10), dict(seeds=[42]), dict(shaping="lvr"),
                    dict(paper_extractor=True)):
        assert T.config_tag(args(**changed)) != base


def test_transfer_integer_and_float_widths_share_a_resume_key():
    assert T.config_tag(args(widths=[45, 50, 55])) == \
        T.config_tag(args(widths=[45.0, 50.0, 55.0]))


def test_one_unit_trains_source_only_and_scores_each_target_test(monkeypatch):
    seen = []

    class Model:
        def learn(self, total_timesteps):
            return self

    monkeypatch.setattr(T, "rolling_steps", lambda n: [
        T.Split(train=[0, 1, 2, 3], val=[4], test=[5])])
    monkeypatch.setattr(T, "load_panel", lambda key: [None] * 9000)
    monkeypatch.setattr(T, "grid_for", lambda algo: [{"learning_rate": 1e-3}])
    monkeypatch.setattr(T, "train_env", lambda key, widths, split, schedule,
                        reward_shaping: seen.append(("fit", key, tuple(split.train))) or object())
    monkeypatch.setattr(T, "make_agent", lambda *a, **k: Model())

    def score(key, model, widths, windows, schedule):
        seen.append(("score", key, tuple(windows)))
        return np.array([1.0])

    monkeypatch.setattr(T, "score_agent", score)
    a = args(seeds=[42], targets=[CORE[0], CORE[1]])
    result = T.run_unit(T.Unit(CORE[0], 0, "tag"), a)
    assert all(key == CORE[0] for stage, key, _ in seen if stage == "fit")
    scores = [(key, windows) for stage, key, windows in seen if stage == "score"]
    assert scores[0] == (CORE[0], (4,)), "selection must score source validation first"
    assert not any(5 in windows for stage, _, windows in seen if stage == "fit")
    assert set(result["targets"]) == {CORE[0], CORE[1]}
    assert ("fit", CORE[0], (0, 1, 2, 3, 4)) in seen


def test_partial_transfer_collection_is_never_reported_final(tmp_path):
    expected = [T.Unit(CORE[0], 0, "abc"), T.Unit(CORE[0], 1, "abc")]
    row = {"unit": {"source": CORE[0], "step": 0, "tag": "abc"}, "targets": {}}
    (tmp_path / f"{expected[0].name}.json").write_text(json.dumps(row))
    text = T.aggregate(tmp_path, expected, CORE, tmp_path / "report", 1, 100, 4)
    assert text.startswith("PARTIAL — NOT A FINAL RESULT")
    assert not (tmp_path / "report").exists()


def test_transfer_step_count_is_dynamic_but_must_align():
    aligned = [
        T.Unit(CORE[0], step, "abc") for step in (0, 1)
    ] + [
        T.Unit(CORE[1], step, "abc") for step in (0, 1)
    ]
    assert T._aligned_steps(aligned) == [0, 1]

    unaligned = aligned[:-1]
    with np.testing.assert_raises_regex(ValueError, "aligned step indices"):
        T._aligned_steps(unaligned)


def test_slurm_transfer_wrapper_is_cpu_resumable_array():
    body = (Path(__file__).resolve().parents[1] / "scripts/slurm/transfer_array.sh").read_text()
    assert "#SBATCH --partition=" not in body
    assert "#SBATCH --array=0-17" in body
    assert "PYTHONNOUSERSITE=1" in body
    assert "transfer_rolling" in body
    assert "--gres=gpu" not in body
