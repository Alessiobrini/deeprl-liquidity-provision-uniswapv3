import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.experiments import ppo_budget_ablation as P  # noqa: E402
from src.deeprl_liquidity_provision_uniswapv3.experiments.validate_ppo_budget_ablation import (  # noqa: E402
    audit,
    collection_hash,
)


def test_budget_units_cover_every_treatment_once():
    units = P.expected_units()
    keys = {(unit.pool, unit.step, unit.budget) for unit in units}
    assert len(units) == len(keys) == 3 * 2 * 24
    assert {unit.budget for unit in units} == {20_000, 100_000, 500_000}
    assert {unit.pool for unit in units} == {"usdc_weth_005", "usdc_weth_030"}


def test_budget_shards_map_one_array_task_to_one_unit():
    units = P.expected_units()
    assigned = [unit for shard in range(144) for unit in units[shard::144]]
    assert assigned == units


def test_config_selection_reads_validation_not_test(monkeypatch):
    split = P.Split(train=[0, 1, 2, 3], val=[4], test=[5])
    calls = []

    class FakeEnv:
        def close(self):
            pass

    class FakeModel:
        def learn(self, total_timesteps):
            assert total_timesteps == 20_000

    monkeypatch.setattr(P, "grid_for", lambda _: [
        {"learning_rate": 1e-3, "ent_coef": 0.0, "net_arch": [4, 2]},
        {"learning_rate": 3e-4, "ent_coef": 0.01, "net_arch": [64, 64]},
    ])
    monkeypatch.setattr(P, "_matched_train_env", lambda *args: FakeEnv())
    monkeypatch.setattr(P, "make_agent", lambda *args, **kwargs: FakeModel())

    values = iter([1.0, 3.0, 4.0, 6.0])

    def fake_score_agent(pool, model, widths, windows, schedule, **kwargs):
        calls.append(list(windows))
        return np.asarray([next(values)])

    monkeypatch.setattr(P, "score_agent", fake_score_agent)
    selected, validation, search = P.select_config("usdc_weth_005", split, 20_000)
    assert calls == [[4], [4], [4], [4]]
    assert selected["learning_rate"] == 3e-4
    assert validation == 5.0
    assert len(search) == 2


def test_behavior_diagnostics_retain_both_actions_and_probability_variation(monkeypatch):
    class FakeEnv:
        unwrapped = object()

        def __init__(self):
            self.i = 0

        def reset(self):
            return np.asarray([0.0]), {}

        def step(self, action):
            self.i += 1
            return (
                np.asarray([float(self.i)]),
                0.0,
                self.i == 2,
                False,
                {"reward_true": float(action + 1), "n_rebalances": self.i},
            )

        def close(self):
            pass

    probabilities = iter([
        np.asarray([0.7, 0.1, 0.1, 0.1]),
        np.asarray([0.1, 0.1, 0.7, 0.1]),
    ])
    monkeypatch.setattr(P, "build_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(P, "_policy_probabilities", lambda *args: next(probabilities))
    result = P.evaluate_behavior("usdc_weth_005", 5, object())
    assert result["reward"] == 4.0
    assert result["n_decisions"] == 2
    assert result["n_actions"] == 2
    assert result["action_histogram"] == {
        "hold": 1,
        "width_45": 0,
        "width_50": 1,
        "width_55": 0,
    }
    assert result["n_rebalances"] == 2
    assert np.isclose(sum(result["mean_action_probability"].values()), 1.0)
    assert result["state_dependence_tv"] > 0


def test_partial_collection_is_labeled_and_not_aggregated(tmp_path):
    text = P.analyze(tmp_path, tmp_path / "report", samples=10)
    assert text.startswith("PARTIAL - NOT A FINAL RESULT")
    assert "0/144 units" in text
    assert not (tmp_path / "report" / "summary.csv").exists()


def test_complete_aggregation_writes_window_level_budget_comparisons(monkeypatch, tmp_path):
    config = {"learning_rate": 3e-4, "ent_coef": 0.0, "net_arch": [4, 2]}
    rows = {}
    for pool_index, pool in enumerate(P.POOLS):
        for budget in P.BUDGETS:
            for step in range(P.EXPECTED_WINDOWS):
                reward = float(pool_index + step + budget / 10_000)
                behavior = {
                    "reward": reward,
                    "seed": 42,
                    "n_decisions": 2,
                    "n_actions": 1,
                    "n_rebalances": 2,
                    "mean_policy_entropy": 0.5,
                    "normalized_policy_entropy": 0.5 / np.log(4),
                    "state_dependence_tv": 0.1,
                    "max_probability_range": 0.2,
                    "action_histogram": {
                        "hold": 2, "width_45": 0, "width_50": 0, "width_55": 0,
                    },
                    "mean_action_probability": {
                        "hold": 0.4, "width_45": 0.2,
                        "width_50": 0.2, "width_55": 0.2,
                    },
                    "std_action_probability": {
                        "hold": 0.1, "width_45": 0.1,
                        "width_50": 0.1, "width_55": 0.1,
                    },
                }
                search = [{
                    "config": config,
                    "seed_validation": [
                        {"seed": 42, "reward": reward},
                        {"seed": 123, "reward": reward},
                    ],
                    "mean_validation": reward,
                }]
                rows[(pool, step, budget)] = {"ppo": {
                    "test": reward,
                    "validation": reward,
                    "selected_config": config,
                    "validation_search": search,
                    "per_seed": [
                        {**behavior, "seed": 42},
                        {**behavior, "seed": 123},
                    ],
                }}
    monkeypatch.setattr(P, "_load_results", lambda _: (rows, []))
    report = tmp_path / "aggregate"
    text = P.analyze(tmp_path, report, samples=100)
    assert text.startswith("COMPLETE - FINAL COLLECTION")
    assert "100,000 steps" in text and "500,000 steps" in text
    assert len((report / "windows.csv").read_text().splitlines()) == 2 * 3 * 24 + 1
    assert len((report / "behavior.csv").read_text().splitlines()) == 2 * 3 * 24 * 2 + 1
    assert len((report / "summary.csv").read_text().splitlines()) == 2 * 3 + 1


def test_budget_audit_labels_empty_collection_partial(tmp_path):
    result = audit(tmp_path)
    assert result["status"] == "PARTIAL"
    assert result["count"] == 0
    assert "0/144 files" in result["problems"][0]


def test_budget_collection_hash_is_deterministic_by_filename(tmp_path):
    second = tmp_path / "b.json"
    first = tmp_path / "a.json"
    first.write_text("first")
    second.write_text("second")
    assert collection_hash([second, first]) == collection_hash([first, second])


def test_completed_result_is_skipped_only_when_both_checkpoints_exist(tmp_path):
    path = tmp_path / "unit.json"
    path.write_text(json.dumps({
        "ppo": {"per_seed": [
            {"checkpoint": "checkpoints/a.zip"},
            {"checkpoint": "checkpoints/b.zip"},
        ]}
    }))
    assert not P._result_is_complete(path, tmp_path)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/a.zip").write_bytes(b"a")
    (tmp_path / "checkpoints/b.zip").write_bytes(b"b")
    assert P._result_is_complete(path, tmp_path)


def test_budget_slurm_wrapper_is_generic_cpu_and_maps_144_units():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/slurm/ppo_budget_ablation_array.sh"
    ).read_text()
    assert "#SBATCH --partition=" not in script
    assert "--gres=gpu" not in script
    assert "#SBATCH --array=0-143%48" in script
    assert "N_UNITS=144" in script
    assert '--shard "$SLURM_ARRAY_TASK_ID"' in script
    assert '--of "$N_UNITS"' in script
    assert "PYTHONNOUSERSITE=1" in script
