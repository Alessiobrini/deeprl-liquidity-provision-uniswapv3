"""Tests for the read-only combined rolling-results analysis."""
import json

import pytest

from src.deeprl_liquidity_provision_uniswapv3.data.pools import CORE
from src.deeprl_liquidity_provision_uniswapv3.experiments.combined import (
    ALGOS, HEURISTICS, EXPECTED_WINDOWS, load_runs, pool_arrays, render,
)


def write_collection(root, omit=None, disagree=None):
    for pool in CORE:
        for step in range(EXPECTED_WINDOWS):
            for ai, algo in enumerate(ALGOS):
                if omit == (pool, step, algo):
                    continue
                arms = {}
                for hi, name in enumerate(HEURISTICS):
                    test = float(step + hi)
                    if disagree == (pool, step, algo, name):
                        test += 1
                    # RecentreWhenOut wins validation, irrespective of test ordering.
                    val = 100.0 if name == "RecentreWhenOut" else float(hi)
                    arms[name] = {"test": test, "val": val, "name": name,
                                  "n_actions": hi + 1}
                label = algo.upper()
                arms[label] = {"test": float(step - ai), "val": 0.0,
                               "name": label, "n_actions": 1}
                row = {"unit": {"pool": pool, "step": step, "tag": algo},
                       "arms": arms, "provenance": {"config": {"algo": algo}}}
                (root / f"{pool}__step{step:03d}__{algo}.json").write_text(json.dumps(row))


def test_complete_collection_aligns_algorithms_and_deduplicates_heuristics(tmp_path):
    write_collection(tmp_path)
    data, complete, diagnostics = load_runs(tmp_path)
    assert complete and not diagnostics
    rewards, validation, _ = pool_arrays(data, CORE[0])
    assert len(rewards) == len(HEURISTICS) + len(ALGOS)
    assert all(len(v) == EXPECTED_WINDOWS for v in rewards.values())
    assert max(validation, key=validation.get) == "RecentreWhenOut"


def test_missing_algorithm_window_is_partial_and_writes_no_final_csv(tmp_path):
    missing = (CORE[0], 3, "qrdqn")
    write_collection(tmp_path, omit=missing)
    report_dir = tmp_path / "combined"
    text = render(tmp_path, report_dir, 1, 100, 4)
    assert text.startswith("PARTIAL — NOT A FINAL RESULT")
    assert "missing ['qrdqn']" in text
    assert not report_dir.exists()


def test_repeated_heuristic_disagreement_is_rejected(tmp_path):
    bad = (CORE[0], 2, "a2c", "Passive")
    write_collection(tmp_path, disagree=bad)
    with pytest.raises(ValueError, match="repeated Passive test values disagree"):
        load_runs(tmp_path)


def test_complete_render_writes_h1_h2_and_machine_readable_artifacts(tmp_path):
    write_collection(tmp_path)
    report_dir = tmp_path / "combined"
    text = render(tmp_path, report_dir, 7, 100, 4)
    assert text.startswith("COMPLETE — FINAL COLLECTION")
    assert text.count("H1 heuristic vs Passive") == len(CORE)
    assert text.count("H2 RL vs validation-selected heuristic") == len(CORE)
    assert "Validation-selected heuristic reference: RecentreWhenOut" in text
    assert (report_dir / "report.txt").read_text() == text
    assert (report_dir / "summary.csv").exists()
    assert (report_dir / "comparisons.csv").exists()
