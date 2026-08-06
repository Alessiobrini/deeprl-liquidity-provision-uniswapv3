"""Independent structural audit for the PPO training-budget collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

from .ppo_budget_ablation import (
    ACTION_LABELS,
    BUDGETS,
    EXECUTION_WIDTHS,
    EXPECTED_WINDOWS,
    POOLS,
    SCHEDULE,
    SEEDS,
    WIDTHS,
    _sha256,
    expected_units,
)
from .rolling import grid_for


def collection_hash(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def audit(root: Path, ignored: set[Path] | None = None) -> dict:
    ignored = {path.resolve() for path in (ignored or set())}
    files = sorted(
        path for path in root.glob("*__budget*.json")
        if path.resolve() not in ignored
    )
    expected = {(unit.pool, unit.step, unit.budget) for unit in expected_units()}
    problems, seen, checkpoint_files = [], set(), []
    valid_configs = grid_for("ppo")
    for path in files:
        try:
            row = json.loads(path.read_text())
        except Exception as exc:
            problems.append(f"{path.name}: invalid JSON: {exc}")
            continue
        unit = row.get("unit", {})
        pool, step, budget = unit.get("pool"), unit.get("step"), unit.get("budget")
        key = (pool, step, budget)
        if key in seen:
            problems.append(f"{path.name}: duplicate {key}")
        seen.add(key)
        expected_split = {
            "train": list(range(step, step + 4)),
            "val": [step + 4],
            "test": [step + 5],
        } if isinstance(step, int) else None
        if row.get("split") != expected_split:
            problems.append(f"{path.name}: split mismatch")

        provenance = row.get("provenance", {})
        checks = {
            "widths": list(WIDTHS),
            "execution_widths": list(EXECUTION_WIDTHS),
            "width_units": "spacing",
            "schedule": SCHEDULE,
            "budget": budget,
            "seeds": list(SEEDS),
        }
        for name, value in checks.items():
            if provenance.get(name) != value:
                problems.append(
                    f"{path.name}: provenance {name}={provenance.get(name)!r}, "
                    f"expected {value!r}"
                )

        result = row.get("ppo", {})
        selected = result.get("selected_config")
        if selected not in valid_configs:
            problems.append(f"{path.name}: invalid selected config")
        search = result.get("validation_search", [])
        if len(search) != len(valid_configs):
            problems.append(f"{path.name}: validation search has {len(search)} candidates")
        else:
            for index, (candidate, expected_config) in enumerate(zip(search, valid_configs)):
                if candidate.get("config") != expected_config:
                    problems.append(f"{path.name}: candidate {index} config mismatch")
                seed_validation = candidate.get("seed_validation", [])
                if [item.get("seed") for item in seed_validation] != list(SEEDS):
                    problems.append(f"{path.name}: candidate {index} seed mismatch")
                values = [item.get("reward", float("nan")) for item in seed_validation]
                if not all(math.isfinite(value) for value in values):
                    problems.append(f"{path.name}: candidate {index} non-finite validation")
                elif not np.isclose(candidate.get("mean_validation", np.nan), np.mean(values)):
                    problems.append(f"{path.name}: candidate {index} mean mismatch")
            means = [candidate["mean_validation"] for candidate in search]
            expected_selected = search[int(np.argmax(means))]["config"]
            if selected != expected_selected:
                problems.append(f"{path.name}: selected config is not validation argmax")
            if not np.isclose(result.get("validation", np.nan), max(means)):
                problems.append(f"{path.name}: selected validation mismatch")

        per_seed = result.get("per_seed", [])
        if [item.get("seed") for item in per_seed] != list(SEEDS):
            problems.append(f"{path.name}: refit seed mismatch")
            continue
        rewards = []
        for seed_row in per_seed:
            seed = seed_row["seed"]
            rewards.append(seed_row.get("reward", np.nan))
            checkpoint_text = seed_row.get("checkpoint", "")
            checkpoint_rel = PurePosixPath(checkpoint_text)
            if checkpoint_rel.is_absolute() or ".." in checkpoint_rel.parts:
                problems.append(f"{path.name}: unsafe checkpoint path for seed {seed}")
                continue
            checkpoint = root / Path(*checkpoint_rel.parts)
            checkpoint_files.append(checkpoint)
            if not checkpoint.is_file():
                problems.append(f"{path.name}: missing checkpoint for seed {seed}")
            elif _sha256(checkpoint) != seed_row.get("checkpoint_sha256"):
                problems.append(f"{path.name}: checkpoint hash mismatch for seed {seed}")

            histogram = seed_row.get("action_histogram", {})
            if set(histogram) != set(ACTION_LABELS):
                problems.append(f"{path.name}: action labels mismatch for seed {seed}")
            elif sum(histogram.values()) != seed_row.get("n_decisions"):
                problems.append(f"{path.name}: action count mismatch for seed {seed}")
            elif sum(value > 0 for value in histogram.values()) != seed_row.get("n_actions"):
                problems.append(f"{path.name}: distinct-action mismatch for seed {seed}")
            probabilities = seed_row.get("mean_action_probability", {})
            if set(probabilities) != set(ACTION_LABELS) or not np.isclose(
                sum(probabilities.values()), 1.0
            ):
                problems.append(f"{path.name}: probability mismatch for seed {seed}")
            for name in ("reward", "mean_policy_entropy", "normalized_policy_entropy",
                         "state_dependence_tv", "max_probability_range"):
                if not math.isfinite(seed_row.get(name, float("nan"))):
                    problems.append(f"{path.name}: non-finite {name} for seed {seed}")
            for name in ("normalized_policy_entropy", "state_dependence_tv",
                         "max_probability_range"):
                value = seed_row.get(name, -1.0)
                if not 0.0 <= value <= 1.0 + 1e-12:
                    problems.append(f"{path.name}: invalid {name} for seed {seed}")
        if all(math.isfinite(value) for value in rewards) and not np.isclose(
            result.get("test", np.nan), np.mean(rewards)
        ):
            problems.append(f"{path.name}: test reward is not the seed mean")

    missing, extra = expected - seen, seen - expected
    if extra:
        problems.append(f"unexpected units: {sorted(extra)[:5]}")
    incomplete = len(files) != len(expected) or bool(missing)
    if incomplete:
        problems.append(
            f"{len(files)}/{len(expected)} files; missing={len(missing)} extra={len(extra)}"
        )
    expected_checkpoints = len(expected) * len(SEEDS)
    if len(checkpoint_files) != len(files) * len(SEEDS):
        problems.append(
            f"{len(checkpoint_files)}/{len(files) * len(SEEDS)} referenced checkpoints"
        )

    invalid = any(
        not problem.startswith(f"{len(files)}/{len(expected)} files;")
        for problem in problems
    )
    status = "INVALID" if invalid else ("PARTIAL" if incomplete else "COMPLETE")
    return {
        "status": status,
        "problems": problems,
        "count": len(files),
        "expected_count": len(expected),
        "referenced_checkpoints": len(checkpoint_files),
        "expected_checkpoints": expected_checkpoints,
        "json_sha256": collection_hash(files) if files else hashlib.sha256(b"").hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, {args.output})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
