#!/usr/bin/env python3
"""Audit complete collections and regenerate paper aggregates without training."""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BLOCKS = (2, 4, 6, 8)
ALGOS = ("ppo", "a2c", "dqn", "qrdqn", "recurrentppo")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artifact_hashes(root: Path) -> dict[str, str]:
    """Hash public tables and figures, excluding reports with machine paths."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".csv", ".pdf", ".png"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/paper_reproduced")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    source, destination = args.input_root.resolve(), args.out.resolve()
    if source == destination or destination in source.parents:
        parser.error("--out must not replace the input root or one of its parents")
    if args.data_dir:
        os.environ["CLEB_DATA_DIR"] = str(args.data_dir.resolve())

    from src.deeprl_liquidity_provision_uniswapv3.data.pools import CORE
    from src.deeprl_liquidity_provision_uniswapv3.experiments import (
        action_geometry, combined, normalized_ppo, oracle_calibration,
        oracle_pool_null, ppo_budget_ablation, retail_frontier,
        retail_frontier_summary, sensitivity, state_headroom,
        transfer_rolling, transfer_robustness_summary, validate_action_geometry,
        validate_normalized_ppo, validate_oracle_calibration,
        validate_ppo_budget_ablation, validate_results,
        validate_retail_frontier, validate_state_headroom,
    )

    primary = source / "algos_v1"
    audits = {"primary_transfer_sensitivity": validate_results.audit(
        primary, source / "transfer_v1", source / "sensitivity_v1"
    )}
    validators = {
        "action_geometry_v1": validate_action_geometry,
        "state_headroom_v1": validate_state_headroom,
        "normalized_ppo_v1": validate_normalized_ppo,
        "oracle_calibration_v1": validate_oracle_calibration,
        "ppo_budget_ablation_v1": validate_ppo_budget_ablation,
    }
    for name, module in validators.items():
        directory = source / name
        audits[name] = module.audit(directory, {(directory / "audit.json").resolve()})
    audits["retail_frontier_v1"] = validate_retail_frontier.audit(
        source / "retail_frontier_v1", primary
    )
    invalid = {name: report for name, report in audits.items() if report["status"] != "COMPLETE"}
    if invalid:
        raise ValueError("Incomplete or invalid collections: " + json.dumps(invalid))
    null_root = source / "oracle_pool_null_v1"
    for pool in CORE:
        for step in range(24):
            path = null_root / f"{pool}__step{step:03d}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing null draws: {path}. Regenerate or retrieve the full collection.")
    # Check plotting dependencies before writing any final aggregates.
    if args.figures:
        from src.deeprl_liquidity_provision_uniswapv3.experiments import plot_extensions

    destination.mkdir(parents=True, exist_ok=True)
    geometry = action_geometry.load_design(primary, source / "action_geometry_v1")
    retail_rows = [
        json.loads(path.read_text())
        for path in sorted((source / "retail_frontier_v1").glob("*__*.json"))
    ]
    for block in BLOCKS:
        combined.render(primary, destination / f"benchmark/aggregate_block{block}", 20260716, 10_000, block)
        action_geometry.analyze(geometry, destination / f"geometry/aggregate_block{block}", block=block)
        retail_frontier.aggregate(retail_rows, destination / f"retail/aggregate_block{block}", block=block)
        state_headroom.analyze(
            source / "state_headroom_v1", source / "action_geometry_v1",
            destination / f"headroom/aggregate_block{block}", block=block,
        )
        normalized_ppo.analyze(
            source / "normalized_ppo_v1", source / "action_geometry_v1",
            destination / f"normalized/aggregate_block{block}", block=block,
        )
        oracle_calibration.analyze(
            source / "oracle_calibration_v1", destination / f"oracle/aggregate_block{block}", block=block,
        )
        ppo_budget_ablation.analyze(
            source / "ppo_budget_ablation_v1", destination / f"budget/aggregate_block{block}", block=block,
        )
        sensitivity.render(
            primary, source / "sensitivity_v1", destination / f"sensitivity/aggregate_block{block}",
            20260716, 10_000, block,
        )
        for algo in ALGOS:
            config = argparse.Namespace(
                algo=algo, widths=[45, 50, 55], schedule="event_driven",
                steps=20_000, seeds=[42, 123], shaping="none", paper_extractor=False,
            )
            units = transfer_rolling.work_units(CORE, transfer_rolling.config_tag(config))
            transfer_rolling.aggregate(
                source / "transfer_v1", units, CORE,
                destination / f"transfer/aggregate_{algo}_block{block}",
                20260716, 10_000, block,
            )
        print(f"Regenerated block length {block}", flush=True)
    write_csv(destination / "retail/counts.csv", retail_frontier_summary.summarize_root(destination / "retail"))
    write_csv(destination / "transfer/counts.csv", transfer_robustness_summary.summarize_root(destination / "transfer"))
    oracle_pool_null.aggregate(null_root, destination / "oracle/pool_randomization.csv")
    if args.figures:
        figures = destination / "figures"
        plot_extensions.plot_geometry(destination / "geometry/aggregate_block4/comparisons.csv", figures)
        plot_extensions.plot_retail(destination / "retail/aggregate_block4/frontier.csv", figures)
    from src.deeprl_liquidity_provision_uniswapv3.experiments.bakeoff import PANEL
    data_hashes = {
        name: hashlib.sha256((PANEL / name).read_bytes()).hexdigest()
        for pool in CORE for name in (f"{pool}_hourly.parquet", f"{pool}_swaps.parquet")
    }
    code_hashes = {}
    for folder in ("src", "scripts", "tests"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh"}:
                code_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "status": "COMPLETE", "operation": "aggregate reproduction; no training",
        "blocks": list(BLOCKS), "bootstrap_samples": 10_000,
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_files_sha256": code_hashes, "data_files_sha256": data_hashes,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "package_versions": {
            name: metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "torch", "gymnasium",
                         "stable-baselines3", "sb3-contrib", "TA-Lib")
        },
        "environment_files_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("environment.yml", "pyproject.toml")
        },
        "collection_audits": audits, "artifacts_sha256": artifact_hashes(destination),
        "pool_null_npz_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(null_root.glob("*.npz"))
        },
    }
    if args.figures:
        manifest["package_versions"]["matplotlib"] = metadata.version("matplotlib")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("COMPLETE: all required collections audited and aggregates regenerated.")


if __name__ == "__main__":
    main()
