"""Matched-window analysis of PPO schedule, shaping, and extractor sensitivities."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..data.pools import CORE
from .rolling import holm_adjust, paired_block_inference

EXPECTED_VARIANTS = {
    "schedule_hourly", "schedule_daily", "schedule_weekly",
    "shaping_shadow", "shaping_lvr", "paper_extractor",
}


def variant_name(config: dict) -> str:
    if config.get("paper_extractor"):
        return "paper_extractor"
    if config.get("shaping", "none") != "none":
        return f"shaping_{config['shaping']}"
    return f"schedule_{config.get('schedule', 'event_driven')}"


def _load(directory: Path) -> dict:
    rows = {}
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text())
        cfg = row.get("provenance", {}).get("config", {})
        if str(cfg.get("algo", "")).lower() != "ppo":
            continue
        key = (row["unit"]["pool"], int(row["unit"]["step"]))
        variant = variant_name(cfg)
        if key in rows.setdefault(variant, {}):
            raise ValueError(f"duplicate {variant} result for {key} in {directory}")
        rows[variant][key] = row
    return rows


def _reward(row: dict) -> float:
    return float(row["arms"]["PPO"]["test"])


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def render(primary_dir: Path, sensitivity_dir: Path, report_dir: Path,
           seed: int, samples: int, block: int) -> str:
    primary_all, variants = _load(primary_dir), _load(sensitivity_dir)
    primary = primary_all.get("schedule_event_driven", {})
    expected = {(p, s) for p in CORE for s in range(24)}
    diagnostics = []
    if set(primary) != expected:
        diagnostics.append(f"primary has {len(primary)}/144 windows")
    missing_variants = EXPECTED_VARIANTS - set(variants)
    unexpected_variants = set(variants) - EXPECTED_VARIANTS
    if missing_variants:
        diagnostics.append("missing variants: " + ", ".join(sorted(missing_variants)))
    if unexpected_variants:
        diagnostics.append("unexpected variants: " +
                           ", ".join(sorted(unexpected_variants)))
    for name, rows in sorted(variants.items()):
        if set(rows) != expected:
            diagnostics.append(f"{name} has {len(rows)}/144 windows")
    status = "COMPLETE — FINAL COLLECTION" if variants and not diagnostics else \
        "PARTIAL — NOT A FINAL RESULT"
    lines = [status, f"Primary: {primary_dir}; sensitivities: {sensitivity_dir}"]
    if diagnostics:
        lines.extend(diagnostics)
        return "\n".join(lines) + "\n"

    summary, comparisons = [], []
    for pool_i, pool in enumerate(CORE):
        base = np.array([_reward(primary[(pool, s)]) for s in range(24)])
        lines.extend(["", f"POOL {pool}; primary event-driven/default extractor/no shaping",
                      f"{'variant':<24} {'mean':>9} {'vs primary':>11} {'wins':>6} "
                      f"{'95% block CI':>23} {'p Holm':>9}", "-" * 86])
        raw = []
        for variant_i, (name, rows) in enumerate(sorted(variants.items())):
            values = np.array([_reward(rows[(pool, s)]) for s in range(24)])
            delta = values - base
            rng = np.random.default_rng(seed + 100 * pool_i + variant_i)
            lo, hi, p = paired_block_inference(delta, rng, samples, block)
            raw.append({"pool": pool, "variant": name, "mean": float(values.mean()),
                        "primary_mean": float(base.mean()), "difference": float(delta.mean()),
                        "wins": float((delta > 0).mean()), "ci_low": lo, "ci_high": hi,
                        "p_raw": p})
            summary.append({"pool": pool, "variant": name,
                            "mean_reward": float(values.mean())})
        for row, adjusted in zip(raw, holm_adjust([r["p_raw"] for r in raw])):
            row["p_holm"] = adjusted; comparisons.append(row)
            ci = f"[{row['ci_low']:+,.0f}, {row['ci_high']:+,.0f}]"
            lines.append(f"{row['variant']:<24} {row['mean']:>9,.0f} "
                         f"{row['difference']:>+11,.0f} {row['wins']:>5.0%} "
                         f"{ci:>23} {adjusted:>9.4f}")
    lines.extend(["", "Paired by seed-averaged test window; moving-block bootstrap; "
                   "Holm correction across sensitivity variants within each pool."])
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "summary.csv", summary)
    _write_csv(report_dir / "comparisons.csv", comparisons)
    text = "\n".join(lines) + "\n"
    (report_dir / "report.txt").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--sensitivities", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--bootstrap-seed", type=int, default=20260716)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--bootstrap-block", type=int, default=4)
    args = ap.parse_args()
    print(render(args.primary, args.sensitivities, args.report_dir,
                 args.bootstrap_seed, args.bootstrap_samples,
                 args.bootstrap_block), end="")


if __name__ == "__main__":
    main()
