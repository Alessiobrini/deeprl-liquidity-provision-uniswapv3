"""Leak-free, resumable walk-forward cross-pool policy transfer.

One work unit is (algorithm, source pool, rolling step). Configuration selection
uses only the source validation window, models are refit on the five source windows
preceding test, and the frozen models are evaluated on the same unseen next window
of every target pool. Training is therefore done once per source and reused across
targets without allowing any target test reward into selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..agents.registry import make_agent
from ..data.pools import CORE, POOLS
from .agent_arm import score_agent, train_env
from .bakeoff import Split, load_panel
from .rolling import (canonical_widths, grid_for, holm_adjust, paired_block_inference,
                      rolling_steps)


def config_tag(args) -> str:
    keys = ("algo", "widths", "schedule", "steps", "seeds", "shaping",
            "paper_extractor")
    values = {k: getattr(args, k) for k in keys}
    values["widths"] = canonical_widths(values["widths"])
    payload = json.dumps(values, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class Unit:
    source: str
    step: int
    tag: str

    @property
    def name(self) -> str:
        return f"src-{self.source}__step{self.step:03d}__{self.tag}"


def work_units(sources: list[str], tag: str) -> list[Unit]:
    out = []
    for source in sorted(sources):
        n_windows = len(load_panel(source)) // 1500
        out.extend(Unit(source, i, tag) for i in range(len(rolling_steps(n_windows))))
    return out


def _provenance(args) -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        sha, dirty = "unknown", True
    config = json.loads(json.dumps(vars(args), default=str))
    return {"git_sha": sha, "git_dirty": dirty, "host": platform.node(),
            "python": platform.python_version(), "config": config}


def run_unit(unit: Unit, args) -> dict:
    split = rolling_steps(len(load_panel(unit.source)) // 1500)[unit.step]

    # Select one source configuration. Target data are unreachable from this loop.
    best_cfg, best_val = None, -np.inf
    for cfg in grid_for(args.algo):
        vals = []
        for seed in args.seeds:
            env = train_env(unit.source, args.widths, split, args.schedule,
                            reward_shaping=args.shaping)
            model = make_agent(args.algo, env, seed=seed,
                               paper_extractor=args.paper_extractor, **cfg)
            model.learn(total_timesteps=args.steps)
            vals.append(float(score_agent(unit.source, model, args.widths,
                                          split.val, args.schedule).mean()))
        value = float(np.mean(vals))
        if value > best_val:
            best_cfg, best_val = cfg, value

    # Refit from scratch on every source window preceding test.
    fit = Split(train=split.train + split.val, val=[], test=split.test)
    models = []
    for seed in args.seeds:
        env = train_env(unit.source, args.widths, fit, args.schedule,
                        reward_shaping=args.shaping)
        model = make_agent(args.algo, env, seed=seed,
                           paper_extractor=args.paper_extractor, **best_cfg)
        model.learn(total_timesteps=args.steps)
        models.append(model)

    targets = {}
    for target in args.targets:
        per_seed = [float(score_agent(target, model, args.widths, split.test,
                                      args.schedule).mean()) for model in models]
        targets[target] = {"test": float(np.mean(per_seed)), "per_seed": per_seed}
    return {"unit": asdict(unit),
            "split": {"train": split.train, "val": split.val, "test": split.test},
            "selection": {"config": best_cfg, "validation": best_val},
            "targets": targets, "provenance": _provenance(args)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aligned_steps(expected: list[Unit]) -> list[int]:
    """Return actual step indices, refusing unpaired source timelines."""
    by_source: dict[str, set[int]] = {}
    for unit in expected:
        by_source.setdefault(unit.source, set()).add(unit.step)
    if not by_source:
        raise ValueError("transfer aggregation has no expected work units")
    reference = next(iter(by_source.values()))
    if any(steps != reference for steps in by_source.values()):
        counts = {source: len(steps) for source, steps in sorted(by_source.items())}
        raise ValueError(
            "transfer aggregation requires aligned step indices across sources; "
            f"got {counts}"
        )
    return sorted(reference)


def aggregate(out_dir: Path, expected: list[Unit], targets: list[str], report_dir: Path,
              seed: int, samples: int, block: int) -> str:
    tag = expected[0].tag
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob(f"*__{tag}.json"))]
    done = {(r["unit"]["source"], r["unit"]["step"]) for r in rows}
    missing = [u for u in expected if (u.source, u.step) not in done]
    status = "COMPLETE — FINAL COLLECTION" if not missing else "PARTIAL — NOT A FINAL RESULT"
    lines = [status, f"{len(rows)}/{len(expected)} source-step units; config {tag}"]
    if missing:
        lines.append("Missing: " + ", ".join(u.name for u in missing[:20]))
        return "\n".join(lines) + "\n"

    by_key = {(r["unit"]["source"], r["unit"]["step"]): r for r in rows}
    steps = _aligned_steps(expected)
    sources = sorted({unit.source for unit in expected})
    summary, comparisons = [], []
    for target_i, target in enumerate(targets):
        if target not in sources:
            raise ValueError(
                f"target {target!r} has no native source run; include it in --sources"
            )
        lines.extend(["", f"TARGET {target}",
                      f"{'source':<18} {'mean':>10} {'vs native':>11} {'wins':>7} "
                      f"{'95% block CI':>23} {'p Holm':>9}", "-" * 84])
        native = np.array([by_key[(target, i)]["targets"][target]["test"]
                           for i in steps])
        eligible = []
        for source in sources:
            ps, pt = POOLS[source], POOLS[target]
            if source != target and (ps.pair == pt.pair or ps.fee_tier == pt.fee_tier):
                eligible.append(source)
        raw = []
        for source_i, source in enumerate(eligible):
            moved = np.array([by_key[(source, i)]["targets"][target]["test"]
                              for i in steps])
            delta = moved - native
            rng = np.random.default_rng(seed + 100 * target_i + source_i)
            lo, hi, p = paired_block_inference(delta, rng, samples, block)
            kind = ("within_pair_across_tier" if POOLS[source].pair == POOLS[target].pair
                    else "across_pair_within_tier")
            raw.append({"target": target, "source": source, "contrast": kind,
                        "mean_transferred": float(moved.mean()),
                        "mean_native": float(native.mean()),
                        "difference": float(delta.mean()),
                        "wins": float((delta > 0).mean()), "ci_low": lo,
                        "ci_high": hi, "p_raw": p})
        for row, adjusted in zip(raw, holm_adjust([r["p_raw"] for r in raw])):
            row["p_holm"] = adjusted
            comparisons.append(row)
            ci = f"[{row['ci_low']:+,.0f}, {row['ci_high']:+,.0f}]"
            lines.append(f"{row['source']:<18} {row['mean_transferred']:>10,.0f} "
                         f"{row['difference']:>+11,.0f} {row['wins']:>6.0%} "
                         f"{ci:>23} {row['p_holm']:>9.4f}")
        for source in sources:
            moved = np.array([by_key[(source, i)]["targets"][target]["test"]
                              for i in steps])
            summary.append({"target": target, "source": source,
                            "mean_reward": float(moved.mean()),
                            "native": source == target})

    lines.extend(["", "Inference is paired by target test window. Seeds are averaged "
                   "within windows; Holm is applied separately within each target.",
                   "Ratios are omitted because negative or near-zero native rewards make "
                   "transfer efficiency unstable."])
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "summary.csv", summary)
    _write_csv(report_dir / "comparisons.csv", comparisons)
    text = "\n".join(lines) + "\n"
    (report_dir / "report.txt").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="*", default=CORE)
    ap.add_argument("--targets", nargs="*", default=CORE)
    ap.add_argument("--widths", nargs="*", type=float, default=[45, 50, 55])
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--schedule", default="event_driven")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123])
    ap.add_argument("--shaping", default="none")
    ap.add_argument("--paper-extractor", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bootstrap-seed", type=int, default=20260716)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--bootstrap-block", type=int, default=4)
    args = ap.parse_args()

    units = work_units(args.sources, config_tag(args))
    if args.aggregate:
        report_dir = args.report_dir or args.out / f"aggregate_{args.algo}"
        print(aggregate(args.out, units, args.targets, report_dir,
                        args.bootstrap_seed, args.bootstrap_samples,
                        args.bootstrap_block), end="")
        return
    assert 0 <= args.shard < args.of
    args.out.mkdir(parents=True, exist_ok=True)
    mine = units[args.shard::args.of]
    print(f"shard {args.shard}/{args.of}: {len(mine)} of {len(units)} units")
    for unit in mine:
        path = args.out / f"{unit.name}.json"
        if path.exists() and not args.force:
            print(f"  {unit.name} skip (done)")
            continue
        result = run_unit(unit, args)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=1))
        tmp.rename(path)
        print(f"  {unit.name} " + " ".join(
            f"{k}:{v['test']:,.0f}" for k, v in result["targets"].items()))


if __name__ == "__main__":
    main()
