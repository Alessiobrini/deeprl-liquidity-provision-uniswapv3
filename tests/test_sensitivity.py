import csv
import json

from src.deeprl_liquidity_provision_uniswapv3.data.pools import CORE
from src.deeprl_liquidity_provision_uniswapv3.experiments.sensitivity import render


def write_rows(root, schedule="event_driven", shaping="none", paper=False,
               omit=None, offset=0.0):
    for pool in CORE:
        for step in range(24):
            if omit == (pool, step):
                continue
            cfg = {"algo": "ppo", "schedule": schedule, "shaping": shaping,
                   "paper_extractor": paper}
            row = {"unit": {"pool": pool, "step": step},
                   "arms": {"PPO": {"test": float(step) + offset}},
                   "provenance": {"config": cfg}}
            tag = f"{schedule}-{shaping}-{paper}"
            (root / f"{pool}-{step}-{tag}.json").write_text(json.dumps(row))


def test_complete_sensitivity_collection_writes_report(tmp_path):
    primary, variants, report = tmp_path / "p", tmp_path / "v", tmp_path / "r"
    primary.mkdir(); variants.mkdir()
    write_rows(primary)
    write_rows(variants, schedule="hourly", offset=10.0)
    write_rows(variants, schedule="daily", offset=-3.0)
    write_rows(variants, schedule="weekly", offset=5.0)
    write_rows(variants, shaping="shadow", offset=2.0)
    write_rows(variants, shaping="lvr", offset=-1.0)
    write_rows(variants, paper=True, offset=4.0)
    text = render(primary, variants, report, 1, 100, 4)
    assert text.startswith("COMPLETE — FINAL COLLECTION")
    assert (report / "comparisons.csv").exists()
    with (report / "comparisons.csv").open(newline="") as handle:
        comparisons = list(csv.DictReader(handle))
    hourly = next(
        row for row in comparisons
        if row["pool"] == CORE[0] and row["variant"] == "schedule_hourly"
    )
    assert float(hourly["mean"]) == 21.5
    assert float(hourly["primary_mean"]) == 11.5
    assert float(hourly["difference"]) == 10.0
    assert float(hourly["wins"]) == 1.0
    assert float(hourly["ci_low"]) == 10.0
    assert float(hourly["ci_high"]) == 10.0
    assert float(hourly["p_raw"]) == 1 / 101
    assert float(hourly["p_holm"]) == 6 / 101


def test_partial_variant_refuses_final_artifacts(tmp_path):
    primary, variants, report = tmp_path / "p", tmp_path / "v", tmp_path / "r"
    primary.mkdir(); variants.mkdir()
    write_rows(primary)
    write_rows(variants, schedule="daily", omit=(CORE[0], 0))
    text = render(primary, variants, report, 1, 100, 4)
    assert text.startswith("PARTIAL — NOT A FINAL RESULT")
    assert not report.exists()


def test_missing_entire_variant_refuses_final_artifacts(tmp_path):
    primary, variants, report = tmp_path / "p", tmp_path / "v", tmp_path / "r"
    primary.mkdir(); variants.mkdir()
    write_rows(primary)
    for schedule in ("hourly", "daily", "weekly"):
        write_rows(variants, schedule=schedule)
    for shaping in ("shadow", "lvr"):
        write_rows(variants, shaping=shaping)
    text = render(primary, variants, report, 1, 100, 4)
    assert "missing variants: paper_extractor" in text
    assert not report.exists()
