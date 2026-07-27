import json

from src.deeprl_liquidity_provision_uniswapv3.data.pools import CORE
from src.deeprl_liquidity_provision_uniswapv3.experiments.sensitivity import EXPECTED_VARIANTS
from src.deeprl_liquidity_provision_uniswapv3.experiments.validate_results import ALGOS, audit


def test_complete_audit_and_native_mismatch(tmp_path):
    p, t, s = (tmp_path / x for x in "pts")
    for d in (p, t, s): d.mkdir()
    for algo in ALGOS:
        for pool in CORE:
            for step in range(24):
                cfg = {"algo": algo, "schedule": "event_driven", "shaping": "none",
                       "paper_extractor": False}
                base = float(step)
                primary = {"unit": {"pool": pool, "step": step},
                           "arms": {algo.upper(): {"test": base}},
                           "provenance": {"config": cfg}}
                (p / f"{algo}-{pool}-{step}.json").write_text(json.dumps(primary))
                targets = {target: {"test": base, "per_seed": [base, base]}
                           for target in CORE}
                transfer = {"unit": {"source": pool, "step": step},
                            "split": {"test": [step + 5]}, "targets": targets,
                            "provenance": {"config": cfg}}
                (t / f"{algo}-{pool}-{step}.json").write_text(json.dumps(transfer))
    for variant in EXPECTED_VARIANTS:
        for pool in CORE:
            for step in range(24):
                cfg = {"algo": "ppo", "schedule": "event_driven", "shaping": "none",
                       "paper_extractor": variant == "paper_extractor"}
                if variant.startswith("schedule_"): cfg["schedule"] = variant[9:]
                if variant.startswith("shaping_"): cfg["shaping"] = variant[8:]
                row = {"unit": {"pool": pool, "step": step},
                       "split": {"test": [step + 5]}, "arms": {"PPO": {"test": 0}},
                       "provenance": {"config": cfg}}
                (s / f"{variant}-{pool}-{step}.json").write_text(json.dumps(row))
    assert audit(p, t, s)["status"] == "COMPLETE"
    broken = json.loads((t / f"ppo-{CORE[0]}-0.json").read_text())
    broken["targets"][CORE[0]]["test"] = 1
    (t / f"ppo-{CORE[0]}-0.json").write_text(json.dumps(broken))
    assert "native-primary mismatches 1" in audit(p, t, s)["problems"]
