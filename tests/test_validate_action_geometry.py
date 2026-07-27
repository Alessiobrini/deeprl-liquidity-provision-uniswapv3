import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.experiments import validate_action_geometry as V  # noqa: E402


def test_empty_collection_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "CORE", ["pool"])
    monkeypatch.setattr(V, "ALGOS", ("ppo",))
    result = V.audit(tmp_path)
    assert result["status"] == "INVALID"
    assert result["count"] == 0
    assert any("0/144" in p for p in result["problems"])


def test_collection_hash_depends_on_name_and_content(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"x": 1}))
    first = V.collection_hash([a])
    a.write_text(json.dumps({"x": 2}))
    assert V.collection_hash([a]) != first


def test_unexpected_json_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "CORE", [])
    monkeypatch.setattr(V, "ALGOS", ())
    (tmp_path / "stray.json").write_text("{}")
    result = V.audit(tmp_path)
    assert result["status"] == "INVALID"
    assert result["count"] == 0
    assert any("unexpected JSON" in p for p in result["problems"])


def test_exact_output_path_can_be_ignored_but_other_json_cannot(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "CORE", [])
    monkeypatch.setattr(V, "ALGOS", ())
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}")
    assert V.audit(tmp_path, {audit_path})["status"] == "COMPLETE"
    (tmp_path / "stray.json").write_text("{}")
    assert V.audit(tmp_path, {audit_path})["status"] == "INVALID"
