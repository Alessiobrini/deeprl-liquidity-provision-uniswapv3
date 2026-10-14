import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.reproduce_paper import artifact_hashes
from src.deeprl_liquidity_provision_uniswapv3.experiments import bakeoff


def test_data_directory_override_preserves_default(monkeypatch, tmp_path):
    original = bakeoff.PANEL
    with monkeypatch.context() as context:
        context.setenv("CLEB_DATA_DIR", str(tmp_path))
        assert importlib.reload(bakeoff).PANEL == tmp_path
    assert importlib.reload(bakeoff).PANEL == original


def test_public_artifact_manifest_excludes_machine_path_reports(tmp_path):
    (tmp_path / "table.csv").write_text("value\n1\n")
    (tmp_path / "report.txt").write_text("private machine path")
    assert set(artifact_hashes(tmp_path)) == {"table.csv"}
    assert len(artifact_hashes(tmp_path)["table.csv"]) == 64


def test_release_protocol_references_resolve():
    root = Path(__file__).resolve().parents[1]
    record = root / "reproducibility/protocols.md"
    assert record.is_file()
    content = record.read_text()
    for anchor in ("State headroom", "Oracle calibration", "Pool-level oracle null"):
        assert f"## {anchor}" in content


def test_frozen_artifact_hashes_match_manifest():
    root = Path(__file__).resolve().parents[1] / "reproducibility/artifacts"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETE"
    assert manifest["artifacts_sha256"] == artifact_hashes(root)
    assert len(manifest["pool_null_npz_sha256"]) == 144
    assert len(manifest["data_files_sha256"]) == 12
