import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.experiments import validate_retail_frontier as V  # noqa: E402


def test_empty_frontier_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "CORE", ["pool"])
    monkeypatch.setattr(V, "CAPITALS", (1_500,))
    result = V.audit(tmp_path, tmp_path)
    assert result["status"] == "INVALID"
    assert result["count"] == 0
    assert any("0/720" in p for p in result["problems"])
