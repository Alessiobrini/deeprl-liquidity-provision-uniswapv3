import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.experiments import normalized_ppo as N  # noqa: E402
from src.deeprl_liquidity_provision_uniswapv3.experiments.validate_normalized_ppo import (  # noqa: E402
    audit,
)


def test_normalization_changes_reward_only(monkeypatch):
    class FakeEnv:
        observation_space = None
        action_space = None

    captured = {}

    class FakeDummy:
        def __init__(self, envs):
            captured["base"] = envs[0]()

    class FakeNormalize:
        def __init__(self, env, **kwargs):
            captured["vector"] = env
            captured["kwargs"] = kwargs

    monkeypatch.setattr(N, "train_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(N, "DummyVecEnv", FakeDummy)
    monkeypatch.setattr(N, "VecNormalize", FakeNormalize)
    split = N.Split(train=[0], val=[1], test=[2])
    N.normalized_train_env("usdc_weth_005", split)
    assert captured["kwargs"] == {
        "training": True,
        "norm_obs": False,
        "norm_reward": True,
        "clip_reward": 10.0,
        "gamma": 0.99,
        "epsilon": 1e-8,
    }


def test_normalized_ppo_slurm_wrapper_has_no_site_specific_partition():
    body = (
        Path(__file__).resolve().parents[1]
        / "scripts/slurm/normalized_ppo_array.sh"
    ).read_text()
    assert "#SBATCH --partition=" not in body


def test_normalized_ppo_audit_rejects_incomplete_collection(tmp_path):
    result = audit(tmp_path)
    assert result["status"] == "INVALID"
    assert "0/48 files" in result["problems"][0]
