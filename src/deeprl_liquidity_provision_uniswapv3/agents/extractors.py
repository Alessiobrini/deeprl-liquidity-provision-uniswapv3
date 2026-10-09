"""The previous paper's policy network, ported.

Its runner did NOT use a default MLP. It built PPO with a `CustomMLPFeatureExtractor`
(in the codebase of Xu and Brini, 2025, arXiv 2501.07508) whose first layer is a
non-affine BatchNorm over the raw observation:

    BatchNorm1d(obs_dim, affine=False)
      -> Linear(obs_dim, h[0]) -> act
      -> Linear(h[0], h[1])    -> act
      -> Linear(h[1], features_dim)

with `features_dim=128`, `hidden_dim` from the Optuna-selected `dim_hidden_layers`
(default `[4, 2]`), and `activation` from the config (default `tanh`).

That BatchNorm is load-bearing, not decoration. The paper's state carries raw prices
(~2,000), raw liquidity, and unscaled TA-Lib indicators on wildly different scales.
Handing that to SB3's default `MlpPolicy`, which has no input normalization, is a
different agent, and it is the one this rebuild has been running while claiming to
compare against the paper's.

Two faithful details worth keeping rather than silently improving:

- `affine=False`, so there are no learnable scale/shift parameters here.
- The original hardcodes exactly TWO hidden layers by indexing `hidden_dim[0]` and
  `hidden_dim[1]`, so a 3-element `dim_hidden_layers` from its own Optuna search
  space (`[4,4,4]`, `[6,6,6]` are both in the YAML) silently drops the third layer.
  Replicated: a 3-layer config in the previous paper's search space never built a
  3-layer net, and reproducing its behaviour means reproducing that too.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PaperMLPExtractor(BaseFeaturesExtractor):
    """BatchNorm -> two hidden layers -> features_dim. The previous paper's extractor."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 128,
                 hidden_dim=(4, 2), activation: str = "tanh"):
        super().__init__(observation_space, features_dim)
        act = nn.ReLU() if activation == "relu" else nn.Tanh()
        n = observation_space.shape[0]
        h = list(hidden_dim)
        if len(h) < 2:
            raise ValueError(f"hidden_dim needs at least 2 entries, got {hidden_dim}")
        # Indexing [0] and [1] only: faithful to the original, which ignores any
        # third layer its own search space could hand it.
        self.net = nn.Sequential(
            nn.BatchNorm1d(n, affine=False),
            nn.Linear(n, h[0]), act,
            nn.Linear(h[0], h[1]), act,
            nn.Linear(h[1], features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
