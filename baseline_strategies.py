import numpy as np
import math

def _nearest_action_index(env, width):
    """
    Given a desired width, find the index of the nearest available width
    in env.action_values. env.action_values is a numpy array of ints/floats.
    """
    diffs = np.abs(env.action_values - width)
    return int(np.argmin(diffs))


class PassiveWidthSweep:
    """Test a set of fixed widths and pick the one with highest PnL."""
    def __init__(self, width_candidates, deposit_action_idx=1):
        """
        width_candidates: list of widths (integers) expressed in tick spacing units.
        deposit_action_idx: index in env.action_values that triggers deposit/withdraw.
                            In Uniswapv3Env this is 1 by default (action=0 means hold).
        """
        self.widths = width_candidates
        self.deposit_action_idx = deposit_action_idx

    def _run_width(self, env, width):
        # find the closest available width index
        act_idx = _nearest_action_index(env, width)
        env.reset()
        done = False
        total_raw = 0.0
        # initial deposit
        _, _, done, _, info = env.step(act_idx)
        total_raw += info.get('raw_reward', 0.0)
        # hold the position until episode ends
        while not done:
            _, _, done, _, info = env.step(0)
            total_raw += info.get('raw_reward', 0.0)
        return total_raw

    def evaluate(self, env):
        best_w = None
        best_reward = -np.inf
        for w in self.widths:
            # skip widths not in action_values
            if w not in env.action_values:
                continue
            reward = self._run_width(env, w)
            if reward > best_reward:
                best_reward = reward
                best_w = w
        return best_w, best_reward
    
    
class VolProportionalWidth:
    """Set width ∝ volatility σ_t; choose nearest available width."""
    def __init__(self, k_values, base_factor=100.0, deposit_action_idx=1):
        """
        k_values: list of multipliers for σ.  width_t ≈ k * σ_t * base_factor.
        base_factor: converts σ into tick units (adjust empirically).
        """
        self.k_values = k_values
        self.base_factor = base_factor
        self.deposit_action_idx = deposit_action_idx

    def _run_k(self, env, k):
        env.reset()
        done = False
        total_raw = 0.0
        # compute width from sigma[0]
        sigma = env.ew_sigma[0]
        width = int(k * sigma * self.base_factor)
        act_idx = _nearest_action_index(env, width)
        # deposit at start
        _, _, done, _, info = env.step(act_idx)
        total_raw += info.get('raw_reward', 0.0)
        # each hour, recompute width based on new sigma and recenter if changed
        while not done:
            sigma = env.ew_sigma[env.count]
            width = int(k * sigma * self.base_factor)
            act_idx = _nearest_action_index(env, width)
            _, _, done, _, info = env.step(act_idx if act_idx != 0 else 0)
            total_raw += info.get('raw_reward', 0.0)
        return total_raw

    def evaluate(self, env):
        best_k, best_reward = None, -np.inf
        for k in self.k_values:
            r = self._run_k(env, k)
            if r > best_reward:
                best_reward = r
                best_k = k
        return best_k, best_reward

class ILMinimizer:
    """
    Choose width to minimize expected Impermanent Loss over a horizon H.
    Uses heuristic width ≈ 2 * σ * sqrt(H) * base_factor.
    """
    def __init__(self, horizon_hours, base_factor=100.0, deposit_action_idx=1):
        self.H = horizon_hours
        self.base_factor = base_factor
        self.deposit_action_idx = deposit_action_idx

    def _optimal_width(self, sigma):
        # width ≈ 2 * σ * sqrt(H)
        return int(2.0 * sigma * math.sqrt(self.H) * self.base_factor)

    def evaluate(self, env):
        env.reset()
        done = False
        total_raw = 0.0
        # compute initial width
        sigma = env.ew_sigma[0]
        width = self._optimal_width(sigma)
        act_idx = _nearest_action_index(env, width)
        _, _, done, _, info = env.step(act_idx)
        total_raw += info.get('raw_reward', 0.0)
        # update width each step
        while not done:
            sigma = env.ew_sigma[env.count]
            width = self._optimal_width(sigma)
            act_idx = _nearest_action_index(env, width)
            _, _, done, _, info = env.step(act_idx if act_idx != 0 else 0)
            total_raw += info.get('raw_reward', 0.0)
        return width, total_raw
    
class ReactiveRecentering:
    """
    Recenter band when volatility exceeds a threshold or large price change.
    Since this env doesn’t expose basis_z, we react to σ_t and price jumps.
    """
    def __init__(self, vol_thresholds, price_jump_thresholds,
                 width_ticks, deposit_action_idx=1):
        """
        vol_thresholds: list of σ_t thresholds (raw σ values) for recentering.
        price_jump_thresholds: list of fractional price-change thresholds (e.g. 0.01 for 1%).
        width_ticks: fixed width to use when (re)deploying.
        """
        self.vol_thresholds = vol_thresholds
        self.price_jump_thresholds = price_jump_thresholds
        self.width_ticks = width_ticks
        self.deposit_action_idx = deposit_action_idx

    def _run_params(self, env, vol_th, jump_th):
        env.reset()
        done = False
        total_raw = 0.0
        last_price = env.market_data[0]
        # initial deposit with fixed width
        act_idx = _nearest_action_index(env, self.width_ticks)
        _, _, done, _, info = env.step(act_idx)
        total_raw += info.get('raw_reward', 0.0)
        while not done:
            sigma = env.ew_sigma[env.count]
            current_price = env.market_data[env.count]
            # compute fractional price change from last hour
            pct_change = abs(current_price - last_price) / max(last_price, 1e-12)
            # recenter if σ or price change exceeds threshold
            if sigma > vol_th or pct_change > jump_th:
                act_idx = _nearest_action_index(env, self.width_ticks)
                last_price = current_price
            else:
                act_idx = 0  # hold
            _, _, done, _, info = env.step(act_idx)
            total_raw += info.get('raw_reward', 0.0)
        return total_raw

    def evaluate(self, env):
        best_params, best_reward = None, -np.inf
        for v_th in self.vol_thresholds:
            for j_th in self.price_jump_thresholds:
                r = self._run_params(env, v_th, j_th)
                if r > best_reward:
                    best_reward = r
                    best_params = (v_th, j_th)
        return best_params, best_reward