import numpy as np

class PassiveWidthSweep:
    def __init__(self, width_candidates, deposit_action_idx=2):
        """
        width_candidates: list of tick widths (e.g. [20, 40, 60, ..., 200])
        deposit_action_idx: which liquidity_move_idx corresponds to depositing.
        """
        self.widths = width_candidates
        self.deposit_action_idx = deposit_action_idx

    def _tick_indices(self, env, width_ticks):
        """Compute lower/upper indices given width in ticks."""
        current_price = env.data['price'][env.current_step]
        current_tick = env.price_to_tick(current_price)
        half_width = width_ticks // 2
        lower_tick = current_tick - half_width * env.tick_step
        upper_tick = current_tick + half_width * env.tick_step
        lower_idx = max(0, (lower_tick - env.min_tick) // env.tick_step)
        upper_idx = min(env.num_tick_choices - 1, (upper_tick - env.min_tick) // env.tick_step)
        return lower_idx, upper_idx

    def run_once(self, env, width_ticks):
        """Simulate one passive strategy over entire window with fixed width."""
        lower_idx, upper_idx = self._tick_indices(env, width_ticks)
        obs = env.reset()
        done = False
        total_reward = 0.0
        # first step: deposit into the pool (liquidity_move_idx = deposit_action_idx)
        obs, r, done, info = env.step((lower_idx, upper_idx, self.deposit_action_idx))
        total_reward += info['raw_reward']
        while not done:
            # hold the same band and don’t rebalance
            obs, r, done, info = env.step((lower_idx, upper_idx, 0))
            total_reward += info['raw_reward']
        return total_reward

    def evaluate(self, env):
        """Try each width and return the best width and its reward."""
        best_width = None
        best_reward = -np.inf
        for w in self.widths:
            reward = self.run_once(env, w)
            if reward > best_reward:
                best_reward = reward
                best_width = w
        return best_width, best_reward
    
    
class VolProportionalWidth:
    def __init__(self, k_values, deposit_action_idx=2):
        """
        k_values: list of multipliers; width_t = k * sigma_t.
        """
        self.k_values = k_values
        self.deposit_action_idx = deposit_action_idx

    def evaluate(self, env):
        best_k = None
        best_reward = -np.inf
        for k in self.k_values:
            obs = env.reset()
            done = False
            total_reward = 0.0
            # initial band based on sigma_0
            sigma = env.data['volatility'][0]
            width = int(k * sigma * 100)  # convert to ticks (CAN EXPERIMENT with scaling)
            lower_idx, upper_idx = self._tick_indices(env, width)
            # deposit once
            obs, r, done, info = env.step((lower_idx, upper_idx, self.deposit_action_idx))
            total_reward += info['raw_reward']
            # recompute width each hour based on new sigma and recenter
            while not done:
                sigma = env.data['volatility'][env.current_step]
                width = int(k * sigma * 100)
                lower_idx, upper_idx = self._tick_indices(env, width)
                obs, r, done, info = env.step((lower_idx, upper_idx, 0))
                total_reward += info['raw_reward']
            if total_reward > best_reward:
                best_reward = total_reward
                best_k = k
        return best_k, best_reward

class ILMinimizer:
    """Exponential-weighted variance model to choose band that minimizes expected IL over a horizon H"""
    def __init__(self, horizon_hours, deposit_action_idx=2):
        self.H = horizon_hours
        self.deposit_action_idx = deposit_action_idx

    def optimal_width(self, sigma):
        """Choose width to minimize expected IL over horizon H"""
        # Heuristic: width ~ 2 * sigma * sqrt(H)
        # Heuristic (width) Optimization: TODO: Derive formulas from Cartea-Drissi-Monga (2023)
        return int(2 * sigma * np.sqrt(self.H) * 100)
    
    def evaluate(self, env):
        obs = env.reset()
        done = False
        total_reard = 0.0
        sigma = env.data['volatility'][0]
        width = self.optimal_width(sigma)
        lower_idx, upper_idx = self._tick_indices(env, width)
        obs, r, done, info = env.step((lower_idx, upper_idx, self.deposit_action_idx))
        total_reward += info['raw_reward']
        while not done:
            sigma = env.data['volatility'][env.current_step]
            width = self.optimal_width(sigma)
            lower_idx, upper_idx = self._tick_indices(env, width)
            obs, r, done, info = env.step((lower_idx, upper_idx, 0))
            total_reward += info['raw_reward']
        return width, total_reard
