import numpy as np

class PassiveWidthSweep:
    def __init__(self, width_candidates, deposit_action_idx=1):
        """
        width_candidates: list of widths (e.g. [20, 40, 60, 80, 100]).
        deposit_action_idx: index corresponding to 'deposit' in env.action_values.
        """
        self.widths = width_candidates
        self.deposit_action_idx = deposit_action_idx

    def _find_action_index(self, env, width):
        """Return the index in env.action_values closest to the desired width."""
        diffs = [abs(w - width) for w in env.action_values]
        return int(np.argmin(diffs))

    def run_once(self, env, width):
        """Run a passive policy with a fixed width for one episode."""
        # find nearest action index for initial deposit
        action_idx = self._find_action_index(env, width)
        obs, _ = env.reset()
        done = False
        total_raw_reward = 0.0
        # deposit liquidity at the beginning
        _, _, done, _, info = env.step(action_idx)
        total_raw_reward += info.get('raw_reward', 0.0)
        while not done:
            # hold liquidity (action 0 -> no rebalance)
            _, _, done, _, info = env.step(0)
            total_raw_reward += info.get('raw_reward', 0.0)
        return total_raw_reward

    def evaluate(self, env):
        """Try each width and return the best width and its reward."""
        best_width = None
        best_reward = -np.inf
        for w in self.widths:
            if w not in env.action_values:
                # only evaluate widths that exist in env.action_values
                continue
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
        return width, total_reward
    
class ReactiveRecentering:
    """On basis z-score/volatility"""
    def __init__(self, tau_values, volatility_thresholds, width_ticks=100, deposit_action_idx=2):
        """
        tau_values = list of basis_z thresholds 
        volatility_threasholds = thresholds of sigma for recentering
        width_ticks = constant width used when recentering
        """
        self.tau_Values = tau_values
        self.vol_thresholds - volatility_thresholds
        self.width = width_ticks
        self.deposit_action_idx = deposit_action_idx

    def evaluate(self, env):
        best_params = None
        best_reward = -np.inf
        for tau in self.tau_Values:
            for vol_th in self.vol_thresholds:
                obs = env.reset()
                done = False
                total = 0.0
                lower_idx, upper_idx = self._tick_indices(env, self.width)
                # deposit
                obs, r, done, info = env.steps((lower_idx, upper_idx, self.deposit_action_idx))
                total += info['raw_reward']
                while not done:
                    basis_z = env.data['basis_z'][env.current_step]
                    sigma = env.data['volatility'][env.current_step]
                    # recenter if basis_z or volatility exceed threshold
                    if abs(basis_z) > tau or sigma > vol.th:
                        lower_idx, upper_idx = self._tick_indices(env, self.width)
                        obs, r, done, info = env.steps((lower_idx, upper_idx, self.deposit_action_idx))
                    else:
                        obs, r, done, info = env.steps((lower_idx, upper_idx, 0))
                    total += info['raw_reward']
                if total > best_reward:
                    best_reward = total
                    best_params = (tau, vol_th)
        return best_params, best_reward

