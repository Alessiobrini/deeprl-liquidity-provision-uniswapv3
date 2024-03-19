# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:24:34 2024

@author: ab978
"""
import pdb
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

'''
PROBLEM STATEMENT:  a liquidity provider (LP) holds capital and want to participate in
an AMM Uniswap v3 pool. She can adjust the position at discrete hourly time steps.

It is an optimal uniform allocation strategy around the current price of the token pair.

There has to be a mapping: action is discrete integer i, then current price is p. 
The interval will be [current_tick - i, current_tick + i] which then translates to
[p_l,p_u]

state variables:
    - tech idx
    - USD in portfolio
    - width of liquidity interval (previous action)
    - value of liquidity position at t in USD
    - central tick of the price interval
    
Value of liquidity position can be initialized in several ways (varying initial funds)

To map ticks to prices one needs to implement the formula at page 161 of Ottina book

Action space is discrete from 0 to N where N is the max width of the liquidity range allowed.

Reward is the result of liquidity reallocation

'''

# check how to express


c_t = 1000000 # cash hold
m_t = 10 # central tick to be inferred from the price
w_t = 0 # width of liquidity interval
l_t = 0 # value of initial liquidity




class Uniswapv3Env(gym.Env):
    """
    A custom environment for simulating interaction in a Uniswapv3 AMM.
    
    Attributes:
        delta (float): The fee tier of the AMM.
        n_actions (int): Choices for price range width
        market_data (np.ndarray): The preorganized data from a pandas DataFrame, used for simulation.
        d (float): The tick spacing of the AMM.
        l (float): Initial liquidity
        gas (float): fixed gas fee
    """
    
    def __init__(self, 
                 delta: float, 
                 n_actions: int, 
                 market_data: pd.DataFrame,
                 l: float,
                 gas: float):
        super(Uniswapv3Env, self).__init__()
        # store array of preorganized data from a pandas dataframe
        self.market_data = market_data.values.astype(np.float32)
        # store the column names of the pandas dataframe
        self.names = market_data.columns.tolist()
        self.names.extend(['c','m','w','l'])
        self.delta = delta
        self.gas = gas
        self.d = self._fee_to_tickspacing(self.delta) # tick spacing
        self.c = 0 # initial cash
        self.l = l # initial liquidity provided
        self.w = 1 # initial interval width
        self.count = 0 # iteration counter

        # Boundaries to choose
        lower_bounds = []
        upper_bounds = []
        for name in self.names:
            lower_bounds.append(-np.inf)
            upper_bounds.append(np.inf)
        lower_bounds = np.array(lower_bounds, dtype=np.float32)
        upper_bounds = np.array(upper_bounds, dtype=np.float32)
        # Define the observation space as a continuous vector space
        self.observation_space = spaces.Box(low=lower_bounds, high=upper_bounds, shape=(len(self.names),), dtype=np.float32)
        # To test it run "self.observation_space.contains(np.array([1,2,1,1,1,1],dtype=np.float32))"
        # while in the env. should return True
        self.action_space = spaces.Discrete(n_actions)
        # Initialize current state
        self.current_state = None
        
            
    def reset(self, **kwargs):
        # Assuming self.market_data[0] is a NumPy array
        m = self._price2tick(self.market_data[self.count,0])  # Convert price to tick
        # Convert m, w, and l into a NumPy array of the same shape as self.market_data[0]
        additional_info = np.array([self.c, m, self.w, self.l])
        # Concatenate the additional_info array with the self.market_data[0] array
        self.current_state = np.concatenate((self.market_data[self.count], additional_info))

        # # Ensure the returned observation is within the observation_space
        # if not self.observation_space.contains(self.current_state):
        #     raise ValueError("The observation returned by the step() method is not within the observation space.")

        
        return self.current_state, {} # a dict of info is needed and I initialized it empty
    
    def step(self, action):
        """
        Advances the environment by one step based on the given action.

        Parameters:
            action (int): The action taken by the external RL agent, representing an interest rate adjustment.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: A tuple containing the next observation, the calculated reward, done flag, and additional info.
        """
        
        self.count += 1
        # calculate mid tick corresponding to AMM market price
        pt_1,pt = self.market_data[self.count-1,0], self.market_data[self.count,0]
        m = self._price2tick(pt)
        # action is the width of the uniform price range
        self.w = action
        
        # calculate price range
        tl, tu = m - self.d*self.w, m + self.d*self.w
        pl, pu = self._tick2price(tl), self._tick2price(tu)
        

        
        # reward as per the original paper
        gas_fee = self._indicator(action)*self.gas
        
        if pt < pl:
            fees = self._calculate_fee(pt_1, pl)
        elif pt > pu:
            fees = self._calculate_fee(pt_1, pu)
        else:
            fees = self._calculate_fee(pt_1, pt)
        self.c += fees
        
        
        xt_1,yt_1 = self._calculate_xy(pt_1, pl, pu)
        xt,yt = self._calculate_xy(pt, pl, pu)
        deltaV = pt*xt + yt - (pt_1 * xt_1 + yt_1)
        lvr = deltaV - xt*(pt-pt_1)
        pdb.set_trace()
        reward = -gas_fee + fees + lvr
        
        additional_info = np.array([self.c, m, self.w, self.l])
        self.current_state = np.concatenate((self.market_data[self.count], additional_info))
        
        # Do we need termination? 
        # terminated = None
        # Do we ever want to implement sparse reward?
        # reward = 1 if terminated else 0  # Binary sparse rewards
        done = False  # Implement your termination logic here
        truncated = False
        info = {}

        # # Ensure the returned observation is within the observation_space
        # if not self.observation_space.contains(self.current_state):
        #     raise ValueError("The observation returned by the step() method is not within the observation space.")


        return self.current_state, reward, done, truncated, info
    
    def _price2tick(self, p: float):
        return math.floor(math.log(p, 1.0001))
    
    def _tick2price(self, t: int):
        return 1.0001**t
    
    def _fee_to_tickspacing(self, fee_tier: float):
        if fee_tier == 0.05:
            return 10
        elif fee_tier == 0.30:
            return 60
        elif fee_tier == 1.00:
            return 200
        else:
            raise ValueError(f"Unsupported fee tier: {fee_tier}")
            
    def _calculate_fee(self, p, p_prime):
        if p <= p_prime:
            fee = (self.delta / (1 - self.delta)) * self.l * (math.sqrt(p) - math.sqrt(p_prime))
        else:
            fee = (self.delta / (1 - self.delta)) * self.l * ((1 / math.sqrt(p)) - (1 / math.sqrt(p_prime))) * p_prime
        return fee
    
    def _indicator(self,a):
        return 1 if a != 0 else 0
    
    def _calculate_xy(self, p, pl, pu):
        if p <= pl:
            x = self.l * (1 / math.sqrt(pl) - 1 / math.sqrt(pu))
            y = 0
        elif p >= pu:
            x = 0
            y = self.l * (math.sqrt(pu) - math.sqrt(pl))
        else:  # pl < p < pu
            x = self.l * (1 / math.sqrt(p) - 1 / math.sqrt(pu))
            y = self.l * (math.sqrt(p) - math.sqrt(pl))
        return x, y



# register(
#     id='EconEnv-v0',
#     entry_point=__name__ + ':EconEnv',  # Adjust 'patorch.to.your_module' to the actual path of your EconEnv class
# )

class CustomMLPFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom MLP feature extractor.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(CustomMLPFeatureExtractor, self).__init__(observation_space, features_dim)

        # Define the architecture of the feature extractor
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)



if __name__ == '__main__':

    np.random.seed(42)
    data = pd.DataFrame(np.abs(np.random.randn(500,2)).astype(np.float32)) + 1000.0
    data.columns = ['tech_idx1','tech_idx2']

    uni_env = Uniswapv3Env(delta=0.05, n_actions=10, 
                           market_data=data,  l=10**6, gas=1)


    obs, _ = uni_env.reset()
    
    action = 1
    uni_env.step(action)

    # # Define policy_kwargs with the custom feature extractor
    # policy_kwargs = dict(
    #     features_extractor_class=CustomMLPFeatureExtractor,
    #     features_extractor_kwargs=dict(features_dim=128),  # This dimension will be the output of the feature extractor
    # )
    
    # env = gym.make('EconEnv-v0', stepsize=0.25, n_actions=10, data=df, econ_model=model)
    # # Create the PPO model with the custom MLP policy and feature extractor
    # rlmodel = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=2)
    
    # # Train the model
    # rlmodel.learn(total_timesteps=10000)
    
    # # Test the trained model
    # obs, _ = env.reset()
    # for _ in range(20):
    #     action, _states = rlmodel.predict(obs, deterministic=False)
    #     obs, rewards, done, truncated, info = env.step(action)
    #     print(f'Action taken: {action}')
    #     print(f'Next State: {obs}')
    #     print(f'Reward: {rewards}\n')