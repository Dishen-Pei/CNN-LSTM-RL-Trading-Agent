#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Step 5A: Set up trading logic


# In[2]:


# rl_trading_agent.py

import gym
import numpy as np
import pandas as pd
from gym import spaces
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler


class TradingEnv(gym.Env):
    """
    Customized reinforcement learning trading environment (compatible with OpenAI Gym) for single-label simulation trading.
    Status: the last N candlesticks and technical indicators
    Action: continuous action space, indicating the current position ratio [0, 1]
    Reward: the rate of return for this period
    """

    def __init__(self, df, window_size=24):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size

        # Establish Features
        self.features = self._build_features(self.df)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # Calculate valid length
        self.valid_length = len(self.features)
        self.max_steps = self.valid_length - self.window_size - 1

        # Gym Environment Setting
        obs_shape = (window_size, self.features.shape[1])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Reset
        self.reset()

    def _build_features(self, df):
        """Build RSI、MACD、EMA、Bollinger Band etc. Tech features"""
        df = df.copy()
        df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
        macd = MACD(close=df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["ema_5"] = EMAIndicator(close=df["close"], window=5).ema_indicator()
        df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
        bb = BollingerBands(close=df["close"], window=20)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["momentum"] = df["close"] - df["close"].shift(10)
        df = df.dropna()
        return df[["open", "high", "low", "close", "Volume BTC", "rsi", "macd", "macd_signal",
                  "ema_5", "ema_20", "bb_upper", "bb_middle", "bb_lower", "momentum"]]

    def reset(self):
        self.step_idx = 0
        self.position = 0.0  # Current Position
        self.cash = 1.0      # Initial Capital as 1
        self.asset_value = 1.0
        self.history = []
        return self._get_observation()
    def _get_observation(self):
        obs = self.features[self.step_idx:self.step_idx + self.window_size]
        # If the window_size is less than 0, add 0
        if obs.shape[0] < self.window_size:
            pad_length = self.window_size - obs.shape[0]
            padding = np.zeros((pad_length, self.features.shape[1]))
            obs = np.vstack((obs, padding))
        
        # ⚠️ Additional insurance mechanism
        assert obs.shape == (self.window_size, self.features.shape[1]), \
        f"Obs shape mismatch: {obs.shape} != {(self.window_size, self.features.shape[1])}"
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action[0], 0.0, 1.0)
        current_price = self.df["close"].iloc[self.step_idx + self.window_size - 1]
        next_price = self.df["close"].iloc[self.step_idx + self.window_size]
        return_rate = (next_price - current_price) / current_price
        
        self.position = action
        portfolio_return = self.position * return_rate
        self.asset_value *= (1 + portfolio_return)
        
        reward = portfolio_return
        self.step_idx += 1
        done = self.step_idx >= self.max_steps
        
    #  Fix observation size issue
        if done:
            obs = np.zeros_like(self._get_observation())  # Keep shape but value 0
        else:
            obs = self._get_observation()
        info = {"asset_value": self.asset_value}
        return obs, reward, done, info

    def render(self):
        print(f"Step: {self.step_idx}, Value: {self.asset_value:.4f}, Position: {self.position:.2f}")


# In[ ]:


# Test the model
import pandas as pd
df = pd.read_csv("xxxxxxxxxx")
env = TradingEnv(df)
obs = env.reset()
done = False
log = []

while not done:
    action = env.action_space.sample()  
    obs, reward, done, info = env.step(action)

    # Record Data
    log.append({
        "step": env.step_idx,
        "position": env.position,
        "reward": reward,
        "asset_value": info["asset_value"]
    })

# Save as CSV
df_log = pd.DataFrame(log)
df_log.to_csv("rl_trading_log.csv", index=False)

print("RL Agent trading process has been saved to rl_trading_log.csv")

