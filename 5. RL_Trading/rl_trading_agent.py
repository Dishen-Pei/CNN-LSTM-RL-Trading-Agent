#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
    自定义强化学习交易环境（兼容 OpenAI Gym），用于单标的模拟交易。
    状态：最近 N 根 K 线和技术指标
    动作：连续动作空间，表示当前仓位比例 [0, 1]
    奖励：本周期的收益率
    """

    def __init__(self, df, window_size=24):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size

        # 构建特征
        self.features = self._build_features(self.df)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # 计算有效步数
        self.valid_length = len(self.features)
        self.max_steps = self.valid_length - self.window_size - 1

        # Gym 环境设置
        obs_shape = (window_size, self.features.shape[1])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 初始化状态
        self.reset()

    def _build_features(self, df):
        """构建 RSI、MACD、EMA、布林带等技术指标"""
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
        self.position = 0.0  # 当前持仓比例
        self.cash = 1.0      # 初始资金 1
        self.asset_value = 1.0
        self.history = []
        return self._get_observation()
    def _get_observation(self):
        obs = self.features[self.step_idx:self.step_idx + self.window_size]
        # 不足 window_size 就补 0
        if obs.shape[0] < self.window_size:
            pad_length = self.window_size - obs.shape[0]
            padding = np.zeros((pad_length, self.features.shape[1]))
            obs = np.vstack((obs, padding))
        
        # ⚠️ 额外添加保险机制
        assert obs.shape == (self.window_size, self.features.shape[1]), \
        f"Obs shape mismatch: {obs.shape} != {(self.window_size, self.features.shape[1])}"
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action[0], 0.0, 1.0)
        # ✅ 当前价格和下一时刻价格
        current_price = self.df["close"].iloc[self.step_idx + self.window_size - 1]
        next_price = self.df["close"].iloc[self.step_idx + self.window_size]
        return_rate = (next_price - current_price) / current_price
        
        # ✅ 更新持仓 → 再计算基于“新仓位”的收益
        self.position = action
        portfolio_return = self.position * return_rate
        self.asset_value *= (1 + portfolio_return)
        
        reward = portfolio_return
        self.step_idx += 1
        done = self.step_idx >= self.max_steps
        
        
    # ✅ 修复 observation 尺寸不足问题
        if done:
            obs = np.zeros_like(self._get_observation())  # 保持 shape 但值为0 
        else:
            obs = self._get_observation()
        info = {"asset_value": self.asset_value}
        return obs, reward, done, info

    def render(self):
        print(f"Step: {self.step_idx}, Value: {self.asset_value:.4f}, Position: {self.position:.2f}")


# In[9]:


import pandas as pd
df = pd.read_csv(r"C:\Users\24716\Downloads\cleaned_crypto_data.csv")
env = TradingEnv(df)
obs = env.reset()
done = False
log = []

while not done:
    action = env.action_space.sample()  # 或者用你的智能体 action
    obs, reward, done, info = env.step(action)

    # 记录数据
    log.append({
        "step": env.step_idx,
        "position": env.position,
        "reward": reward,
        "asset_value": info["asset_value"]
    })

# 保存到CSV
df_log = pd.DataFrame(log)
df_log.to_csv("rl_trading_log.csv", index=False)

print("✅ 已保存 RL Agent 交易过程至 rl_trading_log.csv")


# In[ ]:




