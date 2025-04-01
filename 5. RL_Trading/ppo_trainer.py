#!/usr/bin/env python
# coding: utf-8

# In[43]:


# ppo_trainer.py
get_ipython().system('pip install shimmy>=2.0')
get_ipython().system('pip install gymnasium')
import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_trading_agent import TradingEnv

# 加载数据
raw_df = pd.read_csv(r"C:\Users\24716\Downloads\cleaned_crypto_data.csv")

# 创建 Gym 环境（封装为向量化环境）
def make_env():
    return TradingEnv(raw_df)

env = DummyVecEnv([make_env])

# 初始化 PPO 模型
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./ppo_logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

# 训练模型
model.learn(total_timesteps=50_000)

# 保存模型
model.save("ppo_trading_agent")
print("✅ PPO Agent 已保存为 ppo_trading_agent.zip")




# In[47]:


obs = env.reset()
done = False
log = []

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    log.append({
        "step": env.envs[0].step_idx,
        "position": env.envs[0].position,
        "reward": reward[0],
        "asset_value": info[0]["asset_value"],
        "signal": float(action[0])
    })

    if done[0]:
        break
# 保存为 CSV
log_df = pd.DataFrame(log)
log_df.to_csv("ppo_trading_log.csv", index=False)
print("✅ 策略信号与资产记录保存为 ppo_trading_log.csv")


# In[ ]:




