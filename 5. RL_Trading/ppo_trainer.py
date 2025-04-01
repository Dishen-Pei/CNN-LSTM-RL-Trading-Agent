#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Step 5B: Reinforcement learning with Proximal Policy Optimization (PPO)


# In[17]:


# ppo_trainer.py
get_ipython().system('pip install shimmy>=2.0')
get_ipython().system('pip install gymnasium')
import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_trading_agent import TradingEnv

# Load Data
raw_df = pd.read_csv("xxxxxxxxx")

# Create a Gym environment (encapsulated as a vectorized environment)
def make_env():
    return TradingEnv(raw_df)

env = DummyVecEnv([make_env])

# Initializing the PPO model
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

model.learn(total_timesteps=50_000)
model.save("ppo_trading_agent")
print(" PPO Agent has been saved as ppo_trading_agent.zip")




# In[ ]:


#Record signal and Asset Info
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

log_df = pd.DataFrame(log)
log_df.to_csv("ppo_trading_log.csv", index=False)
print("The strategy signals and asset records are saved as ppo_trading_log.csv")


# In[19]:


# Model Result Visualization
import pandas as pd
import matplotlib.pyplot as plt

log_df = pd.read_csv("ppo_trading_log.csv")

plt.figure(figsize=(10, 5))
plt.plot(log_df["step"], log_df["asset_value"], label="Asset Value")
plt.plot(log_df["step"], log_df["signal"], label="Signal (Position)")
plt.title("PPO Agent Capital & Signal")
plt.xlabel("Step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




