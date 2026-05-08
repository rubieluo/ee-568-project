import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC

# Watch the expert play live
# env = gym.make("CartPole-v1", render_mode="human")
# model = PPO.load("experts/CartPole-v1_expert")

env = gym.make("Pendulum-v1", render_mode="human")
model = SAC.load("experts/Pendulum-v1_expert")

obs, _ = env.reset()
total_reward = 0
while True:
    action, _ = model.predict(obs, deterministic=True) # follow expert policy
    # action = env.action_space.sample() #see random actions
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"Episode return: {total_reward}")
        total_reward = 0
        obs, _ = env.reset()