"""
Phase 1: Train expert policies for CartPole and Pendulum.
Usage:
    python train_expert.py --env CartPole-v1
    python train_expert.py --env Pendulum-v1
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

EXPERT_DIR = "experts"
os.makedirs(EXPERT_DIR, exist_ok=True)

ENV_CONFIG = {
    "CartPole-v1": {
        "algo": PPO,
        "timesteps": 50_000,
        "hyperparams": dict(learning_rate=3e-4, n_steps=2048, batch_size=64),
        "target_return": 475.0,
    },
    "Pendulum-v1": {
        "algo": SAC,
        "timesteps": 50_000,
        "hyperparams": dict(learning_rate=3e-4, buffer_size=100_000, batch_size=256),
        "target_return": -200.0,  # Pendulum: higher (less negative) is better
    },
}


def train_expert(env_name: str, seed: int = 0):
    cfg = ENV_CONFIG[env_name]
    env = gym.make(env_name)

    model = cfg["algo"](
        "MlpPolicy",
        env,
        seed=seed,
        verbose=1,
        **cfg["hyperparams"],
    )
    print(f"\nTraining {cfg['algo'].__name__} on {env_name} ...")
    model.learn(total_timesteps=cfg["timesteps"])

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"\nExpert performance: {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"Target: {cfg['target_return']}")

    # Save
    save_path = os.path.join(EXPERT_DIR, f"{env_name}_expert")
    model.save(save_path)
    print(f"Saved to {save_path}.zip")
    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", choices=list(ENV_CONFIG.keys()))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_expert(args.env, args.seed)
