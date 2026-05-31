import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC

EXPERT_DIR = "experts"
DATA_DIR = "datasets"
os.makedirs(DATA_DIR, exist_ok=True)

ENV_ALGO = {
    "CartPole-v1": PPO,
    "Pendulum-v1": SAC,
}

K_VALUES = [1, 5, 20, 100]  # dataset sizes to sweep over


def load_expert(env_name: str):
    algo = ENV_ALGO[env_name]
    path = os.path.join(EXPERT_DIR, f"{env_name}_expert")
    return algo.load(path)


def collect_trajectories(env_name: str, K: int, seed: int = 0):
    model = load_expert(env_name)
    env = gym.make(env_name)

    all_obs, all_actions, all_next_obs = [], [], []
    all_terminateds, all_truncateds, all_rewards = [], [], []
    traj_returns = []

    for ep in range(K):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_obs, ep_acts, ep_next_obs = [], [], []
        ep_terms, ep_truncs, ep_rews = [], [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_obs.append(obs)
            ep_acts.append(action)
            ep_next_obs.append(next_obs)
            ep_terms.append(float(terminated))   # real terminal state (e.g. pole fell)
            ep_truncs.append(float(truncated))   # time-limit cutoff — should NOT zero the Bellman bootstrap
            ep_rews.append(reward)
            ep_return += reward
            obs = next_obs

        all_obs.extend(ep_obs)
        all_actions.extend(ep_acts)
        all_next_obs.extend(ep_next_obs)
        all_terminateds.extend(ep_terms)
        all_truncateds.extend(ep_truncs)
        all_rewards.extend(ep_rews)
        traj_returns.append(ep_return)
        print(f"  Traj {ep+1}/{K}  return={ep_return:.1f}")

    print(f"\nMean expert return: {np.mean(traj_returns):.1f} ± {np.std(traj_returns):.1f}")

    save_path = os.path.join(DATA_DIR, f"{env_name}_K{K}.npz")
    np.savez(
        save_path,
        obs=np.array(all_obs),
        actions=np.array(all_actions),
        next_obs=np.array(all_next_obs),
        terminateds=np.array(all_terminateds),
        truncateds=np.array(all_truncateds),
        rewards=np.array(all_rewards),
    )
    print(f"Saved {len(all_obs)} transitions to {save_path}\n")
    env.close()


def collect_all(env_name: str):
    """Convenience: collect all K values at once."""
    for K in K_VALUES:
        print(f"\n=== {env_name} | K={K} ===")
        collect_trajectories(env_name, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", choices=list(ENV_ALGO.keys()))
    parser.add_argument("--K", type=int, default=None, help="Single K value; omit to collect all")
    args = parser.parse_args()

    if args.K:
        collect_trajectories(args.env, args.K)
    else:
        collect_all(args.env)
