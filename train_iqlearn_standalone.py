"""
train_iqlearn_standalone.py

Clean IQ-Learn implementation for CartPole-v1 (discrete) and Pendulum-v1 (continuous).
Uses iq_loss() directly from the IQ-Learn repo, builds everything else from scratch
to avoid their broken hydra/wandb/old-gym dependencies.

Usage:
    python train_iqlearn_standalone.py --env CartPole-v1 --K 20 --seed 0
    python train_iqlearn_standalone.py --env Pendulum-v1 --K 100 --seed 1

Results saved to: results/iqlearn_results.csv
"""

import argparse
import csv
import os
import sys
import random
import types
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical, Normal
import gymnasium as gym

# Pull iq_loss from the repo without importing anything else
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "IQ-Learn", "iq_learn"))
from iq import iq_loss

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, "iqlearn_results.csv")

# Hyperparameters
CONFIG = {
    "CartPole-v1": {
        "train_steps": 40_000,
        "batch_size": 256,
        "lr": 3e-4,
        "gamma": 0.99,
        "eval_every": 5_000,
        "eval_eps": 20,
        "hidden": 256,
        "discrete": True,
        "alpha": 0.1, 
    },
    "Pendulum-v1": {
        "train_steps": 50_000,
        "batch_size": 256,
        "lr": 1e-4,
        "gamma": 0.99,
        "eval_every": 5_000,
        "eval_eps": 20,
        "hidden": 256,
        "discrete": False,
        "alpha": 0.2,
    },
}


# Networks

class QNetDiscrete(nn.Module):
    """Q(s, ·) → vector of Q-values, one per action. For CartPole."""
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class QNetContinuous(nn.Module):
    """Q(s, a) → scalar. For Pendulum."""
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class PolicyContinuous(nn.Module):
    """Gaussian policy for Pendulum."""
    def __init__(self, obs_dim, act_dim, hidden, act_limit):
        super().__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs, deterministic=False):
        x = self.net(obs)
        mean = self.mean_head(x)
        if deterministic:
            action = torch.tanh(mean) * self.act_limit
            log_prob = torch.zeros(obs.shape[0], 1, device=obs.device)
        else:
            std = self.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            raw = dist.rsample()
            action = torch.tanh(raw) * self.act_limit
            # log prob with tanh squashing correction
            log_prob = (dist.log_prob(raw)
                        - torch.log(1 - action.pow(2) / self.act_limit**2 + 1e-6)
                        ).sum(dim=-1, keepdim=True)
        return action, log_prob


# Agent wrappers that expose the interface iq_loss() expects:
# agent.gamma, agent.args, agent.getV(obs), agent.get_targetV(obs)

class IQAgentDiscrete:
    """Wraps QNetDiscrete to match the interface expected by iq_loss()."""

    def __init__(self, obs_dim, act_dim, cfg, device):
        self.gamma = cfg["gamma"]
        self.device = device
        self.alpha = torch.tensor(cfg["alpha"]).to(device)
        hidden = cfg["hidden"]

        self.q_net = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.target_net = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = Adam(self.q_net.parameters(), lr=cfg["lr"])

        # args namespace that iq_loss() reads
        self.args = types.SimpleNamespace(
            gamma=self.gamma,
            method=types.SimpleNamespace(
                div="chi", # χ² divergence (phi_grad=1, adds chi2_loss term)
                loss="value", # use both expert + agent states
                chi=True, # enable χ² regularization term
                alpha=0.5, # χ² coefficient
                grad_pen=False,
                regularize=False,
                tanh=False,
                constrain=False,
            )
        )

    def getV(self, obs):
        """Soft value function V(s) = α · logsumexp(Q(s,·) / α)"""
        q = self.q_net(obs)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def get_targetV(self, obs):
        q = self.target_net(obs)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def critic(self, obs, action):
        """Q(s, a) for a specific action index."""
        q = self.q_net(obs)
        return q.gather(1, action.long())

    def choose_action(self, obs_np):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(obs)
            dist = Categorical(F.softmax(q / self.alpha, dim=1))
        return dist.sample().item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


class IQAgentContinuous:
    """Wraps QNetContinuous + PolicyContinuous for Pendulum."""

    def __init__(self, obs_dim, act_dim, act_limit, cfg, device):
        self.gamma = cfg["gamma"]
        self.device = device
        self.alpha = cfg["alpha"]
        hidden = cfg["hidden"]

        self.q_net = QNetContinuous(obs_dim, act_dim, hidden).to(device)
        self.target_net = QNetContinuous(obs_dim, act_dim, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.policy = PolicyContinuous(obs_dim, act_dim, hidden, act_limit).to(device)

        self.q_optimizer = Adam(self.q_net.parameters(), lr=cfg["lr"])
        self.pi_optimizer = Adam(self.policy.parameters(), lr=cfg["lr"])

        self.args = types.SimpleNamespace(
            gamma=self.gamma,
            method=types.SimpleNamespace(
                div="chi",
                loss="value",
                chi=True,
                alpha=0.5,
                grad_pen=False,
                regularize=False,
                tanh=False,
                constrain=False,
            )
        )

    def _value_from_q(self, obs, net):
        action, log_prob = self.policy(obs)
        q = net(obs, action)
        return q - self.alpha * log_prob

    def getV(self, obs):
        return self._value_from_q(obs, self.q_net)

    def get_targetV(self, obs):
        with torch.no_grad():
            return self._value_from_q(obs, self.target_net)

    def critic(self, obs, action):
        return self.q_net(obs, action)

    def choose_action(self, obs_np, deterministic=False):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy(obs, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# Replay buffer

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, obs, action, reward, next_obs, done, is_expert):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (obs, action, reward, next_obs, done, is_expert)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, is_expert = zip(*batch)

        to_t = lambda x: torch.FloatTensor(np.array(x)).to(device)
        obs = to_t(obs)
        next_obs = to_t(next_obs)
        action = to_t(action)
        reward = to_t(reward).unsqueeze(1)
        done = to_t(done).unsqueeze(1)
        is_expert = torch.BoolTensor(np.array(is_expert)).unsqueeze(1).to(device)

        if action.dim() == 1:
            action = action.unsqueeze(1)

        return obs, next_obs, action, reward, done, is_expert

    def __len__(self):
        return len(self.buffer)


# Training

def load_expert_data(env_name, K):
    path = f"datasets/{env_name}_K{K}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}\nRun collect_demos.py first.")
    return np.load(path)


def fill_expert_buffer(replay_buffer, expert_data, max_transitions=50_000):
    """Load expert transitions into the replay buffer, flagged as is_expert=True."""
    obs = expert_data["obs"]
    actions = expert_data["actions"]
    n_obs = expert_data["next_obs"]
    dones = expert_data["dones"]
    n = min(len(obs), max_transitions)

    for i in range(n):
        replay_buffer.push(obs[i], actions[i], 0.0, n_obs[i], dones[i], True)
    print(f"  Loaded {n} expert transitions into buffer")


def evaluate_policy(agent, env_name, n_episodes, discrete):
    env = gym.make(env_name)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.choose_action(obs, deterministic=True) if not discrete \
                       else agent.choose_action(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += r
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def train(env_name, K, seed, cfg):
    print(f"\n{'='*60}")
    print(f"IQ-Learn | {env_name} | K={K} | seed={seed}")
    print(f"{'='*60}")

    # Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discrete = cfg["discrete"]

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    # Build agent
    if discrete:
        act_dim = env.action_space.n
        agent = IQAgentDiscrete(obs_dim, act_dim, cfg, device)
    else:
        act_dim = env.action_space.shape[0]
        act_limit = float(env.action_space.high[0])
        agent = IQAgentContinuous(obs_dim, act_dim, act_limit, cfg, device)

    # Replay buffer: expert + agent transitions together
    replay = ReplayBuffer(capacity=100_000)

    # Fill with expert data
    expert_data = load_expert_data(env_name, K)
    fill_expert_buffer(replay, expert_data)

    # Training loop
    obs, _ = env.reset(seed=seed)
    best_return = -float("inf")
    eval_returns = []

    for step in range(1, cfg["train_steps"] + 1):

        # Collect one agent transition
        action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.push(obs, action, reward, next_obs, float(done), False)
        obs = next_obs if not done else env.reset(seed=seed + step)[0]

        # Need enough samples to form a mixed batch
        if len(replay) < cfg["batch_size"] * 2:
            continue

        # IQ-Learn update
        batch = replay.sample(cfg["batch_size"], device)
        obs_b, next_obs_b, action_b, reward_b, done_b, is_expert_b = batch

        current_Q = agent.critic(obs_b, action_b)
        current_v = agent.getV(obs_b)
        next_v = agent.get_targetV(next_obs_b)

        loss, loss_dict = iq_loss(
            agent,
            current_Q, current_v, next_v,
            (obs_b, next_obs_b, action_b, reward_b, done_b, is_expert_b)
        )

        # Critic update
        if discrete:
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
        else:
            agent.q_optimizer.zero_grad()
            loss.backward()
            agent.q_optimizer.step()

            # Policy update (maximize Q)
            action_pi, log_prob = agent.policy(obs_b)
            q_pi = agent.q_net(obs_b, action_pi)
            pi_loss = (agent.alpha * log_prob - q_pi).mean()
            agent.pi_optimizer.zero_grad()
            pi_loss.backward()
            agent.pi_optimizer.step()

        # Periodic target network update
        if step % 1000 == 0:
            agent.update_target()

        # Evaluation
        if step % cfg["eval_every"] == 0:
            mean_ret = evaluate_policy(agent, env_name, cfg["eval_eps"], discrete)
            eval_returns.append(mean_ret)
            if mean_ret > best_return:
                best_return = mean_ret
            print(f"  step {step:>6d}/{cfg['train_steps']} | "
                  f"eval_return={mean_ret:.1f} | "
                  f"best={best_return:.1f} | "
                  f"loss={loss_dict['total_loss']:.4f}")

    env.close()
    final_return = evaluate_policy(agent, env_name, cfg["eval_eps"], discrete)
    print(f"\nFinal return: {final_return:.1f}")
    return final_return


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1",
                        choices=["CartPole-v1", "Pendulum-v1"])
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true",
                        help="Run full sweep: all K values x 3 seeds")
    args = parser.parse_args()

    runs = (
        [(args.env, K, s) for K in [1, 5, 20, 100] for s in [0, 1, 2]]
        if args.all else
        [(args.env, args.K, args.seed)]
    )

    print(f"Planned runs: {len(runs)}")

    fieldnames = ["env", "algo", "K", "seed", "mean_return", "timestamp"]
    write_header = not os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for env_name, K, seed in runs:
            cfg = CONFIG[env_name]
            mean_return = train(env_name, K, seed, cfg)
            writer.writerow(dict(
                env=env_name, algo="iq_learn", K=K, seed=seed,
                mean_return=mean_return,
                timestamp=datetime.now().isoformat(),
            ))
            f.flush()

    print(f"\nAll results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
