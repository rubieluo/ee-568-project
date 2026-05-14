"""
train_soar_firl_standalone.py

SOAR + f-IRL implementation for CartPole-v1 and Pendulum-v1.

SOAR modification over f-IRL (Algorithm 5 from IL-SOAR paper):
    Instead of one critic Q(s,a), maintain L critics {Q1, Q2, ..., QL}.
    Compute optimistic Q estimate:
        Q_opt(s,a) = mean(Ql(s,a)) - clip(std(Ql(s,a)), 0, sigma)
    Use Q_opt for the policy (actor) update instead of a single critic.
    All critics are trained identically via TD on independent minibatches.

Why this helps:
    std across critics measures uncertainty — states visited rarely have
    high disagreement between critics. Subtracting std makes uncertain
    states look cheaper (better), driving the policy to explore them.
    This is "optimism in the face of uncertainty."

Everything else (discriminator, reward function, outer loop) is identical
to train_firl_standalone.py.

Usage:
    python train_soar_firl_standalone.py --env CartPole-v1 --K 20 --seed 0
    python train_soar_firl_standalone.py --env Pendulum-v1 --K 100 --seed 1
    python train_soar_firl_standalone.py --env CartPole-v1 --all

Results saved to: results/soar_firl_results.csv
"""

import argparse
import csv
import os
import sys
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical, Normal
import gymnasium as gym

# Pull reward model and discriminator from f-IRL repos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "f-IRL"))
from firl.models.reward import MLPReward
from firl.models.discrim import SMMIRLDisc as Discriminator

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, "soar_firl_results.csv")

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
        # f-IRL
        "disc_iter": 20,
        "reward_iter": 50,
        "reward_lr": 3e-4,
        "outer_iters": 40,
        "collect_trajs": 10,
        # SOAR specific
        "n_critics": 4, # from paper
        "sigma": 1.0, # clipping threshold for std
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
        # f-IRL
        "disc_iter": 20,
        "reward_iter": 50,
        "reward_lr": 1e-4,
        "outer_iters": 40,
        "collect_trajs": 10,
        # SOAR specific
        "n_critics": 4,
        "sigma": 1.0,
    },
}


# Networks (same as IQ Learn)

class QNetDiscrete(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, obs):
        return self.net(obs)


class QNetContinuous(nn.Module):
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
            log_prob = (dist.log_prob(raw)
                        - torch.log(1 - action.pow(2) / self.act_limit**2 + 1e-6)
                        ).sum(dim=-1, keepdim=True)
        return action, log_prob


# SOAR Agents — key difference: L critics instead of 1

def optimistic_q(q_values_list, sigma):
    """
    Algorithm 5 from IL-SOAR paper.
    Given a list of Q-value tensors from L critics,
    compute: mean(Ql) - clip(std(Ql), 0, sigma)

    Args:
        q_values_list: list of L tensors, each shape [batch, 1] or [batch, act_dim]
        sigma: clipping threshold for std

    Returns:
        optimistic Q estimate, same shape as each element
    """
    # Stack into [L, batch, ...] then compute mean and std across critics
    stacked = torch.stack(q_values_list, dim=0) # [L, batch, ...]
    mean_q = stacked.mean(dim=0) # [batch, ...]
    std_q = stacked.std(dim=0) # [batch, ...]
    std_q = torch.clamp(std_q, 0, sigma) # clip std
    return mean_q - std_q # optimistic estimate


class SOARAgentDiscrete:
    """
    SOAR agent for CartPole.
    Key difference from FIRLAgentDiscrete: L critic networks instead of 1.
    Policy update uses optimistic Q = mean(Ql) - clip(std(Ql), 0, sigma).
    """
    def __init__(self, obs_dim, act_dim, cfg, device):
        self.gamma = cfg["gamma"]
        self.device = device
        self.alpha = torch.tensor(cfg["alpha"]).to(device)
        self.sigma = cfg["sigma"]
        hidden = cfg["hidden"]
        n_critics = cfg["n_critics"]

        # L critic networks — each trained independently
        self.q_nets = nn.ModuleList([
            QNetDiscrete(obs_dim, act_dim, hidden).to(device)
            for _ in range(n_critics)
        ])
        self.target_nets = nn.ModuleList([
            QNetDiscrete(obs_dim, act_dim, hidden).to(device)
            for _ in range(n_critics)
        ])
        # Initialize target nets to match q_nets
        for q, t in zip(self.q_nets, self.target_nets):
            t.load_state_dict(q.state_dict())

        # One optimizer per critic
        self.optimizers = [
            Adam(q.parameters(), lr=cfg["lr"])
            for q in self.q_nets
        ]

    def _soft_value(self, q_net, obs):
        """Soft value V(s) = α·logsumexp(Q(s,·)/α)"""
        q = q_net(obs)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def getV_optimistic(self, obs):
        """
        Compute optimistic soft value using all critics.
        Used for target computation in critic updates.
        """
        v_list = [self._soft_value(q, obs) for q in self.q_nets]
        return optimistic_q(v_list, self.sigma)

    def get_targetV_optimistic(self, obs):
        """Optimistic soft value from target networks."""
        v_list = [self._soft_value(t, obs) for t in self.target_nets]
        return optimistic_q(v_list, self.sigma)

    def critic_optimistic(self, obs, action):
        """
        Optimistic Q(s,a) for specific action.
        This is what the policy update uses.
        """
        q_list = [q(obs).gather(1, action.long()) for q in self.q_nets]
        return optimistic_q(q_list, self.sigma)

    def choose_action(self, obs_np):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Use optimistic Q for action selection
            q_list = [q(obs) for q in self.q_nets]
            q_opt = optimistic_q(q_list, self.sigma)
            dist = Categorical(F.softmax(q_opt / self.alpha, dim=1))
        return dist.sample().item()

    def update(self, obs_b, action_b, reward_b, next_obs_b, done_b):
        """
        Update all L critics independently.
        Each critic gets a different random minibatch (independent datasets
        as required by the SOAR paper).
        """
        losses = []
        with torch.no_grad():
            next_v = self.get_targetV_optimistic(next_obs_b)
            target_q = reward_b + self.gamma * (1 - done_b) * next_v

        for q_net, optimizer in zip(self.q_nets, self.optimizers):
            current_q = q_net(obs_b).gather(1, action_b.long())
            loss = F.mse_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return np.mean(losses)

    def update_target(self):
        for q, t in zip(self.q_nets, self.target_nets):
            t.load_state_dict(q.state_dict())


class SOARAgentContinuous:
    """
    SOAR agent for Pendulum.
    Key difference: L critic networks, optimistic Q for actor update.
    """
    def __init__(self, obs_dim, act_dim, act_limit, cfg, device):
        self.gamma = cfg["gamma"]
        self.device = device
        self.alpha = cfg["alpha"]
        self.sigma = cfg["sigma"]
        hidden = cfg["hidden"]
        n_critics = cfg["n_critics"]

        # L critic networks
        self.q_nets = nn.ModuleList([
            QNetContinuous(obs_dim, act_dim, hidden).to(device)
            for _ in range(n_critics)
        ])
        self.target_nets = nn.ModuleList([
            QNetContinuous(obs_dim, act_dim, hidden).to(device)
            for _ in range(n_critics)
        ])
        for q, t in zip(self.q_nets, self.target_nets):
            t.load_state_dict(q.state_dict())

        self.policy = PolicyContinuous(obs_dim, act_dim, hidden, act_limit).to(device)

        self.q_optimizers = [
            Adam(q.parameters(), lr=cfg["lr"])
            for q in self.q_nets
        ]
        self.pi_optimizer = Adam(self.policy.parameters(), lr=cfg["lr"])

    def _value_from_q(self, q_net, obs):
        action, log_prob = self.policy(obs)
        q = q_net(obs, action)
        return q - self.alpha * log_prob

    def get_targetV_optimistic(self, obs):
        with torch.no_grad():
            v_list = [self._value_from_q(t, obs) for t in self.target_nets]
        return optimistic_q(v_list, self.sigma)

    def critic_optimistic(self, obs, action):
        q_list = [q(obs, action) for q in self.q_nets]
        return optimistic_q(q_list, self.sigma)

    def choose_action(self, obs_np, deterministic=False):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy(obs, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, obs_b, action_b, reward_b, next_obs_b, done_b):
        """
        Update all L critics + policy.
        Critics use optimistic target value.
        Policy update uses optimistic Q — this is the key SOAR contribution.
        """
        # Update all critics
        with torch.no_grad():
            next_v = self.get_targetV_optimistic(next_obs_b)
            target_q = reward_b + self.gamma * (1 - done_b) * next_v

        for q_net, optimizer in zip(self.q_nets, self.q_optimizers):
            current_q = q_net(obs_b, action_b)
            q_loss = F.mse_loss(current_q, target_q)
            optimizer.zero_grad()
            q_loss.backward()
            optimizer.step()

        # Policy update using OPTIMISTIC Q
        # Normal f-IRL: pi_loss = alpha * log_prob - Q_single(s, pi(s))
        # SOAR: pi_loss = alpha * log_prob - Q_opt(s, pi(s))
        # Q_opt = mean(Ql) - clip(std(Ql), 0, sigma)
        # Subtracting std encourages exploration of uncertain states.
        action_pi, log_prob = self.policy(obs_b)
        q_opt = self.critic_optimistic(obs_b, action_pi)
        pi_loss = (self.alpha * log_prob - q_opt).mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        return pi_loss.item()

    def update_target(self):
        for q, t in zip(self.q_nets, self.target_nets):
            t.load_state_dict(q.state_dict())


# Replay buffer (same as f-irl)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, obs, action, reward, next_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (obs, action, reward, next_obs, done)
        self.pos = (self.pos + 1) % self.capacity

    def update_rewards(self, reward_func, device):
        if len(self.buffer) == 0:
            return
        obs_arr = np.array([t[0] for t in self.buffer if t is not None])
        with torch.no_grad():
            new_rewards = reward_func.get_scalar_reward(obs_arr)
        for i, t in enumerate(self.buffer):
            if t is not None:
                obs, action, _, next_obs, done = t
                self.buffer[i] = (obs, action, float(new_rewards[i]), next_obs, done)

    def sample(self, batch_size, device):
        batch = random.sample([t for t in self.buffer if t is not None], batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        to_t = lambda x: torch.FloatTensor(np.array(x)).to(device)
        obs = to_t(obs)
        next_obs = to_t(next_obs)
        action = to_t(action)
        reward = to_t(reward).unsqueeze(1)
        done = to_t(done).unsqueeze(1)
        if action.dim() == 1:
            action = action.unsqueeze(1)
        return obs, next_obs, action, reward, done

    def __len__(self):
        return len([t for t in self.buffer if t is not None])

def load_expert_data(env_name, K):
    path = f"datasets/{env_name}_K{K}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path)
    print(f"  Loaded expert dataset: {len(data['obs'])} transitions from {K} trajectories")
    return data


def collect_agent_states(agent, env_name, n_trajs, discrete):
    env = gym.make(env_name)
    states = []
    for ep in range(n_trajs):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = agent.choose_action(obs) if discrete \
                     else agent.choose_action(obs, deterministic=False)
            states.append(obs.copy())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
    return np.array(states)


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
    print(f"SOAR+f-IRL | {env_name} | K={K} | seed={seed} | L={cfg['n_critics']} critics")
    print(f"{'='*60}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discrete = cfg["discrete"]

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    if discrete:
        act_dim = env.action_space.n
        agent = SOARAgentDiscrete(obs_dim, act_dim, cfg, device)
    else:
        act_dim = env.action_space.shape[0]
        act_limit = float(env.action_space.high[0])
        agent = SOARAgentContinuous(obs_dim, act_dim, act_limit, cfg, device)

    reward_func = MLPReward(
        input_dim = obs_dim,
        hidden_sizes = (cfg["hidden"], cfg["hidden"]),
        hid_act = "tanh",
        device = device,
    ).to(device)
    reward_optimizer = Adam(reward_func.parameters(), lr=cfg["reward_lr"])

    disc = Discriminator(
        input_dim = obs_dim,
        hid_dim = cfg["hidden"],
        batch_size = cfg["batch_size"],
        device = device,
    )

    expert_data = load_expert_data(env_name, K)
    expert_states = expert_data["obs"]

    replay = ReplayBuffer(capacity=100_000)

    obs, _ = env.reset(seed=seed)
    for _ in range(cfg["batch_size"] * 2):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.push(obs, action, 0.0, next_obs, float(done))
        obs = next_obs if not done else env.reset(seed=seed)[0]

    best_return = -float("inf")
    sac_step = 0
    steps_per_outer = cfg["train_steps"] // cfg["outer_iters"]

    for outer_itr in range(cfg["outer_iters"]):
        print(f"\n--- Outer iter {outer_itr+1}/{cfg['outer_iters']} ---")

        # Run SOAR agent with current reward
        obs, _ = env.reset(seed=seed + outer_itr)
        for _ in range(steps_per_outer):
            action = agent.choose_action(obs)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                reward = float(reward_func(obs_t).cpu().item())

            replay.push(obs, action, reward, next_obs, float(done))
            obs = next_obs if not done else env.reset(seed=seed + sac_step)[0]
            sac_step += 1

            if len(replay) >= cfg["batch_size"]:
                obs_b, next_obs_b, action_b, reward_b, done_b = replay.sample(
                    cfg["batch_size"], device)
                agent.update(obs_b, action_b, reward_b, next_obs_b, done_b)

            if sac_step % 1000 == 0:
                agent.update_target()

        # Collect agent states
        agent_states = collect_agent_states(
            agent, env_name, cfg["collect_trajs"], discrete)
        print(f"  Collected {len(agent_states)} agent states")

        # Train discriminator
        disc_loss = disc.learn(expert_states, agent_states, iter=cfg["disc_iter"])
        print(f"  Disc loss: {np.mean(disc_loss[-10:]):.4f}")

        # Update reward function
        reward_losses = []
        for _ in range(cfg["reward_iter"]):
            idx = np.random.choice(len(agent_states), cfg["batch_size"])
            s_batch = torch.FloatTensor(agent_states[idx]).to(device)
            with torch.no_grad():
                log_ratio = disc.log_density_ratio(agent_states[idx])
            pred_reward = reward_func(s_batch).squeeze()
            reward_loss = F.mse_loss(pred_reward, log_ratio)
            reward_optimizer.zero_grad()
            reward_loss.backward()
            reward_optimizer.step()
            reward_losses.append(reward_loss.item())
        print(f"  Reward loss: {np.mean(reward_losses):.4f}")

        # Re-label replay buffer
        replay.update_rewards(reward_func, device)

        # Evaluate
        mean_ret = evaluate_policy(agent, env_name, cfg["eval_eps"], discrete)
        if mean_ret > best_return:
            best_return = mean_ret
        print(f"  eval_return={mean_ret:.1f} | best={best_return:.1f}")

    env.close()
    final_return = evaluate_policy(agent, env_name, cfg["eval_eps"], discrete)
    print(f"\nFinal return: {final_return:.1f}")
    return final_return, best_return


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1",
                        choices=["CartPole-v1", "Pendulum-v1"])
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    runs = (
        [(args.env, K, s) for K in [1, 5, 20, 100] for s in [0, 1, 2]]
        if args.all else
        [(args.env, args.K, args.seed)]
    )

    print(f"Planned runs: {len(runs)}")

    fieldnames = ["env", "algo", "K", "seed", "mean_return", "best_return", "timestamp"]
    write_header = not os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for env_name, K, seed in runs:
            cfg = CONFIG[env_name]
            mean_return, best_return = train(env_name, K, seed, cfg)
            writer.writerow(dict(
                env=env_name, algo="soar_f_irl", K=K, seed=seed,
                mean_return=mean_return,
                best_return=best_return,
                timestamp=datetime.now().isoformat(),
            ))
            f.flush()

    print(f"\nAll results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()