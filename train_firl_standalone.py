# f-IRL for CartPole and Pendulum
# based on: https://arxiv.org/abs/2011.04709
# uses discriminator from f-IRL repo, everything else written from scratch

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "f-IRL"))
from firl.models.discrim import SMMIRLDisc as Discriminator

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, "firl_results.csv")

# hyperparams
CONFIG = {
    "CartPole-v1": {
        "train_steps": 40_000,
        "batch_size": 256,
        "lr": 3e-4,
        "gamma": 0.99,
        "eval_eps": 20,
        "hidden": 256,
        "discrete": True,
        "alpha": 0.1,
        "disc_iter": 20,
        "outer_iters": 40,
        "collect_trajs": 10,
    },
    "Pendulum-v1": {
        "train_steps": 50_000,
        "batch_size": 256,
        "lr": 1e-4,
        "gamma": 0.99,
        "eval_eps": 20,
        "hidden": 256,
        "discrete": False,
        "alpha": 0.2,
        "disc_iter": 20,
        "outer_iters": 40,
        "collect_trajs": 10,
    },
}


# Q network for discrete actions (cartpole)
# input: state, output: Q value for each action
class QNetDiscrete(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)


# Q network for continuous actions (pendulum)
# takes state AND action as input since we cant enumerate all actions
class QNetContinuous(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


# gaussian policy for pendulum - outputs a distribution over actions
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
            # tanh squashing correction for log prob
            log_prob = (dist.log_prob(raw)
                        - torch.log(1 - action.pow(2) / self.act_limit**2 + 1e-6)
                        ).sum(dim=-1, keepdim=True)
        return action, log_prob


# SAC agent for cartpole
# reward comes from discriminator density ratio, not from environment
class FIRLAgentDiscrete:
    def __init__(self, obs_dim, act_dim, cfg, device):
        self.gamma = cfg["gamma"]
        self.device = device
        self.alpha = torch.tensor(cfg["alpha"]).to(device)
        hidden = cfg["hidden"]

        self.q_net = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.target_net = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = Adam(self.q_net.parameters(), lr=cfg["lr"])

    def getV(self, obs):
        # soft value function V(s) = alpha * logsumexp(Q/alpha)
        q = self.q_net(obs)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def get_targetV(self, obs):
        q = self.target_net(obs)
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def critic(self, obs, action):
        q = self.q_net(obs)
        return q.gather(1, action.long())

    def choose_action(self, obs_np):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(obs)
            dist = Categorical(F.softmax(q / self.alpha, dim=1))
        return dist.sample().item()

    def update(self, obs_b, action_b, reward_b, next_obs_b, done_b):
        # bellman target using discriminator reward
        with torch.no_grad():
            next_v = self.get_targetV(next_obs_b)
            target_q = reward_b + self.gamma * (1 - done_b) * next_v

        current_q = self.critic(obs_b, action_b)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# SAC agent for pendulum (continuous actions)
class FIRLAgentContinuous:
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

    def getV(self, obs):
        action, log_prob = self.policy(obs)
        q = self.q_net(obs, action)
        return q - self.alpha * log_prob

    def get_targetV(self, obs):
        with torch.no_grad():
            action, log_prob = self.policy(obs)
            q = self.target_net(obs, action)
            return q - self.alpha * log_prob

    def critic(self, obs, action):
        return self.q_net(obs, action)

    def choose_action(self, obs_np, deterministic=False):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy(obs, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, obs_b, action_b, reward_b, next_obs_b, done_b):
        # critic update
        with torch.no_grad():
            next_v = self.get_targetV(next_obs_b)
            target_q = reward_b + self.gamma * (1 - done_b) * next_v

        current_q = self.critic(obs_b, action_b)
        q_loss = F.mse_loss(current_q, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # actor update - maximize Q while staying entropic
        action_pi, log_prob = self.policy(obs_b)
        q_pi = self.q_net(obs_b, action_pi)
        pi_loss = (self.alpha * log_prob - q_pi).mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        return q_loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


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

    def sample(self, batch_size, device):
        batch = random.sample([t for t in self.buffer if t is not None], batch_size)
        obs, action, reward, next_obs, done = zip(*batch)

        def to_t(x):
            return torch.FloatTensor(np.array(x)).to(device)

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
    print(f"loaded {len(data['obs'])} expert transitions (K={K})")
    return data


def collect_agent_states(agent, env_name, n_trajs, discrete):
    # roll out current policy and save visited states
    # used to train discriminator each outer iteration
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
    print(f"\n{'='*55}")
    print(f"f-IRL | {env_name} | K={K} | seed={seed}")
    print(f"{'='*55}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discrete = cfg["discrete"]

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    if discrete:
        act_dim = env.action_space.n
        agent = FIRLAgentDiscrete(obs_dim, act_dim, cfg, device)
    else:
        act_dim = env.action_space.shape[0]
        act_limit = float(env.action_space.high[0])
        agent = FIRLAgentContinuous(obs_dim, act_dim, act_limit, cfg, device)

    # discriminator from f-IRL repo
    # classifies expert states as 1, agent states as 0
    # output logits = log(p_expert / p_agent) = reward signal
    disc = Discriminator(
        input_dim=obs_dim,
        hid_dim=cfg["hidden"],
        batch_size=cfg["batch_size"],
        device=device,
    )

    expert_data = load_expert_data(env_name, K)
    expert_states = expert_data["obs"]

    buf = ReplayBuffer(capacity=100_000)

    # fill buffer with random transitions before training starts
    obs, _ = env.reset(seed=seed)
    for _ in range(cfg["batch_size"]):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buf.push(obs, action, 0.0, next_obs, float(done))
        obs = next_obs if not done else env.reset(seed=seed)[0]

    best_return = -float("inf")
    sac_step = 0
    steps_per_outer = cfg["train_steps"] // cfg["outer_iters"]

    for outer_itr in range(cfg["outer_iters"]):
        print(f"\n--- outer iter {outer_itr+1}/{cfg['outer_iters']} ---")

        # step 1: run SAC using current discriminator reward
        obs, _ = env.reset(seed=seed + outer_itr)
        for _ in range(steps_per_outer):
            action = agent.choose_action(obs)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            with torch.no_grad():
                reward = float(disc.log_density_ratio(np.array([obs])).cpu().item())

            buf.push(obs, action, reward, next_obs, float(done))
            obs = next_obs if not done else env.reset(seed=seed + sac_step)[0]
            sac_step += 1

            if len(buf) >= cfg["batch_size"]:
                obs_b, next_obs_b, action_b, reward_b, done_b = buf.sample(
                    cfg["batch_size"], device)
                agent.update(obs_b, action_b, reward_b, next_obs_b, done_b)

            if sac_step % 1000 == 0:
                agent.update_target()

        # step 2: update discriminator with fresh agent states
        agent_states = collect_agent_states(agent, env_name, cfg["collect_trajs"], discrete)
        print(f"  collected {len(agent_states)} agent states")

        disc_loss = disc.learn(expert_states, agent_states, iter=cfg["disc_iter"])
        print(f"  disc loss: {np.mean(disc_loss[-10:]):.4f}")

        # step 3: re-label buffer with updated discriminator
        obs_arr = np.array([t[0] for t in buf.buffer if t is not None])
        with torch.no_grad():
            new_rewards = disc.log_density_ratio(obs_arr).cpu().numpy()
        idx = 0
        for i, t in enumerate(buf.buffer):
            if t is not None:
                o, a, _, no, d = t
                buf.buffer[i] = (o, a, float(new_rewards[idx]), no, d)
                idx += 1

        mean_ret = evaluate_policy(agent, env_name, cfg["eval_eps"], discrete)
        if mean_ret > best_return:
            best_return = mean_ret
        print(f"  eval={mean_ret:.1f} | best={best_return:.1f}")

    env.close()
    final_return = evaluate_policy(agent, env_name, cfg["eval_eps"], discrete)
    print(f"\nfinal return: {final_return:.1f}")
    return final_return, best_return


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

    print(f"planned runs: {len(runs)}")

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
                env=env_name, algo="f_irl", K=K, seed=seed,
                mean_return=mean_return,
                best_return=best_return,
                timestamp=datetime.now().isoformat(),
            ))
            f.flush()

    print(f"\nresults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()