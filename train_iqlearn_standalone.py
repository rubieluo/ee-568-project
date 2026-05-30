# IQ-Learn for CartPole and Pendulum
# based on: https://arxiv.org/abs/2106.12142
# uses iq_loss from the IQ-Learn repo, everything else written from scratch

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "IQ-Learn", "iq_learn"))
from iq import iq_loss

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, "iqlearn_results.csv")
TRACE_CSV = os.path.join(RESULTS_DIR, "training_trace.csv")
TRACE_FIELDS = ["algo", "env", "K", "seed", "step", "eval_return"]


def append_trace(algo, env, K, seed, step, eval_return):
    write_header = not os.path.exists(TRACE_CSV)
    with open(TRACE_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(dict(algo=algo, env=env, K=K, seed=seed, step=step, eval_return=eval_return))

CONFIG = {
    "CartPole-v1": {
        "train_steps": 30_000,   # smoke-test verified
        "batch_size": 256,
        "lr": 3e-4,
        "gamma": 0.99,
        "eval_every": 5_000,
        "eval_eps": 20,
        "hidden": 256,
        "discrete": True,
        "alpha": 0.1,
        "tau": 0.005,
    },
    "Pendulum-v1": {
        "train_steps": 30_000,   # smoke-test verified
        "batch_size": 256,
        "lr": 1e-4,
        "gamma": 0.99,
        "eval_every": 5_000,
        "eval_eps": 20,
        "hidden": 256,
        "discrete": False,
        "alpha": 0.2,
        "tau": 0.005,
    },
}


# Q network for discrete actions (cartpole)
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


# gaussian policy for pendulum
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
            # tanh squashing correction
            log_prob = (dist.log_prob(raw)
                        - torch.log(1 - action.pow(2) / self.act_limit**2 + 1e-6)
                        ).sum(dim=-1, keepdim=True)
        return action, log_prob


# soft polyak update
def soft_update(source, target, tau):
    for s, t in zip(source.parameters(), target.parameters()):
        t.data.mul_(1 - tau)
        t.data.add_(tau * s.data)


# agent for cartpole - double Q
class IQAgentDiscrete:
    def __init__(self, obs_dim, act_dim, cfg, device):
        self.gamma = cfg["gamma"]
        self.device = device
        self.alpha = torch.tensor(cfg["alpha"]).to(device)
        self.tau = cfg["tau"]
        hidden = cfg["hidden"]

        self.q1 = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.q2 = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.q1_target = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.q2_target = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg["lr"]
        )

        # iq_loss reads these from agent.args
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

    def _v_from_q(self, q_vals):
        return self.alpha * torch.logsumexp(q_vals / self.alpha, dim=1, keepdim=True)

    def getV(self, obs):
        q1 = self.q1(obs)
        q2 = self.q2(obs)
        q_min = torch.min(q1, q2)
        return self._v_from_q(q_min)

    def get_targetV(self, obs):
        q1 = self.q1_target(obs)
        q2 = self.q2_target(obs)
        q_min = torch.min(q1, q2)
        return self._v_from_q(q_min)

    def critic(self, obs, action):
        # min over the pair, as required by the iq_loss objective
        q1 = self.q1(obs).gather(1, action.long())
        q2 = self.q2(obs).gather(1, action.long())
        return torch.min(q1, q2)

    def choose_action(self, obs_np, deterministic=False):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q1 = self.q1(obs)
            q2 = self.q2(obs)
            q = torch.min(q1, q2)
            if deterministic:
                return int(q.argmax(dim=1).item())
            dist = Categorical(F.softmax(q / self.alpha, dim=1))
        return dist.sample().item()

    def update_target(self):
        soft_update(self.q1, self.q1_target, self.tau)
        soft_update(self.q2, self.q2_target, self.tau)


# agent for pendulum
class IQAgentContinuous:
    def __init__(self, obs_dim, act_dim, act_limit, cfg, device):
        self.gamma = cfg["gamma"]
        self.device = device
        self.alpha = cfg["alpha"]
        self.tau = cfg["tau"]
        hidden = cfg["hidden"]

        self.q1 = QNetContinuous(obs_dim, act_dim, hidden).to(device)
        self.q2 = QNetContinuous(obs_dim, act_dim, hidden).to(device)
        self.q1_target = QNetContinuous(obs_dim, act_dim, hidden).to(device)
        self.q2_target = QNetContinuous(obs_dim, act_dim, hidden).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.policy = PolicyContinuous(obs_dim, act_dim, hidden, act_limit).to(device)

        self.q_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg["lr"]
        )
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

    def _val(self, obs, q_a, q_b):
        action, log_prob = self.policy(obs)
        q = torch.min(q_a(obs, action), q_b(obs, action))
        return q - self.alpha * log_prob

    def getV(self, obs):
        return self._val(obs, self.q1, self.q2)

    def get_targetV(self, obs):
        with torch.no_grad():
            return self._val(obs, self.q1_target, self.q2_target)

    def critic(self, obs, action):
        return torch.min(self.q1(obs, action), self.q2(obs, action))

    def choose_action(self, obs_np, deterministic=False):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy(obs, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update_target(self):
        soft_update(self.q1, self.q1_target, self.tau)
        soft_update(self.q2, self.q2_target, self.tau)


# ring buffer that stores transitions, is_expert flag managed externally
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, obs, action, reward, next_obs, done_no_max):
        # done_no_max: 1 only on a true termination (not truncation), used to mask bootstrap
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (obs, action, reward, next_obs, done_no_max)
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


def sample_iq_batch(expert_buf, policy_buf, half_batch, device):
    """draw equal-size expert and policy minibatches, concat with is_expert flag."""
    obs_e, next_obs_e, action_e, reward_e, done_e = expert_buf.sample(half_batch, device)
    obs_p, next_obs_p, action_p, reward_p, done_p = policy_buf.sample(half_batch, device)

    obs = torch.cat([obs_p, obs_e], dim=0)
    next_obs = torch.cat([next_obs_p, next_obs_e], dim=0)
    action = torch.cat([action_p, action_e], dim=0)
    reward = torch.cat([reward_p, reward_e], dim=0)
    done = torch.cat([done_p, done_e], dim=0)

    is_expert = torch.cat(
        [torch.zeros(half_batch, 1, dtype=torch.bool, device=device),
         torch.ones(half_batch, 1, dtype=torch.bool, device=device)],
        dim=0
    )
    return obs, next_obs, action, reward, done, is_expert


def load_expert_data(env_name, K):
    path = f"datasets/{env_name}_K{K}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path)
    print(f"loaded {len(data['obs'])} expert transitions (K={K})")
    return data


def evaluate_policy(agent, env_name, n_episodes, discrete):
    env = gym.make(env_name)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.choose_action(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += r
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def env_is_time_limited_only(env_name):
    # Pendulum-v1 has no terminal state; CartPole-v1 does.
    return env_name == "Pendulum-v1"


def train(env_name, K, seed, cfg):
    print(f"\n{'='*55}")
    print(f"IQ-Learn | {env_name} | K={K} | seed={seed}")
    print(f"{'='*55}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discrete = cfg["discrete"]
    pendulum_like = env_is_time_limited_only(env_name)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]

    if discrete:
        act_dim = env.action_space.n
        agent = IQAgentDiscrete(obs_dim, act_dim, cfg, device)
    else:
        act_dim = env.action_space.shape[0]
        act_limit = float(env.action_space.high[0])
        agent = IQAgentContinuous(obs_dim, act_dim, act_limit, cfg, device)

    # separate buffers: expert is never overwritten by agent data
    expert_buf = ReplayBuffer(capacity=200_000)
    policy_buf = ReplayBuffer(capacity=100_000)

    data = load_expert_data(env_name, K)
    # materialize arrays once: NpzFile views re-read from disk on every index,
    # which silently turned 50k pushes into a 200s loop that the OS killed.
    ds_obs = np.asarray(data["obs"])
    ds_act = np.asarray(data["actions"])
    ds_nob = np.asarray(data["next_obs"])
    ds_don = np.asarray(data["dones"])
    n = len(ds_obs)
    for i in range(n):
        # for Pendulum the dataset's "done" is a truncation -> do not mask bootstrap
        ds_done = float(ds_don[i])
        done_no_max = 0.0 if pendulum_like else ds_done
        expert_buf.push(
            ds_obs[i], ds_act[i], 0.0,
            ds_nob[i], done_no_max,
        )
    print(f"  loaded {n} expert transitions into protected expert buffer")

    obs, _ = env.reset(seed=seed)
    best_return = -float("inf")
    half_batch = cfg["batch_size"] // 2

    for step in range(1, cfg["train_steps"] + 1):
        # collect agent transition
        action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # bootstrap mask: only true terminations zero the future value
        done_no_max = float(terminated)
        policy_buf.push(obs, action, reward, next_obs, done_no_max)
        obs = next_obs if not done else env.reset(seed=seed + step)[0]

        if len(policy_buf) < half_batch:
            continue

        # balanced IQ-Learn batch from two separate buffers
        obs_b, next_obs_b, action_b, reward_b, done_b, is_expert_b = sample_iq_batch(
            expert_buf, policy_buf, half_batch, device
        )

        current_Q = agent.critic(obs_b, action_b)
        current_v = agent.getV(obs_b)
        next_v = agent.get_targetV(next_obs_b)

        loss, loss_dict = iq_loss(
            agent, current_Q, current_v, next_v,
            (obs_b, next_obs_b, action_b, reward_b, done_b, is_expert_b)
        )

        if discrete:
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
        else:
            agent.q_optimizer.zero_grad()
            loss.backward()
            agent.q_optimizer.step()

            action_pi, log_prob = agent.policy(obs_b)
            q_pi = torch.min(
                agent.q1(obs_b, action_pi),
                agent.q2(obs_b, action_pi),
            )
            pi_loss = (agent.alpha * log_prob - q_pi).mean()
            agent.pi_optimizer.zero_grad()
            pi_loss.backward()
            agent.pi_optimizer.step()

        # soft target update every step
        agent.update_target()

        if step % cfg["eval_every"] == 0:
            mean_ret = evaluate_policy(agent, env_name, cfg["eval_eps"], discrete)
            if mean_ret > best_return:
                best_return = mean_ret
            append_trace("iq_learn", env_name, K, seed, step, mean_ret)
            print(f"  step {step}/{cfg['train_steps']} | eval={mean_ret:.1f} | best={best_return:.1f} | loss={loss_dict['total_loss']:.4f}")

    env.close()
    final_return = evaluate_policy(agent, env_name, cfg["eval_eps"], discrete)
    print(f"\nfinal return: {final_return:.1f}")
    return final_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1",
                        choices=["CartPole-v1", "Pendulum-v1"])
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--train_steps", type=int, default=None,
                        help="override CONFIG train_steps (useful for smoke tests)")
    args = parser.parse_args()

    runs = (
        [(args.env, K, s) for K in [1, 5, 20, 100] for s in [0, 1, 2]]
        if args.all else
        [(args.env, args.K, args.seed)]
    )

    print(f"planned runs: {len(runs)}")

    fieldnames = ["env", "algo", "K", "seed", "mean_return", "timestamp"]
    write_header = not os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for env_name, K, seed in runs:
            cfg = dict(CONFIG[env_name])
            if args.train_steps is not None:
                cfg["train_steps"] = args.train_steps
            mean_return = train(env_name, K, seed, cfg)
            writer.writerow(dict(
                env=env_name, algo="iq_learn", K=K, seed=seed,
                mean_return=mean_return,
                timestamp=datetime.now().isoformat(),
            ))
            f.flush()

    print(f"\nresults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
