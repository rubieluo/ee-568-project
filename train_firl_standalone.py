# f-IRL for CartPole and Pendulum
# based on: https://arxiv.org/abs/2011.04709
# uses discriminator + MLPReward from f-IRL repo; SAC loop written from scratch

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
from firl.models.reward import MLPReward

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
        "tau": 0.005,
        "disc_iter": 20,
        "outer_iters": 40,
        "collect_trajs": 10,
        "reward_grad_steps": 10,
        "reward_lr": 3e-4,
        "div": "fkl",
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
        "tau": 0.005,
        "disc_iter": 20,
        "outer_iters": 40,
        "collect_trajs": 10,
        "reward_grad_steps": 10,
        "reward_lr": 3e-4,
        "div": "fkl",
    },
}


# Q network for discrete actions
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


# Q network for continuous actions
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


# gaussian policy for continuous envs
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


def soft_update(source, target, tau):
    for s, t in zip(source.parameters(), target.parameters()):
        t.data.mul_(1 - tau)
        t.data.add_(tau * s.data)


# discrete SAC with double-Q
class FIRLAgentDiscrete:
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

    def _v_from_q(self, q):
        return self.alpha * torch.logsumexp(q / self.alpha, dim=1, keepdim=True)

    def get_targetV(self, obs):
        q1 = self.q1_target(obs)
        q2 = self.q2_target(obs)
        return self._v_from_q(torch.min(q1, q2))

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

    def update(self, obs_b, action_b, reward_b, next_obs_b, done_b):
        with torch.no_grad():
            next_v = self.get_targetV(next_obs_b)
            target_q = reward_b + self.gamma * (1 - done_b) * next_v

        q1 = self.q1(obs_b).gather(1, action_b.long())
        q2 = self.q2(obs_b).gather(1, action_b.long())
        loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.q1, self.q1_target, self.tau)
        soft_update(self.q2, self.q2_target, self.tau)
        return loss.item()


# continuous SAC with double-Q
class FIRLAgentContinuous:
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

    def choose_action(self, obs_np, deterministic=False):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy(obs, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, obs_b, action_b, reward_b, next_obs_b, done_b):
        # critic
        with torch.no_grad():
            next_action, next_log_prob = self.policy(next_obs_b)
            q1_t = self.q1_target(next_obs_b, next_action)
            q2_t = self.q2_target(next_obs_b, next_action)
            next_v = torch.min(q1_t, q2_t) - self.alpha * next_log_prob
            target_q = reward_b + self.gamma * (1 - done_b) * next_v

        q1 = self.q1(obs_b, action_b)
        q2 = self.q2(obs_b, action_b)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # actor
        action_pi, log_prob = self.policy(obs_b)
        q_pi = torch.min(self.q1(obs_b, action_pi), self.q2(obs_b, action_pi))
        pi_loss = (self.alpha * log_prob - q_pi).mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        soft_update(self.q1, self.q1_target, self.tau)
        soft_update(self.q2, self.q2_target, self.tau)

        return q_loss.item()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, obs, action, reward, next_obs, done_no_max):
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


def load_expert_data(env_name, K):
    path = f"datasets/{env_name}_K{K}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path)
    print(f"loaded {len(data['obs'])} expert transitions (K={K})")
    return data


def collect_agent_trajectories(agent, env_name, n_trajs, discrete, max_T):
    """returns (trajs (n, T, d) padded, flat_states (real only), lengths (n,))."""
    env = gym.make(env_name)
    trajs = []
    flat_states = []
    lengths = []
    for ep in range(n_trajs):
        obs, _ = env.reset(seed=ep)
        traj = []
        done = False
        steps = 0
        while not done and steps < max_T:
            traj.append(obs.copy())
            flat_states.append(obs.copy())
            if discrete:
                action = agent.choose_action(obs)
            else:
                action = agent.choose_action(obs, deterministic=False)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        lengths.append(steps)
        # pad to rectangular (n, T, d) so we can reshape; padded steps are masked out in the loss
        while len(traj) < max_T:
            traj.append(obs.copy())
        trajs.append(np.array(traj))
    env.close()
    return np.array(trajs), np.array(flat_states), np.array(lengths)


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
    return env_name == "Pendulum-v1"


def reward_loss_firl(div, agent_trajs, traj_lengths, disc, reward_func, device):
    """f-IRL covariance-style objective with per-step mask for early-terminating episodes.

    agent_trajs: numpy (N, T, d) padded to max_T.
    traj_lengths: numpy (N,) actual visited length per trajectory.
    Implements firl/divs/f_div_disc.py:f_div_disc_loss for the no-IS branch,
    summing only over real (non-padded) steps so terminated episodes don't bias
    the covariance with hundreds of repeated terminal states.
    """
    N, T, d = agent_trajs.shape
    s_vec = agent_trajs.reshape(-1, d)
    logits = disc.log_density_ratio(s_vec)  # tensor, no grad through disc

    if div == "fkl":
        t1_per = torch.exp(logits)
    elif div == "rkl":
        t1_per = logits
    elif div == "js":
        t1_per = F.softplus(logits)
    else:
        raise ValueError(f"unknown div {div}")

    mask = torch.zeros(N, T, device=device)
    for i, l in enumerate(traj_lengths):
        mask[i, :int(l)] = 1.0

    t1 = ((-t1_per).view(N, T) * mask).sum(dim=1)  # (N,)

    s_tensor = torch.FloatTensor(s_vec).to(device)
    t2 = (reward_func.r(s_tensor).view(N, T) * mask).sum(dim=1)  # (N,)

    surrogate = (t1 * t2).mean() - t1.mean() * t2.mean()
    return surrogate / T


def train(env_name, K, seed, cfg):
    print(f"\n{'='*55}")
    print(f"f-IRL | {env_name} | K={K} | seed={seed}")
    print(f"{'='*55}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discrete = cfg["discrete"]
    pendulum_like = env_is_time_limited_only(env_name)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    max_T = env.spec.max_episode_steps

    if discrete:
        act_dim = env.action_space.n
        agent = FIRLAgentDiscrete(obs_dim, act_dim, cfg, device)
    else:
        act_dim = env.action_space.shape[0]
        act_limit = float(env.action_space.high[0])
        agent = FIRLAgentContinuous(obs_dim, act_dim, act_limit, cfg, device)

    disc = Discriminator(
        input_dim=obs_dim,
        hid_dim=cfg["hidden"],
        batch_size=cfg["batch_size"],
        device=device,
    )

    # neural reward function that SAC actually optimizes against
    reward_func = MLPReward(
        input_dim=obs_dim,
        hidden_sizes=(cfg["hidden"], cfg["hidden"]),
        hid_act="tanh",
        clamp_magnitude=10.0,
        device=device,
    ).to(device)
    reward_optimizer = Adam(reward_func.parameters(), lr=cfg["reward_lr"], weight_decay=1e-4)

    expert_data = load_expert_data(env_name, K)
    expert_states = expert_data["obs"]

    buf = ReplayBuffer(capacity=100_000)

    # random warm-up
    obs, _ = env.reset(seed=seed)
    for _ in range(cfg["batch_size"]):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        done_no_max = 0.0 if pendulum_like else float(terminated)
        buf.push(obs, action, 0.0, next_obs, done_no_max)
        obs = next_obs if not done else env.reset(seed=seed)[0]

    best_return = -float("inf")
    sac_step = 0
    steps_per_outer = cfg["train_steps"] // cfg["outer_iters"]

    for outer_itr in range(cfg["outer_iters"]):
        print(f"\n--- outer iter {outer_itr+1}/{cfg['outer_iters']} ---")

        # step 1: SAC inner loop with reward_func as reward
        obs, _ = env.reset(seed=seed + outer_itr)
        for _ in range(steps_per_outer):
            action = agent.choose_action(obs)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            done_no_max = float(terminated) if not pendulum_like else 0.0

            # SAC reward comes from reward_func (the f-IRL state-only reward), not disc
            with torch.no_grad():
                r = float(reward_func.get_scalar_reward(np.array([obs]))[0])

            buf.push(obs, action, r, next_obs, done_no_max)
            obs = next_obs if not done else env.reset(seed=seed + sac_step)[0]
            sac_step += 1

            if len(buf) >= cfg["batch_size"]:
                obs_b, next_obs_b, action_b, reward_b, done_b = buf.sample(
                    cfg["batch_size"], device)
                agent.update(obs_b, action_b, reward_b, next_obs_b, done_b)

        # step 2: collect agent trajectories (rectangular + per-traj real lengths for the masked covariance loss)
        agent_trajs, agent_flat, traj_lengths = collect_agent_trajectories(
            agent, env_name, cfg["collect_trajs"], discrete, max_T
        )
        print(f"  collected {agent_trajs.shape[0]} trajs, mean real length {traj_lengths.mean():.1f}")

        # step 3: train discriminator on expert vs agent states
        disc_loss = disc.learn(expert_states, agent_flat, iter=cfg["disc_iter"])
        print(f"  disc loss: {np.mean(disc_loss[-10:]):.4f}")

        # step 4: train reward_func via f-IRL covariance gradient
        for _ in range(cfg["reward_grad_steps"]):
            loss = reward_loss_firl(cfg["div"], agent_trajs, traj_lengths, disc, reward_func, device)
            reward_optimizer.zero_grad()
            loss.backward()
            reward_optimizer.step()
        print(f"  reward loss: {loss.item():.4f}")

        # NOTE: no buffer relabel — matches f-IRL reference reinitialize=False branch.
        # SAC adapts to the new reward through fresh transitions only; old Bellman targets
        # stay self-consistent (they reflect the policy under whatever reward was in force when they were stored).

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
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--outer_iters", type=int, default=None)
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
            cfg = dict(CONFIG[env_name])
            if args.train_steps is not None:
                cfg["train_steps"] = args.train_steps
            if args.outer_iters is not None:
                cfg["outer_iters"] = args.outer_iters
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
