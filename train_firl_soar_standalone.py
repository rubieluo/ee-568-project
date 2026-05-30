# SOAR + f-IRL for CartPole and Pendulum
# based on IL-SOAR paper: https://arxiv.org/abs/2502.19859
# SOAR = ensemble of L critics + optimistic Q in actor only
#   - reward-maximization framing => mean + clip(std, 0, sigma)
#   - critic backup is plain SAC (no optimism bonus)
#   - each of the L critics is itself a double-Q pair
#   - each critic has its own replay buffer (independent samples)

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
RESULTS_CSV = os.path.join(RESULTS_DIR, "soar_firl_results.csv")
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
        "n_critics": 4,
        "sigma": 1.0,
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
        "n_critics": 4,
        "sigma": 10.0,  # paper grid-searches sigma per env; larger for continuous control
    },
}


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


# Algorithm 5 from the paper, written for reward-maximization framing.
# Paper uses cost-minimization with mean - std; for reward we flip to mean + std.
# High std = critics disagree = uncertain state => optimistic upward bonus encourages exploration.
def optimistic_q(q_list, sigma):
    stacked = torch.stack(q_list, dim=0)
    mean_q = stacked.mean(dim=0)
    std_q = torch.clamp(stacked.std(dim=0), 0, sigma)
    return mean_q + std_q


# One SOAR critic = a SAC double-Q pair (Q1, Q2) with its own target copies and optimizer.
class DiscreteDoubleQ:
    def __init__(self, obs_dim, act_dim, hidden, lr, device):
        self.device = device
        self.q1 = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.q2 = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.q1_target = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.q2_target = QNetDiscrete(obs_dim, act_dim, hidden).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

    def soft_v(self, obs, alpha, target=False):
        if target:
            q = torch.min(self.q1_target(obs), self.q2_target(obs))
        else:
            q = torch.min(self.q1(obs), self.q2(obs))
        return alpha * torch.logsumexp(q / alpha, dim=1, keepdim=True)

    def q_min(self, obs):
        return torch.min(self.q1(obs), self.q2(obs))

    def update_targets(self, tau):
        soft_update(self.q1, self.q1_target, tau)
        soft_update(self.q2, self.q2_target, tau)


class ContinuousDoubleQ:
    def __init__(self, obs_dim, act_dim, hidden, lr, device):
        self.device = device
        self.q1 = QNetContinuous(obs_dim, act_dim, hidden).to(device)
        self.q2 = QNetContinuous(obs_dim, act_dim, hidden).to(device)
        self.q1_target = QNetContinuous(obs_dim, act_dim, hidden).to(device)
        self.q2_target = QNetContinuous(obs_dim, act_dim, hidden).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

    def q_min(self, obs, action):
        return torch.min(self.q1(obs, action), self.q2(obs, action))

    def q_min_target(self, obs, action):
        return torch.min(self.q1_target(obs, action), self.q2_target(obs, action))

    def update_targets(self, tau):
        soft_update(self.q1, self.q1_target, tau)
        soft_update(self.q2, self.q2_target, tau)


class SOARAgentDiscrete:
    def __init__(self, obs_dim, act_dim, cfg, device):
        self.gamma = cfg["gamma"]
        self.device = device
        self.alpha = torch.tensor(cfg["alpha"]).to(device)
        self.sigma = cfg["sigma"]
        self.tau = cfg["tau"]
        self.n_critics = cfg["n_critics"]
        hidden = cfg["hidden"]

        self.critics = [
            DiscreteDoubleQ(obs_dim, act_dim, hidden, cfg["lr"], device)
            for _ in range(self.n_critics)
        ]

    def choose_action(self, obs_np, deterministic=False):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_list = [c.q_min(obs) for c in self.critics]
            q_opt = optimistic_q(q_list, self.sigma)
            if deterministic:
                return int(q_opt.argmax(dim=1).item())
            dist = Categorical(F.softmax(q_opt / self.alpha, dim=1))
        return dist.sample().item()

    def update_one(self, critic, obs_b, action_b, reward_b, next_obs_b, done_b):
        # plain SAC backup (no optimism in the target — Algorithm 7)
        with torch.no_grad():
            next_v = critic.soft_v(next_obs_b, self.alpha, target=True)
            target_q = reward_b + self.gamma * (1 - done_b) * next_v

        q1 = critic.q1(obs_b).gather(1, action_b.long())
        q2 = critic.q2(obs_b).gather(1, action_b.long())
        loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        critic.optimizer.zero_grad()
        loss.backward()
        critic.optimizer.step()
        critic.update_targets(self.tau)


class SOARAgentContinuous:
    def __init__(self, obs_dim, act_dim, act_limit, cfg, device):
        self.gamma = cfg["gamma"]
        self.device = device
        self.alpha = cfg["alpha"]
        self.sigma = cfg["sigma"]
        self.tau = cfg["tau"]
        self.n_critics = cfg["n_critics"]
        hidden = cfg["hidden"]

        self.critics = [
            ContinuousDoubleQ(obs_dim, act_dim, hidden, cfg["lr"], device)
            for _ in range(self.n_critics)
        ]
        self.policy = PolicyContinuous(obs_dim, act_dim, hidden, act_limit).to(device)
        self.pi_optimizer = Adam(self.policy.parameters(), lr=cfg["lr"])

    def choose_action(self, obs_np, deterministic=False):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy(obs, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update_one(self, critic, obs_b, action_b, reward_b, next_obs_b, done_b):
        # plain SAC critic backup, no optimism in target (Algorithm 7)
        with torch.no_grad():
            next_action, next_log_prob = self.policy(next_obs_b)
            next_q = critic.q_min_target(next_obs_b, next_action)
            next_v = next_q - self.alpha * next_log_prob
            target_q = reward_b + self.gamma * (1 - done_b) * next_v

        q1 = critic.q1(obs_b, action_b)
        q2 = critic.q2(obs_b, action_b)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        critic.optimizer.zero_grad()
        q_loss.backward()
        critic.optimizer.step()
        critic.update_targets(self.tau)

    def update_actor(self, obs_b):
        # SOAR's actor update uses optimistic Q over the L critics (Algorithm 6 lines 14-16)
        action_pi, log_prob = self.policy(obs_b)
        q_list = [c.q_min(obs_b, action_pi) for c in self.critics]
        q_opt = optimistic_q(q_list, self.sigma)
        pi_loss = (self.alpha * log_prob - q_opt).mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()


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
    """f-IRL covariance objective with per-step mask for early-terminating episodes."""
    N, T, d = agent_trajs.shape
    s_vec = agent_trajs.reshape(-1, d)
    logits = disc.log_density_ratio(s_vec)

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

    t1 = ((-t1_per).view(N, T) * mask).sum(dim=1)

    s_tensor = torch.FloatTensor(s_vec).to(device)
    t2 = (reward_func.r(s_tensor).view(N, T) * mask).sum(dim=1)

    surrogate = (t1 * t2).mean() - t1.mean() * t2.mean()
    return surrogate / T


def train(env_name, K, seed, cfg):
    print(f"\n{'='*55}")
    print(f"SOAR+f-IRL | {env_name} | K={K} | seed={seed} | L={cfg['n_critics']} | sigma={cfg['sigma']}")
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
        agent = SOARAgentDiscrete(obs_dim, act_dim, cfg, device)
    else:
        act_dim = env.action_space.shape[0]
        act_limit = float(env.action_space.high[0])
        agent = SOARAgentContinuous(obs_dim, act_dim, act_limit, cfg, device)

    disc = Discriminator(
        input_dim=obs_dim,
        hid_dim=cfg["hidden"],
        batch_size=cfg["batch_size"],
        device=device,
    )

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

    # one buffer per critic (independent samples — required by Corollary 4.11)
    bufs = [ReplayBuffer(capacity=100_000) for _ in range(cfg["n_critics"])]
    n_critics = cfg["n_critics"]

    # warm-up: write random transitions to all buffers so each critic has something to train on
    obs, _ = env.reset(seed=seed)
    for _ in range(cfg["batch_size"]):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        done_no_max = 0.0 if pendulum_like else float(terminated)
        # Each transition goes to ONE randomly selected buffer to keep them independent.
        bufs[random.randrange(n_critics)].push(obs, action, 0.0, next_obs, done_no_max)
        obs = next_obs if not done else env.reset(seed=seed)[0]

    best_return = -float("inf")
    sac_step = 0
    steps_per_outer = cfg["train_steps"] // cfg["outer_iters"]

    for outer_itr in range(cfg["outer_iters"]):
        print(f"\n--- outer iter {outer_itr+1}/{cfg['outer_iters']} ---")

        obs, _ = env.reset(seed=seed + outer_itr)
        for _ in range(steps_per_outer):
            action = agent.choose_action(obs)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            done_no_max = float(terminated) if not pendulum_like else 0.0

            with torch.no_grad():
                r = float(reward_func.get_scalar_reward(np.array([obs]))[0])

            # send this transition to one critic's buffer (round-robin via random pick)
            buf_idx = random.randrange(n_critics)
            bufs[buf_idx].push(obs, action, r, next_obs, done_no_max)
            obs = next_obs if not done else env.reset(seed=seed + sac_step)[0]
            sac_step += 1

            # each critic trains on its own buffer
            for i, c in enumerate(agent.critics):
                if len(bufs[i]) >= cfg["batch_size"]:
                    obs_b, next_obs_b, action_b, reward_b, done_b = bufs[i].sample(
                        cfg["batch_size"], device)
                    agent.update_one(c, obs_b, action_b, reward_b, next_obs_b, done_b)

            # continuous actor update: sample from any buffer (pooled is fine for the actor),
            # but we use buf 0 by convention. Optimism is applied only here.
            if not discrete:
                if len(bufs[0]) >= cfg["batch_size"]:
                    obs_b, _, _, _, _ = bufs[0].sample(cfg["batch_size"], device)
                    agent.update_actor(obs_b)

        # discriminator + reward_func update
        agent_trajs, agent_flat, traj_lengths = collect_agent_trajectories(
            agent, env_name, cfg["collect_trajs"], discrete, max_T
        )
        print(f"  collected {agent_trajs.shape[0]} trajs, mean real length {traj_lengths.mean():.1f}")

        disc_loss = disc.learn(expert_states, agent_flat, iter=cfg["disc_iter"])
        print(f"  disc loss: {np.mean(disc_loss[-10:]):.4f}")

        for _ in range(cfg["reward_grad_steps"]):
            loss = reward_loss_firl(cfg["div"], agent_trajs, traj_lengths, disc, reward_func, device)
            reward_optimizer.zero_grad()
            loss.backward()
            reward_optimizer.step()
        print(f"  reward loss: {loss.item():.4f}")

        # NOTE: no buffer relabel — matches f-IRL reference reinitialize=False branch.
        # Each per-critic buffer's Bellman targets stay self-consistent with the reward
        # in force when the transitions were stored; SAC adapts via fresh transitions.

        mean_ret = evaluate_policy(agent, env_name, cfg["eval_eps"], discrete)
        if mean_ret > best_return:
            best_return = mean_ret
        append_trace("soar_f_irl", env_name, K, seed, sac_step, mean_ret)
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
    parser.add_argument("--sigma", type=float, default=None)
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
            if args.sigma is not None:
                cfg["sigma"] = args.sigma
            mean_return, best_return = train(env_name, K, seed, cfg)
            writer.writerow(dict(
                env=env_name, algo="soar_f_irl", K=K, seed=seed,
                mean_return=mean_return,
                best_return=best_return,
                timestamp=datetime.now().isoformat(),
            ))
            f.flush()

    print(f"\nresults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
