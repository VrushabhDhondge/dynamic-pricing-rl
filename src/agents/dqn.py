"""Deep Q-Network pricing agent (Week 3).

Replaces the Q-table with a small neural network that maps the state
[inventory, days_left] to one Q-value per price level. Two tricks keep
training stable:
  * Experience replay : learn from random past transitions, not just the latest
  * Target network    : a slowly-updated copy of the net provides the learning target
Exploration uses epsilon-greedy. Double-DQN targets curb overestimation, and the
best-scoring weights seen during training are kept (not just the final ones).
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (torch.tensor(s, dtype=torch.float32), torch.tensor(a, dtype=torch.int64),
                torch.tensor(r, dtype=torch.float32), torch.tensor(s2, dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


class QNetwork(nn.Module):
    def __init__(self, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    REWARD_SCALE = 1000.0  # keeps targets small (season return ~ 20) so training is stable

    def __init__(self, capacity, horizon, n_actions, gamma=1.0, lr=5e-4,
                 batch_size=64, target_sync=1000, eps_start=1.0, eps_end=0.10, seed=0):
        self.name = "DQN"
        self.capacity, self.horizon, self.n_actions = capacity, horizon, n_actions
        self.gamma, self.batch_size, self.target_sync = gamma, batch_size, target_sync
        self.eps_start, self.eps_end = eps_start, eps_end
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        self.q = QNetwork(n_actions)
        self.target = QNetwork(n_actions)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.steps = 0

    def _norm(self, obs):
        """Scale state to [0, 1] so both inputs are on the same footing."""
        return np.array([obs[0] / self.capacity, obs[1] / self.horizon], dtype=np.float32)

    def act(self, obs):
        """Greedy action (evaluation)."""
        with torch.no_grad():
            return int(self.q(torch.tensor(self._norm(obs))).argmax())

    def act_explore(self, obs, eps):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        return self.act(obs)

    def _learn(self):
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_next = self.q(s2).argmax(1, keepdim=True)            # online net picks the action
            next_q = self.target(s2).gather(1, best_next).squeeze(1)  # target net scores it
            target = r + self.gamma * next_q * (1 - d)
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

    def evaluate_greedy(self, env, n=100):
        """Average revenue of the greedy policy over n seasons (no exploration)."""
        totals = []
        for _ in range(n):
            obs, _ = env.reset()
            done, total = False, 0.0
            while not done:
                obs, r, done, _, _ = env.step(self.act(obs))
                total += r
            totals.append(total)
        return float(np.mean(totals))

    def train(self, env, eval_env, episodes=4000, log_every=250, eval_every=100, warmup=1000):
        history, best_score, best_state = [], -np.inf, None
        for ep in range(episodes):
            eps = max(self.eps_end,
                      self.eps_start - (self.eps_start - self.eps_end) * ep / (0.7 * episodes))
            obs, _ = env.reset()
            done, total = False, 0.0
            while not done:
                a = self.act_explore(obs, eps)
                next_obs, r, done, _, _ = env.step(a)
                self.buffer.push(self._norm(obs), a, r / self.REWARD_SCALE,
                                 self._norm(next_obs), float(done))
                obs, total = next_obs, total + r
                self.steps += 1
                if len(self.buffer) >= warmup:
                    self._learn()
                if self.steps % self.target_sync == 0:
                    self.target.load_state_dict(self.q.state_dict())
            history.append(total)
            if (ep + 1) % eval_every == 0 and len(self.buffer) >= warmup:
                score = self.evaluate_greedy(eval_env)
                if score > best_score:   # keep the best policy seen so far
                    best_score = score
                    best_state = {k: v.clone() for k, v in self.q.state_dict().items()}
            if (ep + 1) % log_every == 0:
                print(f"episode {ep+1:>6} | eps {eps:.2f} | train revenue ${np.mean(history[-log_every:]):,.0f}"
                      f" | best greedy eval ${best_score:,.0f}", flush=True)
        if best_state is not None:
            self.q.load_state_dict(best_state)
            self.target.load_state_dict(best_state)
            print(f"Restored best weights (greedy eval ${best_score:,.0f})")
        return history

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path))
        self.target.load_state_dict(self.q.state_dict())