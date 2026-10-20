"""Tabular Q-learning pricing agent.

Q-table shape: (inventory+1, days_left+1, n_prices).
Entry Q[i, d, a] = expected total revenue from now until departure if we
are in state (i seats left, d days left), charge price a today, and act
optimally afterwards.
"""
import numpy as np


class QLearningAgent:
    def __init__(self, n_inventory, n_days, n_actions,
                 alpha=0.5, gamma=1.0, eps_start=1.0, eps_end=0.05, seed=0):
        self.name = "Q-Learning"
        self.Q = np.zeros((n_inventory + 1, n_days + 1, n_actions))
        self.N = np.zeros_like(self.Q)          # visit counts per (state, action)
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _s(obs):
        return int(obs[0]), int(obs[1])

    def act(self, obs):
        """Greedy action (used for evaluation, no exploration)."""
        i, d = self._s(obs)
        return int(np.argmax(self.Q[i, d]))

    def act_explore(self, obs, eps):
        """Epsilon-greedy: random price with probability eps, else the best known."""
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return self.act(obs)

    def update(self, obs, action, reward, next_obs, done):
        """Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max Q(s') - Q(s,a))"""
        i, d = self._s(obs)
        target = reward
        if not done:
            ni, nd = self._s(next_obs)
            target += self.gamma * np.max(self.Q[ni, nd])
        self.N[i, d, action] += 1
        lr = max(0.02, self.alpha / (1 + 0.02 * self.N[i, d, action]))  # decaying learning rate
        self.Q[i, d, action] += lr * (target - self.Q[i, d, action])

    def train(self, env, episodes=100_000, log_every=10_000):
        history = []
        for ep in range(episodes):
            # epsilon decays linearly over the first 80% of training
            eps = max(self.eps_end,
                      self.eps_start - (self.eps_start - self.eps_end) * ep / (0.8 * episodes))
            obs, _ = env.reset()
            done, total = False, 0.0
            while not done:
                a = self.act_explore(obs, eps)
                next_obs, r, done, _, _ = env.step(a)
                self.update(obs, a, r, next_obs, done)
                obs, total = next_obs, total + r
            history.append(total)
            if (ep + 1) % log_every == 0:
                print(f"episode {ep+1:>7} | eps {eps:.2f} | avg revenue (last {log_every}) "
                      f"${np.mean(history[-log_every:]):,.0f}")
        return history

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)