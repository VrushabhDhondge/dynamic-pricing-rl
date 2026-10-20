"""Dynamic pricing environment (airline seats / hotel rooms) for RL.

MDP
---
State  : [remaining_inventory, days_until_departure]
Action : index into a discrete price ladder
Reward : revenue earned that day (price * units sold)
Done   : departure reached (days_left == 0) or inventory sold out
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DynamicPricingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        capacity: int = 100,
        horizon: int = 60,
        prices=(80, 120, 160, 200, 250, 300, 400),
        base_demand: float = 2.0,      # expected daily buyers at the reference price
        ref_price: float = 200.0,
        elasticity: float = 1.5,       # higher = demand drops faster as price rises
        seed: int | None = None,
    ):
        super().__init__()
        self.capacity = capacity
        self.horizon = horizon
        self.prices = np.asarray(prices, dtype=np.float32)
        self.base_demand = base_demand
        self.ref_price = ref_price
        self.elasticity = elasticity

        self.action_space = spaces.Discrete(len(self.prices))
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([capacity, horizon], dtype=np.float32),
            dtype=np.float32,
        )
        self.inventory = capacity
        self.days_left = horizon
        self.reset(seed=seed)

    # ---- demand model --------------------------------------------------
    def time_multiplier(self, days_left: int) -> float:
        """Demand ramps from 0.6x (far out) to 1.4x (departure day); mean ~1."""
        progress = 1.0 - days_left / self.horizon
        return 0.6 + 0.8 * progress

    def expected_demand(self, price: float, days_left: int) -> float:
        """Poisson mean: falls exponentially with price, shifts with time left."""
        price_effect = np.exp(-self.elasticity * (price / self.ref_price - 1.0))
        return self.base_demand * self.time_multiplier(days_left) * price_effect

    # ---- gym API -------------------------------------------------------
    def _obs(self):
        return np.array([self.inventory, self.days_left], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = self.capacity
        self.days_left = self.horizon
        return self._obs(), {}

    def step(self, action: int):
        price = float(self.prices[action])
        lam = self.expected_demand(price, self.days_left)
        demand = int(self.np_random.poisson(lam))
        sold = min(demand, self.inventory)

        reward = price * sold
        self.inventory -= sold
        self.days_left -= 1

        terminated = self.inventory == 0 or self.days_left == 0
        info = {"price": price, "sold": sold, "demand": demand}
        return self._obs(), float(reward), terminated, False, info

    def render(self):
        print(f"inventory={self.inventory:3d} days_left={self.days_left:2d}")


if __name__ == "__main__":
    # Sanity test: random agent and each fixed-price agent, 500 seasons each.
    env = DynamicPricingEnv(seed=0)

    def run(policy, n=500):
        totals, unsold = [], []
        for _ in range(n):
            obs, _ = env.reset()
            done, total = False, 0.0
            while not done:
                obs, r, done, _, _ = env.step(policy(obs))
                total += r
            totals.append(total)
            unsold.append(env.inventory)
        return np.mean(totals), np.mean(unsold)

    rev, left = run(lambda o: env.action_space.sample())
    print(f"{'random':>12}: revenue ${rev:9,.0f} | unsold seats {left:5.1f}")
    for a, p in enumerate(env.prices):
        rev, left = run(lambda o, a=a: a)
        print(f"{'fixed $' + str(int(p)):>12}: revenue ${rev:9,.0f} | unsold seats {left:5.1f}")