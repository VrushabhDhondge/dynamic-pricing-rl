"""Naive heuristic pricing agents (Week 2 baselines).

Every agent has the same interface: act(obs) -> price index,
where obs = [remaining_inventory, days_left].
"""
import numpy as np


class FixedPrice:
    """Always charge the same price."""
    def __init__(self, prices, price=250):
        self.name = f"Fixed ${price}"
        self.idx = int(np.argmin(np.abs(np.asarray(prices) - price)))

    def act(self, obs):
        return self.idx


class DailyDiscount:
    """Start at the top price and cut it by `rate` every day (default 10%)."""
    def __init__(self, prices, horizon=60, rate=0.10):
        self.name = f"Discount {int(rate*100)}%/day"
        self.prices = np.asarray(prices, dtype=float)
        self.horizon, self.rate = horizon, rate

    def act(self, obs):
        days_elapsed = self.horizon - int(obs[1])
        target = self.prices.max() * (1 - self.rate) ** days_elapsed
        return int(np.argmin(np.abs(self.prices - target)))


class TimeBased:
    """Fixed schedule by phase of the season: high early, lower as departure nears."""
    def __init__(self, prices, horizon=60, schedule=(400, 300, 250, 160)):
        self.name = "Time-based"
        self.prices = np.asarray(prices, dtype=float)
        self.horizon = horizon
        self.schedule = schedule  # one price per equal slice of the season

    def act(self, obs):
        elapsed = self.horizon - int(obs[1])
        phase = min(int(elapsed / self.horizon * len(self.schedule)), len(self.schedule) - 1)
        return int(np.argmin(np.abs(self.prices - self.schedule[phase])))


class DemandBased:
    """Flat demand-based rule: price rises when inventory is scarce relative
    to time left, falls when there is a surplus. Ignores demand seasonality."""
    def __init__(self, prices, capacity=100, horizon=60):
        self.name = "Demand-based"
        self.prices = np.asarray(prices, dtype=float)
        self.capacity, self.horizon = capacity, horizon

    def act(self, obs):
        inv_frac = obs[0] / self.capacity
        time_frac = max(obs[1], 1) / self.horizon
        pressure = inv_frac / time_frac          # >1 = surplus, <1 = scarce
        # map pressure 0.5..1.5 onto the price ladder, high price when scarce
        level = np.clip((1.5 - pressure), 0, 1) * (len(self.prices) - 1)
        return int(round(level))