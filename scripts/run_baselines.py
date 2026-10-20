import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # project root

from src.envs.pricing_env import DynamicPricingEnv
from src.agents.baselines import FixedPrice, DailyDiscount, TimeBased, DemandBased
from src.utils.evaluation import evaluate

env = DynamicPricingEnv(seed=0)
agents = [FixedPrice(env.prices, 250), DailyDiscount(env.prices, env.horizon),
          TimeBased(env.prices, env.horizon), DemandBased(env.prices, env.capacity, env.horizon)]

print(f"{'Agent':<20}{'Revenue':>12}{'Std':>9}{'Unsold':>9}")
for a in agents:
    r = evaluate(env, a, 1000)
    print(f"{r['name']:<20}${r['mean_revenue']:>10,.0f}{r['std_revenue']:>9,.0f}{r['mean_unsold']:>9.1f}")