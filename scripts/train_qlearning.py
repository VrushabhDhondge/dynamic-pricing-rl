import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.envs.pricing_env import DynamicPricingEnv
from src.agents.q_learning import QLearningAgent
from src.agents.baselines import FixedPrice, DemandBased
from src.utils.evaluation import evaluate

EPISODES = 300_000

env = DynamicPricingEnv(seed=1)
agent = QLearningAgent(env.capacity, env.horizon, env.action_space.n)
agent.train(env, episodes=EPISODES)
agent.save(pathlib.Path(__file__).resolve().parents[1] / "models" / "q_table.npy")

# Fair comparison on a fresh environment
test_env = DynamicPricingEnv(seed=123)
print(f"\n{'Agent':<20}{'Revenue':>12}{'Std':>9}{'Unsold':>9}")
for a in [FixedPrice(test_env.prices, 250),
          DemandBased(test_env.prices, test_env.capacity, test_env.horizon), agent]:
    r = evaluate(test_env, a, 1000)
    print(f"{r['name']:<20}${r['mean_revenue']:>10,.0f}{r['std_revenue']:>9,.0f}{r['mean_unsold']:>9.1f}")