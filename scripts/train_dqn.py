import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.envs.pricing_env import DynamicPricingEnv
from src.agents.dqn import DQNAgent
from src.agents.q_learning import QLearningAgent
from src.agents.baselines import FixedPrice, DemandBased
from src.utils.evaluation import evaluate

ROOT = pathlib.Path(__file__).resolve().parents[1]
EPISODES = 4000

env = DynamicPricingEnv(seed=1)
agent = DQNAgent(env.capacity, env.horizon, env.action_space.n)
eval_env = DynamicPricingEnv(seed=777)   # separate env used only to pick the best checkpoint
agent.train(env, eval_env, episodes=EPISODES)
(ROOT / "models").mkdir(exist_ok=True)
agent.save(ROOT / "models" / "dqn.pt")

# Load the Q-learning table trained earlier, for comparison
ql = QLearningAgent(env.capacity, env.horizon, env.action_space.n)
ql.load(ROOT / "models" / "q_table.npy")

test_env = DynamicPricingEnv(seed=123)
print(f"\n{'Agent':<20}{'Revenue':>12}{'Std':>9}{'Unsold':>9}")
for a in [FixedPrice(test_env.prices, 250),
          DemandBased(test_env.prices, test_env.capacity, test_env.horizon), ql, agent]:
    r = evaluate(test_env, a, 1000)
    print(f"{r['name']:<20}${r['mean_revenue']:>10,.0f}{r['std_revenue']:>9,.0f}{r['mean_unsold']:>9.1f}")