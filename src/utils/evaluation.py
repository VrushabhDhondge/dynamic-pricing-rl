"""Run many simulated seasons for an agent and summarise the results."""
import numpy as np


def run_season(env, agent):
    """Play one full season. Returns total revenue, unsold seats, price per day."""
    obs, _ = env.reset()
    done, total, prices = False, 0.0, []
    while not done:
        obs, reward, done, _, info = env.step(agent.act(obs))
        total += reward
        prices.append(info["price"])
    return total, env.inventory, prices


def evaluate(env, agent, n_seasons=1000):
    revs, unsold = [], []
    for _ in range(n_seasons):
        r, u, _ = run_season(env, agent)
        revs.append(r)
        unsold.append(u)
    return {"name": agent.name, "mean_revenue": np.mean(revs),
            "std_revenue": np.std(revs), "mean_unsold": np.mean(unsold)}