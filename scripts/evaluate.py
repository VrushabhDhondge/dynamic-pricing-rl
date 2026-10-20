"""Week 4: evaluate all agents over 1,000 seasons and make the business plots.

Run from the project root:  python scripts\\evaluate.py
Saves figures to results/figures and a summary table to results/summary.csv
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.envs.pricing_env import DynamicPricingEnv
from src.agents.baselines import FixedPrice, DailyDiscount, TimeBased, DemandBased
from src.agents.q_learning import QLearningAgent
from src.utils.evaluation import run_season

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
N_SEASONS = 1000

env = DynamicPricingEnv(seed=2024)
agents = [FixedPrice(env.prices, 250), DailyDiscount(env.prices, env.horizon),
          TimeBased(env.prices, env.horizon), DemandBased(env.prices, env.capacity, env.horizon)]

# Learned agents (skipped with a message if the model file is missing)
q_path = ROOT / "models" / "q_table.npy"
if q_path.exists():
    ql = QLearningAgent(env.capacity, env.horizon, env.action_space.n)
    ql.load(q_path)
    agents.append(ql)
else:
    print("! models/q_table.npy not found - skipping Q-Learning")

dqn_path = ROOT / "models" / "dqn.pt"
if dqn_path.exists():
    try:
        from src.agents.dqn import DQNAgent
        dqn = DQNAgent(env.capacity, env.horizon, env.action_space.n)
        dqn.load(dqn_path)
        agents.append(dqn)
    except ImportError:
        print("! torch not installed - skipping DQN")
else:
    print("! models/dqn.pt not found - skipping DQN")

# ---- run the seasons ----------------------------------------------------
results, price_paths = {}, {}
for a in agents:
    revs, unsold, paths = [], [], []
    for _ in range(N_SEASONS):
        r, u, p = run_season(env, a)
        revs.append(r); unsold.append(u)
        paths.append(p + [np.nan] * (env.horizon - len(p)))   # pad seasons that sold out early
    results[a.name] = {"revenue": np.array(revs), "unsold": np.array(unsold)}
    price_paths[a.name] = np.nanmean(np.array(paths, dtype=float), axis=0)

# ---- summary table ------------------------------------------------------
rows = []
best_base = max((n for n in results if n not in ("Q-Learning", "DQN")),
                key=lambda n: results[n]["revenue"].mean())
for n, r in results.items():
    rows.append({"Agent": n, "Mean revenue": r["revenue"].mean(), "Std": r["revenue"].std(),
                 "Mean unsold seats": r["unsold"].mean(),
                 f"vs best baseline ({best_base})":
                     f"{(r['revenue'].mean() / results[best_base]['revenue'].mean() - 1) * 100:+.1f}%"})
summary = pd.DataFrame(rows).sort_values("Mean revenue", ascending=False)
summary.to_csv(ROOT / "results" / "summary.csv", index=False)
print(summary.round(1).to_string(index=False))

sns.set_theme(style="whitegrid")
rl = [n for n in results if n in ("Q-Learning", "DQN")]

# Plot 1: price trajectory (average price charged on each day of the season)
plt.figure(figsize=(10, 5))
days = np.arange(1, env.horizon + 1)
for n in price_paths:
    plt.plot(days, price_paths[n], lw=3 if n in rl else 1.3, label=n, alpha=1 if n in rl else .7)
plt.xlabel("Days since sales opened (60 = departure day)")
plt.ylabel("Average price charged ($)")
plt.title("Price trajectory over the selling season")
plt.legend(); plt.tight_layout(); plt.savefig(FIG / "price_trajectory.png", dpi=150); plt.close()

# Plot 2: revenue comparison with 95% confidence bars
order = list(summary["Agent"])
means = [results[n]["revenue"].mean() for n in order]
ci = [1.96 * results[n]["revenue"].std() / np.sqrt(N_SEASONS) for n in order]
plt.figure(figsize=(9, 5))
colors = ["#1D9E75" if n in rl else "#888780" for n in order]
plt.bar(order, means, yerr=ci, color=colors, capsize=4)
plt.ylabel("Average revenue per season ($)")
plt.title(f"Revenue over {N_SEASONS:,} simulated seasons (95% CI)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout(); plt.savefig(FIG / "revenue_comparison.png", dpi=150); plt.close()

# Plot 3: revenue distribution
plt.figure(figsize=(9, 5))
for n in order:
    sns.kdeplot(results[n]["revenue"], label=n, lw=2.5 if n in rl else 1.2, fill=False)
plt.xlabel("Revenue per season ($)"); plt.title("Distribution of season revenue")
plt.legend(); plt.tight_layout(); plt.savefig(FIG / "revenue_distribution.png", dpi=150); plt.close()

# Plot 4: learned policy map - which price does each RL agent pick in each situation?
for n in rl:
    agent = next(a for a in agents if a.name == n)
    inv_levels = np.arange(0, env.capacity + 1, 2)[1:]       # seats left (2..100)
    day_levels = np.arange(1, env.horizon + 1)                # days left (1..60)
    grid = np.array([[env.prices[agent.act(np.array([i, d]))] for d in day_levels]
                     for i in inv_levels])
    plt.figure(figsize=(10, 5.5))
    ax = sns.heatmap(grid, cmap="viridis", cbar_kws={"label": "Price chosen ($)"},
                     xticklabels=5, yticklabels=5)
    ax.set_xticklabels(day_levels[::5]); ax.set_yticklabels(inv_levels[::5])
    ax.invert_yaxis(); ax.invert_xaxis()    # left = season start, right = departure
    plt.xlabel("Days left until departure  (season start  ->  departure)")
    plt.ylabel("Seats remaining")
    plt.title(f"{n}: learned pricing policy")
    plt.tight_layout(); plt.savefig(FIG / f"policy_map_{n.lower().replace('-', '_')}.png", dpi=150)
    plt.close()

print(f"\nSaved figures to {FIG}")