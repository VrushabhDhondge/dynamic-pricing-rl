"""Revenue Manager dashboard for the RL dynamic pricing agent.

Run locally:  streamlit run app.py
"""
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

from src.envs.pricing_env import DynamicPricingEnv
from src.agents.baselines import FixedPrice, DailyDiscount, TimeBased, DemandBased
from src.agents.q_learning import QLearningAgent

st.set_page_config(page_title="RL Dynamic Pricing", page_icon="✈️", layout="wide")


# ---------------------------------------------------------------- helpers
class Bounded:
    """Revenue Manager safety bounds: clip any agent's price to [floor, ceiling]."""
    def __init__(self, agent, lo, hi):
        self.agent, self.lo, self.hi, self.name = agent, lo, hi, agent.name

    def act(self, obs):
        return int(np.clip(self.agent.act(obs), self.lo, self.hi))


@st.cache_resource
def load_rl_agents():
    """Load trained agents. DQN is optional (needs torch + models/dqn.pt)."""
    env = DynamicPricingEnv()
    agents = {}
    q_path = ROOT / "models" / "q_table.npy"
    if q_path.exists():
        ql = QLearningAgent(env.capacity, env.horizon, env.action_space.n)
        ql.load(q_path)
        agents["Q-Learning"] = ql
    dqn_path = ROOT / "models" / "dqn.pt"
    if dqn_path.exists():
        try:
            from src.agents.dqn import DQNAgent
            dqn = DQNAgent(env.capacity, env.horizon, env.action_space.n)
            dqn.load(dqn_path)
            agents["DQN"] = dqn
        except ImportError:
            pass  # torch not installed on this host: run without the DQN
    return agents


def all_agents(env, lo, hi):
    agents = [FixedPrice(env.prices, 250), DailyDiscount(env.prices, env.horizon),
              TimeBased(env.prices, env.horizon), DemandBased(env.prices, env.capacity, env.horizon)]
    agents += list(load_rl_agents().values())
    return [Bounded(a, lo, hi) for a in agents]


def simulate(agent, seed):
    """Play one season; return a day-by-day table."""
    env = DynamicPricingEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    rows, done, total, day = [], False, 0.0, 0
    while not done:
        obs, r, done, _, info = env.step(agent.act(obs))
        day, total = day + 1, total + r
        rows.append({"Day": day, "Price ($)": info["price"], "Seats sold": info["sold"],
                     "Seats left": int(obs[0]), "Cumulative revenue ($)": total})
    return pd.DataFrame(rows)


def season_chart(series: dict, column: str, ytitle: str):
    """Line chart with a fixed 1-60 day axis and distinct colours per strategy."""
    long = pd.concat([d[["Day", column]].rename(columns={column: "value"}).assign(Strategy=name)
                      for name, d in series.items()])
    names = list(series.keys())
    return (alt.Chart(long).mark_line(strokeWidth=2.5)
            .encode(x=alt.X("Day:Q", scale=alt.Scale(domain=[1, 60], nice=False), title="Day of season"),
                    y=alt.Y("value:Q", title=ytitle),
                    color=alt.Color("Strategy:N", legend=alt.Legend(orient="bottom", title=None),
                                    scale=alt.Scale(domain=names, range=["#1D9E75", "#EF9F27"])))
            .properties(height=300))


@st.cache_data(show_spinner="Simulating seasons...")
def compare(lo, hi, n_seasons):
    env = DynamicPricingEnv(seed=2024)
    out = []
    for a in all_agents(env, lo, hi):
        revs, unsold = [], []
        for _ in range(n_seasons):
            obs, _ = env.reset()
            done, total = False, 0.0
            while not done:
                obs, r, done, _, _ = env.step(a.act(obs))
                total += r
            revs.append(total); unsold.append(env.inventory)
        out.append({"Agent": a.name, "Avg revenue ($)": np.mean(revs),
                    "Std ($)": np.std(revs), "Avg unsold seats": np.mean(unsold)})
    return pd.DataFrame(out).sort_values("Avg revenue ($)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- layout
env0 = DynamicPricingEnv()
prices = [int(p) for p in env0.prices]
rl_agents = load_rl_agents()

st.title("✈️ Dynamic Pricing with Reinforcement Learning")
st.caption(f"Simulated flight: {env0.capacity} seats, {env0.horizon}-day booking window, "
           f"{len(prices)} price levels. Unsold seats are worth nothing at departure.")

if not rl_agents:
    st.error("No trained model found. Add `models/q_table.npy` (run `scripts/train_qlearning.py`).")
    st.stop()

with st.sidebar:
    st.header("Revenue Manager controls")
    floor, ceiling = st.select_slider("Allowed price range ($)", options=prices,
                                      value=(prices[0], prices[-1]))
    lo, hi = prices.index(floor), prices.index(ceiling)
    st.caption("Safety bounds: every agent's price is clipped to this range.")
    agent_name = st.selectbox("Agent to simulate", list(rl_agents.keys()))
    seed = st.number_input("Random seed (demand scenario)", 0, 99999, 42)

tab1, tab2, tab3 = st.tabs(["Simulate a season", "Compare strategies", "Learned policy"])

# ---- Tab 1
with tab1:
    agent = Bounded(rl_agents[agent_name], lo, hi)
    base = Bounded(DemandBased(env0.prices, env0.capacity, env0.horizon), lo, hi)
    df, df_b = simulate(agent, seed), simulate(base, seed)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{agent_name} revenue", f"${df['Cumulative revenue ($)'].iloc[-1]:,.0f}",
              f"{df['Cumulative revenue ($)'].iloc[-1] - df_b['Cumulative revenue ($)'].iloc[-1]:+,.0f} vs demand-based")
    c2.metric("Seats unsold", int(df["Seats left"].iloc[-1]))
    c3.metric("Average price", f"${df['Price ($)'].mean():,.0f}")
    c4.metric("Sold out on day", int(df["Day"].iloc[-1]) if df["Seats left"].iloc[-1] == 0 else "not sold out")

    left, right = st.columns(2)
    with left:
        st.subheader("Price trajectory")
        st.altair_chart(season_chart({agent_name: df, "Demand-based": df_b}, "Price ($)", "Price ($)"),
                        width="stretch")
    with right:
        st.subheader("Seats remaining")
        st.altair_chart(season_chart({agent_name: df, "Demand-based": df_b}, "Seats left", "Seats remaining"),
                        width="stretch")
    with st.expander("Day-by-day table"):
        st.dataframe(df, width="stretch", hide_index=True)

# ---- Tab 2
with tab2:
    n = st.slider("Seasons to simulate per strategy", 100, 1000, 300, step=100)
    res = compare(lo, hi, n)
    chart_df = res.rename(columns={"Avg revenue ($)": "Revenue"})
    st.altair_chart(
        alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Agent:N", sort=None, title=None, axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("Revenue:Q", title="Average revenue per season ($)"),
            color=alt.condition(alt.FieldOneOfPredicate(field="Agent", oneOf=list(rl_agents.keys())),
                                alt.value("#1D9E75"), alt.value("#888780")),
            tooltip=["Agent", alt.Tooltip("Revenue:Q", format=",.0f")],
        ).properties(height=320), width="stretch")
    st.caption("Green = reinforcement learning agents, grey = rule-based baselines.")
    st.dataframe(res.round(1), width="stretch", hide_index=True)
    best_base = res[~res["Agent"].isin(rl_agents.keys())].iloc[0]
    top_rl = res[res["Agent"].isin(rl_agents.keys())].iloc[0]
    st.info(f"Best RL agent ({top_rl['Agent']}) vs best baseline ({best_base['Agent']}): "
            f"{(top_rl['Avg revenue ($)'] / best_base['Avg revenue ($)'] - 1) * 100:+.1f}% revenue "
            f"under the \\${floor}-\\${ceiling} price range.")

# ---- Tab 3
with tab3:
    name = st.selectbox("Agent", list(rl_agents.keys()), key="map_agent")
    a = rl_agents[name]
    seats, days = np.arange(1, env0.capacity + 1), np.arange(1, env0.horizon + 1)
    bounded = Bounded(a, lo, hi)   # the map reflects the Revenue Manager's price range
    grid = np.array([[prices[bounded.act(np.array([s, d]))] for d in days] for s in seats], dtype=float)
    if name == "Q-Learning":   # hide states the agent never experienced in training
        grid[np.all(a.Q[1:env0.capacity + 1, 1:env0.horizon + 1] == 0, axis=2)] = np.nan
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   extent=[days[0] - .5, days[-1] + .5, seats[0] - .5, seats[-1] + .5])
    ax.invert_xaxis()
    ax.set_xlabel("Days left until departure (season start on the left)", fontsize=9)
    ax.set_ylabel("Seats remaining", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, label="Price chosen ($)")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    map_col, _ = st.columns([3, 2])
    with map_col:
        st.pyplot(fig)
    st.caption("Few seats + lots of time → high prices. Many seats + little time → discounts. "
               "Blank cells are situations never seen in training.")