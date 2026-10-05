"""IPL Auction Intelligence & Player Valuation System  --  run:  streamlit run app.py"""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from features import FULL_FEATURES, PERF_FEATURES, load_clean, make_X, row_from_inputs

st.set_page_config(page_title="IPL Auction Intelligence", page_icon="🏏", layout="wide")

NAVY, ORANGE, TEAL, GREY = "#0B2447", "#F5841F", "#19A7CE", "#8A94A6"
TEAM_COLORS = {
    "Chennai Super Kings": "#F9CD05", "Mumbai Indians": "#004BA0", "Royal Challengers Bengaluru": "#D1171B",
    "Kolkata Knight Riders": "#3A225D", "Delhi Capitals": "#17479E", "Punjab Kings": "#DD1F2D",
    "Rajasthan Royals": "#EA1A85", "Sunrisers Hyderabad": "#F26522", "Lucknow Super Giants": "#00A3E0",
    "Gujarat Titans": "#1C2C5B",
}

st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1250px;}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, #1B3A6B 60%, {ORANGE} 140%); color: white;
        padding: 22px 28px; border-radius: 16px; margin-bottom: 18px;}}
.hero h1 {{margin: 0; font-size: 1.9rem; color: white;}}
.hero p {{margin: 4px 0 0 0; opacity: .85;}}
.kpi {{background: rgba(128,128,128,.08); border-left: 5px solid {ORANGE}; padding: 12px 16px; border-radius: 10px;}}
.kpi .l {{font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; opacity: .7;}}
.kpi .v {{font-size: 1.7rem; font-weight: 700;}}
.kpi .s {{font-size: .8rem; opacity: .65;}}
.predbox {{background: linear-gradient(135deg, {NAVY}, #1B3A6B); color: white; padding: 26px; border-radius: 18px; text-align: center;}}
.predbox .big {{font-size: 3.2rem; font-weight: 800; color: {ORANGE}; line-height: 1.1;}}
.predbox .rng {{opacity: .85; margin-top: 4px;}}
.pill {{display:inline-block; padding:3px 12px; border-radius:20px; font-size:.8rem; font-weight:600; color:white;}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data / model
@st.cache_data
def get_data():
    d = load_clean("data.csv")
    oof = pd.read_csv("oof_predictions.csv")
    d = d.merge(oof, on=["Player_ID", "Season", "Team"], how="left").drop_duplicates(
        subset=["Player_ID", "Season", "Team", "Matches_raw", "Runs_raw", "Wickets"])
    d["Value_Gap"] = d["Pred_perf"] - d["Auction_Price"]  # + = performance suggests worth more than paid
    return d


@st.cache_resource
def get_models():
    return joblib.load("models.joblib"), json.load(open("metrics.json"))


df = get_data()
models, metrics = get_models()


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def cr(x):
    return f"₹{x:,.2f} Cr"


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=45, b=10), legend_title_text="")
    return fig


def predict(row, use_prev):
    tag = "full" if use_prev else "perf"
    p = float(models[tag].predict(make_X(row, use_prev))[0])
    lo = max(p + metrics[tag]["resid_q10"], 0.2)
    hi = p + metrics[tag]["resid_q90"]
    return max(p, 0.2), lo, max(hi, p), tag


# ------------------------------------------------------------------ sidebar
st.markdown('<div class="hero"><h1>🏏 IPL Auction Intelligence & Player Valuation</h1>'
            '<p>Player performance analytics + ML-based auction price estimation · IPL 2020–2025</p></div>',
            unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["📊 Overview", "🔍 Player Explorer", "💰 Price Predictor",
                                     "🏆 Top Valuable Players", "🏟️ Team & Season Analysis",
                                     "🧪 Model Performance"])
st.sidebar.markdown("---")
nv = int(df.Verified.sum())
st.sidebar.caption(f"Dataset: {len(df):,} player-seasons · {df.Player_ID.nunique()} players.\n\n"
                   f"**{nv}** prices are verified (2025); the other {len(df) - nv:,} are simulated demo values.")

# ================================================================== OVERVIEW
if page == "📊 Overview":
    c = st.columns(4)
    kpi(c[0], "Players", df.Player_ID.nunique(), "unique in dataset")
    kpi(c[1], "Player-seasons", f"{len(df):,}", "2020 – 2025")
    kpi(c[2], "Avg auction price", cr(df.Auction_Price.mean()), "all records")
    top = df.loc[df.Auction_Price.idxmax()]
    kpi(c[3], "Highest price", cr(top.Auction_Price), f"{top.Player_Name} · {top.Season}")
    st.write("")
    a, b = st.columns(2)
    yearly = df.groupby("Season").Auction_Price.agg(["mean", "max"]).reset_index()
    fig = go.Figure()
    fig.add_bar(x=yearly.Season, y=yearly["mean"], name="Average", marker_color=TEAL)
    fig.add_scatter(x=yearly.Season, y=yearly["max"], name="Maximum", mode="lines+markers", line=dict(color=ORANGE, width=3))
    fig.update_layout(title="Auction price by season (₹ Cr)")
    a.plotly_chart(style(fig), width="stretch")
    fig = px.histogram(df, x="Auction_Price", color="Auction_Status", nbins=30, barmode="overlay", opacity=.75,
                       color_discrete_map={"Sold (Demo)": TEAL, "Sold (Verified)": ORANGE},
                       title="Distribution of auction prices (₹ Cr)")
    b.plotly_chart(style(fig), width="stretch")
    a, b = st.columns(2)
    fig = px.scatter(df, x="Runs_raw", y="Auction_Price", color="Role", opacity=.7, hover_name="Player_Name",
                     title="Runs scored vs auction price", labels={"Runs_raw": "Runs", "Auction_Price": "Price (₹ Cr)"})
    b.plotly_chart(style(fig), width="stretch")
    role = df.groupby("Role").Auction_Price.mean().sort_values().reset_index()
    fig = px.bar(role, x="Auction_Price", y="Role", orientation="h", title="Average price by player role (₹ Cr)",
                 color_discrete_sequence=[ORANGE])
    a.plotly_chart(style(fig), width="stretch")
    st.caption("Role is derived from performance: Batter ≥75 runs, Bowler ≥4 wickets, All-rounder = both.")

# ================================================================== PLAYER EXPLORER
elif page == "🔍 Player Explorer":
    names = sorted(df.Player_Name.unique())
    name = st.selectbox("Search / select a player", names, index=names.index("Virat Kohli") if "Virat Kohli" in names else 0)
    p = df[df.Player_Name == name].sort_values("Season")
    last = p.iloc[-1]
    c = st.columns(5)
    kpi(c[0], "Latest team", last.Team.split()[0] + " " + (last.Team.split()[1] if len(last.Team.split()) > 1 else ""), f"{last.Season}")
    kpi(c[1], "Role", last.Role, f"{len(p)} season(s)")
    kpi(c[2], "Total runs", f"{int(p.Runs.sum()):,}", f"{int(p.Wickets.sum())} wickets")
    kpi(c[3], "Latest price", cr(last.Auction_Price), last.Auction_Status)
    kpi(c[4], "Model estimate", cr(last.Pred_full), f"vs actual {cr(last.Auction_Price)}")
    st.write("")
    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=p.Season, y=p.Runs, name="Runs", marker_color=TEAL)
    fig.add_scatter(x=p.Season, y=p.Strike_Rate_raw, name="Strike rate", yaxis="y2", mode="lines+markers", line=dict(color=ORANGE, width=3))
    fig.update_layout(title="Batting: runs & strike rate", yaxis2=dict(overlaying="y", side="right", showgrid=False))
    a.plotly_chart(style(fig, 340), width="stretch")
    fig = go.Figure()
    fig.add_bar(x=p.Season, y=p.Wickets, name="Wickets", marker_color=NAVY)
    fig.add_scatter(x=p.Season, y=p.Economy_raw, name="Economy", yaxis="y2", mode="lines+markers", line=dict(color=ORANGE, width=3))
    fig.update_layout(title="Bowling: wickets & economy", yaxis2=dict(overlaying="y", side="right", showgrid=False))
    b.plotly_chart(style(fig, 340), width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=p.Season, y=p.Auction_Price, name="Actual / demo price", mode="lines+markers", line=dict(color=NAVY, width=3))
    fig.add_scatter(x=p.Season, y=p.Pred_full, name="Model prediction", mode="lines+markers", line=dict(color=ORANGE, width=3, dash="dash"))
    fig.update_layout(title="Auction price: actual vs predicted (₹ Cr)")
    st.plotly_chart(style(fig, 340), width="stretch")
    show = p[["Season", "Team", "Matches_raw", "Runs_raw", "Batting_Average_raw", "Strike_Rate_raw", "Wickets",
              "Economy_raw", "Base_Price", "Auction_Price", "Pred_full", "Auction_Status"]].rename(columns={
        "Matches_raw": "Matches", "Runs_raw": "Runs", "Batting_Average_raw": "Bat Avg", "Strike_Rate_raw": "SR",
        "Economy_raw": "Economy", "Base_Price": "Base", "Auction_Price": "Price", "Pred_full": "Predicted"})
    st.dataframe(show.round(2), width="stretch", hide_index=True)
    st.caption("Predicted values here are out-of-fold: the model never saw this player when predicting them.")

# ================================================================== PREDICTOR
elif page == "💰 Price Predictor":
    st.subheader("Estimate an auction price")
    mode = st.radio("Start from", ["An existing player (auto-fill)", "A new player (enter manually)"], horizontal=True)
    d = dict(matches=10, runs=250, avg=25.0, sr=135.0, wk=0, econ=0.0, base=1.0, prev=6.0)
    if mode.startswith("An existing"):
        a, b = st.columns(2)
        nm = a.selectbox("Player", sorted(df.Player_Name.unique()))
        pl = df[df.Player_Name == nm].sort_values("Season")
        sn = b.selectbox("Use stats from season", list(pl.Season)[::-1])
        r = pl[pl.Season == sn].iloc[0]
        d = dict(matches=int(r.Matches), runs=int(r.Runs), avg=float(r.Batting_Average), sr=float(r.Strike_Rate),
                 wk=int(r.Wickets), econ=float(r.Economy), base=float(r.Base_Price), prev=float(r.Previous_Auction_Price))
    key = f"{mode}-{nm if mode.startswith('An existing') else 'new'}-{sn if mode.startswith('An existing') else 0}"
    left, right = st.columns([3, 2], gap="large")
    with left:
        c = st.columns(3)
        matches = c[0].slider("Matches", 0, 18, min(d["matches"], 18), key=f"m{key}")
        runs = c[1].slider("Runs", 0, 900, min(d["runs"], 900), key=f"r{key}")
        wk = c[2].slider("Wickets", 0, 35, min(d["wk"], 35), key=f"w{key}")
        c = st.columns(3)
        avg = c[0].number_input("Batting average", 0.0, 80.0, min(d["avg"], 80.0), 0.5, key=f"a{key}")
        sr = c[1].number_input("Strike rate", 0.0, 250.0, min(d["sr"], 250.0), 1.0, key=f"s{key}")
        econ = c[2].number_input("Economy (0 = didn't bowl)", 0.0, 20.0, min(d["econ"], 20.0), 0.1, key=f"e{key}")
        c = st.columns(2)
        base = c[0].select_slider("Base price (₹ Cr)", [0.3, 0.5, 0.75, 1.0, 1.5, 2.0], value=min([0.3, 0.5, 0.75, 1.0, 1.5, 2.0], key=lambda v: abs(v - d["base"])), key=f"b{key}")
        use_prev = c[1].toggle("I know the previous auction price", value=mode.startswith("An existing"), key=f"t{key}")
        prev = st.slider("Previous auction price (₹ Cr)", 0.2, 20.0, float(np.clip(d["prev"], 0.2, 20.0)), 0.25, key=f"p{key}") if use_prev else None
    row = row_from_inputs(matches, runs, avg, sr, wk, econ, base, prev)
    price, lo, hi, tag = predict(row, use_prev)
    with right:
        st.markdown(f'<div class="predbox"><div style="opacity:.8">ESTIMATED AUCTION PRICE</div>'
                    f'<div class="big">₹{price:.2f} Cr</div>'
                    f'<div class="rng">likely range ₹{lo:.2f} – ₹{hi:.2f} Cr</div></div>', unsafe_allow_html=True)
        st.caption(f"Model: {metrics[tag]['best']} "
                   f"({'with' if use_prev else 'without'} previous price) · typical error ±₹{metrics[tag]['results'][metrics[tag]['best']]['MAE']:.2f} Cr")
        if mode.startswith("An existing"):
            act = r.Auction_Price
            diff = price - act
            st.metric(f"Actual / demo price ({r.Auction_Status})", cr(act), f"{diff:+.2f} Cr model vs actual", delta_color="off")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=price, number=dict(suffix=" Cr"),
                                       gauge=dict(axis=dict(range=[0, 20]), bar=dict(color=ORANGE),
                                                  steps=[dict(range=[0, 5], color="#E8F1FA"), dict(range=[5, 10], color="#C7DFF2"),
                                                         dict(range=[10, 20], color="#9CC5E6")])))
        st.plotly_chart(style(gauge, 230), width="stretch")
    # similar players: nearest on performance
    pool = df.copy()
    z = lambda s: (s - df[s.name].mean()) / df[s.name].std()
    dist = ((z(pool.Runs) - (runs - df.Runs.mean()) / df.Runs.std()) ** 2 + (z(pool.Wickets) - (wk - df.Wickets.mean()) / df.Wickets.std()) ** 2
            + (z(pool.Strike_Rate) - (sr - df.Strike_Rate.mean()) / df.Strike_Rate.std()) ** 2).pow(.5)
    near = pool.assign(Distance=dist).nsmallest(6, "Distance")[["Player_Name", "Season", "Team", "Runs_raw", "Wickets", "Auction_Price"]]
    st.markdown("**Most similar past player-seasons**")
    st.dataframe(near.rename(columns={"Runs_raw": "Runs", "Auction_Price": "Price (₹ Cr)"}).round(2), width="stretch", hide_index=True)

# ================================================================== TOP PLAYERS
elif page == "🏆 Top Valuable Players":
    c = st.columns([1, 1, 1])
    season = c[0].selectbox("Season", ["All"] + sorted(df.Season.unique(), reverse=True))
    role = c[1].selectbox("Role", ["All"] + sorted(df.Role.unique()))
    n = c[2].slider("Show top", 5, 25, 10)
    f = df.copy()
    if season != "All":
        f = f[f.Season == season]
    if role != "All":
        f = f[f.Role == role]
    t1, t2, t3 = st.tabs(["💎 Most expensive", "📈 Undervalued (performance > price)", "📉 Overpaid (price > performance)"])
    with t1:
        t = f.nlargest(n, "Auction_Price")
        fig = px.bar(t.iloc[::-1], x="Auction_Price", y=t.iloc[::-1].Player_Name + " (" + t.iloc[::-1].Season.astype(str) + ")",
                     orientation="h", color="Team", color_discrete_map=TEAM_COLORS, title="Highest auction prices (₹ Cr)")
        st.plotly_chart(style(fig, 80 + 32 * n), width="stretch")
    with t2:
        t = f.nlargest(n, "Value_Gap")
        fig = px.bar(t.iloc[::-1], x="Value_Gap", y=t.iloc[::-1].Player_Name + " (" + t.iloc[::-1].Season.astype(str) + ")",
                     orientation="h", color_discrete_sequence=[TEAL], title="Fair value minus price paid (₹ Cr)")
        st.plotly_chart(style(fig, 80 + 32 * n), width="stretch")
    with t3:
        t = f.nsmallest(n, "Value_Gap")
        fig = px.bar(t.iloc[::-1], x="Value_Gap", y=t.iloc[::-1].Player_Name + " (" + t.iloc[::-1].Season.astype(str) + ")",
                     orientation="h", color_discrete_sequence=[ORANGE], title="Fair value minus price paid (₹ Cr)")
        st.plotly_chart(style(fig, 80 + 32 * n), width="stretch")
    cols = ["Player_Name", "Season", "Team", "Role", "Runs_raw", "Wickets", "Auction_Price", "Pred_perf", "Value_Gap"]
    st.dataframe(f.sort_values("Value_Gap", ascending=False)[cols].rename(columns={
        "Runs_raw": "Runs", "Auction_Price": "Price", "Pred_perf": "Performance-based fair price", "Value_Gap": "Value gap"}).round(2),
        width="stretch", hide_index=True, height=300)
    st.caption("Fair price = out-of-fold estimate from the performance-only model (ignores previous price and player name), "
               "so it answers: what do the stats alone say this player is worth?")

# ================================================================== TEAM & SEASON
elif page == "🏟️ Team & Season Analysis":
    a, b = st.columns(2)
    spend = df.groupby(["Season", "Team"]).Auction_Price.sum().reset_index()
    fig = px.bar(spend, x="Season", y="Auction_Price", color="Team", color_discrete_map=TEAM_COLORS,
                 title="Total squad price by season (₹ Cr, players in dataset)")
    a.plotly_chart(style(fig, 420), width="stretch")
    avg = df.groupby("Team").Auction_Price.mean().sort_values().reset_index()
    fig = px.bar(avg, x="Auction_Price", y="Team", orientation="h", color="Team", color_discrete_map=TEAM_COLORS,
                 title="Average player price by team (₹ Cr)")
    fig.update_layout(showlegend=False)
    b.plotly_chart(style(fig, 420), width="stretch")
    team = st.selectbox("Drill into a team", sorted(df.Team.unique()))
    t = df[df.Team == team]
    c = st.columns(4)
    kpi(c[0], "Player-seasons", len(t), team)
    kpi(c[1], "Avg price", cr(t.Auction_Price.mean()), "")
    kpi(c[2], "Total runs", f"{int(t.Runs.sum()):,}", "")
    kpi(c[3], "Total wickets", f"{int(t.Wickets.sum()):,}", "")
    st.write("")
    a, b = st.columns(2)
    ts = t.groupby("Season").agg(Avg_Price=("Auction_Price", "mean"), Runs=("Runs", "sum")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=ts.Season, y=ts.Runs, name="Total runs", marker_color=TEAM_COLORS[team])
    fig.add_scatter(x=ts.Season, y=ts.Avg_Price, name="Avg price (Cr)", yaxis="y2", mode="lines+markers", line=dict(color=ORANGE, width=3))
    fig.update_layout(title=f"{team}: runs vs average price", yaxis2=dict(overlaying="y", side="right", showgrid=False))
    a.plotly_chart(style(fig, 340), width="stretch")
    tt = t.nlargest(8, "Auction_Price").drop_duplicates("Player_Name")
    fig = px.bar(tt.iloc[::-1], x="Auction_Price", y="Player_Name", orientation="h", title="Most expensive players",
                 color_discrete_sequence=[TEAM_COLORS[team]])
    b.plotly_chart(style(fig, 340), width="stretch")
    pivot = df.pivot_table(index="Team", columns="Season", values="Auction_Price", aggfunc="mean").round(2)
    fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Blues", title="Average price heatmap: team × season (₹ Cr)")
    st.plotly_chart(style(fig, 420), width="stretch")

# ================================================================== MODEL PERFORMANCE
else:
    st.subheader("How good is the model?")
    st.write(f"Trained on {metrics['n_rows']:,} player-seasons. Scores use **5-fold cross-validation grouped by player**, "
             "so the model is always tested on players it has never seen.")
    for tag, title in [("full", "With previous auction price"), ("perf", "Performance only (new players / fair-value ranking)")]:
        m = metrics[tag]
        res = pd.DataFrame(m["results"]).T.rename(columns={"MAE": "MAE (Cr)", "RMSE": "RMSE (Cr)", "R2": "R²", "MAE_verified": "MAE on verified (Cr)"})
        st.markdown(f"**{title}** — best: `{m['best']}`")
        st.dataframe(res.round(3), width="stretch")
    a, b = st.columns(2)
    imp = pd.Series(metrics["full"]["importance"]).sort_values().reset_index()
    imp.columns = ["Feature", "Importance"]
    a.plotly_chart(style(px.bar(imp, x="Importance", y="Feature", orientation="h", title="Feature importance (full model)",
                                color_discrete_sequence=[ORANGE]), 420), width="stretch")
    fig = px.scatter(df, x="Auction_Price", y="Pred_full", color="Auction_Status", opacity=.6, hover_name="Player_Name",
                     color_discrete_map={"Sold (Demo)": TEAL, "Sold (Verified)": ORANGE},
                     title="Actual vs predicted price (out-of-fold, ₹ Cr)", labels={"Auction_Price": "Actual", "Pred_full": "Predicted"})
    fig.add_shape(type="line", x0=0, y0=0, x1=27, y1=27, line=dict(color=GREY, dash="dash"))
    b.plotly_chart(style(fig, 420), width="stretch")
    st.info(f"**Be aware:** a model that always guesses the mean is off by ₹{metrics['baseline_MAE_mean']:.2f} Cr on average. "
            "Our models beat that clearly on the full dataset, but much of this comes from the simulated demo prices, "
            "which closely follow previous price. On the 72 *verified* 2025 prices the error is far larger (≈₹5 Cr), because "
            "real mega-auction prices reset sharply. Treat predictions as a demonstration of the method, not a real bidding tool.")
