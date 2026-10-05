"""Shared data cleaning + feature engineering for the IPL Auction Intelligence demo."""
import numpy as np
import pandas as pd

RAW_NUM = ["Matches", "Runs", "Batting_Average", "Strike_Rate", "Wickets", "Economy"]

PERF_FEATURES = [
    "Matches", "Runs", "Batting_Average", "Strike_Rate", "Wickets", "Economy",
    "Batted", "Bowled", "Runs_per_Match", "Wickets_per_Match", "Base_Price",
]
FULL_FEATURES = PERF_FEATURES + ["Previous_Auction_Price"]


def role_from_stats(runs, wickets):
    bat, bowl = runs >= 75, wickets >= 4
    if bat and bowl:
        return "All-rounder"
    if bat:
        return "Batter"
    if bowl:
        return "Bowler"
    return "Squad / Limited"


def load_clean(path="data.csv"):
    """Missing stats mean 'did not bat / did not bowl', so they become 0 with flags.
    Display copies of the original columns are kept with a _raw suffix."""
    df = pd.read_csv(path)
    df["Auction_Status"] = df["Auction_Status"].str.strip()
    df["Verified"] = df["Auction_Status"].str.contains("Verified")
    for c in RAW_NUM:
        df[c + "_raw"] = df[c]
    df["Batted"] = df["Runs"].notna().astype(int)
    df["Bowled"] = (df["Economy"].fillna(0) > 0).astype(int)
    for c in RAW_NUM:
        df[c] = df[c].fillna(0)
    # clip implausible small-sample extremes for modelling only
    df["Strike_Rate"] = df["Strike_Rate"].clip(upper=250)
    df["Economy"] = df["Economy"].clip(upper=20)
    df["Batting_Average"] = df["Batting_Average"].clip(upper=80)
    df = add_derived(df)
    df["Role"] = [role_from_stats(r, w) for r, w in zip(df["Runs"], df["Wickets"])]
    return df


def add_derived(df):
    m = df["Matches"].replace(0, np.nan)
    df["Runs_per_Match"] = (df["Runs"] / m).fillna(0)
    df["Wickets_per_Match"] = (df["Wickets"] / m).fillna(0)
    return df


def make_X(df, with_prev=True):
    return df[FULL_FEATURES if with_prev else PERF_FEATURES].astype(float)


def row_from_inputs(matches, runs, bat_avg, sr, wickets, econ, base_price, prev_price=None):
    """Build a one-row frame from the prediction-form inputs."""
    d = pd.DataFrame([{
        "Matches": matches, "Runs": runs, "Batting_Average": min(bat_avg, 80),
        "Strike_Rate": min(sr, 250), "Wickets": wickets, "Economy": min(econ, 20),
        "Batted": int(runs > 0 or matches > 0 and bat_avg > 0), "Bowled": int(econ > 0),
        "Base_Price": base_price,
        "Previous_Auction_Price": prev_price if prev_price is not None else np.nan,
    }])
    return add_derived(d)
