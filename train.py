"""Train the auction price models. Run once:  python train.py"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import load_clean, make_X

df = load_clean("data.csv")
y = df["Auction_Price"].values
groups = df["Player_ID"].values  # same player never in train and test fold -> honest scores

CANDIDATES = {
    "Ridge Regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42),
}


def evaluate(with_prev):
    X = make_X(df, with_prev)
    cv = GroupKFold(n_splits=5)
    results, oof_store = {}, {}
    for name, mk in CANDIDATES.items():
        oof = cross_val_predict(mk(), X, y, groups=groups, cv=cv)
        v = df["Verified"].values
        results[name] = {
            "MAE": float(mean_absolute_error(y, oof)),
            "RMSE": float(np.sqrt(np.mean((y - oof) ** 2))),
            "R2": float(r2_score(y, oof)),
            "MAE_verified": float(mean_absolute_error(y[v], oof[v])),
        }
        oof_store[name] = oof
    best = min(results, key=lambda k: results[k]["MAE"])
    return results, best, oof_store[best]


out = {"n_rows": int(len(df)), "n_players": int(df.Player_ID.nunique()),
       "baseline_MAE_mean": float(np.mean(np.abs(y - y.mean())))}
models = {}
for tag, with_prev in [("full", True), ("perf", False)]:
    res, best, oof = evaluate(with_prev)
    resid = y - oof
    model = CANDIDATES[best]()
    model.fit(make_X(df, with_prev), y)
    models[tag] = model
    out[tag] = {"results": res, "best": best,
                "resid_q10": float(np.quantile(resid, 0.10)), "resid_q90": float(np.quantile(resid, 0.90))}
    df[f"Pred_{tag}"] = oof  # out-of-fold predictions: never seen the player in training
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=make_X(df, with_prev).columns).sort_values(ascending=False)
        out[tag]["importance"] = imp.round(4).to_dict()
    elif hasattr(model[-1], "coef_"):
        coef = pd.Series(np.abs(model[-1].coef_), index=make_X(df, with_prev).columns)
        out[tag]["importance"] = (coef / coef.sum()).sort_values(ascending=False).round(4).to_dict()
    print(f"\n== {tag} model ({'with' if with_prev else 'without'} previous price) ==")
    print(pd.DataFrame(res).T.round(3).to_string())
    print("best:", best)

joblib.dump(models, "models.joblib")
df[["Player_ID", "Season", "Team", "Pred_full", "Pred_perf"]].to_csv("oof_predictions.csv", index=False)
json.dump(out, open("metrics.json", "w"), indent=2)
print("\nbaseline MAE (always guess the mean):", round(out["baseline_MAE_mean"], 3))
