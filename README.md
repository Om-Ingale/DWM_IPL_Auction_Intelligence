# IPL Auction Intelligence & Player Valuation System

College demo: explore IPL 2020–2025 player performance and estimate auction prices with machine learning.

## Run it

```bash
pip install -r requirements.txt
python train.py          # optional: models.joblib + metrics.json are already included
streamlit run app.py     # opens http://localhost:8501
```

## Pages

| Page | What it shows |
|---|---|
| Overview | KPIs, price by season, price distribution, runs vs price, price by role |
| Player Explorer | Search any of the 300 players: batting/bowling trends, price vs model prediction, full stats table |
| Price Predictor | Auto-fill from a real player-season or enter a new player; get an estimate with a likely range, gauge, and similar past players |
| Top Valuable Players | Most expensive, undervalued, and overpaid players (filter by season and role) |
| Team & Season Analysis | Spend by team/season, team drill-down, team × season heatmap |
| Model Performance | Model comparison, feature importance, actual vs predicted, honest limitations |

## Files

- `data.csv` – the provided dataset, unchanged
- `features.py` – cleaning + feature engineering (shared by training and the app)
- `train.py` – compares Ridge, Random Forest and Gradient Boosting, saves the best of each
- `app.py` – Streamlit dashboard
- `models.joblib`, `metrics.json`, `oof_predictions.csv` – outputs of `train.py`
- `test_app.py` – headless smoke test of every page

## Method (for your viva)

- **Cleaning:** missing runs/matches/economy mean the player did not bat/bowl, so they become 0 plus `Batted`/`Bowled` flags (no rows dropped). Extreme small-sample values (strike rate > 250, economy > 20) are clipped for modelling only; the dashboard still shows raw values.
- **Two models:** one *with* previous auction price, one *performance-only* (for new players and for fair-value ranking).
- **Validation:** 5-fold GroupKFold by player, so a player is never in both train and test. Predictions shown in the app are out-of-fold.
- **Likely range:** 10th–90th percentile of cross-validated errors added to the point estimate.
- **Value gap:** performance-only fair price minus price paid (positive = undervalued).

## Honest limitations

- 968 of 1,040 prices are simulated; they closely follow previous price (correlation 0.85), so much of the model's accuracy reflects that.
- The 72 verified prices are all from the 2025 mega auction. On those the model's error is about ₹5 Cr (vs about ₹0.9 Cr overall), because real prices reset sharply.
- Season is not used as a feature, so the model has no notion of price inflation or auction format.
- This is a demonstration of the method, not a real bidding tool.
