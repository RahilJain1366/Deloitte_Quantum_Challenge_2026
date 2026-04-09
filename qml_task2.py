"""
=============================================================================
 TASK 2 — TIME SERIES MODEL: INSURANCE PREMIUM PREDICTION FOR 2021
 Data:     Dataset2.csv  (2018-2021, ZIP-level insurance + fire risk)
 Target:   Earned Premium (total) per ZIP for 2021

 Two-stage approach:
   Stage 1 — predict pure premium ($/exposure unit) using ML time series
   Stage 2 — multiply by actual 2021 earned exposure to get total premium
 This eliminates noise from the 2020 exposure reporting format change.

 Models: Gradient Boosting (50%) + Random Forest (35%) + Ridge (15%)
=============================================================================
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import json

DATASET_PATH = "tast2_data/Dataset2.csv"
OUTPUT_DIR   = "outputs_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  TASK 2 — INSURANCE PREMIUM TIME SERIES FORECASTER")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & AGGREGATE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Loading and aggregating to ZIP × Year level…")
raw = pd.read_csv(DATASET_PATH, low_memory=False)

SUM_COLS = [
    "Earned Exposure", "Earned Premium",
    "CAT Cov A Fire -  Incurred Losses", "CAT Cov A Fire -  Number of Claims",
    "CAT Cov A Smoke -  Incurred Losses",
    "CAT Cov C Fire -  Incurred Losses",
    "Non-CAT Cov A Fire -  Incurred Losses", "Non-CAT Cov A Fire -  Number of Claims",
    "Non-CAT Cov C Fire -  Incurred Losses",
    "Number of High Fire Risk Exposure", "Number of Very High Fire Risk Exposure",
    "Number of Moderate Fire Risk Exposure", "Number of Low Fire Risk Exposure",
]
MEAN_COLS = [
    "Avg Fire Risk Score", "Avg PPC",
    "Cov A Amount Weighted Avg", "Cov C Amount Weighted Avg",
    "median_income", "housing_value", "total_population",
    "median_monthly_housing_costs",
]

for c in SUM_COLS:
    if c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)
for c in MEAN_COLS:
    if c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

agg_dict = {c: "sum"  for c in SUM_COLS  if c in raw.columns}
agg_dict.update({c: "mean" for c in MEAN_COLS if c in raw.columns})

df = raw.groupby(["Year", "ZIP"]).agg(agg_dict).reset_index()
df = df.sort_values(["ZIP", "Year"]).reset_index(drop=True)
print(f"   Shape: {df.shape} | Years: {sorted(df['Year'].unique())}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
#    Key insight: model pure_premium ($/exposure) — stable across years
#    Final prediction = predicted_pure_premium × actual_2021_exposure
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Engineering features…")

# Pure premium = the actuarially stable quantity
df["pure_premium"] = np.where(df["Earned Exposure"] > 0,
    df["Earned Premium"] / df["Earned Exposure"], np.nan)

# Total incurred losses
df["total_incurred"] = (
    df.get("CAT Cov A Fire -  Incurred Losses",   pd.Series(0, index=df.index)) +
    df.get("Non-CAT Cov A Fire -  Incurred Losses", pd.Series(0, index=df.index)) +
    df.get("Non-CAT Cov C Fire -  Incurred Losses", pd.Series(0, index=df.index))
)
# Incurred loss per exposure (loss cost) — another stable signal
df["loss_cost"] = np.where(df["Earned Exposure"] > 0,
    df["total_incurred"] / df["Earned Exposure"], 0)

# Claims frequency
df["total_claims"] = (
    df.get("CAT Cov A Fire -  Number of Claims",   pd.Series(0, index=df.index)) +
    df.get("Non-CAT Cov A Fire -  Number of Claims", pd.Series(0, index=df.index))
)
df["claim_freq"] = np.where(df["Earned Exposure"] > 0,
    df["total_claims"] / df["Earned Exposure"], 0)

# High-risk exposure fraction
df["high_risk_share"] = np.where(df["Earned Exposure"] > 0,
    (df["Number of High Fire Risk Exposure"] + df["Number of Very High Fire Risk Exposure"])
    / df["Earned Exposure"], 0).clip(0, 1)

# Lag features (shift within ZIP — vectorized)
LAG_COLS = ["pure_premium", "Earned Exposure", "Avg Fire Risk Score",
            "loss_cost", "claim_freq", "high_risk_share"]
for col in LAG_COLS:
    df[f"{col}_lag1"] = df.groupby("ZIP")[col].shift(1)
    df[f"{col}_lag2"] = df.groupby("ZIP")[col].shift(2)

# YoY growth in pure premium
df["pp_growth_yoy"] = (
    (df["pure_premium"] - df["pure_premium_lag1"]) /
    df["pure_premium_lag1"].replace(0, np.nan)
).fillna(0).clip(-2, 5)

# Rolling 2-year mean pure premium
df["pp_rolling2"] = (
    df["pure_premium_lag1"].fillna(0) + df["pure_premium_lag2"].fillna(0)
) / 2

# OLS trend slope of pure premium per ZIP (vectorized)
pivot = df.pivot_table(index="ZIP", columns="Year", values="pure_premium")
train_years = [c for c in pivot.columns if c in [2018, 2019, 2020]]
if len(train_years) >= 2:
    yr_arr = np.array(train_years) - min(train_years)
    pp_arr = pivot[train_years].values
    x_mean = yr_arr.mean()
    y_mean = np.nanmean(pp_arr, axis=1, keepdims=True)
    cov    = np.nanmean((pp_arr - y_mean) * (yr_arr - x_mean), axis=1)
    var_x  = np.mean((yr_arr - x_mean) ** 2)
    slopes = np.where(var_x > 0, cov / var_x, 0)
    slope_map = dict(zip(pivot.index, slopes))
    df["pp_trend"] = df["ZIP"].map(slope_map).fillna(0)
else:
    df["pp_trend"] = 0

print(f"   Features engineered. Shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. QML RISK INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
qml_paths = [
    "/home/claude/results/qml_results_v5.json",
    "/home/claude/results/qml_results_v4.json",
]
for qml_path in qml_paths:
    if os.path.exists(qml_path):
        print(f"\n   Loading QML risk from {os.path.basename(qml_path)}…")
        with open(qml_path) as f:
            qml = json.load(f)
        qml_df = pd.DataFrame(qml["predictions_2023"])
        zip_key = "zip" if "zip" in qml_df.columns else "zip_code"
        qml_df = qml_df.rename(columns={zip_key: "ZIP_q", "risk_probability": "qml_risk_prob"})
        qml_df["ZIP_q"] = qml_df["ZIP_q"].astype(float).astype(int)
        df["ZIP_int"] = df["ZIP"].astype(int)
        df = df.merge(qml_df[["ZIP_q", "qml_risk_prob"]],
                      left_on="ZIP_int", right_on="ZIP_q", how="left")
        df = df.drop(columns=["ZIP_q", "ZIP_int"], errors="ignore")
        df["qml_risk_prob"] = df["qml_risk_prob"].fillna(df["Avg Fire Risk Score"])
        print(f"   QML merged. Coverage: {df['qml_risk_prob'].notna().mean()*100:.0f}%")
        break
else:
    print("\n   QML results not found — using Avg Fire Risk Score as proxy.")
    df["qml_risk_prob"] = df["Avg Fire Risk Score"]

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
#    Train on pure_premium for 2019-2020 (have lag1 from 2018, lag2 for 2020)
#    Test: predict 2021 pure_premium, then × 2021 exposure
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Splitting data…")

FEATURE_COLS = [c for c in [
    # Core time-series signal
    "pure_premium_lag1", "pure_premium_lag2",
    "pp_growth_yoy", "pp_rolling2", "pp_trend",
    # Loss signals
    "loss_cost_lag1", "claim_freq_lag1",
    # Risk signals
    "Avg Fire Risk Score", "qml_risk_prob",
    "Avg Fire Risk Score_lag1",
    "high_risk_share", "high_risk_share_lag1",
    "Number of High Fire Risk Exposure", "Number of Very High Fire Risk Exposure",
    # Coverage & market
    "Cov A Amount Weighted Avg", "Earned Exposure",
    # Socioeconomic
    "median_income", "housing_value",
    # Avg PPC (fire protection quality)
    "Avg PPC",
] if c in df.columns]

TARGET = "pure_premium"

# Keep only rows with valid exposure and non-null pure_premium
df_valid = df[(df["Earned Exposure"] > 0) & df["pure_premium"].notna()].copy()

# Fill NaN features with column median
for c in FEATURE_COLS:
    med = df_valid[c].median()
    df_valid[c] = df_valid[c].fillna(med)

# Clip extreme pure premiums (data quality guard: >$10k/unit is anomalous)
PP_CAP = 10000
df_valid["pure_premium"] = df_valid["pure_premium"].clip(0, PP_CAP)

train  = df_valid[df_valid["Year"].isin([2019, 2020])].copy()
test   = df_valid[df_valid["Year"] == 2021].copy()
common = set(train["ZIP"]) & set(test["ZIP"])
train  = train[train["ZIP"].isin(common)]
test   = test[test["ZIP"].isin(common)]

X_train = train[FEATURE_COLS].values
y_train = train[TARGET].values   # pure premium
X_test  = test[FEATURE_COLS].values
y_test_pp  = test[TARGET].values
y_test_exp = test["Earned Exposure"].values
y_test_tot = test["Earned Premium"].values   # ground truth total premium

print(f"   Train: {len(X_train)} rows | Test: {len(X_test)} rows | ZIPs: {len(common)}")
print(f"   Pure premium range (train): [{y_train.min():.0f}, {y_train.max():.0f}]")

# Log-transform (pure premium is right-skewed but less extreme than total)
y_train_log = np.log1p(y_train)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Training ensemble models…")

models = {
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1),
    "Ridge Regression": Ridge(alpha=10.0),
}
WEIGHTS = {"Gradient Boosting": 0.50, "Random Forest": 0.35, "Ridge Regression": 0.15}

trained = {}
for name, model in models.items():
    model.fit(X_train_s, y_train_log)
    trained[name] = model
    cv = cross_val_score(model, X_train_s, y_train_log, cv=3, scoring="r2")
    print(f"   {name}: CV R² = {cv.mean():.4f} ± {cv.std():.4f}")

# Stage 1: predict pure premium
pred_pp_log  = sum(WEIGHTS[n] * trained[n].predict(X_test_s) for n in models)
pred_pp      = np.clip(np.expm1(pred_pp_log), 0, PP_CAP)

# Stage 2: total premium = predicted pure premium × actual 2021 exposure
pred_total   = pred_pp * y_test_exp

# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Evaluating on 2021 hold-out…")

# Pure premium metrics
mae_pp  = mean_absolute_error(y_test_pp, pred_pp)
rmse_pp = np.sqrt(mean_squared_error(y_test_pp, pred_pp))
r2_pp   = r2_score(y_test_pp, pred_pp)
mask    = y_test_pp > 0
mape_pp = np.mean(np.abs((y_test_pp[mask] - pred_pp[mask]) / y_test_pp[mask])) * 100

# Total premium metrics
mae_tot  = mean_absolute_error(y_test_tot, pred_total)
rmse_tot = np.sqrt(mean_squared_error(y_test_tot, pred_total))
r2_tot   = r2_score(y_test_tot, pred_total)
mask_t   = y_test_tot > 0
mape_tot = np.mean(np.abs((y_test_tot[mask_t] - pred_total[mask_t]) / y_test_tot[mask_t])) * 100

print(f"\n   ── STAGE 1: PURE PREMIUM ($/exposure unit) ────────────────────")
print(f"   MAE   : ${mae_pp:>10,.2f}/unit")
print(f"   RMSE  : ${rmse_pp:>10,.2f}/unit")
print(f"   MAPE  : {mape_pp:>10.1f}%")
print(f"   R²    : {r2_pp:>10.4f}")

print(f"\n   ── STAGE 2: TOTAL EARNED PREMIUM ($) ──────────────────────────")
print(f"   MAE   : ${mae_tot:>15,.0f}")
print(f"   RMSE  : ${rmse_tot:>15,.0f}")
print(f"   MAPE  : {mape_tot:>14.1f}%")
print(f"   R²    : {r2_tot:>15.4f}")
print("   ────────────────────────────────────────────────────────────────")

# Feature importances
fi = pd.Series(trained["Gradient Boosting"].feature_importances_,
               index=FEATURE_COLS).sort_values(ascending=False)
print(f"\n   ── TOP 10 FEATURE IMPORTANCES (Gradient Boosting) ─────────────")
for feat, imp in fi.head(10).items():
    print(f"   {feat:<45} {imp:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Saving results…")

results_df = test[["ZIP", "Year"]].copy().reset_index(drop=True)
results_df["actual_premium_2021"]       = y_test_tot
results_df["predicted_pure_premium"]    = pred_pp.round(2)
results_df["actual_2021_exposure"]      = y_test_exp
results_df["predicted_total_premium"]   = pred_total.round(2)
results_df["abs_error"]  = np.abs(y_test_tot - pred_total)
results_df["pct_error"]  = np.where(y_test_tot > 0,
    results_df["abs_error"] / y_test_tot * 100, np.nan)
results_df["fire_risk_score"] = test["Avg Fire Risk Score"].values
results_df["qml_risk_prob"]   = test["qml_risk_prob"].values
results_df = results_df.sort_values("predicted_total_premium", ascending=False)

print(f"\n   ── TOP 20 ZIPs BY PREDICTED 2021 PREMIUM ───────────────────────")
print(f"   {'ZIP':>7}  {'Actual':>13}  {'Predicted':>13}  {'Err%':>7}  {'Risk':>6}")
print("   " + "-" * 58)
for _, r in results_df.head(20).iterrows():
    print(f"   {int(r['ZIP']):>7}  "
          f"${r['actual_premium_2021']:>12,.0f}  "
          f"${r['predicted_total_premium']:>12,.0f}  "
          f"{r['pct_error']:>6.1f}%  "
          f"{r['qml_risk_prob']:>6.3f}")

print(f"\n   ZIPs predicted   : {len(results_df)}")
print(f"   Within 20% error : {(results_df['pct_error']<20).sum()} ({(results_df['pct_error']<20).mean()*100:.1f}%)")
print(f"   Within 50% error : {(results_df['pct_error']<50).sum()} ({(results_df['pct_error']<50).mean()*100:.1f}%)")

out_csv  = os.path.join(OUTPUT_DIR, "premium_predictions_2021.csv")
out_json = os.path.join(OUTPUT_DIR, "premium_predictions_2021.json")
results_df.to_csv(out_csv, index=False)

summary = {
    "model": "Two-stage Ensemble (GB 50% + RF 35% + Ridge 15%)",
    "stage1_target": "Pure premium ($/exposure unit)",
    "stage2": "Predicted pure premium × actual 2021 exposure",
    "train_years": [2019, 2020], "predict_year": 2021,
    "n_zips": len(results_df), "n_features": len(FEATURE_COLS),
    "features": FEATURE_COLS,
    "metrics_pure_premium": {
        "MAE": round(mae_pp, 2), "RMSE": round(rmse_pp, 2),
        "MAPE_pct": round(mape_pp, 2), "R2": round(r2_pp, 4)},
    "metrics_total_premium": {
        "MAE": round(mae_tot, 2), "RMSE": round(rmse_tot, 2),
        "MAPE_pct": round(mape_tot, 2), "R2": round(r2_tot, 4)},
    "top_features": fi.head(10).round(5).to_dict(),
    "predictions": results_df.round(2).to_dict(orient="records"),
}
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n   → premium_predictions_2021.csv")
print(f"   → premium_predictions_2021.json")
print("=" * 70)
print("  DONE")
print("=" * 70)