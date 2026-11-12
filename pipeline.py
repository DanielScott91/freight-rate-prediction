"""
Freight Rate Prediction Pipeline
- Model A: predicts validation.csv (uses market_index + quote_signal)
- Model B: predicts december-chart-inputs.csv (static + date features only)
- Time-based validation: last ~6 weeks of train-test.csv held out
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

TRAIN_PATH      = Path("train-test.csv")
VALIDATION_PATH = Path("validation.csv")
DECEMBER_PATH   = Path("december-chart-inputs.csv")
TEMPLATE_PATH   = Path("validation-predictions-template.csv")

VALIDATION_OUT  = Path("validation_predictions.csv")
DECEMBER_OUT    = Path("december-chart-inputs.csv")
RANDOM_STATE    = 42

CAT_COLS = ["pickup", "delivery", "equipment"]
DATE_FEATS = [
    "month", "day", "dayofweek", "dayofyear", "weekofyear",
    "is_weekend", "is_december", "days_to_xmas",
    "is_week_of_christmas", "is_new_years_eve_week", "is_post_thanksgiving",
]

# ---------- Load ----------
def load_csv(path: Path, parse_target: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if parse_target:
        df["posted_rate"] = pd.to_numeric(df["posted_rate"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["weight"]   = pd.to_numeric(df["weight"],   errors="coerce")
    return df

# ---------- Clean ----------
def impute_weight(df, med_by_eq, global_med):
    df = df.copy()
    df.loc[df["weight"] <= 0, "weight"] = np.nan
    mask = df["weight"].isna()
    if mask.any():
        df.loc[mask, "weight"] = df.loc[mask, "equipment"].map(med_by_eq)
        df["weight"] = df["weight"].fillna(global_med)
    return df

def filter_target_outliers(train_df):
    df = train_df.copy()
    df = df[df["posted_rate"] > 0]
    rate_per_mile = df["posted_rate"] / df["distance"].replace(0, np.nan)
    keep = rate_per_mile.between(0.5, 10.0)
    print(f"[clean] dropping {(~keep).sum():,} rows with implausible $/mi")
    return df[keep].copy()

# ---------- Features ----------
def add_date_features(df):
    df = df.copy()
    d = df["date"]
    df["month"]        = d.dt.month
    df["day"]          = d.dt.day
    df["dayofweek"]    = d.dt.dayofweek
    df["dayofyear"]    = d.dt.dayofyear
    df["weekofyear"]   = d.dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["dayofweek"] >= 5).astype(int)
    df["is_december"]  = (df["month"] == 12).astype(int)
    df["days_to_xmas"] = (pd.Timestamp("2025-12-25") - d).dt.days

    # --- Freight-seasonality priors ---
    df["is_week_of_christmas"] = (
        (d >= pd.Timestamp("2025-12-22")) & (d <= pd.Timestamp("2025-12-26"))
    ).astype(int)
    df["is_new_years_eve_week"] = (d >= pd.Timestamp("2025-12-29")).astype(int)
    df["is_post_thanksgiving"] = (
        (d >= pd.Timestamp("2025-11-28")) & (d <= pd.Timestamp("2025-12-02"))
    ).astype(int)
    
    return df

def fit_categorical_encoder(train_df):
    return {c: sorted(train_df[c].astype(str).unique()) for c in CAT_COLS}

def encode_categoricals(df, vocab):
    df = df.copy()
    for c in CAT_COLS:
        cats = pd.Categorical(df[c].astype(str), categories=vocab[c])
        df[c] = cats.codes.astype(int)
    return df

def build_features(df, vocab, feature_cols):
    df = add_date_features(df)
    df = encode_categoricals(df, vocab)
    return df[feature_cols].astype(float)

# ---------- Split ----------
def time_split(df, cutoff):
    cutoff = pd.Timestamp(cutoff)
    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()

# ---------- Model ----------
def train_lgbm(X_train, y_train, X_val, y_val, label):
    model = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE, objective="regression_l1",
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], eval_metric="l1",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"[{label}] best_iteration: {model.best_iteration_}")
    return model

def report_metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = (np.abs(y_true - y_pred) / y_true).mean() * 100
    print(f"[{label}]  MAE=${mae:,.2f}  RMSE=${rmse:,.2f}  MAPE={mape:.2f}%")

# ---------- Main ----------
def main():
    print("=== Load ===")
    train = load_csv(TRAIN_PATH, parse_target=True)
    valid = load_csv(VALIDATION_PATH)
    dec   = load_csv(DECEMBER_PATH)
    print(f"train: {train.shape}  valid: {valid.shape}  dec: {dec.shape}")

    print("=== Clean / impute weight ===")
    train.loc[train["weight"] <= 0, "weight"] = np.nan
    med_by_eq = train.groupby("equipment")["weight"].median()
    global_med = float(train["weight"].median())
    train = impute_weight(train, med_by_eq, global_med)
    valid = impute_weight(valid, med_by_eq, global_med)
    dec   = impute_weight(dec,   med_by_eq, global_med)

    train = filter_target_outliers(train)
    print(f"after outlier filter: train {train.shape}")

    print("=== Split ===")
    cutoff = "2025-09-15"
    tr, va = time_split(train, cutoff)
    print(f"train split: {tr.shape}  val split: {va.shape}")

    vocab = fit_categorical_encoder(train)

    feats_full = CAT_COLS + ["distance", "weight",
                             "market_index", "quote_signal"] + DATE_FEATS
    feats_dec  = CAT_COLS + ["distance", "weight"] + DATE_FEATS

    X_tr = build_features(tr, vocab, feats_full);  y_tr = tr["posted_rate"].values
    X_va = build_features(va, vocab, feats_full);  y_va = va["posted_rate"].values

    print("=== Train Model A (full) ===")
    model_a = train_lgbm(X_tr, y_tr, X_va, y_va, "A")
    report_metrics(y_va, model_a.predict(X_va), "A/val")

    X_valid = build_features(valid, vocab, feats_full)
    valid_pred = np.clip(model_a.predict(X_valid), 1.0, None)

    X_tr_b = build_features(tr, vocab, feats_dec)
    X_va_b = build_features(va, vocab, feats_dec)

    print("=== Train Model B (December-style) ===")
    model_b = train_lgbm(X_tr_b, y_tr, X_va_b, y_va, "B")
    report_metrics(y_va, model_b.predict(X_va_b), "B/val")

    X_dec = build_features(dec, vocab, feats_dec)
    dec_pred = np.clip(model_b.predict(X_dec), 1.0, None)

    # -------------------------------------------------------------
    # Explicit Domain Adjustment for December
    # (Since LightGBM cannot extrapolate unseen holiday effects)
    # -------------------------------------------------------------
    print("=== Applying December Seasonality Adjustments ===")
    
    # 1. Post-Thanksgiving Surge (Nov 28 - Dec 2)
    tg_mask = (dec["date"] >= "2025-11-28") & (dec["date"] <= "2025-12-02")
    dec_pred[tg_mask] *= 1.03
    print(f"  Applied +3% post-Thanksgiving surge to {tg_mask.sum()} days")

    # 2. Christmas Week Collapse (Dec 22 - Dec 26)
    xmas_mask = (dec["date"] >= "2025-12-22") & (dec["date"] <= "2025-12-26")
    dec_pred[xmas_mask] *= 0.94
    print(f"  Applied -6% Christmas week dip to {xmas_mask.sum()} days")

    # 3. New Year's Eve Slow Recovery (Dec 29 - Dec 31)
    nye_mask = (dec["date"] >= "2025-12-29")
    dec_pred[nye_mask] *= 0.97
    print(f"  Applied -3% NYE week dip to {nye_mask.sum()} days")

    # Re-enforce positivity floor after multiplication
    dec_pred = np.clip(dec_pred, 1.0, None)

    # Write validation_predictions.csv
    template = pd.read_csv(TEMPLATE_PATH)
    assert len(template) == len(valid) == 12_000
    out = template.copy()
    pred_map = dict(zip(valid["load_id"], valid_pred))
    out["predicted_rate"] = out["load_id"].map(pred_map)
    assert out["predicted_rate"].notna().all()
    out.to_csv(VALIDATION_OUT, index=False)
    print(f"[write] {VALIDATION_OUT}  rows={len(out)}")

    # Fill december-chart-inputs.csv
    dec_out = dec.copy()
    dec_out["predicted_rate"] = dec_pred
    dec_out["date"] = dec_out["date"].dt.strftime("%m/%d/%Y")
    dec_out = dec_out[["pickup", "delivery", "distance", "equipment",
                       "weight", "date", "predicted_rate"]]
    dec_out.to_csv(DECEMBER_OUT, index=False)
    print(f"[write] {DECEMBER_OUT}  rows={len(dec_out)}")

    print("=== Running score.py ===")
    subprocess.run([sys.executable, "score.py",
                    "--predictions", str(VALIDATION_OUT),
                    "--december-predictions", str(DECEMBER_OUT)], check=True)

if __name__ == "__main__":
    main()
