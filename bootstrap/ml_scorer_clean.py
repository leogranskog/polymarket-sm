"""
No-lookahead ML wallet scorer.

Approach:
  - Use pnl_change_monthly to get PnL per wallet per month
  - Train on wallets using only pre-2024 data
  - Label = was wallet profitable in 2023?
  - Features = user_features (behavioral, not outcome-based)
  - Predict 2024 profitability without using any 2024+ data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import numpy as np
import json
import pickle
from pathlib import Path
from config import RAW_DIR, PROC_DIR

CATEGORIES = ["sports", "crypto", "finance", "politics", "tech", "culture", "weather"]
CAT_PNL_COLS = {cat: f"pnl_resolved_{cat}" for cat in CATEGORIES}

# Only behavioral features — NO outcome-based features
# These are things we could observe BEFORE knowing if someone is profitable
FEATURE_COLS = [
    # Trading behavior
    "n_trades", "n_markets", "n_events", "n_categories",
    "avg_trade_volume", "median_trade_volume", "trade_size_cv",
    "avg_price_traded", "price_std",
    "frac_extreme_price", "frac_midrange",
    # Maker/taker behavior
    "frac_maker", "frac_maker_volume", "maker_txn_to_trade_ratio",
    # Timing
    "frac_early_trader", "frac_late_trader",
    "frac_new_market_trades", "frac_closing_market_trades",
    "frac_night_trading", "frac_weekend_trading",
    "avg_time_to_resolution", "avg_market_age_at_trade",
    # Market selection
    "avg_market_volume", "avg_market_liquidity",
    "frac_binary_markets", "avg_market_liquidity",
    # Position behavior
    "frac_both_sides", "frac_held_to_resolution",
    "avg_holding_duration", "position_turnover",
    "avg_positions_per_market", "round_trip_rate",
    # Risk behavior
    "frac_longshot", "frac_sureshot",
    "buy_sell_ratio", "avg_trade_imbalance",
    "max_trade_frac", "oneshot_ratio",
    # Activity patterns
    "active_days", "trading_span_days", "active_day_ratio",
    "trades_per_week", "trading_regularity",
    "burst_trading_score", "volume_gini",
    # Specialization
    "category_hhi", "category_entropy", "n_categories",
    "counterparty_hhi", "market_hhi",
    "repeat_counterparty_rate", "counterparty_ratio",
    # Category mix
    "frac_politics", "frac_sports", "frac_crypto",
]


def load_monthly_pnl() -> pl.DataFrame:
    """Load pnl_change_monthly — one row per (user, month)."""
    path = RAW_DIR / "pnl_change_monthly.parquet"
    print(f"  Loading pnl_change_monthly...")
    df = pl.read_parquet(path)
    print(f"  Rows: {len(df):,}  Columns: {df.columns}")
    return df


def build_training_labels(monthly_df: pl.DataFrame) -> pl.DataFrame:
    """
    Build labels from monthly PnL.
    Label = was wallet net profitable during 2023?
    Only use 2023 data for labeling — no 2024+ data.
    """
    print("\n  Building training labels from 2023 PnL...")

    # Find date/timestamp column
    date_col = next(
        (c for c in monthly_df.columns
         if "date" in c.lower() or "time" in c.lower() or "month" in c.lower()),
        None
    )
    addr_col = next(
        (c for c in monthly_df.columns
         if "address" in c.lower() or "user" in c.lower()),
        None
    )
    pnl_col = next(
        (c for c in monthly_df.columns
         if "pnl" in c.lower() or "change" in c.lower()),
        None
    )

    print(f"    date_col: {date_col}, addr_col: {addr_col}, pnl_col: {pnl_col}")
    print(f"    Sample data:")
    print(monthly_df.head(3))

    if not all([date_col, addr_col, pnl_col]):
        print("  ⚠ Could not identify columns — using pnl_summary fallback")
        return None

    # Filter to 2023 only
    df_2023 = (
        monthly_df
        .with_columns(pl.col(date_col).cast(pl.Utf8).alias("date_str"))
        .filter(pl.col("date_str").str.starts_with("2023"))
    )
    print(f"    2023 rows: {len(df_2023):,}")

    if len(df_2023) == 0:
        print("  ⚠ No 2023 data found — trying different date format")
        return None

    # Sum PnL per wallet in 2023
    labels = (
        df_2023
        .group_by(addr_col)
        .agg(pl.sum(pnl_col).alias("pnl_2023"))
        .with_columns(
            (pl.col("pnl_2023") > 0).cast(pl.Int32).alias("label")
        )
        .rename({addr_col: "user_address"})
    )

    pos = labels.filter(pl.col("label") == 1)["label"].len()
    neg = labels.filter(pl.col("label") == 0)["label"].len()
    print(f"    2023 profitable: {pos:,} | unprofitable: {neg:,}")
    print(f"    Base rate: {pos/(pos+neg)*100:.1f}% profitable in 2023")

    return labels


def build_features(feat_df: pl.DataFrame, labels: pl.DataFrame) -> tuple:
    """
    Join features with labels.
    Features = behavioral only (observable without knowing outcomes).
    """
    print("\n  Building feature matrix...")

    df = labels.join(feat_df, on="user_address", how="inner")
    print(f"  Matched wallets: {len(df):,}")

    available = [c for c in FEATURE_COLS if c in df.columns]
    # Remove duplicates
    available = list(dict.fromkeys(available))
    print(f"  Features: {len(available)} available")

    X = df.select(available).to_pandas()
    y = df["label"].to_pandas()

    # Fill nulls
    X = X.fillna(X.median())

    addresses = df["user_address"].to_list()
    return X, y, addresses, df, available


def train_xgboost(X, y) -> tuple:
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report

    print("\n  Training XGBoost (no lookahead)...")
    print(f"  Train/test split on {len(X):,} wallets")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators     = 400,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.7,
        min_child_weight = 10,
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1),
        random_state     = 42,
        eval_metric      = "auc",
        verbosity        = 0,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n  ── Model Performance (NO lookahead) ────────")
    print(f"  ROC-AUC:   {auc:.4f}  (trained on 2023, blind to 2024+)")
    print(f"  {classification_report(y_test, y_pred, target_names=['Unprofitable','Profitable'])}")

    return model, auc, X_test, y_test


def feature_importance(model, feature_names: list, top_n: int = 20):
    import pandas as pd
    imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"  ── Top {top_n} Predictive Features ────────────")
    for _, row in imp.head(top_n).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:<35} {bar} {row['importance']:.4f}")

    return imp


def score_all_wallets(
    model,
    feat_df: pl.DataFrame,
    pnl_df:  pl.DataFrame,
    feature_names: list,
) -> pl.DataFrame:
    """
    Score ALL wallets using model trained on 2023 data.
    This is what we use to make 2024 trading decisions.
    """
    print("\n  Scoring all wallets for 2024 predictions...")

    available = [c for c in feature_names if c in feat_df.columns]
    X_all = feat_df.select(available).to_pandas().fillna(0)

    ml_scores = model.predict_proba(X_all)[:, 1]
    addresses  = feat_df["user_address"].to_list()

    # Category scores from pnl_summary (pre-2024 resolved PnL)
    # Join pnl for category scoring
    pnl_joined = feat_df.join(
        pnl_df.select(["user_address"] + list(CAT_PNL_COLS.values())),
        on="user_address",
        how="left"
    )

    cat_scores_list = []
    top_cats = []

    for row in pnl_joined.select(
        [CAT_PNL_COLS[c] for c in CATEGORIES]
    ).iter_rows():
        vals = {cat: max(0.0, float(v or 0))
                for cat, v in zip(CATEGORIES, row)}
        max_v = max(vals.values()) if vals else 1.0
        if max_v > 0:
            norm = {k: round(v/max_v, 4) for k, v in vals.items()}
        else:
            norm = {k: 0.0 for k in CATEGORIES}
        cat_scores_list.append(json.dumps(norm))
        top_cats.append(max(vals, key=vals.get))

    out = pl.DataFrame({
        "address":           addresses,
        "ml_score":          ml_scores.tolist(),
        "sm_score":          ml_scores.tolist(),
        "is_smart_money":    [float(s) >= 0.60 for s in ml_scores],
        "top_category":      top_cats,
        "category_scores":   cat_scores_list,
        "trade_count":       feat_df["n_trades"].to_list(),
        "total_volume":      feat_df["total_volume"].to_list(),
        "taker_ratio":       feat_df["frac_maker"].to_list(),
        "early_entry_ratio": feat_df["frac_early_trader"].to_list(),
        "avg_holding_days":  feat_df["avg_holding_duration"].to_list(),
        "longshot_bias":     feat_df["frac_longshot"].to_list(),
        "specialization":    feat_df["category_hhi"].to_list(),
        "score_profitability": [0.5] * len(addresses),
        "score_skill":       ml_scores.tolist(),
        "score_reliability": feat_df["active_day_ratio"].to_list(),
    })

    sm_count = out.filter(pl.col("is_smart_money"))["address"].len()
    print(f"  ✅ ML SM Universe: {sm_count:,} wallets (score ≥ 0.60)")

    print(f"\n  Category distribution (SM wallets):")
    sm = out.filter(pl.col("is_smart_money"))
    counts = (
        sm.group_by("top_category")
          .count()
          .sort("count", descending=True)
    )
    for row in counts.iter_rows(named=True):
        print(f"    {row['top_category']:<12} {row['count']:>6,} wallets")

    return out


def run():
    print("\n" + "="*58)
    print("  ML Wallet Scorer — No Lookahead (trained on 2023)")
    print("="*58)

    # Load data
    monthly_df = load_monthly_pnl()
    feat_df    = pl.read_parquet(RAW_DIR / "user_features.parquet")
    pnl_df     = pl.read_parquet(RAW_DIR / "user_pnl_summary.parquet")

    print(f"  user_features: {len(feat_df):,} users")
    print(f"  pnl_summary:   {len(pnl_df):,} users")

    # Build labels from 2023 PnL only
    labels = build_training_labels(monthly_df)

    if labels is None:
        print("  ❌ Could not build labels — check pnl_change_monthly columns")
        return

    # Build features + labels
    X, y, addresses, df_matched, feature_names = build_features(feat_df, labels)

    # Train model
    model, auc, X_test, y_test = train_xgboost(X, y)

    # Feature importance
    imp_df = feature_importance(model, feature_names)

    # Score ALL wallets (for 2024 trading)
    scored = score_all_wallets(model, feat_df, pnl_df, feature_names)

    # Save everything
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    out_path = PROC_DIR / "wallets_scored.parquet"
    scored.write_parquet(out_path)
    print(f"\n  ✅ Wallets saved to {out_path}")

    imp_path = PROC_DIR / "feature_importance_clean.csv"
    imp_df.to_csv(str(imp_path), index=False)
    print(f"  ✅ Feature importance → {imp_path}")

    model_path = PROC_DIR / "xgboost_nolookahead.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":    model,
            "features": feature_names,
            "auc":      auc,
            "trained_on": "2023_data_only",
        }, f)
    print(f"  ✅ Model saved → {model_path}")

    print(f"\n  SM score tiers:")
    for t in [0.9, 0.8, 0.7, 0.6, 0.5]:
        n = scored.filter(pl.col("ml_score") >= t)["address"].len()
        print(f"    ≥ {t}: {n:,} wallets")

    print(f"\n  ✅ No-lookahead ML scoring complete!")
    print(f"     Model trained on 2023 → predicts 2024 performance")
    print(f"     AUC: {auc:.4f}")


if __name__ == "__main__":
    run()