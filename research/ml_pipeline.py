"""
Full ML Pipeline for Paper 1.
"Identifying Skilled Traders in Prediction Markets:
 Evidence from Behavioral Features"

Data splits:
  Train:     2022-11 to 2023-12
  Val:       2024-01 to 2024-06  (Optuna tuning)
  WF Test 1: 2024-07 to 2024-12
  WF Test 2: 2024-01 to 2024-12
  TRUE OOS:  2025-01 to 2025-12  (touch once at end)

Outputs:
  - Model comparison table (CSV + LaTeX)
  - Walk-forward AUC plot
  - SHAP feature importance plot
  - Decile analysis plot
  - Calibration curves
  - All figures publication-ready
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import numpy as np
import pandas as pd
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import date
from config import RAW_DIR, PROC_DIR

# ── Output directories ────────────────────────────────────────────────────────
RESEARCH_DIR = Path(__file__).parent
FIGURES_DIR  = RESEARCH_DIR / "figures"
TABLES_DIR   = RESEARCH_DIR / "tables"
MODELS_DIR   = RESEARCH_DIR / "models"

for d in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORIES = ["sports", "crypto", "finance", "politics", "tech", "culture", "weather"]
CAT_PNL    = {c: f"pnl_resolved_{c}" for c in CATEGORIES}

RANDOM_STATE = 42

# ── Feature columns ───────────────────────────────────────────────────────────
BEHAVIORAL_FEATURES = [
    # Volume & size
    "n_trades", "n_markets", "n_events", "n_categories",
    "total_volume", "avg_trade_volume", "median_trade_volume",
    "max_trade_volume", "volume_std", "trade_size_cv",
    # Price behavior
    "avg_price_traded", "price_std",
    "frac_extreme_price", "frac_midrange",
    "frac_longshot", "frac_sureshot",
    # Maker/taker
    "frac_maker", "frac_maker_volume", "maker_txn_to_trade_ratio",
    # Timing
    "frac_early_trader", "frac_late_trader",
    "frac_new_market_trades", "frac_closing_market_trades",
    "frac_night_trading", "frac_weekend_trading",
    "avg_time_to_resolution", "avg_market_age_at_trade",
    # Market selection
    "avg_market_volume", "avg_market_liquidity",
    "avg_trades_per_market", "avg_volume_per_market",
    "frac_binary_markets",
    # Position behavior
    "frac_both_sides", "frac_held_to_resolution",
    "avg_holding_duration", "position_turnover",
    "avg_positions_per_market", "round_trip_rate",
    "oneshot_ratio",
    # Activity
    "active_days", "trading_span_days", "active_day_ratio",
    "trades_per_week", "trading_regularity",
    "burst_trading_score", "volume_gini",
    # Specialization
    "category_hhi", "category_entropy",
    "category_hhi_inverse", "n_categories",
    # Counterparty
    "counterparty_hhi", "market_hhi",
    "repeat_counterparty_rate", "counterparty_ratio",
    # Direction
    "frac_buys", "buy_sell_ratio", "avg_trade_imbalance",
    "max_trade_frac", "net_volume",
    # Category mix
    "frac_politics", "frac_sports", "frac_crypto",
    "frac_day_trading", "frac_multi_outcome",
    # Interaction features (engineered)
    "specialist_maker",       # frac_maker × category_hhi
    "early_liquid",           # frac_early_trader × avg_market_liquidity
    "longshot_winrate_proxy", # frac_longshot × frac_held_to_resolution
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print("  Loading data...")
    pnl  = pl.read_parquet(RAW_DIR / "user_pnl_summary.parquet")
    feat = pl.read_parquet(RAW_DIR / "user_features.parquet")
    monthly = pl.read_parquet(RAW_DIR / "pnl_change_monthly.parquet")

    print(f"  PnL rows:     {len(pnl):,}")
    print(f"  Feature rows: {len(feat):,}")
    print(f"  Monthly rows: {len(monthly):,}")
    print(f"  Monthly cols: {monthly.columns}")

    return pnl, feat, monthly


def build_monthly_labels(monthly: pl.DataFrame, period: str) -> pl.DataFrame:
    """
    Build labels from a specific time period.
    period: '2023', '2024-H1', '2024-H2', '2024', '2025'
    """
    date_col = next((c for c in monthly.columns
                     if any(x in c.lower() for x in ["date","time","month","period"])), None)
    addr_col = next((c for c in monthly.columns
                     if any(x in c.lower() for x in ["address","user"])), None)
    pnl_col  = next((c for c in monthly.columns
                     if any(x in c.lower() for x in ["pnl","change","delta"])), None)

    if not all([date_col, addr_col, pnl_col]):
        raise ValueError(f"Cannot find columns. Available: {monthly.columns}")

    df = monthly.with_columns(
        pl.col(date_col).cast(pl.Utf8).alias("date_str")
    )

    if period == "2023":
        df = df.filter(pl.col("date_str").str.starts_with("2023"))
    elif period == "2024-H1":
        df = df.filter(
            pl.col("date_str").str.starts_with("2024-01") |
            pl.col("date_str").str.starts_with("2024-02") |
            pl.col("date_str").str.starts_with("2024-03") |
            pl.col("date_str").str.starts_with("2024-04") |
            pl.col("date_str").str.starts_with("2024-05") |
            pl.col("date_str").str.starts_with("2024-06")
        )
    elif period == "2024-H2":
        df = df.filter(
            pl.col("date_str").str.starts_with("2024-07") |
            pl.col("date_str").str.starts_with("2024-08") |
            pl.col("date_str").str.starts_with("2024-09") |
            pl.col("date_str").str.starts_with("2024-10") |
            pl.col("date_str").str.starts_with("2024-11") |
            pl.col("date_str").str.starts_with("2024-12")
        )
    elif period == "2024":
        df = df.filter(pl.col("date_str").str.starts_with("2024"))
    elif period == "2025":
        df = df.filter(pl.col("date_str").str.starts_with("2025"))
    else:
        raise ValueError(f"Unknown period: {period}")

    labels = (
        df
        .group_by(addr_col)
        .agg(pl.sum(pnl_col).alias(f"pnl_{period}"))
        .with_columns(
            (pl.col(f"pnl_{period}") > 0).cast(pl.Int32).alias("label")
        )
        .rename({addr_col: "user_address"})
    )

    return labels


def build_feature_matrix(
    feat: pl.DataFrame,
    labels: pl.DataFrame,
    min_trades: int = 50,
) -> tuple:
    """Join features with labels, engineer interactions."""

    df = labels.join(feat, on="user_address", how="inner")
    df = df.filter(pl.col("n_trades") >= min_trades)

    # Engineer interaction features
    df = df.with_columns([
        (pl.col("frac_maker") * pl.col("category_hhi")).alias("specialist_maker"),
        (pl.col("frac_early_trader") * pl.col("avg_market_liquidity")).alias("early_liquid"),
        (pl.col("frac_longshot") * pl.col("frac_held_to_resolution")).alias("longshot_winrate_proxy"),
    ])

    available = [c for c in BEHAVIORAL_FEATURES if c in df.columns]
    available = list(dict.fromkeys(available))  # deduplicate

    X = df.select(available).to_pandas().fillna(0)
    y = df["label"].to_pandas()
    addresses = df["user_address"].to_list()

    pos = int(y.sum())
    neg = int((y == 0).sum())
    print(f"    N={len(X):,}  Profitable={pos:,} ({pos/len(X)*100:.1f}%)  "
          f"Features={len(available)}")

    return X, y, addresses, available


# ── Models ────────────────────────────────────────────────────────────────────

def get_all_models():
    """Return all models with default params for initial comparison."""
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.dummy import DummyClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    models = {
        "Random":          DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "Logistic":        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Ridge":           RidgeClassifier(),
        "Decision Tree":   DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "KNN":             KNeighborsClassifier(n_neighbors=50),
        "Random Forest":   RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost":         XGBClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                         verbosity=0, eval_metric="auc"),
        "LightGBM":        LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                          verbose=-1),
        "CatBoost":        CatBoostClassifier(iterations=300, random_state=RANDOM_STATE,
                                              verbose=False),
    }
    return models


def optimize_model(model_name: str, X_train, y_train, X_val, y_val,
                   n_trials: int = 50) -> dict:
    """Bayesian hyperparameter optimization with Optuna."""
    import optuna
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if model_name == "XGBoost":
            from xgboost import XGBClassifier
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
                "random_state": RANDOM_STATE, "verbosity": 0, "eval_metric": "auc",
            }
            model = XGBClassifier(**params)

        elif model_name == "LightGBM":
            from lightgbm import LGBMClassifier
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": RANDOM_STATE, "verbose": -1,
            }
            model = LGBMClassifier(**params)

        elif model_name == "CatBoost":
            from catboost import CatBoostClassifier
            params = {
                "iterations":       trial.suggest_int("iterations", 100, 500),
                "depth":            trial.suggest_int("depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "random_state": RANDOM_STATE, "verbose": False,
            }
            model = CatBoostClassifier(**params)

        elif model_name == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
                "max_depth":     trial.suggest_int("max_depth", 3, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features":  trial.suggest_categorical("max_features", ["sqrt","log2"]),
                "random_state": RANDOM_STATE, "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)

        elif model_name == "Logistic":
            from sklearn.linear_model import LogisticRegression
            params = {
                "C":       trial.suggest_float("C", 1e-4, 100.0, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1","l2"]),
                "solver":  "liblinear",
                "max_iter": 1000, "random_state": RANDOM_STATE,
            }
            model = LogisticRegression(**params)

        else:
            return 0.5

        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val)[:, 1]
        else:
            proba = model.decision_function(X_val)
        return roc_auc_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_optimized_model(model_name: str, best_params: dict,
                          X_train, y_train, scale_pos_weight: float):
    """Train final model with optimized hyperparameters."""
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        model = XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight,
                              random_state=RANDOM_STATE, verbosity=0, eval_metric="auc")
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbose=-1)
    elif model_name == "CatBoost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**best_params, random_state=RANDOM_STATE, verbose=False)
    elif model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "Logistic":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**best_params, max_iter=1000,
                                   random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute all evaluation metrics."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, brier_score_loss, log_loss,
        matthews_corrcoef, precision_score, recall_score,
    )
    from scipy.stats import spearmanr

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        from sklearn.preprocessing import MinMaxScaler
        df_vals = model.decision_function(X_test)
        proba = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-8)
    else:
        proba = model.predict(X_test).astype(float)

    pred = (proba >= 0.5).astype(int)

    metrics = {
        "model":     model_name,
        "roc_auc":   roc_auc_score(y_test, proba),
        "pr_auc":    average_precision_score(y_test, proba),
        "f1":        f1_score(y_test, pred),
        "brier":     brier_score_loss(y_test, proba),
        "log_loss":  log_loss(y_test, proba),
        "mcc":       matthews_corrcoef(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall":    recall_score(y_test, pred),
        "n_test":    len(y_test),
    }
    return metrics, proba


def build_stacking_ensemble(base_models: dict, X_train, y_train,
                             X_val, y_val, X_test, y_test):
    """Train stacking ensemble on top of base models."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    print("  Building stacking ensemble...")

    # Get val predictions from base models (train meta-learner on these)
    val_preds = {}
    test_preds = {}

    for name, model in base_models.items():
        if hasattr(model, "predict_proba"):
            val_preds[name]  = model.predict_proba(X_val)[:, 1]
            test_preds[name] = model.predict_proba(X_test)[:, 1]

    val_meta  = pd.DataFrame(val_preds)
    test_meta = pd.DataFrame(test_preds)

    # Meta-learner
    meta = LogisticRegression(C=1.0, random_state=RANDOM_STATE)
    meta.fit(val_meta, y_val)

    proba = meta.predict_proba(test_meta)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    print(f"  Stacking AUC: {auc:.4f}")

    return meta, proba, auc


# ── Statistical tests ─────────────────────────────────────────────────────────

def delong_test(y_true, proba1, proba2) -> float:
    """DeLong test for AUC comparison. Returns p-value."""
    from scipy.stats import norm
    n1 = int(y_true.sum())
    n2 = int((y_true == 0).sum())

    def auc_var(y, pred):
        pos = pred[y == 1]
        neg = pred[y == 0]
        auc = np.mean([np.mean(p > neg) for p in pos])
        v10 = np.var([np.mean(p > neg) for p in pos])
        v01 = np.var([np.mean(pos > n) for n in neg])
        var = v10/len(pos) + v01/len(neg)
        return auc, var

    auc1, var1 = auc_var(y_true.values, proba1)
    auc2, var2 = auc_var(y_true.values, proba2)
    z = (auc1 - auc2) / np.sqrt(var1 + var2 + 1e-12)
    return 2 * (1 - norm.cdf(abs(z)))


def permutation_test(model, X_test, y_test, n_permutations=1000) -> float:
    """Test if AUC is above chance via permutation."""
    from sklearn.metrics import roc_auc_score
    if hasattr(model, "predict_proba"):
        true_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        true_auc = roc_auc_score(y_test, model.predict(X_test))

    null_aucs = []
    rng = np.random.RandomState(RANDOM_STATE)
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_test)
        if hasattr(model, "predict_proba"):
            null_aucs.append(roc_auc_score(y_perm, model.predict_proba(X_test)[:, 1]))
        else:
            null_aucs.append(roc_auc_score(y_perm, model.predict(X_test)))

    p_value = np.mean(np.array(null_aucs) >= true_auc)
    return p_value


def bootstrap_ci(y_true, proba, n_bootstrap=1000, ci=0.95) -> tuple:
    """Bootstrap confidence interval for AUC."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(RANDOM_STATE)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            aucs.append(roc_auc_score(y_true.iloc[idx], proba[idx]))
        except Exception:
            pass
    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    return lower, upper


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame, save=True):
    """Figure 1: Model comparison bar chart."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("Model Comparison: Predicting Trading Skill in Prediction Markets",
                 fontsize=14, fontweight="bold", y=1.02)

    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1 Score")]
    colors  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(results_df)))

    for ax, (metric, label) in zip(axes, metrics):
        df_sorted = results_df.sort_values(metric, ascending=True)
        bars = ax.barh(df_sorted["model"], df_sorted[metric], color=colors)
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Random baseline")
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlim(0.4, 1.0)
        for bar, val in zip(bars, df_sorted[metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "fig1_model_comparison.pdf"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")
    plt.close()


def plot_walkforward_auc(wf_results: list, save=True):
    """Figure 2: Walk-forward AUC over time."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12, 6))

    periods = [r["period"] for r in wf_results]
    x = range(len(periods))

    for model_name in wf_results[0]["model_aucs"].keys():
        aucs = [r["model_aucs"][model_name] for r in wf_results]
        style = "-o" if model_name in ["XGBoost", "LightGBM", "Stacking"] else "--s"
        lw    = 2.5 if model_name in ["XGBoost", "LightGBM", "Stacking"] else 1.0
        ax.plot(x, aucs, style, label=model_name, linewidth=lw, markersize=6)

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random baseline")
    ax.fill_between(x, 0.5, 0.55, alpha=0.1, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=15)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Walk-Forward ROC-AUC: Skill Prediction Stability Over Time",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.45, 0.85)
    ax.grid(True, alpha=0.3)

    # Shade true OOS period
    ax.axvspan(len(periods)-1.5, len(periods)-0.5,
               alpha=0.15, color="gold", label="True OOS (2025)")

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "fig2_walkforward_auc.pdf"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")
    plt.close()


def plot_shap(model, X_test, feature_names, model_name="XGBoost", save=True):
    """Figure 3: SHAP feature importance."""
    import shap
    import matplotlib.pyplot as plt

    print(f"  Computing SHAP values for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: bar plot of mean |SHAP|
    mean_shap = np.abs(shap_vals).mean(axis=0)
    idx       = np.argsort(mean_shap)[-20:]
    axes[0].barh(
        [feature_names[i] for i in idx],
        mean_shap[idx],
        color="steelblue"
    )
    axes[0].set_xlabel("Mean |SHAP Value|", fontsize=11)
    axes[0].set_title("Feature Importance (Mean |SHAP|)",
                       fontsize=12, fontweight="bold")

    # Right: beeswarm
    shap.summary_plot(
        shap_vals, X_test,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type="dot",
        ax=axes[1] if hasattr(shap.summary_plot, "ax") else None,
    )
    axes[1].set_title("SHAP Beeswarm Plot", fontsize=12, fontweight="bold")

    fig.suptitle(f"SHAP Feature Importance — {model_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "fig3_shap_importance.pdf"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")
    plt.close()

    return shap_vals


def plot_decile_analysis(y_true, proba, model_name="Best Model", save=True):
    """Figure 4: Decile analysis — do top-scored wallets actually outperform?"""
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"label": y_true, "proba": proba})
    df["decile"] = pd.qcut(df["proba"], q=10, labels=False) + 1

    decile_stats = (
        df.groupby("decile")
          .agg(win_rate=("label","mean"), n=("label","count"))
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        decile_stats["decile"],
        decile_stats["win_rate"] * 100,
        color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, 10)),
        edgecolor="black", linewidth=0.5
    )
    ax.axhline(
        y=float(y_true.mean()) * 100,
        color="red", linestyle="--",
        label=f"Baseline ({float(y_true.mean())*100:.1f}%)"
    )
    ax.set_xlabel("Decile (1=lowest score, 10=highest score)", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title(
        f"Decile Analysis: Predicted Score vs Actual Profitability\n({model_name})",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.set_xticks(range(1, 11))

    for bar, (_, row) in zip(bars, decile_stats.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f"{row['win_rate']*100:.1f}%",
            ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "fig4_decile_analysis.pdf"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")
    plt.close()
    return decile_stats


def plot_calibration(models_probas: dict, y_test, save=True):
    """Figure 5: Calibration curves."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0,1],[0,1],"k--", label="Perfect calibration")

    for name, proba in models_probas.items():
        try:
            fraction_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
            ax.plot(mean_pred, fraction_pos, "-o", label=name, linewidth=1.5)
        except Exception:
            pass

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves: Predicted vs Actual Profitability",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "fig5_calibration.pdf"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")
    plt.close()


def plot_specialist_vs_generalist(df_results: pd.DataFrame, save=True):
    """Figure 6: Specialist vs generalist win rates per category."""
    import matplotlib.pyplot as plt

    if "category" not in df_results.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    cats = df_results["category"].unique()
    x = np.arange(len(cats))
    width = 0.35

    specialist   = df_results[df_results["type"] == "specialist"]
    generalist   = df_results[df_results["type"] == "generalist"]

    ax.bar(x - width/2, specialist["win_rate"]*100,
           width, label="Specialist", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, generalist["win_rate"]*100,
           width, label="Generalist", color="salmon", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Specialist vs Generalist Win Rate by Category",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "fig6_specialist_generalist.pdf"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight", dpi=300)
        print(f"  Saved: {path}")
    plt.close()


# ── Tables ────────────────────────────────────────────────────────────────────

def save_model_comparison_table(results: list, ci_dict: dict, pval_dict: dict):
    """Table 1: Full model comparison with CI and p-values."""
    rows = []
    for r in results:
        name = r["model"]
        ci_lo, ci_hi = ci_dict.get(name, (0, 0))
        pval = pval_dict.get(name, 1.0)
        rows.append({
            "Model":       name,
            "ROC-AUC":     f"{r['roc_auc']:.4f}",
            "95% CI":      f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            "PR-AUC":      f"{r['pr_auc']:.4f}",
            "F1":          f"{r['f1']:.4f}",
            "Brier":       f"{r['brier']:.4f}",
            "MCC":         f"{r['mcc']:.4f}",
            "p-value":     f"{pval:.4f}" if pval > 0.001 else "<0.001",
        })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = TABLES_DIR / "table1_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save LaTeX
    latex = df.to_latex(index=False, escape=False,
                         caption="Model Comparison: Predicting Trading Skill",
                         label="tab:model_comparison")
    latex_path = TABLES_DIR / "table1_model_comparison.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {latex_path}")

    print(f"\n  {df.to_string(index=False)}")
    return df


def save_walkforward_table(wf_results: list):
    """Table 2: Walk-forward results."""
    rows = []
    for r in wf_results:
        row = {"Period": r["period"], "N_test": r["n_test"]}
        for model, auc in r["model_aucs"].items():
            row[model] = f"{auc:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path  = TABLES_DIR / "table2_walkforward.csv"
    latex_path = TABLES_DIR / "table2_walkforward.tex"
    df.to_csv(csv_path, index=False)
    latex = df.to_latex(index=False, escape=False,
                         caption="Walk-Forward ROC-AUC by Period",
                         label="tab:walkforward")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {csv_path}, {latex_path}")
    print(f"\n  {df.to_string(index=False)}")
    return df


def save_precision_at_k_table(y_true, probas: dict):
    """Table 3: Precision@K for each model."""
    ks = [50, 100, 250, 500, 1000]
    rows = []
    for model_name, proba in probas.items():
        row = {"Model": model_name}
        idx_sorted = np.argsort(proba)[::-1]
        for k in ks:
            top_k = y_true.iloc[idx_sorted[:k]]
            row[f"P@{k}"] = f"{top_k.mean()*100:.1f}%"
        rows.append(row)

    baseline_row = {"Model": "Random baseline"}
    for k in ks:
        baseline_row[f"P@{k}"] = f"{float(y_true.mean())*100:.1f}%"
    rows.append(baseline_row)

    df = pd.DataFrame(rows)
    csv_path = TABLES_DIR / "table3_precision_at_k.csv"
    df.to_csv(csv_path, index=False)
    latex = df.to_latex(index=False, escape=False,
                         caption="Precision@K: Top-K Wallet Accuracy",
                         label="tab:precision_at_k")
    with open(TABLES_DIR / "table3_precision_at_k.tex", "w") as f:
        f.write(latex)
    print(f"  Saved: {csv_path}")
    print(f"\n  {df.to_string(index=False)}")
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(optimize: bool = True, n_trials: int = 50, true_oos: bool = False):
    print("\n" + "="*65)
    print("  ML PIPELINE — Predicting Trading Skill in Prediction Markets")
    print("="*65)

    # ── Load data ──────────────────────────────────────────────────────
    pnl, feat, monthly = load_data()

    # ── Build data splits ──────────────────────────────────────────────
    print("\n  Building data splits (no lookahead)...")

    # Training labels: 2023
    print("  Train period (2023):")
    labels_train = build_monthly_labels(monthly, "2023")

    # Validation labels: 2024-H1
    print("  Val period (2024-H1):")
    labels_val = build_monthly_labels(monthly, "2024-H1")

    # Test period 1: 2024-H2
    print("  Test period 1 (2024-H2):")
    labels_test1 = build_monthly_labels(monthly, "2024-H2")

    # Test period 2: full 2024
    print("  Test period 2 (2024):")
    labels_test2 = build_monthly_labels(monthly, "2024")

    # True OOS: 2025 (only used at the end)
    if true_oos:
        print("  TRUE OOS (2025) — loading but not touching:")
        labels_oos = build_monthly_labels(monthly, "2025")

    # ── Build feature matrices ─────────────────────────────────────────
    print("\n  Building feature matrices...")
    print("  Train:")
    X_train, y_train, addr_train, features = build_feature_matrix(feat, labels_train)
    print("  Val:")
    X_val,   y_val,   addr_val,   _        = build_feature_matrix(feat, labels_val)
    print("  Test 1:")
    X_test1, y_test1, addr_test1, _        = build_feature_matrix(feat, labels_test1)
    print("  Test 2:")
    X_test2, y_test2, addr_test2, _        = build_feature_matrix(feat, labels_test2)

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # ── Initial model comparison ───────────────────────────────────────
    print("\n" + "="*65)
    print("  PHASE 1: Initial Model Comparison (default params)")
    print("="*65)

    base_models = get_all_models()
    initial_results = []
    initial_probas  = {}

    for name, model in base_models.items():
        print(f"  Training {name}...")
        try:
            model.fit(X_train, y_train)
            metrics, proba = evaluate_model(model, X_test2, y_test2, name)
            initial_results.append(metrics)
            initial_probas[name] = proba
            print(f"    AUC: {metrics['roc_auc']:.4f}  "
                  f"PR-AUC: {metrics['pr_auc']:.4f}  "
                  f"F1: {metrics['f1']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")

    # ── Hyperparameter optimization ────────────────────────────────────
    print("\n" + "="*65)
    print(f"  PHASE 2: Hyperparameter Optimization (Optuna, {n_trials} trials each)")
    print("="*65)

    models_to_optimize = ["XGBoost", "LightGBM", "CatBoost",
                          "Random Forest", "Logistic"]
    optimized_models  = {}
    optimized_results = []
    optimized_probas  = {}

    for name in models_to_optimize:
        print(f"\n  Optimizing {name} ({n_trials} trials)...")
        try:
            best_params = optimize_model(
                name, X_train, y_train, X_val, y_val, n_trials=n_trials
            )
            print(f"    Best params: {best_params}")

            model = train_optimized_model(name, best_params, X_train, y_train, spw)
            metrics, proba = evaluate_model(model, X_test2, y_test2, f"{name} (opt)")
            optimized_results.append(metrics)
            optimized_probas[f"{name} (opt)"] = proba
            optimized_models[name] = model
            print(f"    AUC: {metrics['roc_auc']:.4f}  "
                  f"PR-AUC: {metrics['pr_auc']:.4f}")

            # Save model
            with open(MODELS_DIR / f"{name.lower().replace(' ','_')}_optimized.pkl","wb") as f:
                pickle.dump({"model": model, "params": best_params,
                             "features": features}, f)
        except Exception as e:
            print(f"    Failed: {e}")

    # ── Stacking ensemble ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PHASE 3: Stacking Ensemble")
    print("="*65)

    if len(optimized_models) >= 3:
        meta, stack_proba, stack_auc = build_stacking_ensemble(
            optimized_models, X_train, y_train, X_val, y_val, X_test2, y_test2
        )
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, log_loss, matthews_corrcoef, precision_score, recall_score
        stack_pred = (stack_proba >= 0.5).astype(int)
        stack_metrics = {
            "model":     "Stacking Ensemble",
            "roc_auc":   stack_auc,
            "pr_auc":    average_precision_score(y_test2, stack_proba),
            "f1":        f1_score(y_test2, stack_pred),
            "brier":     brier_score_loss(y_test2, stack_proba),
            "log_loss":  log_loss(y_test2, stack_proba),
            "mcc":       matthews_corrcoef(y_test2, stack_pred),
            "precision": precision_score(y_test2, stack_pred),
            "recall":    recall_score(y_test2, stack_pred),
            "n_test":    len(y_test2),
        }
        optimized_results.append(stack_metrics)
        optimized_probas["Stacking Ensemble"] = stack_proba

    # ── Statistical tests ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PHASE 4: Statistical Tests")
    print("="*65)

    ci_dict   = {}
    pval_dict = {}
    best_proba = optimized_probas.get(
        "Stacking Ensemble",
        list(optimized_probas.values())[-1] if optimized_probas else None
    )

    all_results = initial_results + optimized_results
    all_probas  = {**initial_probas, **optimized_probas}

    for name, proba in all_probas.items():
        print(f"  Testing {name}...")
        lo, hi = bootstrap_ci(y_test2, proba)
        ci_dict[name] = (lo, hi)
        print(f"    95% CI: [{lo:.4f}, {hi:.4f}]")

        if best_proba is not None and name != "Stacking Ensemble":
            try:
                pval = delong_test(y_test2, proba, best_proba)
                pval_dict[name] = pval
                print(f"    DeLong vs best: p={pval:.4f}")
            except Exception:
                pval_dict[name] = 1.0

    # ── Walk-forward evaluation ────────────────────────────────────────
    print("\n" + "="*65)
    print("  PHASE 5: Walk-Forward Evaluation")
    print("="*65)

    best_model_name = max(
        optimized_models.keys(),
        key=lambda n: optimized_results[[r["model"] for r in optimized_results]
                                         .index(f"{n} (opt)") 
                                         if f"{n} (opt)" in [r["model"] for r in optimized_results]
                                         else 0]["roc_auc"]
        if any(f"{n} (opt)" == r["model"] for r in optimized_results) else 0
    ) if optimized_models else "XGBoost"

    wf_periods = [
        ("2023→2024-H1", labels_train, labels_val,   X_train, y_train, X_val,   y_val),
        ("2023→2024-H2", labels_train, labels_test1, X_train, y_train, X_test1, y_test1),
        ("2023→2024",    labels_train, labels_test2, X_train, y_train, X_test2, y_test2),
    ]

    wf_results = []
    for period_name, _, _, X_tr, y_tr, X_te, y_te in wf_periods:
        print(f"\n  Period: {period_name}")
        period_aucs = {}
        for name, model in {**base_models, **optimized_models}.items():
            try:
                model.fit(X_tr, y_tr)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_te)[:, 1]
                else:
                    proba = model.predict(X_te).astype(float)
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_te, proba)
                period_aucs[name] = auc
                print(f"    {name}: AUC={auc:.4f}")
            except Exception as e:
                print(f"    {name}: failed ({e})")

        wf_results.append({
            "period":     period_name,
            "model_aucs": period_aucs,
            "n_test":     len(y_te),
        })

    # ── True OOS ──────────────────────────────────────────────────────
    if true_oos and "labels_oos" in dir():
        print("\n" + "="*65)
        print("  TRUE OOS: 2025 (first and only look)")
        print("="*65)

        print("  Building 2025 feature matrix...")
        X_oos, y_oos, addr_oos, _ = build_feature_matrix(feat, labels_oos)

        oos_results = []
        for name, model in optimized_models.items():
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_oos)[:, 1]
                else:
                    proba = model.predict(X_oos).astype(float)
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_oos, proba)
                oos_results.append({"model": name, "oos_auc": auc})
                print(f"  {name}: TRUE OOS AUC = {auc:.4f}")
            except Exception as e:
                print(f"  {name}: failed ({e})")

        oos_df = pd.DataFrame(oos_results)
        oos_df.to_csv(TABLES_DIR / "table_true_oos.csv", index=False)
        print(f"\n  Saved TRUE OOS results")

        wf_results.append({
            "period":     "TRUE OOS 2025",
            "model_aucs": {r["model"]: r["oos_auc"] for r in oos_results},
            "n_test":     len(y_oos),
        })

    # ── Generate all figures and tables ───────────────────────────────
    print("\n" + "="*65)
    print("  PHASE 6: Generating Figures and Tables")
    print("="*65)

    # Get best model and proba for figures
    best_name  = "Stacking Ensemble" if "Stacking Ensemble" in all_probas else \
                  max(all_probas.keys(),
                      key=lambda n: all_results[[r["model"] for r in all_results].index(n)]["roc_auc"]
                      if n in [r["model"] for r in all_results] else 0)
    best_proba = all_probas[best_name]

    results_df = pd.DataFrame(all_results)

    print("\n  Figure 1: Model comparison...")
    plot_model_comparison(results_df)

    print("  Figure 2: Walk-forward AUC...")
    plot_walkforward_auc(wf_results)

    # SHAP for best tree model
    best_tree = optimized_models.get("XGBoost") or \
                optimized_models.get("LightGBM") or \
                base_models.get("XGBoost")
    if best_tree is not None:
        print("  Figure 3: SHAP importance...")
        X_shap = X_test2.sample(min(2000, len(X_test2)), random_state=RANDOM_STATE)
        shap_vals = plot_shap(best_tree, X_shap, features)

    print("  Figure 4: Decile analysis...")
    decile_df = plot_decile_analysis(y_test2, best_proba, best_name)

    print("  Figure 5: Calibration curves...")
    plot_calibration(
        {k: v for k, v in all_probas.items()
         if k in ["XGBoost (opt)", "LightGBM (opt)", "Stacking Ensemble",
                  "Logistic", "Random Forest (opt)"]},
        y_test2
    )

    print("\n  Table 1: Model comparison...")
    save_model_comparison_table(all_results, ci_dict, pval_dict)

    print("  Table 2: Walk-forward...")
    save_walkforward_table(wf_results)

    print("  Table 3: Precision@K...")
    save_precision_at_k_table(y_test2, all_probas)

    # Save best model as new wallets_scored
    if best_tree is not None:
        print("\n  Saving best model scores to wallets_scored.parquet...")
        X_all = feat.select([c for c in features if c in feat.columns]).to_pandas().fillna(0)
        ml_scores = best_tree.predict_proba(X_all)[:, 1]
        scored = pl.DataFrame({
            "address":        feat["user_address"].to_list(),
            "ml_score":       ml_scores.tolist(),
            "sm_score":       ml_scores.tolist(),
            "is_smart_money": [float(s) >= 0.60 for s in ml_scores],
            "trade_count":    feat["n_trades"].to_list(),
            "taker_ratio":    feat["frac_maker"].to_list(),
            "early_entry_ratio": feat["frac_early_trader"].to_list(),
            "avg_holding_days":  feat["avg_holding_duration"].to_list(),
            "total_volume":   feat["total_volume"].to_list(),
            "specialization": feat["category_hhi"].to_list(),
            "score_profitability": [0.5] * len(feat),
            "score_skill":    ml_scores.tolist(),
            "score_reliability": feat["active_day_ratio"].to_list(),
            "top_category":   ["unknown"] * len(feat),
            "category_scores": ["{}"] * len(feat),
        })
        scored.write_parquet(PROC_DIR / "wallets_scored.parquet")
        print(f"  SM wallets (>=0.60): {scored.filter(pl.col('is_smart_money'))['address'].len():,}")

    print("\n" + "="*65)
    print("  PIPELINE COMPLETE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables:  {TABLES_DIR}")
    print(f"  Models:  {MODELS_DIR}")
    print("="*65)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--optimize",  action="store_true", default=True)
    p.add_argument("--trials",    type=int, default=50)
    p.add_argument("--true-oos",  action="store_true",
                   help="Include 2025 true OOS evaluation")
    args = p.parse_args()
    run(optimize=args.optimize, n_trials=args.trials, true_oos=args.true_oos)