"""
Extension 2 — Adjacent-window (short-horizon) training.

Rationale: our main model trains on 2023 (n small, tiny/early-era
population) and predicts 2025 (100x larger, different population). This
tests whether behavioral signal exists over a SHORT, adjacent horizon,
where the population is more comparable. Either outcome is a finding:
signal at short horizon + none at long horizon => regime decay.
No signal even here => population-level attributes (specialization) are
the durable story, not cross-sectional prediction of individuals.

TRUE OOS untouched (2025-H2 not used here).

Usage: python -m research.ml_pipeline_v3_adjacent --trials 50
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from config import PROC_DIR

PIT_DIR = PROC_DIR / "pit"
TAB_DIR = Path(__file__).parent / "tables_v2"
TAB_DIR.mkdir(exist_ok=True)
SEED = 42

META_COLS  = {"wallet", "asof", "first_trade", "last_trade", "wash_flag"}
LABEL_COLS = {"fwd_n_trades", "fwd_clv_vw", "fwd_clv_hitrate", "label_skilled"}

# Adjacent design: features@2024-06-30, split its OWN label window
# (2024-07..2024-12) 80/20 for train/val, predict fully-held-out 2025-H1
# using features computed at the LATER cutoff (2024-12-31), which is the
# next feature snapshot -> genuinely adjacent, non-overlapping windows.
TRAIN_CUTOFF, TRAIN_HORIZON = "2024-06-30", "2024-12-31"
TEST_CUTOFF,  TEST_HORIZON  = "2024-12-31", "2025-06-30"


def load(cutoff, horizon):
    f = pl.read_parquet(PIT_DIR / f"features_asof_{cutoff}.parquet")
    l = pl.read_parquet(PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet")
    return f.join(l, on="wallet", how="inner")


def get_feature_cols(df):
    return sorted(set(c for c in df.columns
                  if c not in META_COLS | LABEL_COLS
                  and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64,
                                      pl.Int32, pl.UInt32, pl.Boolean)))


def to_xy(df, cols, label_col):
    pdf = df.to_pandas()
    X = pdf[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = pdf[label_col]
    return X, y, pdf


def bootstrap_ci(y, p, metric_fn, n=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); p = np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        try:
            vals.append(metric_fn(y[idx], p[idx]))
        except Exception:
            pass
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def run(n_trials=50):
    from lightgbm import LGBMClassifier
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import train_test_split

    print("=" * 65)
    print("  EXTENSION 2 — Adjacent-window (short-horizon) training")
    print(f"  Train: features@{TRAIN_CUTOFF} -> labels to {TRAIN_HORIZON}")
    print(f"  Test:  features@{TEST_CUTOFF} -> labels to {TEST_HORIZON}")
    print("=" * 65)

    train_full = load(TRAIN_CUTOFF, TRAIN_HORIZON)
    test       = load(TEST_CUTOFF, TEST_HORIZON)
    cols = get_feature_cols(train_full)
    print(f"  train_full={len(train_full):,}  test={len(test):,}  "
          f"features={len(cols)}")

    X_full, y_full, _ = to_xy(train_full, cols, "label_skilled")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=SEED, stratify=y_full)
    X_te, y_te, pdf_te = to_xy(test, cols, "label_skilled")

    def obj(t):
        m = LGBMClassifier(
            n_estimators=t.suggest_int("n_estimators", 100, 500),
            max_depth=t.suggest_int("max_depth", 3, 7),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples=t.suggest_int("min_child_samples", 5, 80),
            random_state=SEED, verbose=-1,
        )
        m.fit(X_tr, y_tr)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    print(f"\n  Tuning ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  best params: {best}")

    m = LGBMClassifier(**best, random_state=SEED, verbose=-1)
    m.fit(X_full, y_full)
    p = m.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p)
    lo, hi = bootstrap_ci(y_te, p, roc_auc_score)

    bench = pd.to_numeric(pdf_te["past_clv_vw"], errors="coerce").fillna(0)
    auc_b = roc_auc_score(y_te, bench)
    lo_b, hi_b = bootstrap_ci(y_te, bench, roc_auc_score)

    print(f"\n  Adjacent-window LightGBM: AUC={auc:.4f} [{lo:.4f},{hi:.4f}]")
    print(f"  Benchmark (past CLV):     AUC={auc_b:.4f} [{lo_b:.4f},{hi_b:.4f}]")
    print(f"\n  Interpretation:")
    print(f"    Main pipeline (2023->2025, long horizon): AUC ~0.47-0.51")
    print(f"    This (adjacent, short horizon):           AUC {auc:.4f}")
    if auc > 0.55:
        print("    -> Signal recovers at short horizon: consistent with "
              "regime decay / population drift, not absence of skill.")
    else:
        print("    -> No recovery at short horizon either: the null "
              "(no persistence) is not a horizon artifact.")

    out = pd.DataFrame([
        {"config": "adjacent_short_horizon", "auc": auc, "ci_lo": lo,
         "ci_hi": hi, "n": len(y_te)},
        {"config": "benchmark_past_clv", "auc": auc_b, "ci_lo": lo_b,
         "ci_hi": hi_b, "n": len(y_te)},
    ])
    out.to_csv(TAB_DIR / "t11_adjacent_window.csv", index=False)
    with open(TAB_DIR / "t11_adjacent_window.tex", "w") as f:
        f.write(out.to_latex(index=False, float_format="%.4f",
                caption="Adjacent-window (short-horizon) training vs main "
                        "pipeline", label="tab:adjacent"))
    print(f"\n  ✓ -> {TAB_DIR / 't11_adjacent_window.csv'}")


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--trials", type=int, default=50)
    args = a.parse_args()
    run(n_trials=args.trials)