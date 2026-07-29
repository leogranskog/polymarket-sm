"""
Extension 1 — Continuous CLV target (regression instead of binary label).

Rationale: the binary top-quartile label discards information and its
threshold shifted ~24x across windows (0.0284 -> 0.0012 -> 0.0074), so it
means something different in each era. Regress on continuous forward CLV
directly and evaluate with rank correlation + decile spread, which are
threshold-free.

Uses the SAME leak-free panel as ml_pipeline_v2. TRUE OOS untouched.

Usage: python -m research.ml_pipeline_v3_continuous --trials 50
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
from config import PROC_DIR

PIT_DIR = PROC_DIR / "pit"
FIG_DIR = Path(__file__).parent / "figures_v2"
TAB_DIR = Path(__file__).parent / "tables_v2"
FIG_DIR.mkdir(exist_ok=True); TAB_DIR.mkdir(exist_ok=True)

SEED = 42
META_COLS  = {"wallet", "asof", "first_trade", "last_trade", "wash_flag"}
LABEL_COLS = {"fwd_n_trades", "fwd_clv_vw", "fwd_clv_hitrate", "label_skilled"}

PANEL = {
    "train": [("2023-06-30", "2023-12-31"),
              ("2023-09-30", "2024-03-31"),
              ("2023-12-31", "2024-06-30")],
    "val":   [("2024-06-30", "2024-12-31")],
    "test":  [("2024-12-31", "2025-06-30")],
}


def load_split(pairs):
    frames = []
    for cutoff, horizon in pairs:
        f = pl.read_parquet(PIT_DIR / f"features_asof_{cutoff}.parquet")
        l = pl.read_parquet(PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet")
        frames.append(f.join(l, on="wallet", how="inner"))
    return pl.concat(frames, how="diagonal")


def get_feature_cols(df):
    return sorted(set(c for c in df.columns
                  if c not in META_COLS | LABEL_COLS
                  and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64,
                                      pl.Int32, pl.UInt32, pl.Boolean)))


def to_xy_reg(df, cols):
    pdf = df.to_pandas()
    X = pdf[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = pdf["fwd_clv_vw"].astype(float)
    return X, y, pdf


def bootstrap_rho_ci(y, p, n=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); p = np.asarray(p)
    rhos = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        r, _ = spearmanr(y[idx], p[idx])
        rhos.append(r)
    return np.percentile(rhos, 2.5), np.percentile(rhos, 97.5)


def decile_spread(y_true, y_pred, tag):
    d = pd.DataFrame({"pred": y_pred, "actual": y_true})
    d["decile"] = pd.qcut(d["pred"].rank(method="first"), 10,
                          labels=False) + 1
    g = d.groupby("decile")["actual"].agg(["mean", "count"]).reset_index()
    spread = g["mean"].iloc[-1] - g["mean"].iloc[0]
    print(f"\n  Decile spread ({tag}):")
    for _, r in g.iterrows():
        print(f"    D{int(r['decile']):>2}: actual fwd CLV = {r['mean']:+.4f} "
              f"(n={int(r['count'])})")
    print(f"    D10-D1 = {spread:+.4f}")
    return g, spread


def run(n_trials=50):
    from lightgbm import LGBMRegressor
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 65)
    print("  EXTENSION 1 — Continuous CLV regression (threshold-free)")
    print("=" * 65)

    train = load_split(PANEL["train"])
    val   = load_split(PANEL["val"])
    test  = load_split(PANEL["test"])
    cols  = get_feature_cols(train)
    print(f"  train={len(train):,} val={len(val):,} test={len(test):,} "
          f"features={len(cols)}")

    X_tr, y_tr, _   = to_xy_reg(train, cols)
    X_val, y_val, _ = to_xy_reg(val, cols)
    X_te, y_te, pdf_te = to_xy_reg(test, cols)

    def obj(t):
        m = LGBMRegressor(
            n_estimators=t.suggest_int("n_estimators", 100, 600),
            max_depth=t.suggest_int("max_depth", 3, 8),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples=t.suggest_int("min_child_samples", 5, 80),
            reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10, log=True),
            reg_lambda=t.suggest_float("reg_lambda", 1e-8, 10, log=True),
            random_state=SEED, verbose=-1,
        )
        m.fit(X_tr, y_tr)
        p = m.predict(X_val)
        rho, _ = spearmanr(y_val, p)
        return rho

    print(f"\n  Tuning LightGBM regressor ({n_trials} trials, validation)...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  best params: {best}")

    # Disclosure: train-only
    m_trainonly = LGBMRegressor(**best, random_state=SEED, verbose=-1)
    m_trainonly.fit(X_tr, y_tr)
    p_trainonly = m_trainonly.predict(X_te)
    rho_t, pval_t = spearmanr(y_te, p_trainonly)
    lo_t, hi_t = bootstrap_rho_ci(y_te, p_trainonly)
    print(f"\n  Train-only fit:  Spearman rho={rho_t:.4f} "
          f"[{lo_t:.4f},{hi_t:.4f}] (p={pval_t:.2e})")

    # Primary: train+val
    X_fit = pd.concat([X_tr, X_val], ignore_index=True)
    y_fit = pd.concat([y_tr, y_val], ignore_index=True)
    m = LGBMRegressor(**best, random_state=SEED, verbose=-1)
    m.fit(X_fit, y_fit)
    p = m.predict(X_te)
    rho, pval = spearmanr(y_te, p)
    lo, hi = bootstrap_rho_ci(y_te, p)
    print(f"  Train+val fit:   Spearman rho={rho:.4f} [{lo:.4f},{hi:.4f}] "
          f"(p={pval:.2e})")

    # Benchmark: past CLV alone
    bench = pd.to_numeric(pdf_te["past_clv_vw"], errors="coerce").fillna(
        pdf_te["past_clv_vw"].median())
    rho_b, pval_b = spearmanr(y_te, bench)
    lo_b, hi_b = bootstrap_rho_ci(y_te, bench)
    print(f"  Benchmark (past CLV): rho={rho_b:.4f} [{lo_b:.4f},{hi_b:.4f}] "
          f"(p={pval_b:.2e})")

    results = pd.DataFrame([
        {"model": "Benchmark: past CLV", "spearman_rho": rho_b,
         "ci_lo": lo_b, "ci_hi": hi_b, "p": pval_b, "n": len(y_te)},
        {"model": "LightGBM (train-only)", "spearman_rho": rho_t,
         "ci_lo": lo_t, "ci_hi": hi_t, "p": pval_t, "n": len(y_te)},
        {"model": "LightGBM (train+val)", "spearman_rho": rho, "ci_lo": lo,
         "ci_hi": hi, "p": pval, "n": len(y_te)},
    ])
    results.to_csv(TAB_DIR / "t10_continuous_target.csv", index=False)
    with open(TAB_DIR / "t10_continuous_target.tex", "w") as f:
        f.write(results.to_latex(index=False, float_format="%.4f",
                caption="Continuous CLV target: rank correlation "
                        "(threshold-free)", label="tab:continuous"))

    g, spread = decile_spread(y_te.values, p, "train+val fit, test 2025-H1")
    g.to_csv(TAB_DIR / "t10b_continuous_deciles.csv", index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(g)))
    ax.bar(g["decile"], g["mean"], color=colors, edgecolor="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Predicted-CLV decile (continuous target)")
    ax.set_ylabel("Actual forward CLV")
    ax.set_title("Continuous-target model: predicted vs actual (test 2025-H1)")
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(FIG_DIR / f"f7_continuous_deciles{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()

    print(f"\n  ✓ tables -> {TAB_DIR}")
    print(f"  ✓ figure -> {FIG_DIR / 'f7_continuous_deciles.pdf'}")


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--trials", type=int, default=50)
    args = a.parse_args()
    run(n_trials=args.trials)