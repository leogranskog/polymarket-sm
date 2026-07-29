"""
ML Pipeline v2 — LEAK-FREE, publication-grade.

Design (pre-registered):
  Primary model:      LightGBM (tuned on validation only, hyperparams frozen)
  Final fit:          train + validation, frozen hyperparams
                      (expanding-window refit, cf. Gu, Kelly & Xiu 2020)
  Disclosure:         train-only fit reported alongside
  Primary label:      top-quartile forward volume-weighted CLV
  Primary benchmark:  ranking by past_clv_vw alone (track record)
  Key ablation:       behavior-only features (no past-CLV block)
  Robustness:         exclude wash_flag wallets; cohort-matched (new wallets)
  Placebo:            shuffled labels -> AUC must collapse to ~0.5
  TRUE OOS:           2025-H2, evaluated ONCE with --true-oos

Usage:
    python run.py research2 --trials 50
    python run.py research2 --trials 100 --true-oos   # final run only
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
import pickle
from pathlib import Path
from config import PROC_DIR

PIT_DIR      = PROC_DIR / "pit"
RESEARCH_DIR = Path(__file__).parent
FIG_DIR      = RESEARCH_DIR / "figures_v2"
TAB_DIR      = RESEARCH_DIR / "tables_v2"
MODEL_DIR    = RESEARCH_DIR / "models_v2"
for d in (FIG_DIR, TAB_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

META_COLS  = {"wallet", "asof", "first_trade", "last_trade", "wash_flag"}
LABEL_COLS = {"fwd_n_trades", "fwd_clv_vw", "fwd_clv_hitrate", "label_skilled"}
PAST_CLV_BLOCK = {"past_clv_vw", "past_clv_mean", "past_clv_hitrate",
                  "past_clv_std", "clv_when_late", "clv_when_early",
                  "insider_gap"}

PANEL = {
    "train": [("2023-06-30", "2023-12-31"),
              ("2023-09-30", "2024-03-31"),
              ("2023-12-31", "2024-06-30")],
    "val":   [("2024-06-30", "2024-12-31")],
    "test":  [("2024-12-31", "2025-06-30")],
    "oos":   [("2025-06-30", "2025-12-31")],
}


# ── Data assembly ─────────────────────────────────────────────────────────────

def load_split(pairs) -> pl.DataFrame:
    frames = []
    for cutoff, horizon in pairs:
        f = pl.read_parquet(PIT_DIR / f"features_asof_{cutoff}.parquet")
        l = pl.read_parquet(PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet")
        frames.append(f.join(l, on="wallet", how="inner"))
    return pl.concat(frames, how="diagonal")


def to_xy(df: pl.DataFrame, feature_cols: list, exclude_wash=False):
    if exclude_wash and "wash_flag" in df.columns:
        df = df.filter(pl.col("wash_flag") != True)
    pdf = df.to_pandas()
    X = pdf[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = pdf["label_skilled"].astype(int)
    return X, y, pdf


def get_feature_cols(df: pl.DataFrame, drop_past_clv=False) -> list:
    cols = [c for c in df.columns
            if c not in META_COLS | LABEL_COLS
            and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64,
                                pl.Int32, pl.UInt32, pl.Boolean)]
    if drop_past_clv:
        cols = [c for c in cols if c not in PAST_CLV_BLOCK]
    return sorted(set(cols))


# ── Stats helpers ─────────────────────────────────────────────────────────────

def bootstrap_auc_ci(y, p, n=1000, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    aucs = []
    y = np.asarray(y); p = np.asarray(p)
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if y[idx].sum() in (0, len(idx)):
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def delong_p(y, p1, p2):
    """Approximate DeLong test p-value for AUC(p1) vs AUC(p2)."""
    from scipy.stats import norm
    y = np.asarray(y)
    pos1, neg1 = np.asarray(p1)[y == 1], np.asarray(p1)[y == 0]
    pos2, neg2 = np.asarray(p2)[y == 1], np.asarray(p2)[y == 0]

    def auc_var(pos, neg):
        v10 = np.array([np.mean(pp > neg) + 0.5 * np.mean(pp == neg)
                        for pp in pos])
        v01 = np.array([np.mean(pos > nn) + 0.5 * np.mean(pos == nn)
                        for nn in neg])
        return v10.mean(), v10.var() / len(pos) + v01.var() / len(neg)

    a1, v1 = auc_var(pos1, neg1)
    a2, v2 = auc_var(pos2, neg2)
    z = (a1 - a2) / np.sqrt(v1 + v2 + 1e-12)
    return float(2 * (1 - norm.cdf(abs(z))))


# ── Models ────────────────────────────────────────────────────────────────────

def tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials):
    import optuna
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(t):
        m = LGBMClassifier(
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
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def fit_models(X_tr, y_tr, best_lgbm_params):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    models = {
        "Random":        DummyClassifier(strategy="stratified",
                                         random_state=SEED),
        "Logistic (L2)": LogisticRegression(max_iter=2000, C=1.0,
                                            class_weight="balanced",
                                            random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=8,
                                                min_samples_leaf=10,
                                                class_weight="balanced",
                                                random_state=SEED, n_jobs=-1),
        "XGBoost":       XGBClassifier(n_estimators=300, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       colsample_bytree=0.8,
                                       scale_pos_weight=spw,
                                       random_state=SEED, verbosity=0,
                                       eval_metric="auc"),
        "LightGBM*":     LGBMClassifier(**best_lgbm_params,
                                        random_state=SEED, verbose=-1),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)
    return models


def proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


# ── Evaluation blocks ─────────────────────────────────────────────────────────

def evaluate_all(models, X_te, y_te, past_clv_te, tag):
    from sklearn.metrics import roc_auc_score, average_precision_score
    rows, probas = [], {}

    bench = np.asarray(past_clv_te, dtype=float)
    bench = np.nan_to_num(bench, nan=np.nanmedian(bench))
    probas["Benchmark: past CLV"] = bench
    lo, hi = bootstrap_auc_ci(y_te, bench)
    rows.append({"model": "Benchmark: past CLV",
                 "auc": roc_auc_score(y_te, bench),
                 "pr_auc": average_precision_score(y_te, bench),
                 "ci_lo": lo, "ci_hi": hi})

    for name, m in models.items():
        p = proba(m, X_te)
        probas[name] = p
        lo, hi = bootstrap_auc_ci(y_te, p)
        rows.append({"model": name,
                     "auc": roc_auc_score(y_te, p),
                     "pr_auc": average_precision_score(y_te, p),
                     "ci_lo": lo, "ci_hi": hi})

    df = pd.DataFrame(rows)
    df["p_vs_benchmark"] = [
        delong_p(y_te, probas[r["model"]], probas["Benchmark: past CLV"])
        if r["model"] != "Benchmark: past CLV" else np.nan
        for _, r in df.iterrows()
    ]
    print(f"\n  ── {tag} ──")
    for _, r in df.iterrows():
        pv = "" if np.isnan(r["p_vs_benchmark"]) else \
             f"  p_vs_bench={r['p_vs_benchmark']:.4f}"
        print(f"  {r['model']:<26} AUC={r['auc']:.4f} "
              f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]{pv}")
    return df, probas


def placebo_test(X_tr, y_tr, X_te, y_te, n=5):
    from sklearn.metrics import roc_auc_score
    from lightgbm import LGBMClassifier
    rng = np.random.RandomState(SEED)
    aucs = []
    for i in range(n):
        y_shuf = pd.Series(rng.permutation(y_tr.values))
        m = LGBMClassifier(n_estimators=200, random_state=SEED + i, verbose=-1)
        m.fit(X_tr, y_shuf)
        aucs.append(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
    print(f"\n  Placebo (shuffled labels, n={n}): "
          f"mean AUC = {np.mean(aucs):.4f} (should be ~0.50)")
    return float(np.mean(aucs))


def decile_forward_clv(pdf_test, p, tag):
    d = pd.DataFrame({"score": p, "fwd_clv": pdf_test["fwd_clv_vw"].values})
    d["decile"] = pd.qcut(d["score"].rank(method="first"), 10,
                          labels=False) + 1
    g = d.groupby("decile")["fwd_clv"].agg(["mean", "count"]).reset_index()
    spread = g["mean"].iloc[-1] - g["mean"].iloc[0]
    rng = np.random.RandomState(SEED)
    sp = []
    for _ in range(1000):
        idx = rng.choice(len(d), len(d), replace=True)
        gg = d.iloc[idx].groupby("decile")["fwd_clv"].mean()
        if 1 in gg.index and 10 in gg.index:
            sp.append(gg.loc[10] - gg.loc[1])
    lo, hi = np.percentile(sp, [2.5, 97.5])
    print(f"\n  Decile analysis ({tag}):")
    for _, r in g.iterrows():
        print(f"    D{int(r['decile']):>2}: fwd CLV = {r['mean']:+.4f}  "
              f"(n={int(r['count'])})")
    print(f"    D10-D1 spread = {spread:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"(~{spread*10000:+.0f} bps per unit stake)")
    return g, spread, (lo, hi)


# ── Figures / tables ──────────────────────────────────────────────────────────

def fig_model_comparison(df, fname):
    import matplotlib.pyplot as plt
    d = df.sort_values("auc")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(d["model"], d["auc"],
            xerr=[d["auc"] - d["ci_lo"], d["ci_hi"] - d["auc"]],
            color="steelblue", capsize=3)
    ax.axvline(0.5, color="red", ls="--", alpha=0.7, label="Random")
    ax.set_xlabel("ROC-AUC (95% bootstrap CI)")
    ax.set_title("Predicting forward CLV skill — leak-free test panel")
    ax.set_xlim(0.45, max(0.75, float(d["auc"].max()) + 0.05))
    ax.legend()
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(FIG_DIR / f"{fname}{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  fig -> {FIG_DIR / fname}.pdf")


def fig_decile(g, fname, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(g)))
    ax.bar(g["decile"], g["mean"], color=colors, edgecolor="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Model-score decile (1 = lowest)")
    ax.set_ylabel("Forward volume-weighted CLV")
    ax.set_title(title)
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(FIG_DIR / f"{fname}{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  fig -> {FIG_DIR / fname}.pdf")


def fig_shap(model, X, fname):
    import shap, matplotlib.pyplot as plt
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, X, max_display=20, show=False)
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(FIG_DIR / f"{fname}{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  fig -> {FIG_DIR / fname}.pdf")


def save_table(df: pd.DataFrame, name: str, caption: str):
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)
    with open(TAB_DIR / f"{name}.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f",
                            caption=caption, label=f"tab:{name}"))
    print(f"  table -> {TAB_DIR / name}.csv/.tex")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(n_trials=50, true_oos=False):
    from sklearn.metrics import roc_auc_score
    from lightgbm import LGBMClassifier

    print("=" * 65)
    print("  ML PIPELINE v2 — leak-free, CLV labels, pre-registered LightGBM")
    print("=" * 65)

    train = load_split(PANEL["train"])
    val   = load_split(PANEL["val"])
    test  = load_split(PANEL["test"])
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    feats_full = get_feature_cols(train)
    feats_beh  = get_feature_cols(train, drop_past_clv=True)
    print(f"  features: full={len(feats_full)}  "
          f"behavior-only={len(feats_beh)}")

    X_tr, y_tr, _      = to_xy(train, feats_full)
    X_val, y_val, _    = to_xy(val,   feats_full)
    X_te, y_te, pdf_te = to_xy(test,  feats_full)

    # 1. Tune primary model on validation ONLY (hyperparams frozen here)
    print(f"\n  Tuning LightGBM ({n_trials} trials, validation panel)...")
    best = tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials)
    print(f"  best params: {best}")

    # 2a. Disclosure fit: train only
    m_trainonly = LGBMClassifier(**best, random_state=SEED, verbose=-1)
    m_trainonly.fit(X_tr, y_tr)
    p_trainonly = m_trainonly.predict_proba(X_te)[:, 1]
    auc_trainonly = roc_auc_score(y_te, p_trainonly)
    lo_t, hi_t = bootstrap_auc_ci(y_te, p_trainonly)
    print(f"  LightGBM (train-only fit): AUC={auc_trainonly:.4f} "
          f"[{lo_t:.4f},{hi_t:.4f}]")

    # 2b. Primary fit: train + validation, frozen hyperparams
    X_fit = pd.concat([X_tr, X_val], ignore_index=True)
    y_fit = pd.concat([y_tr, y_val], ignore_index=True)
    print(f"  Final fit on train+val: {len(X_fit):,} rows")
    models = fit_models(X_fit, y_fit, best)

    # 3. Main evaluation on test panel
    main_df, probas = evaluate_all(models, X_te, y_te,
                                   pdf_te["past_clv_vw"], "TEST 2025-H1")
    main_df = pd.concat([main_df, pd.DataFrame([{
        "model": "LightGBM* (train-only fit)",
        "auc": auc_trainonly, "pr_auc": np.nan,
        "ci_lo": lo_t, "ci_hi": hi_t,
        "p_vs_benchmark": delong_p(y_te, p_trainonly,
                                   probas["Benchmark: past CLV"]),
    }])], ignore_index=True)
    save_table(main_df, "t1_main_test",
               "Model comparison on leak-free test panel (2025-H1 fwd CLV)")
    fig_model_comparison(main_df, "f1_model_comparison")

    # 4. Ablation: behavior-only features (no past-CLV block)
    print("\n  Ablation: behavior-only features (no track record)...")
    Xb_tr, yb_tr, _ = to_xy(train, feats_beh)
    Xb_val, yb_val, _ = to_xy(val, feats_beh)
    Xb_te, yb_te, _ = to_xy(test,  feats_beh)
    Xb_fit = pd.concat([Xb_tr, Xb_val], ignore_index=True)
    yb_fit = pd.concat([yb_tr, yb_val], ignore_index=True)
    m_beh = LGBMClassifier(**best, random_state=SEED, verbose=-1)
    m_beh.fit(Xb_fit, yb_fit)
    p_beh = m_beh.predict_proba(Xb_te)[:, 1]
    auc_beh = roc_auc_score(yb_te, p_beh)
    lo, hi = bootstrap_auc_ci(yb_te, p_beh)
    p_abl = delong_p(y_te, p_beh, probas["Benchmark: past CLV"])
    print(f"  Behavior-only AUC = {auc_beh:.4f} [{lo:.4f},{hi:.4f}]  "
          f"vs track-record benchmark: p={p_abl:.4f}")
    save_table(pd.DataFrame([{
        "config": "behavior-only", "auc": auc_beh,
        "ci_lo": lo, "ci_hi": hi, "p_vs_track_record": p_abl}]),
        "t2_ablation", "Behavior-only vs track-record benchmark")

    # 5. Cohort-matched: brand-new wallets (first trade after 2024-06-30)
    print("\n  Cohort-matched evaluation (new wallets only)...")
    new_mask = pdf_te["first_trade"] > pd.Timestamp("2024-06-30", tz="UTC")
    if new_mask.sum() > 200:
        p_new = probas["LightGBM*"][new_mask.values]
        y_new = y_te[new_mask.values]
        auc_new = roc_auc_score(y_new, p_new)
        lo, hi = bootstrap_auc_ci(y_new, p_new)
        print(f"  New-wallet cohort: n={int(new_mask.sum()):,}, "
              f"AUC={auc_new:.4f} [{lo:.4f},{hi:.4f}]")
        save_table(pd.DataFrame([{
            "cohort": "new wallets (first trade > 2024-06-30)",
            "n": int(new_mask.sum()), "auc": auc_new,
            "ci_lo": lo, "ci_hi": hi}]),
            "t3_cohort_matched",
            "Cohort-matched evaluation on unseen wallets")
    else:
        print(f"  Only {int(new_mask.sum())} new wallets — appendix note")

    # 6. Wash-trade robustness
    print("\n  Robustness: excluding wash-flagged wallets...")
    Xw, yw, pdfw = to_xy(test, feats_full, exclude_wash=True)
    p_w = proba(models["LightGBM*"], Xw)
    auc_w = roc_auc_score(yw, p_w)
    print(f"  ex-wash AUC = {auc_w:.4f}  (n={len(yw):,})")

    # 7. Placebo
    placebo_test(X_tr, y_tr, X_te, y_te)

    # 8. Economic significance
    g, spread, ci = decile_forward_clv(pdf_te, probas["LightGBM*"],
                                       "test 2025-H1")
    save_table(g, "t4_deciles", "Forward CLV by model-score decile")
    fig_decile(g, "f2_deciles",
               "Forward CLV by predicted-skill decile (test, 2025-H1)")

    # 9. SHAP
    print("\n  SHAP...")
    fig_shap(models["LightGBM*"],
             X_te.sample(min(3000, len(X_te)), random_state=SEED),
             "f3_shap")

    # Save primary model
    with open(MODEL_DIR / "lgbm_primary.pkl", "wb") as f:
        pickle.dump({"model": models["LightGBM*"], "features": feats_full,
                     "params": best}, f)

    # 10. TRUE OOS (only with explicit flag)
    if true_oos:
        print("\n" + "=" * 65)
        print("  TRUE OOS — 2025-H2. This is the one and only look.")
        print("=" * 65)
        oos = load_split(PANEL["oos"])
        X_o, y_o, pdf_o = to_xy(oos, feats_full)
        p_o = proba(models["LightGBM*"], X_o)
        auc_o = roc_auc_score(y_o, p_o)
        lo, hi = bootstrap_auc_ci(y_o, p_o)
        print(f"  TRUE OOS AUC = {auc_o:.4f} [{lo:.4f},{hi:.4f}]  "
              f"(n={len(y_o):,})")
        g_o, sp_o, ci_o = decile_forward_clv(pdf_o, p_o, "TRUE OOS 2025-H2")
        save_table(pd.DataFrame([{
            "panel": "TRUE OOS 2025-H2", "n": len(y_o),
            "auc": auc_o, "ci_lo": lo, "ci_hi": hi,
            "d10_d1_spread": sp_o}]),
            "t5_true_oos", "True out-of-sample evaluation (single look)")
        fig_decile(g_o, "f4_deciles_oos",
                   "Forward CLV by decile — TRUE OOS 2025-H2")

    print("\n✅ Pipeline v2 complete.")
    print(f"   figures: {FIG_DIR}\n   tables:  {TAB_DIR}")


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--trials", type=int, default=50)
    a.add_argument("--true-oos", action="store_true")
    args = a.parse_args()
    run(n_trials=args.trials, true_oos=args.true_oos)