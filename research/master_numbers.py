"""
MASTER NUMBERS SCRIPT — the single source of truth for every number
cited in the paper. Run once, produces one timestamped, hashed report.
No other script should be used to generate paper-facing numbers after
this point; if a number changes, it changes because this script was
rerun (documented, timestamped), not because a different script
produced a different one-off result.

Design principles enforced throughout:
  1. ONE frozen model, defined once (Section 1), saved once, and
     reused (never retrained under a different name) everywhere the
     paper refers to "the frozen model" (Section 4).
  2. ONE tuning procedure (Optuna, same trial budget) applied
     identically to every model in the comparison, so no model gets
     a hyperparameter advantage over another by accident.
  3. ONE panel (the corrected, tiered-CLV panel) used throughout;
     no mixing of pre- and post-correction data anywhere in this file.
  4. Every section's output is written to ONE report file with a
     timestamp and a hash of this script, so any future question of
     "which run produced this number" has a single, checkable answer.
  5. FIXED (this version): each confirmatory window's refit uses ALL
     genuinely available pre-cutoff training data, including every
     prior confirmatory/test window that has become "past" data by
     the time that window is being predicted. The previous version
     omitted this for Confirmatory Window 1, causing its "refit"
     model to be trained on identical data to the frozen model with
     only a noisier tuning procedure, an invalid comparison. Fixed
     below.

Usage: python -m research.master_numbers
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import warnings
warnings.filterwarnings("ignore")

import hashlib
import numpy as np
import pandas as pd
import polars as pl
import pickle
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, norm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from config import PROC_DIR

optuna.logging.set_verbosity(optuna.logging.WARNING)

PIT_DIR     = PROC_DIR / "pit"
TAB_DIR     = Path(__file__).parent / "tables_v2"
MODEL_DIR   = Path(__file__).parent / "models_v2"
REPORT_PATH = Path(__file__).parent / "MASTER_NUMBERS.md"
TAB_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
N_TRIALS = 40   # same tuning budget for every model, documented once here
META_COLS  = {"wallet", "asof", "first_trade", "last_trade", "wash_flag"}
LABEL_COLS = {"fwd_n_trades", "fwd_clv_vw", "fwd_clv_hitrate", "label_skilled"}

# Panel definition, used identically everywhere in this script
TRAIN_PAIRS = [
    ("2023-06-30", "2023-12-31"),
    ("2023-09-30", "2024-03-31"),
    ("2023-12-31", "2024-06-30"),
]
VAL_CUTOFF, VAL_HORIZON   = "2024-06-30", "2024-12-31"
TEST_CUTOFF, TEST_HORIZON = "2024-12-31", "2025-06-30"

# FIXED: each confirmatory window's refit now includes EVERY prior
# window that is genuinely available pre-cutoff data by the time that
# window is being predicted, matching the expanding-window design used
# in true_oos_final.py / true_oos_second_window.py exactly.
CONFIRMATORY = [
    {"name": "Confirmatory 1 (2025-H2)",
     "cutoff": "2025-06-30", "horizon": "2025-12-31",
     "extra_train_for_refit": [(TEST_CUTOFF, TEST_HORIZON)]},
    {"name": "Confirmatory 2 (2026-Q1)",
     "cutoff": "2025-12-31", "horizon": "2026-03-29",
     "extra_train_for_refit": [(TEST_CUTOFF, TEST_HORIZON),
                               ("2025-06-30", "2025-12-31")]},
]


def log(msg, f):
    print(msg)
    f.write(msg + "\n")


def script_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_panel(cutoff, horizon) -> pl.DataFrame:
    feat = pl.read_parquet(PIT_DIR / f"features_asof_{cutoff}.parquet")
    lab  = pl.read_parquet(PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet")
    return feat.join(lab, on="wallet", how="inner")


def get_feature_cols(df: pl.DataFrame) -> list:
    return sorted(set(c for c in df.columns
                  if c not in META_COLS | LABEL_COLS
                  and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64,
                                      pl.Int32, pl.UInt32, pl.Boolean)))


def to_xy(df: pl.DataFrame, cols: list):
    pdf = df.to_pandas()
    X = pdf[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = pdf["label_skilled"].astype(int)
    return X, y, pdf


def bootstrap_ci(y, p, n=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); p = np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(roc_auc_score(y[idx], p[idx]))
        except Exception:
            pass
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def delong_p(y, p1, p2):
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


# ── Shared Optuna tuning, identical procedure for every model ──────────────

def tune_model(model_name: str, X_tr, y_tr, X_val, y_val, n_trials=N_TRIALS):
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    def objective(trial):
        if model_name == "Logistic regression":
            C = trial.suggest_float("C", 1e-4, 100.0, log=True)
            m = LogisticRegression(C=C, max_iter=2000,
                                   class_weight="balanced",
                                   random_state=SEED)
        elif model_name == "Random forest":
            m = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                class_weight="balanced", random_state=SEED, n_jobs=-1,
            )
        elif model_name == "XGBoost":
            m = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3,
                                                  log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree",
                                                     0.5, 1.0),
                scale_pos_weight=spw, random_state=SEED, verbosity=0,
                eval_metric="auc",
            )
        elif model_name == "LightGBM":
            m = LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 600),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3,
                                                  log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree",
                                                     0.5, 1.0),
                min_child_samples=trial.suggest_int("min_child_samples",
                                                    5, 80),
                random_state=SEED, verbose=-1,
            )
        else:
            raise ValueError(model_name)

        m.fit(X_tr, y_tr)
        p = (m.predict_proba(X_val)[:, 1] if hasattr(m, "predict_proba")
             else m.decision_function(X_val))
        return roc_auc_score(y_val, p)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def build_model(model_name: str, params: dict, spw: float):
    if model_name == "Logistic regression":
        return LogisticRegression(**params, max_iter=2000,
                                  class_weight="balanced", random_state=SEED)
    if model_name == "Random forest":
        return RandomForestClassifier(**params, class_weight="balanced",
                                      random_state=SEED, n_jobs=-1)
    if model_name == "XGBoost":
        return XGBClassifier(**params, scale_pos_weight=spw,
                             random_state=SEED, verbosity=0,
                             eval_metric="auc")
    if model_name == "LightGBM":
        return LGBMClassifier(**params, random_state=SEED, verbose=-1)
    raise ValueError(model_name)


# ── SECTION 1: model comparison + THE frozen model (defined once) ─────────

def section1_model_comparison(f):
    log("\n" + "=" * 70, f)
    log("  SECTION 1: MODEL COMPARISON (Optuna-tuned, identical procedure "
        "for every model)", f)
    log("=" * 70, f)

    train_frames = [load_panel(c, h) for c, h in TRAIN_PAIRS]
    train_all = pl.concat(train_frames, how="diagonal")
    val_df  = load_panel(VAL_CUTOFF, VAL_HORIZON)
    test_df = load_panel(TEST_CUTOFF, TEST_HORIZON)

    cols = get_feature_cols(train_all)
    X_train, y_train, _        = to_xy(train_all, cols)
    X_val,   y_val,   _        = to_xy(val_df,   cols)
    X_test,  y_test,  test_pdf = to_xy(test_df,  cols)

    log(f"\n  train n={len(X_train):,}  val n={len(X_val):,}  "
        f"test n={len(X_test):,}  features={len(cols)}  "
        f"Optuna trials per model={N_TRIALS}", f)

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    bench = pd.to_numeric(test_pdf["past_clv_vw"], errors="coerce").fillna(
        test_pdf["past_clv_vw"].median())
    auc_bench = roc_auc_score(y_test, bench)
    lo_b, hi_b = bootstrap_ci(y_test.values, bench.values)
    log(f"\n  Past-CLV benchmark: AUC={auc_bench:.4f} "
        f"[{lo_b:.4f},{hi_b:.4f}]", f)

    rows = [{"model": "Past-CLV benchmark", "auc": auc_bench,
             "ci_lo": lo_b, "ci_hi": hi_b, "p_vs_bench": np.nan,
             "best_params": None}]

    frozen_model_ref = None
    frozen_cols_ref = cols

    for name in ["Logistic regression", "Random forest", "XGBoost", "LightGBM"]:
        log(f"\n  Tuning {name} ({N_TRIALS} trials on validation)...", f)
        best_params = tune_model(name, X_train, y_train, X_val, y_val)

        # Final fit: train + validation, frozen tuned hyperparameters
        X_fit = pd.concat([X_train, X_val], ignore_index=True)
        y_fit = pd.concat([y_train, y_val], ignore_index=True)
        model = build_model(name, best_params, spw)
        model.fit(X_fit, y_fit)

        p = (model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba")
             else model.decision_function(X_test))
        auc = roc_auc_score(y_test, p)
        lo, hi = bootstrap_ci(y_test.values, p)
        p_delong = delong_p(y_test.values, p, bench.values)

        log(f"    {name}: AUC={auc:.4f} [{lo:.4f},{hi:.4f}]  "
            f"DeLong p vs benchmark={p_delong:.4f}", f)
        log(f"    Best params: {best_params}", f)

        rows.append({"model": name, "auc": auc, "ci_lo": lo, "ci_hi": hi,
                     "p_vs_bench": p_delong, "best_params": str(best_params)})

        if name == "LightGBM":
            frozen_model_ref = model

    # Save THE frozen model, once, this exact object is what "frozen"
    # means everywhere else in this script and in the paper.
    with open(MODEL_DIR / "MASTER_frozen_model.pkl", "wb") as fh:
        pickle.dump({"model": frozen_model_ref, "features": frozen_cols_ref,
                    "trained_at": datetime.now().isoformat(),
                    "train_pairs": TRAIN_PAIRS,
                    "val_window": (VAL_CUTOFF, VAL_HORIZON)}, fh)
    log(f"\n  ✓ THE frozen model saved -> "
        f"{MODEL_DIR / 'MASTER_frozen_model.pkl'}", f)
    log(f"  (this exact object is reused, unmodified, in Section 4)", f)

    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "MASTER_table_model_comparison.csv", index=False)
    log(f"  Saved -> {TAB_DIR / 'MASTER_table_model_comparison.csv'}", f)
    return rows


# ── SECTION 2: persistence (full population + fixed cohort + survivors) ────

def section2_persistence(f):
    log("\n" + "=" * 70, f)
    log("  SECTION 2: PERSISTENCE (full population, fixed cohort, "
        "survivor cohort)", f)
    log("=" * 70, f)

    windows = [
        ("H2-2023->H1-2024",
         "labels_2023-06-30_to_2023-12-31.parquet",
         "labels_2023-12-31_to_2024-06-30.parquet"),
        ("H1-2024->H2-2024",
         "labels_2023-12-31_to_2024-06-30.parquet",
         "labels_2024-06-30_to_2024-12-31.parquet"),
        ("H2-2024->H1-2025",
         "labels_2024-06-30_to_2024-12-31.parquet",
         "labels_2024-12-31_to_2025-06-30.parquet"),
        ("Confirmatory 1 (2025-H2)",
         "labels_2024-12-31_to_2025-06-30.parquet",
         "labels_2025-06-30_to_2025-12-31.parquet"),
        ("Confirmatory 2 (2026-Q1)",
         "labels_2025-06-30_to_2025-12-31.parquet",
         "labels_2025-12-31_to_2026-03-29.parquet"),
    ]

    earliest = pl.read_parquet(PIT_DIR / windows[0][1]).select("wallet")
    cohort = set(earliest["wallet"].to_list())
    log(f"\n  Fixed cohort (earliest-period wallets): {len(cohort):,}", f)

    all_label_files = sorted(set(
        [w[1] for w in windows] + [w[2] for w in windows]))
    all_sets = [set(pl.read_parquet(PIT_DIR / lf)["wallet"].to_list())
                for lf in all_label_files]
    survivors = set.intersection(*all_sets)
    log(f"  Survivor cohort (present in ALL periods): {len(survivors):,}", f)

    rows = []
    for name, f1, f2 in windows:
        a = pl.read_parquet(PIT_DIR / f1).select(["wallet", "fwd_clv_vw"])
        b = pl.read_parquet(PIT_DIR / f2).select(["wallet", "fwd_clv_vw"])
        j_full = a.join(b, on="wallet", suffix="_next").to_pandas()
        if len(j_full) < 30:
            continue
        rho_full, _ = spearmanr(j_full["fwd_clv_vw"], j_full["fwd_clv_vw_next"])

        a_cohort = a.filter(pl.col("wallet").is_in(list(cohort)))
        j_cohort = a_cohort.join(b, on="wallet", suffix="_next").to_pandas()
        rho_cohort = (spearmanr(j_cohort["fwd_clv_vw"],
                                j_cohort["fwd_clv_vw_next"])[0]
                     if len(j_cohort) >= 30 else np.nan)

        a_surv = a.filter(pl.col("wallet").is_in(list(survivors)))
        j_surv = a_surv.join(b, on="wallet", suffix="_next").to_pandas()
        rho_surv = (spearmanr(j_surv["fwd_clv_vw"],
                              j_surv["fwd_clv_vw_next"])[0]
                   if len(j_surv) >= 20 else np.nan)

        log(f"\n  {name}:", f)
        log(f"    Full:      n={len(j_full):,}   rho={rho_full:+.4f}", f)
        log(f"    Cohort:    n={len(j_cohort):,}   "
            f"rho={rho_cohort:+.4f}" if not np.isnan(rho_cohort)
            else f"    Cohort:    n={len(j_cohort):,}   n/a", f)
        log(f"    Survivors: n={len(j_surv):,}   "
            f"rho={rho_surv:+.4f}" if not np.isnan(rho_surv)
            else f"    Survivors: n={len(j_surv):,}   n/a", f)

        rows.append({"window": name, "n_full": len(j_full),
                     "rho_full": rho_full, "n_cohort": len(j_cohort),
                     "rho_cohort": rho_cohort, "n_survivors": len(j_surv),
                     "rho_survivors": rho_surv})

    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "MASTER_table_persistence.csv", index=False)
    log(f"\n  Saved -> {TAB_DIR / 'MASTER_table_persistence.csv'}", f)
    return rows


# ── SECTION 3: decile portfolio sort ────────────────────────────────────────

def section3_decile_sort(f):
    log("\n" + "=" * 70, f)
    log("  SECTION 3: DECILE PORTFOLIO SORT", f)
    log("=" * 70, f)

    df = load_panel(TEST_CUTOFF, TEST_HORIZON).to_pandas()
    df = df.dropna(subset=["past_clv_vw", "fwd_clv_vw"])
    df["decile"] = pd.qcut(df["past_clv_vw"].rank(method="first"), 10,
                           labels=False) + 1
    g = df.groupby("decile")["fwd_clv_vw"].agg(["mean", "count"]).reset_index()

    log(f"\n  Panel: features@{TEST_CUTOFF}, forward through {TEST_HORIZON}", f)
    for _, r in g.iterrows():
        log(f"    D{int(r['decile']):>2}: mean fwd CLV = {r['mean']:+.4f}  "
            f"(n={int(r['count'])})", f)

    spread = g.loc[g["decile"] == 10, "mean"].values[0] - \
             g.loc[g["decile"] == 1, "mean"].values[0]
    log(f"\n  D10-D1 spread: {spread:+.4f}", f)

    g.to_csv(TAB_DIR / "MASTER_table_deciles.csv", index=False)
    log(f"  Saved -> {TAB_DIR / 'MASTER_table_deciles.csv'}", f)
    return spread


# ── SECTION 4: staleness — reuses THE frozen model from Section 1 ─────────

def section4_staleness(f):
    log("\n" + "=" * 70, f)
    log("  SECTION 4: STALENESS (frozen model from Section 1, refit with "
        "identical tuning procedure and ALL genuinely available "
        "pre-cutoff data, placebo)", f)
    log("=" * 70, f)

    frozen_path = MODEL_DIR / "MASTER_frozen_model.pkl"
    with open(frozen_path, "rb") as fh:
        saved = pickle.load(fh)
    frozen_model = saved["model"]
    frozen_cols  = saved["features"]
    log(f"\n  Loaded frozen model (trained {saved['trained_at']})", f)

    rows = []
    for config in CONFIRMATORY:
        log(f"\n  {config['name']}", f)
        test_df = load_panel(config["cutoff"], config["horizon"])

        # Frozen: evaluate THE SAME saved model object, no retraining
        X_test_frozen, y_test, _ = to_xy(
            test_df, [c for c in frozen_cols if c in test_df.columns])
        p_frozen = frozen_model.predict_proba(X_test_frozen)[:, 1]
        auc_frozen = roc_auc_score(y_test, p_frozen)
        lo_f, hi_f = bootstrap_ci(y_test.values, p_frozen)

        # Refit: same tuning procedure as Section 1, applied to ALL
        # pre-cutoff data genuinely available for this specific window
        refit_train_pairs = TRAIN_PAIRS + [(VAL_CUTOFF, VAL_HORIZON)] + \
                            config["extra_train_for_refit"]
        log(f"    Refit training windows: {refit_train_pairs}", f)
        refit_frames = [load_panel(c, h) for c, h in refit_train_pairs]
        refit_train_all = pl.concat(refit_frames, how="diagonal")
        refit_cols = get_feature_cols(refit_train_all)

        X_refit_train, y_refit_train, _ = to_xy(refit_train_all, refit_cols)
        X_refit_test, y_refit_test, _   = to_xy(test_df, refit_cols)

        # Internal validation split for tuning: use the ORIGINAL
        # validation window's wallets specifically (not a positional
        # slice), matching Section 1's tuning methodology exactly
        val_wallets = set(
            load_panel(VAL_CUTOFF, VAL_HORIZON)["wallet"].to_list())
        refit_pdf = refit_train_all.to_pandas()
        is_val_row = refit_pdf["wallet"].isin(val_wallets)

        X_tr_inner = X_refit_train[~is_val_row.values]
        y_tr_inner = y_refit_train[~is_val_row.values]
        X_val_inner = X_refit_train[is_val_row.values]
        y_val_inner = y_refit_train[is_val_row.values]

        if len(X_val_inner) < 50 or y_val_inner.nunique() < 2:
            log(f"    ⚠ Validation subset too small/degenerate, falling "
                f"back to a random 15% holdout for tuning only", f)
            rng_split = np.random.RandomState(SEED)
            idx = rng_split.permutation(len(X_refit_train))
            split = int(len(idx) * 0.85)
            X_tr_inner  = X_refit_train.iloc[idx[:split]]
            y_tr_inner  = y_refit_train.iloc[idx[:split]]
            X_val_inner = X_refit_train.iloc[idx[split:]]
            y_val_inner = y_refit_train.iloc[idx[split:]]

        best_params = tune_model("LightGBM", X_tr_inner, y_tr_inner,
                                 X_val_inner, y_val_inner)
        refit_model = build_model("LightGBM", best_params,
                                  spw=(y_refit_train==0).sum() /
                                      max((y_refit_train==1).sum(), 1))
        refit_model.fit(X_refit_train, y_refit_train)
        p_refit = refit_model.predict_proba(X_refit_test)[:, 1]
        auc_refit = roc_auc_score(y_refit_test, p_refit)
        lo_r, hi_r = bootstrap_ci(y_refit_test.values, p_refit)

        # Placebo: shuffle labels, same refit procedure, same test set
        rng = np.random.RandomState(SEED)
        y_shuf = pd.Series(rng.permutation(y_refit_train.values))
        placebo_model = build_model("LightGBM", best_params, spw=1.0)
        placebo_model.fit(X_refit_train, y_shuf)
        p_placebo = placebo_model.predict_proba(X_refit_test)[:, 1]
        auc_placebo = roc_auc_score(y_refit_test, p_placebo)

        log(f"    Frozen AUC:  {auc_frozen:.4f} [{lo_f:.4f},{hi_f:.4f}]", f)
        log(f"    Refit AUC:   {auc_refit:.4f} [{lo_r:.4f},{hi_r:.4f}]  "
            f"(n_train={len(X_refit_train):,}, params: {best_params})", f)
        log(f"    Placebo AUC: {auc_placebo:.4f}  "
            f"{'<-- still elevated, needs review' if auc_placebo > 0.55 or auc_placebo < 0.45 else '(clean)'}", f)

        rows.append({"window": config["name"],
                     "auc_frozen": auc_frozen, "frozen_ci_lo": lo_f,
                     "frozen_ci_hi": hi_f,
                     "auc_refit": auc_refit, "refit_ci_lo": lo_r,
                     "refit_ci_hi": hi_r,
                     "auc_placebo": auc_placebo, "n": len(y_test),
                     "n_refit_train": len(X_refit_train)})

    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "MASTER_table_staleness.csv", index=False)
    log(f"\n  Saved -> {TAB_DIR / 'MASTER_table_staleness.csv'}", f)
    return rows


def run():
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        log("=" * 70, f)
        log("  MASTER NUMBERS — single source of truth for the paper", f)
        log(f"  Generated: {datetime.now().isoformat()}", f)
        log(f"  Script hash: {script_hash()}", f)
        log(f"  Optuna trials per model: {N_TRIALS}, seed: {SEED}", f)
        log("=" * 70, f)

        section1_model_comparison(f)
        section2_persistence(f)
        section3_decile_sort(f)
        section4_staleness(f)

        log("\n" + "=" * 70, f)
        log("  ALL SECTIONS COMPLETE. This file is the only source for", f)
        log("  every number cited in the paper. If any number changes,", f)
        log("  it is because this script was rerun; check the timestamp", f)
        log("  and script hash above against what is cited in the paper.", f)
        log("=" * 70, f)

    print(f"\nMaster report written to: {REPORT_PATH}")


if __name__ == "__main__":
    run()