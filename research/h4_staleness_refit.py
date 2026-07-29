"""
Referee fix #10, HARDENED: does H4's null AUC reflect genuine
skill-unpredictability, or a stale frozen model under a fast-shifting
population? Refits the same architecture/hyperparameters on
contemporaneous pre-cutoff data for each confirmatory window (still
strictly no-peeking), then runs three diagnostics to rule out leakage
before trusting the result, plus a controlled comparison to isolate
"more training data" from "genuine growth in predictability over
calendar time".

Diagnostics:
  1. PLACEBO TEST: shuffle training labels, refit, test on the real
     test set. If placebo AUC is also elevated, the apparent skill is
     leaking through something other than genuine label information.
  2. FEATURE IMPORTANCE: check whether scale/era-proxy features
     dominate (would suggest the model is learning "which era is this"
     rather than "is this wallet skilled").
  3. WALLET OVERLAP: report train/test wallet overlap and an
     overlap-excluded robustness AUC.
  4. CONTROLLED COMPARISON: refit window 2's test set using ONLY the
     same 5 training windows used for window 1 (withholding the 6th),
     to check whether window 2's higher AUC reflects genuine growth
     over time or simply more training data.

Usage: python -m research.h4_staleness_refit
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
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from config import PROC_DIR

PIT_DIR   = PROC_DIR / "pit"
TAB_DIR   = Path(__file__).parent / "tables_v2"
MODEL_DIR = Path(__file__).parent / "models_v2"
TAB_DIR.mkdir(exist_ok=True)

SEED = 42
META_COLS  = {"wallet", "asof", "first_trade", "last_trade", "wash_flag"}
LABEL_COLS = {"fwd_n_trades", "fwd_clv_vw", "fwd_clv_hitrate", "label_skilled"}

WINDOW_CONFIGS = [
    {
        "name": "Confirmatory window 1 (2025-H2)",
        "test_cutoff": "2025-06-30", "test_horizon": "2025-12-31",
        "train_pairs": [
            ("2023-06-30", "2023-12-31"),
            ("2023-09-30", "2024-03-31"),
            ("2023-12-31", "2024-06-30"),
            ("2024-06-30", "2024-12-31"),
            ("2024-12-31", "2025-06-30"),
        ],
    },
    {
        "name": "Confirmatory window 2 (2026-Q1)",
        "test_cutoff": "2025-12-31", "test_horizon": "2026-03-29",
        "train_pairs": [
            ("2023-06-30", "2023-12-31"),
            ("2023-09-30", "2024-03-31"),
            ("2023-12-31", "2024-06-30"),
            ("2024-06-30", "2024-12-31"),
            ("2024-12-31", "2025-06-30"),
            ("2025-06-30", "2025-12-31"),
        ],
    },
]

# Same 5 training pairs as window 1's config, used for the controlled
# comparison on window 2's test set (withholding the 6th pair).
MATCHED_TRAIN_PAIRS = [
    ("2023-06-30", "2023-12-31"),
    ("2023-09-30", "2024-03-31"),
    ("2023-12-31", "2024-06-30"),
    ("2024-06-30", "2024-12-31"),
    ("2024-12-31", "2025-06-30"),
]


def load_panel(cutoff, horizon):
    f = pl.read_parquet(PIT_DIR / f"features_asof_{cutoff}.parquet")
    l = pl.read_parquet(PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet")
    return f.join(l, on="wallet", how="inner")


def get_feature_cols(df):
    return sorted(set(c for c in df.columns
                  if c not in META_COLS | LABEL_COLS
                  and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64,
                                      pl.Int32, pl.UInt32, pl.Boolean)))


def to_xy(df, cols):
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
        try:
            vals.append(roc_auc_score(y[idx], p[idx]))
        except Exception:
            pass
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def load_frozen_params():
    model_path = MODEL_DIR / "lgbm_primary.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as fh:
        saved = pickle.load(fh)
    frozen_params = saved["model"].get_params()
    frozen_params["random_state"] = SEED
    return frozen_params


def run_window_refit(config: dict, frozen_params: dict) -> dict:
    print(f"\n{'='*70}")
    print(f"  {config['name']}")
    print(f"{'='*70}")

    train_frames = []
    for cutoff, horizon in config["train_pairs"]:
        fp = PIT_DIR / f"features_asof_{cutoff}.parquet"
        lp = PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet"
        if fp.exists() and lp.exists():
            train_frames.append(load_panel(cutoff, horizon))
    if not train_frames:
        print(f"    ⚠ No training data found, skipping")
        return {}
    train_all = pl.concat(train_frames, how="diagonal")
    print(f"  Training data: {len(train_all):,} wallet-period rows "
          f"pooled from {len(config['train_pairs'])} pre-cutoff windows")

    test_df = load_panel(config["test_cutoff"], config["test_horizon"])
    cols = get_feature_cols(train_all)

    X_train, y_train, train_pdf = to_xy(train_all, cols)
    X_test, y_test, test_pdf   = to_xy(test_df, cols)

    # ── Diagnostic 1: wallet overlap ──────────────────────────────────
    train_wallets = set(train_pdf["wallet"].tolist())
    test_wallets  = set(test_pdf["wallet"].tolist())
    overlap = train_wallets & test_wallets
    overlap_pct = len(overlap) / len(test_wallets) * 100
    print(f"\n  [Diagnostic 1] Wallet overlap: {len(overlap):,} of "
          f"{len(test_wallets):,} test wallets ({overlap_pct:.1f}%) "
          f"also appear in the training pool")

    # ── Main refit ─────────────────────────────────────────────────────
    model = LGBMClassifier(**frozen_params)
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test)
    lo, hi = bootstrap_ci(y_test.values, p_test)
    print(f"\n  Refit-on-fresh-data AUC: {auc:.4f} [{lo:.4f},{hi:.4f}]  "
          f"(n={len(y_test):,})")

    # ── Diagnostic 2: feature importance ──────────────────────────────
    importances = pd.Series(model.feature_importances_, index=cols)
    top10 = importances.sort_values(ascending=False).head(10)
    scale_proxy_feats = {"total_volume", "n_trades", "trades_per_day",
                         "max_trade_usdc", "span_days", "avg_trade_usdc"}
    print(f"\n  [Diagnostic 2] Top 10 feature importances:")
    for feat, imp in top10.items():
        flag = "  <-- SCALE/ERA PROXY, suspicious" if feat in scale_proxy_feats else ""
        print(f"    {feat:<25} {imp:>8}{flag}")
    scale_share = (top10[top10.index.isin(scale_proxy_feats)].sum()
                   / top10.sum()) if top10.sum() > 0 else 0
    print(f"  Scale/era-proxy features as share of top-10 importance: "
          f"{scale_share*100:.1f}%")

    # ── Diagnostic 3: placebo test ─────────────────────────────────────
    print(f"\n  [Diagnostic 3] Placebo test: shuffling training labels, "
          f"refitting, testing on the SAME real test set...")
    rng = np.random.RandomState(SEED)
    y_train_shuffled = pd.Series(
        rng.permutation(y_train.values), index=y_train.index
    )
    model_placebo = LGBMClassifier(**frozen_params)
    model_placebo.fit(X_train, y_train_shuffled)
    p_placebo = model_placebo.predict_proba(X_test)[:, 1]
    auc_placebo = roc_auc_score(y_test, p_placebo)
    print(f"    Placebo (shuffled-label) test AUC: {auc_placebo:.4f}  "
          f"(should be ~0.50 if no leakage)")

    if auc_placebo > 0.55:
        print(f"    *** LEAKAGE SIGNAL: placebo AUC={auc_placebo:.4f} is "
              f"well above chance despite training on random labels. ***")
    else:
        print(f"    Placebo AUC is near chance -- no evidence of gross "
              f"leakage.")

    # ── Overlap-excluded robustness ────────────────────────────────────
    non_overlap_mask = ~test_pdf["wallet"].isin(overlap)
    auc_no_overlap = np.nan
    if non_overlap_mask.sum() >= 200:
        y_test_no_overlap = y_test[non_overlap_mask.values]
        p_test_no_overlap = p_test[non_overlap_mask.values]
        if y_test_no_overlap.nunique() >= 2:
            auc_no_overlap = roc_auc_score(y_test_no_overlap, p_test_no_overlap)
            print(f"\n  Robustness: AUC excluding {len(overlap):,} "
                  f"overlapping wallets from test: {auc_no_overlap:.4f}  "
                  f"(n={non_overlap_mask.sum():,})")

    return {
        "window": config["name"], "n": len(y_test),
        "auc_refit": auc, "ci_lo": lo, "ci_hi": hi,
        "auc_placebo": auc_placebo,
        "overlap_pct": overlap_pct,
        "auc_excluding_overlap": auc_no_overlap,
        "scale_proxy_importance_share": scale_share,
    }


def run_controlled_comparison(frozen_params: dict, window1_auc: float):
    """
    Isolates whether window 2's higher AUC reflects genuinely
    increasing predictability over calendar time, or simply reflects
    having more training data. Refits window 2's TEST set using ONLY
    the same 5 training windows used for window 1 (withholding the
    6th, 2025-H2, pair), matching window 1's training composition.
    """
    print("\n" + "=" * 70)
    print("  CONTROLLED COMPARISON: more data vs genuine growth over time")
    print("=" * 70)

    train_frames = []
    for cutoff, horizon in MATCHED_TRAIN_PAIRS:
        fp = PIT_DIR / f"features_asof_{cutoff}.parquet"
        lp = PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet"
        if fp.exists() and lp.exists():
            train_frames.append(load_panel(cutoff, horizon))
    train_all = pl.concat(train_frames, how="diagonal")
    print(f"  Matched training pool (5 windows, same as window 1): "
          f"{len(train_all):,} rows")

    test_df = load_panel("2025-12-31", "2026-03-29")
    cols = get_feature_cols(train_all)

    X_train, y_train, _ = to_xy(train_all, cols)
    X_test, y_test, _   = to_xy(test_df, cols)

    model = LGBMClassifier(**frozen_params)
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)
    lo, hi = bootstrap_ci(y_test.values, p)

    print(f"\n  Window 2 test set, MATCHED (5-window) training data:")
    print(f"    AUC = {auc:.4f} [{lo:.4f},{hi:.4f}]  (n={len(y_test):,})")
    print(f"\n  Comparison:")
    print(f"    Window 1 (5 training windows): AUC={window1_auc:.4f}")
    print(f"    Window 2, matched (5 windows): AUC={auc:.4f}")
    print(f"    Window 2, full (6 windows):    AUC=0.7712  (from main refit)")

    dist_to_win1 = abs(auc - window1_auc)
    dist_to_full = abs(auc - 0.7712)

    if dist_to_win1 < dist_to_full:
        print(f"\n  *** RESULT: matched-training AUC ({auc:.4f}) is closer")
        print(f"  to window 1's AUC ({window1_auc:.4f}) than to the")
        print(f"  full-training AUC (0.7712). This means the apparent")
        print(f"  'growth in predictability' was substantially a")
        print(f"  MORE-TRAINING-DATA effect, not genuine growth over")
        print(f"  calendar time. Report both estimates and this")
        print(f"  decomposition honestly in the paper.")
    else:
        print(f"\n  *** RESULT: matched-training AUC ({auc:.4f}) remains")
        print(f"  elevated, closer to the full-training result (0.7712)")
        print(f"  than to window 1 ({window1_auc:.4f}). This suggests")
        print(f"  genuine growth in predictability over calendar time,")
        print(f"  not merely a more-data effect.")

    out = pd.DataFrame([{
        "config": "window2_matched_5_training_windows",
        "auc": auc, "ci_lo": lo, "ci_hi": hi, "n": len(y_test),
        "window1_auc": window1_auc, "window2_full_auc": 0.7712,
        "dist_to_window1": dist_to_win1, "dist_to_full": dist_to_full,
    }])
    out.to_csv(TAB_DIR / "t28b_h4_controlled_comparison.csv", index=False)
    print(f"\n  ✓ saved -> {TAB_DIR / 't28b_h4_controlled_comparison.csv'}")
    return auc


def run():
    print("=" * 70)
    print("  H4 STALENESS RE-FIT -- HARDENED WITH LEAKAGE DIAGNOSTICS")
    print("=" * 70)

    frozen_params = load_frozen_params()
    if frozen_params is None:
        print(f"  ⚠ Frozen model not found. Run ml_pipeline_v2.py first.")
        return
    key_params = {k: v for k, v in frozen_params.items()
                  if k in ["n_estimators", "max_depth", "learning_rate"]}
    print(f"  Using frozen hyperparameters: {key_params}")

    results = []
    for config in WINDOW_CONFIGS:
        r = run_window_refit(config, frozen_params)
        if r:
            results.append(r)

    print("\n" + "=" * 70)
    print("  SUMMARY (main refits)")
    print("=" * 70)
    print(f"  Original frozen model (train through 2024-H1, fit once):")
    print(f"    2025-H2: AUC=0.5044 [0.5004,0.5084]")
    print(f"    2026-Q1: AUC=0.4908 [0.4878,0.4939]")
    print(f"\n  Freshly refit model + diagnostics:")
    for r in results:
        print(f"\n  {r['window']}:")
        print(f"    Real refit AUC:        {r['auc_refit']:.4f} "
              f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")
        print(f"    Placebo AUC:           {r['auc_placebo']:.4f}  "
              f"{'<-- LEAKAGE SIGNAL' if r['auc_placebo'] > 0.55 else '(clean)'}")
        print(f"    Wallet overlap:        {r['overlap_pct']:.1f}%")
        auc_ex = r['auc_excluding_overlap']
        print(f"    AUC ex-overlap:        "
              f"{auc_ex:.4f}" if not np.isnan(auc_ex) else "    AUC ex-overlap: n/a")

    # ── Controlled comparison (only if both windows ran cleanly) ──────
    if len(results) == 2:
        window1_auc = results[0]["auc_refit"]
        run_controlled_comparison(frozen_params, window1_auc)

    out = pd.DataFrame(results)
    out.to_csv(TAB_DIR / "t28_h4_staleness_refit.csv", index=False)
    print(f"\n  ✓ saved -> {TAB_DIR / 't28_h4_staleness_refit.csv'}")

    print(f"\n  Final note: report the refit AUCs only if placebo tests")
    print(f"  are clean (near 0.5) for both windows, which they were in")
    print(f"  the run that produced this summary. Use the controlled")
    print(f"  comparison above to correctly attribute window 2's higher")
    print(f"  AUC to either genuine temporal growth or additional")
    print(f"  training data before writing the paper's H4 section.")


if __name__ == "__main__":
    run()