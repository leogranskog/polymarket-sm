"""
SECOND CONFIRMATORY WINDOW — 2026-Q1 (Jan-Mar 2026).

Single look, guarded by PRE_ANALYSIS_PLAN_2.md and its own run log.
Mirrors true_oos_final.py's hypothesis tests exactly, applied to the
new window: features@2025-12-31, labels 2026-01-01 to 2026-03-29.

Uses freeze_hash_2 (analysis-only hash) rather than freeze_hash,
since pit_features.py is a data-preparation utility that necessarily
changes to build each new confirmatory window and is not part of the
hypothesis-testing methodology being frozen here.

Usage: python -m research.true_oos_second_window
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import polars as pl
import pickle
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, norm
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
import statsmodels.api as sm

from config import RAW_DIR, PROC_DIR
from research.freeze_hash_2 import compute_hash

PIT_DIR   = PROC_DIR / "pit"
TAB_DIR   = Path(__file__).parent / "tables_v2"
MODEL_DIR = Path(__file__).parent / "models_v2"
ROOT      = Path(__file__).parent.parent
PLAN_PATH = Path(__file__).parent / "PRE_ANALYSIS_PLAN_2.md"
LOG_PATH  = Path(__file__).parent / "TRUE_OOS_2_RUN_LOG.txt"

SEED = 42
CUTOFF, HORIZON = "2025-12-31", "2026-03-29"
PRIOR_CUTOFF, PRIOR_HORIZON = "2025-06-30", "2025-12-31"  # for persistence


def enforce_guards():
    if LOG_PATH.exists():
        raise RuntimeError(f"{LOG_PATH} already exists. Already run once.")
    if not PLAN_PATH.exists():
        raise RuntimeError(f"{PLAN_PATH} not found. Write the plan first.")
    text = PLAN_PATH.read_text()
    m = re.search(r"FROZEN_HASH:\s*([0-9a-f]{64})", text)
    if not m:
        raise RuntimeError("No FROZEN_HASH in PRE_ANALYSIS_PLAN_2.md.")
    expected, actual = m.group(1), compute_hash(ROOT)
    if expected != actual:
        raise RuntimeError(
            f"HASH MISMATCH.\n  expected: {expected}\n  actual: {actual}"
        )
    print("  Pre-analysis plan (addendum 2) found and hash verified "
          "(analysis-only scope).")
    print("  No prior second-window run log exists.")


def log(msg, f):
    print(msg)
    f.write(msg + "\n")


def bootstrap_dist(y, p, metric_fn, n=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); p = np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        try:
            vals.append(metric_fn(y[idx], p[idx]))
        except Exception:
            pass
    return np.array(vals)


def holm_bonferroni(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    adjusted, running_max = {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


def probit_effect_size(auc):
    auc = np.clip(auc, 1e-6, 1 - 1e-6)
    return np.sqrt(2) * norm.ppf(auc)


def run():
    enforce_guards()

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        log("=" * 65, f)
        log("  SECOND CONFIRMATORY WINDOW — 2026-Q1 (Jan-Mar 2026)", f)
        log(f"  Timestamp: {datetime.now().isoformat()}", f)
        log("  Hash-verified (analysis-only) against PRE_ANALYSIS_PLAN_2.md. "
            "Single look.", f)
        log("=" * 65, f)

        # ── H1/H2/H3 ──────────────────────────────────────────────────
        prior = pl.read_parquet(
            PIT_DIR / f"labels_{PRIOR_CUTOFF}_to_{PRIOR_HORIZON}.parquet"
        ).select(["wallet", "fwd_clv_vw"])
        window2 = pl.read_parquet(
            PIT_DIR / f"labels_{CUTOFF}_to_{HORIZON}.parquet"
        ).select(["wallet", "fwd_clv_vw"])
        j = prior.join(window2, on="wallet", suffix="_next").to_pandas()

        rho, _ = spearmanr(j["fwd_clv_vw"], j["fwd_clv_vw_next"])
        boot = bootstrap_dist(j["fwd_clv_vw"].values, j["fwd_clv_vw_next"].values,
                              lambda a, b: spearmanr(a, b)[0])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_h1 = min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
        pass_h1 = (abs(rho) < 0.05) and (lo <= 0 <= hi)
        log(f"\n  H1 Persistence (H2-2025 -> Q1-2026): n={len(j):,}  "
            f"rho={rho:+.4f} [{lo:+.4f},{hi:+.4f}]  p={p_h1:.4f}  "
            f"-> {'PASS' if pass_h1 else 'FAIL'}", f)

        feats = pl.read_parquet(PIT_DIR / f"features_asof_{CUTOFF}.parquet")
        labels = pl.read_parquet(PIT_DIR / f"labels_{CUTOFF}_to_{HORIZON}.parquet")
        df = feats.join(labels, on="wallet", how="inner").to_pandas()
        t1, t2 = df["category_hhi"].quantile([1/3, 2/3])
        d = df[(df["category_hhi"] >= t2) | (df["category_hhi"] <= t1)].copy()
        d["specialist"] = (d["category_hhi"] >= t2).astype(int)
        d["log_trades"] = np.log1p(d["n_trades"])
        d["log_volume"] = np.log1p(d["total_volume"])
        for c in ["past_clv_vw", "frac_maker"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
        ols = smf.ols("fwd_clv_vw ~ specialist + log_trades + log_volume "
                      "+ frac_maker + past_clv_vw", data=d).fit(cov_type="HC3")

        coef_h2, p_h2 = ols.params["past_clv_vw"], ols.pvalues["past_clv_vw"]
        pass_h2 = (coef_h2 < 0) and (p_h2 < 0.05)
        coef_h3, p_h3 = ols.params["specialist"], ols.pvalues["specialist"]
        pass_h3 = (coef_h3 > 0) and (p_h3 < 0.05)
        log(f"\n  H2 Reversal: n={len(d):,}  coef={coef_h2:+.4f}  p={p_h2:.4f}  "
            f"-> {'PASS' if pass_h2 else 'FAIL'}", f)
        log(f"  H3 Specialization: coef={coef_h3:+.4f}  p={p_h3:.4f}  "
            f"-> {'PASS' if pass_h3 else 'FAIL'}", f)

        # ── H4: frozen ML model ──────────────────────────────────────
        model_path = MODEL_DIR / "lgbm_primary.pkl"
        h4_result = {}
        if model_path.exists():
            with open(model_path, "rb") as fh:
                saved = pickle.load(fh)
            model, feature_cols = saved["model"], saved["features"]
            X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
            X = X.fillna(X.median(numeric_only=True)).fillna(0)
            y = df["label_skilled"].astype(int)
            p = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, p)
            boot_auc = bootstrap_dist(y.values, p, roc_auc_score)
            lo4, hi4 = np.percentile(boot_auc, [2.5, 97.5])
            pass_h4 = (lo4 <= 0.5 <= hi4)
            log(f"\n  H4 ML AUC: n={len(y):,}  AUC={auc:.4f} "
                f"[{lo4:.4f},{hi4:.4f}]  -> {'PASS' if pass_h4 else 'FAIL'}", f)
            h4_result = {"h4_auc": auc, "h4_ci_lo": lo4, "h4_ci_hi": hi4,
                        "h4_pass": pass_h4, "h4_n": len(y)}
        else:
            log("\n  H4: frozen model not found, skipped.", f)

        # ── H5: market-level insider flow, 2026-Q1 window ────────────
        TRADES_GLOB = str(RAW_DIR / "trades" / "**" / "*.parquet")
        LATE_WINDOW_HOURS, MIN_TRADES = 48, 20
        OOS_START, OOS_END = "2026-01-01 00:00:00", "2026-03-29 23:59:59"

        log("\n  H5: scanning 2026-Q1 trades (streaming)...", f)
        lf = (
            pl.scan_parquet(TRADES_GLOB)
            .filter(pl.col("timestamp") >= pl.lit(OOS_START).str.to_datetime(time_zone="UTC"))
            .filter(pl.col("timestamp") <= pl.lit(OOS_END).str.to_datetime(time_zone="UTC"))
            .select(["prediction_id", "event_id", "timestamp", "price",
                     "quantity", "taker_bought", "winner"])
        )
        close_info = (
            lf.group_by("prediction_id")
            .agg([pl.col("timestamp").max().alias("last_ts"),
                  pl.len().alias("n_trades"),
                  pl.col("event_id").first().alias("event_id"),
                  pl.col("winner").drop_nulls().first().alias("win_bool")])
            .filter(pl.col("n_trades") >= MIN_TRADES)
            .filter(pl.col("win_bool").is_not_null())
            .with_columns(pl.col("win_bool").cast(pl.Int32).alias("win"))
            .collect(engine="streaming")
        )
        log(f"  {len(close_info):,} resolved predictions in 2026-Q1", f)

        h5_result = {}
        if len(close_info) >= 200:
            trades = (
                lf.join(close_info.lazy().select(["prediction_id", "last_ts"]),
                        on="prediction_id", how="inner")
                .with_columns([
                    (pl.col("last_ts") - pl.col("timestamp")).dt.total_hours()
                        .alias("hours_to_close"),
                    (pl.col("price") * pl.col("quantity")).alias("usdc"),
                ])
                .with_columns([
                    pl.when(pl.col("taker_bought")).then(pl.col("usdc"))
                      .otherwise(-pl.col("usdc")).alias("signed_usdc"),
                ])
            )
            late = (trades.filter(pl.col("hours_to_close") <= LATE_WINDOW_HOURS)
                    .group_by("prediction_id")
                    .agg([pl.col("signed_usdc").sum().alias("late_net_flow"),
                          pl.col("usdc").sum().alias("late_total_flow"),
                          pl.len().alias("late_n")]))
            pre = (trades.filter(pl.col("hours_to_close") > LATE_WINDOW_HOURS)
                   .group_by("prediction_id").agg(pl.len().alias("pre_n")))
            combined = (
                late.join(pre, on="prediction_id", how="inner")
                .join(close_info.lazy().select(["prediction_id", "event_id", "win"]),
                      on="prediction_id", how="inner")
                .filter((pl.col("late_n") >= 5) & (pl.col("pre_n") >= 5))
                .filter(pl.col("late_total_flow") > 0)
                .with_columns((pl.col("late_net_flow") / pl.col("late_total_flow"))
                             .alias("late_imbalance"))
                .select(["prediction_id", "event_id", "late_imbalance", "win"])
                .collect(engine="streaming").to_pandas()
            )
            combined = combined.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["late_imbalance", "win", "event_id"])
            combined["win"] = combined["win"].astype(int)

            if len(combined) >= 200 and combined["win"].nunique() >= 2:
                auc5 = roc_auc_score(combined["win"], combined["late_imbalance"])
                reset = combined.reset_index(drop=True)
                idx_by_event = reset.groupby("event_id").indices
                event_ids = np.array(list(idx_by_event.keys()))
                rng = np.random.RandomState(SEED)
                boot5 = []
                for _ in range(500):
                    sampled = rng.choice(event_ids, size=len(event_ids), replace=True)
                    idx = np.concatenate([idx_by_event[e] for e in sampled])
                    bd = reset.iloc[idx]
                    if bd["win"].nunique() < 2:
                        continue
                    try:
                        boot5.append(roc_auc_score(bd["win"], bd["late_imbalance"]))
                    except Exception:
                        pass
                lo5, hi5 = np.percentile(boot5, [2.5, 97.5])
                pass_h5 = (lo5 > 0.5) and (auc5 > 0.5)

                X5 = sm.add_constant(combined["late_imbalance"])
                logit5 = sm.Logit(combined["win"], X5).fit(
                    cov_type="cluster", cov_kwds={"groups": combined["event_id"]},
                    disp=0,
                )
                log(f"\n  H5 Insider flow: n={len(combined):,} "
                    f"events={combined['event_id'].nunique():,}  "
                    f"AUC={auc5:.4f} [{lo5:.4f},{hi5:.4f}]  "
                    f"coef={logit5.params['late_imbalance']:+.4f}  "
                    f"p={logit5.pvalues['late_imbalance']:.2e}  "
                    f"-> {'PASS' if pass_h5 else 'FAIL'}", f)
                h5_result = {"h5_auc": auc5, "h5_ci_lo": lo5, "h5_ci_hi": hi5,
                            "h5_pass": pass_h5, "h5_n": len(combined),
                            "h5_events": combined["event_id"].nunique()}
            else:
                log("  H5: insufficient/degenerate sample in 2026-Q1.", f)
        else:
            log("  H5: insufficient predictions in 2026-Q1.", f)

        # ── Summary ───────────────────────────────────────────────────
        log("\n" + "=" * 65, f)
        log("  SECOND-WINDOW SUMMARY vs FIRST CONFIRMATORY WINDOW (2025-H2)", f)
        log("=" * 65, f)
        log(f"  H1 Persistence: 2025-H2 rho=+0.053 (FAIL) | "
            f"2026-Q1 rho={rho:+.4f} ({'PASS' if pass_h1 else 'FAIL'})", f)
        log(f"  H2 Reversal:    2025-H2 coef=+0.031 (FAIL) | "
            f"2026-Q1 coef={coef_h2:+.4f} ({'PASS' if pass_h2 else 'FAIL'})", f)
        log(f"  H3 Specialization: 2025-H2 coef=+0.0060 (PASS) | "
            f"2026-Q1 coef={coef_h3:+.4f} "
            f"({'PASS' if pass_h3 else 'FAIL'})", f)
        if h4_result:
            log(f"  H4 ML AUC:      2025-H2 AUC=0.5044 (FAIL) | "
                f"2026-Q1 AUC={h4_result['h4_auc']:.4f} "
                f"({'PASS' if h4_result['h4_pass'] else 'FAIL'})", f)
        if h5_result:
            log(f"  H5 Insider flow: 2025-H2 AUC=0.5579 (PASS) | "
                f"2026-Q1 AUC={h5_result['h5_auc']:.4f} "
                f"({'PASS' if h5_result['h5_pass'] else 'FAIL'})", f)

        result = {
            "h1_rho": rho, "h1_ci_lo": lo, "h1_ci_hi": hi, "h1_pass": pass_h1,
            "h1_n": len(j),
            "h2_coef": coef_h2, "h2_p": p_h2, "h2_pass": pass_h2,
            "h3_coef": coef_h3, "h3_p": p_h3, "h3_pass": pass_h3,
            "ols_n": len(d),
            **h4_result, **h5_result,
        }
        pd.DataFrame([result]).to_csv(
            TAB_DIR / "t21_second_window_summary.csv", index=False)
        log(f"\n  Saved -> {TAB_DIR / 't21_second_window_summary.csv'}", f)

    print(f"\n  Permanent log -> {LOG_PATH}")
    print("  Do not delete or re-run.")


if __name__ == "__main__":
    run()