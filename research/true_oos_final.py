"""
TRUE OUT-OF-SAMPLE — the single, final, hash-verified confirmatory run.

Preconditions enforced by this script:
  1. research/PRE_ANALYSIS_PLAN.md must exist and contain a FROZEN_HASH
     line matching a fresh hash of the frozen analysis scripts (guards
     against silent edits between "freezing" and this run).
  2. research/TRUE_OOS_RUN_LOG.txt must NOT already exist (guards
     against quietly re-running after a first look).

Reports, per pre-registered hypothesis: point estimate, CI, PASS/FAIL
against the criteria in PRE_ANALYSIS_PLAN.md, Holm-Bonferroni-corrected
p-values across H1/H2/H3/H5, and probit-link effect sizes for the two
AUC-based tests (H4, H5). Archives all frozen code + outputs into a
timestamped folder as a permanent record.

Usage: python -m research.true_oos_final
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import warnings
warnings.filterwarnings("ignore")

import re
import shutil
import hashlib
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
from research.freeze_hash import compute_hash, FROZEN_FILES

PIT_DIR   = PROC_DIR / "pit"
TAB_DIR   = Path(__file__).parent / "tables_v2"
FIG_DIR   = Path(__file__).parent / "figures_v2"
MODEL_DIR = Path(__file__).parent / "models_v2"
ROOT      = Path(__file__).parent.parent
PLAN_PATH = Path(__file__).parent / "PRE_ANALYSIS_PLAN.md"
LOG_PATH  = Path(__file__).parent / "TRUE_OOS_RUN_LOG.txt"

SEED = 42
TRUE_OOS_CUTOFF, TRUE_OOS_HORIZON = "2025-06-30", "2025-12-31"
FROZEN_TEST_CUTOFF, FROZEN_TEST_HORIZON = "2024-12-31", "2025-06-30"

META_COLS  = {"wallet", "asof", "first_trade", "last_trade", "wash_flag"}
LABEL_COLS = {"fwd_n_trades", "fwd_clv_vw", "fwd_clv_hitrate", "label_skilled"}


# ── Integrity guards ──────────────────────────────────────────────────────

def enforce_guards():
    if LOG_PATH.exists():
        raise RuntimeError(
            f"{LOG_PATH} already exists. TRUE-OOS has already been run "
            f"once. Delete it manually only if you understand this "
            f"invalidates the pre-registration guarantee."
        )
    if not PLAN_PATH.exists():
        raise RuntimeError(
            f"{PLAN_PATH} not found. Write the pre-analysis plan first."
        )
    plan_text = PLAN_PATH.read_text()
    m = re.search(r"FROZEN_HASH:\s*([0-9a-f]{64})", plan_text)
    if not m:
        raise RuntimeError("No FROZEN_HASH found in PRE_ANALYSIS_PLAN.md.")
    expected = m.group(1)
    actual = compute_hash(ROOT)
    if expected != actual:
        raise RuntimeError(
            f"HASH MISMATCH. Frozen scripts changed since the plan was "
            f"written.\n  expected: {expected}\n  actual:   {actual}\n"
            f"Do not proceed — re-freeze and write a new plan instead, "
            f"documenting why."
        )
    print("  ✓ Pre-analysis plan found and hash verified.")
    print("  ✓ No prior TRUE-OOS run log exists.")


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


def empirical_p_two_sided(vals, null_value):
    vals = np.asarray(vals)
    p_lo = np.mean(vals <= null_value)
    p_hi = np.mean(vals >= null_value)
    return min(1.0, 2 * min(p_lo, p_hi))


def probit_effect_size(auc):
    """Cohen's-d-equivalent from AUC via the probit link."""
    auc = np.clip(auc, 1e-6, 1 - 1e-6)
    return np.sqrt(2) * norm.ppf(auc)


def holm_bonferroni(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


def load_panel(cutoff, horizon):
    f = pl.read_parquet(PIT_DIR / f"features_asof_{cutoff}.parquet")
    l = pl.read_parquet(PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet")
    return f.join(l, on="wallet", how="inner")


# ── H1 + H2 + H3: persistence, reversal, specialization ──────────────────

def check_h1_h2_h3(f):
    log("\n" + "=" * 65, f)
    log("  H1/H2/H3 — Persistence, reversal, specialization: TRUE OOS", f)
    log("=" * 65, f)

    prior = pl.read_parquet(
        PIT_DIR / f"labels_{FROZEN_TEST_CUTOFF}_to_{FROZEN_TEST_HORIZON}.parquet"
    ).select(["wallet", "fwd_clv_vw", "fwd_n_trades"])
    oos = pl.read_parquet(
        PIT_DIR / f"labels_{TRUE_OOS_CUTOFF}_to_{TRUE_OOS_HORIZON}.parquet"
    ).select(["wallet", "fwd_clv_vw", "fwd_n_trades"])
    j = prior.join(oos, on="wallet", suffix="_next").to_pandas()

    rho, pval_naive = spearmanr(j["fwd_clv_vw"], j["fwd_clv_vw_next"])
    boot_rho = bootstrap_dist(
        j["fwd_clv_vw"].values, j["fwd_clv_vw_next"].values,
        lambda a, b: spearmanr(a, b)[0]
    )
    lo, hi = np.percentile(boot_rho, [2.5, 97.5])
    p_h1 = empirical_p_two_sided(boot_rho, 0.0)
    pass_h1 = (abs(rho) < 0.05) and (lo <= 0 <= hi)

    log(f"\n  H1 Persistence: n={len(j):,}  rho={rho:+.4f} [{lo:+.4f},{hi:+.4f}]  "
        f"p={p_h1:.4f}  -> {'PASS (null replicates)' if pass_h1 else 'FAIL (persistence emerges)'}",
        f)

    feats = pl.read_parquet(PIT_DIR / f"features_asof_{TRUE_OOS_CUTOFF}.parquet")
    labels = pl.read_parquet(
        PIT_DIR / f"labels_{TRUE_OOS_CUTOFF}_to_{TRUE_OOS_HORIZON}.parquet"
    )
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

    log(f"\n  H2 Reversal (past_clv_vw coef): n={len(d):,}  "
        f"coef={coef_h2:+.4f}  p={p_h2:.4f}  "
        f"-> {'PASS' if pass_h2 else 'FAIL'}", f)
    log(f"  H3 Specialization (specialist coef): coef={coef_h3:+.4f}  "
        f"p={p_h3:.4f}  -> {'PASS' if pass_h3 else 'FAIL'}", f)
    log(f"\n  Frozen comparisons: H2 coef=-0.0381 (p=8.4e-04) | "
        f"H3 coef=+0.0061 (p=3.6e-13)", f)

    return {
        "h1_rho": rho, "h1_ci_lo": lo, "h1_ci_hi": hi, "h1_p": p_h1,
        "h1_pass": pass_h1, "h1_n": len(j),
        "h2_coef": coef_h2, "h2_p": p_h2, "h2_pass": pass_h2,
        "h3_coef": coef_h3, "h3_p": p_h3, "h3_pass": pass_h3,
        "ols_n": len(d),
    }


# ── H4: frozen ML pipeline ─────────────────────────────────────────────────

def check_h4(f):
    log("\n" + "=" * 65, f)
    log("  H4 — Frozen behavioral-ML model: TRUE OOS", f)
    log("=" * 65, f)

    model_path = MODEL_DIR / "lgbm_primary.pkl"
    if not model_path.exists():
        log("  ⚠ Frozen model not found. Skipping H4.", f)
        return {}

    with open(model_path, "rb") as fh:
        saved = pickle.load(fh)
    model, feature_cols = saved["model"], saved["features"]

    oos = load_panel(TRUE_OOS_CUTOFF, TRUE_OOS_HORIZON).to_pandas()
    X = oos[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = oos["label_skilled"].astype(int)

    p = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p)
    boot_auc = bootstrap_dist(y.values, p, roc_auc_score)
    lo, hi = np.percentile(boot_auc, [2.5, 97.5])
    pass_h4 = (lo <= 0.5 <= hi)
    d_eff = probit_effect_size(auc)

    log(f"\n  H4 ML AUC: n={len(y):,}  AUC={auc:.4f} [{lo:.4f},{hi:.4f}]  "
        f"probit-d={d_eff:+.4f}  "
        f"-> {'PASS (null replicates)' if pass_h4 else 'FAIL (signal emerges)'}",
        f)
    log(f"  Frozen comparison: AUC=0.4758 [0.4714,0.4800]", f)

    return {"h4_auc": auc, "h4_ci_lo": lo, "h4_ci_hi": hi,
            "h4_effect_size_d": d_eff, "h4_pass": pass_h4, "h4_n": len(y)}


# ── H5: market-level insider flow ─────────────────────────────────────────

def check_h5(f):
    log("\n" + "=" * 65, f)
    log("  H5 — Market-level informed order flow: TRUE OOS", f)
    log("=" * 65, f)

    TRADES_GLOB = str(RAW_DIR / "trades" / "**" / "*.parquet")
    LATE_WINDOW_HOURS, MIN_TRADES = 48, 20
    OOS_START, OOS_END = "2025-07-01 00:00:00", "2025-12-31 23:59:59"

    lf = (
        pl.scan_parquet(TRADES_GLOB)
        .filter(pl.col("timestamp") >= pl.lit(OOS_START)
                .str.to_datetime(time_zone="UTC"))
        .filter(pl.col("timestamp") <= pl.lit(OOS_END)
                .str.to_datetime(time_zone="UTC"))
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
    log(f"  {len(close_info):,} resolved predictions in TRUE-OOS window", f)
    if len(close_info) < 200:
        log("  Insufficient data. Skipping H5.", f)
        return {}

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

    if len(combined) < 200 or combined["win"].nunique() < 2:
        log("  Insufficient/degenerate sample. Skipping H5.", f)
        return {}

    auc = roc_auc_score(combined["win"], combined["late_imbalance"])

    rng = np.random.RandomState(SEED)
    reset = combined.reset_index(drop=True)
    idx_by_event = reset.groupby("event_id").indices
    event_ids = np.array(list(idx_by_event.keys()))
    boot_auc = []
    for _ in range(500):
        sampled = rng.choice(event_ids, size=len(event_ids), replace=True)
        idx = np.concatenate([idx_by_event[e] for e in sampled])
        bd = reset.iloc[idx]
        if bd["win"].nunique() < 2:
            continue
        try:
            boot_auc.append(roc_auc_score(bd["win"], bd["late_imbalance"]))
        except Exception:
            pass
    boot_auc = np.array(boot_auc)
    lo, hi = np.percentile(boot_auc, [2.5, 97.5])
    p_h5 = empirical_p_two_sided(boot_auc, 0.5)
    pass_h5 = (lo <= 0.5) is False and (lo > 0.5) and (auc > 0.5)
    # correct pass condition: CI excludes 0.5 AND auc > 0.5
    pass_h5 = (lo > 0.5) and (auc > 0.5)
    d_eff = probit_effect_size(auc)

    log(f"\n  H5 Insider-flow AUC: n={len(combined):,} "
        f"events={combined['event_id'].nunique():,}  "
        f"AUC={auc:.4f} [{lo:.4f},{hi:.4f}]  p={p_h5:.4f}  "
        f"probit-d={d_eff:+.4f}  "
        f"-> {'PASS (signal replicates)' if pass_h5 else 'FAIL'}", f)
    log(f"  Frozen comparison: AUC=0.5412 [0.5323,0.5501]", f)

    return {"h5_auc": auc, "h5_ci_lo": lo, "h5_ci_hi": hi, "h5_p": p_h5,
            "h5_effect_size_d": d_eff, "h5_pass": pass_h5,
            "h5_n": len(combined),
            "h5_events": combined["event_id"].nunique()}


# ── Archive ────────────────────────────────────────────────────────────────

def archive_run():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(__file__).parent / f"oos_archive_{stamp}"
    dest.mkdir(exist_ok=True)
    for rel in FROZEN_FILES + ["research/PRE_ANALYSIS_PLAN.md",
                               "research/TRUE_OOS_RUN_LOG.txt"]:
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, dest / src.name)
    for sub in ["tables_v2", "figures_v2"]:
        src_dir = Path(__file__).parent / sub
        if src_dir.exists():
            shutil.copytree(src_dir, dest / sub, dirs_exist_ok=True)
    print(f"\n  ✓ Archived frozen code + all outputs -> {dest}")


# ── Main ────────────────────────────────────────────────────────────────────

def run():
    enforce_guards()

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        log("=" * 65, f)
        log("  TRUE OUT-OF-SAMPLE CONFIRMATORY RUN — 2025-H2", f)
        log(f"  Timestamp: {datetime.now().isoformat()}", f)
        log("  Hash-verified against PRE_ANALYSIS_PLAN.md. Single look.", f)
        log("=" * 65, f)

        r = {}
        r.update(check_h1_h2_h3(f))
        r.update(check_h4(f))
        r.update(check_h5(f))

        pvals = {}
        if "h1_p" in r: pvals["H1"] = r["h1_p"]
        if "h2_p" in r: pvals["H2"] = r["h2_p"]
        if "h3_p" in r: pvals["H3"] = r["h3_p"]
        if "h5_p" in r: pvals["H5"] = r["h5_p"]
        adj = holm_bonferroni(pvals) if pvals else {}

        log("\n" + "=" * 65, f)
        log("  FINAL SUMMARY — PASS/FAIL against pre-registered criteria", f)
        log("  (p-values Holm-Bonferroni corrected across H1,H2,H3,H5)", f)
        log("=" * 65, f)
        for h in ["H1", "H2", "H3", "H5"]:
            key = h.lower() + "_pass"
            if key in r:
                raw_p = pvals.get(h, float("nan"))
                adj_p = adj.get(h, float("nan"))
                log(f"  {h}: {'PASS' if r[key] else 'FAIL'}  "
                    f"(raw p={raw_p:.4f}, Holm-adj p={adj_p:.4f})", f)
        if "h4_pass" in r:
            log(f"  H4: {'PASS' if r['h4_pass'] else 'FAIL'}  "
                f"(CI-based criterion, no p-value)", f)

        pd.DataFrame([r]).to_csv(TAB_DIR / "t20_true_oos_summary.csv",
                                 index=False)
        log(f"\n  ✓ -> {TAB_DIR / 't20_true_oos_summary.csv'}", f)

    archive_run()
    print(f"\n  ✓ Permanent log -> {LOG_PATH}")
    print("  Do not delete or re-run. This is the confirmatory record.")


if __name__ == "__main__":
    run()