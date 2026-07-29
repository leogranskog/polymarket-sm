"""
Referee-mandated fix for H5: does late order-flow imbalance predict
the outcome INCREMENTAL to the concurrent market price, or is the
unconditional AUC entirely explained by price already being
informative (price alone achieves AUC ~0.85-0.94, since price
mechanically approaches the outcome as resolution nears)?

Adds price at the START of the late window (the last trade before the
48h imbalance-measurement window begins) as a control. Reports, for
the development sample AND both confirmatory windows separately:
  - price-only AUC and McFadden pseudo-R^2 (baseline)
  - price + imbalance AUC and pseudo-R^2 (does imbalance add anything?)
  - imbalance coefficient controlling for price (event-clustered SEs)
  - likelihood-ratio test: does adding imbalance significantly improve
    the model beyond price alone?
  - event-clustered bootstrap CI on the incremental pseudo-R^2
  - a weighted linear trend test across the three sequential windows

IMPORTANT INTERPRETIVE NOTE: AUC is a poor tool for detecting a real
incremental predictor once the baseline model is already strong,
because there is little room left between a high baseline AUC and the
ceiling of 1.0 for ANY additional variable to move the needle, however
genuinely informative it is. The likelihood-ratio test and McFadden
pseudo-R^2 are the correct tools for this comparison; AUC deltas are
reported for completeness but should not be used alone to declare an
effect absent. This mirrors standard practice in market microstructure
(Kyle 1985; Easley & O'Hara 1987), which define informed order flow
precisely as flow that predicts value BEYOND what is already reflected
in price.

Usage: python -m research.market_insider_flow_v3
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2, t as t_dist
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

from config import RAW_DIR

TRADES_GLOB = str(RAW_DIR / "trades" / "**" / "*.parquet")
TAB_DIR = Path(__file__).parent / "tables_v2"
TAB_DIR.mkdir(exist_ok=True)

LATE_WINDOW_HOURS = 48
MIN_TRADES = 20
SEED = 42


def build_combined_with_price(end_boundary: str, start_boundary: str = None):
    """
    end_boundary: upper bound on trade timestamp.
    start_boundary: optional lower bound, used to isolate a specific
    confirmatory window rather than all data up to end_boundary.
    """
    print(f"  Scanning trades (streaming)... "
          f"[{start_boundary or 'start'} , {end_boundary}]")
    lf = (
        pl.scan_parquet(TRADES_GLOB)
        .filter(pl.col("timestamp") <= pl.lit(end_boundary)
                .str.to_datetime(time_zone="UTC"))
    )
    if start_boundary:
        lf = lf.filter(
            pl.col("timestamp") >= pl.lit(start_boundary)
                .str.to_datetime(time_zone="UTC")
        )
    lf = lf.select(["prediction_id", "event_id", "timestamp", "price",
                     "quantity", "taker_bought", "winner"])

    close_info = (
        lf.group_by("prediction_id")
        .agg([
            pl.col("timestamp").max().alias("last_ts"),
            pl.len().alias("n_trades"),
            pl.col("event_id").first().alias("event_id"),
            pl.col("winner").drop_nulls().first().alias("win_bool"),
        ])
        .filter(pl.col("n_trades") >= MIN_TRADES)
        .filter(pl.col("win_bool").is_not_null())
        .with_columns(pl.col("win_bool").cast(pl.Int32).alias("win"))
        .collect(engine="streaming")
    )
    print(f"  {len(close_info):,} resolved predictions with "
          f">= {MIN_TRADES} trades and a known winner")

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

    late = (
        trades.filter(pl.col("hours_to_close") <= LATE_WINDOW_HOURS)
        .group_by("prediction_id")
        .agg([
            pl.col("signed_usdc").sum().alias("late_net_flow"),
            pl.col("usdc").sum().alias("late_total_flow"),
            pl.len().alias("late_n"),
        ])
    )
    pre = (
        trades.filter(pl.col("hours_to_close") > LATE_WINDOW_HOURS)
        .group_by("prediction_id")
        .agg(pl.len().alias("pre_n"))
    )

    pre_window_price = (
        trades.filter(pl.col("hours_to_close") > LATE_WINDOW_HOURS)
        .sort("hours_to_close")
        .group_by("prediction_id")
        .agg(pl.col("price").first().alias("pre_window_price"))
    )

    combined = (
        late.join(pre, on="prediction_id", how="inner")
        .join(pre_window_price, on="prediction_id", how="inner")
        .join(close_info.lazy().select(["prediction_id", "event_id", "win"]),
              on="prediction_id", how="inner")
        .filter((pl.col("late_n") >= 5) & (pl.col("pre_n") >= 5))
        .filter(pl.col("late_total_flow") > 0)
        .with_columns(
            (pl.col("late_net_flow") / pl.col("late_total_flow"))
                .alias("late_imbalance")
        )
        .select(["prediction_id", "event_id", "late_imbalance",
                  "pre_window_price", "win"])
        .collect(engine="streaming")
        .to_pandas()
    )

    before = len(combined)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["late_imbalance", "pre_window_price", "win", "event_id"])
    combined["win"] = combined["win"].astype(int)
    dropped = before - len(combined)
    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with missing price/imbalance/outcome")
    return combined


def run_price_conditioned_test(combined: pd.DataFrame, tag: str) -> dict:
    n_events = combined["event_id"].nunique()
    print(f"\n  ── {tag} ──")
    print(f"  {len(combined):,} predictions across {n_events:,} events")
    print(f"  Overall win rate: {combined['win'].mean():.3f}")

    if len(combined) < 200 or combined["win"].nunique() < 2:
        print("  Insufficient data. Skipping.")
        return {}

    X_price = sm.add_constant(combined["pre_window_price"])
    m_price = sm.Logit(combined["win"], X_price).fit(disp=0)
    pred_price = m_price.predict(X_price)
    auc_price = roc_auc_score(combined["win"], pred_price)

    print(f"\n  Model A (price alone):")
    print(f"    AUC = {auc_price:.4f}")
    print(f"    price coef = {m_price.params['pre_window_price']:+.4f}  "
          f"p={m_price.pvalues['pre_window_price']:.2e}")

    X_both = sm.add_constant(combined[["pre_window_price", "late_imbalance"]])
    m_both = sm.Logit(combined["win"], X_both).fit(
        cov_type="cluster", cov_kwds={"groups": combined["event_id"]}, disp=0
    )
    pred_both = m_both.predict(X_both)
    auc_both = roc_auc_score(combined["win"], pred_both)

    print(f"\n  Model B (price + imbalance, event-clustered SEs):")
    print(f"    AUC = {auc_both:.4f}")
    print(f"    price coef     = {m_both.params['pre_window_price']:+.4f}  "
          f"p={m_both.pvalues['pre_window_price']:.2e}")
    print(f"    imbalance coef = {m_both.params['late_imbalance']:+.4f}  "
          f"p={m_both.pvalues['late_imbalance']:.2e}")

    marginal_auc = auc_both - auc_price
    print(f"\n  Marginal AUC gain from adding imbalance: {marginal_auc:+.4f}")
    print(f"  (Note: AUC deltas are insensitive once the baseline AUC is "
          f"already high -- see LR test and pseudo-R^2 below.)")

    m_price_nocluster = sm.Logit(combined["win"], X_price).fit(disp=0)
    lr_stat = 2 * (m_both.llf - m_price_nocluster.llf)
    lr_p = 1 - chi2.cdf(lr_stat, df=1)

    print(f"\n  Likelihood-ratio test (imbalance adds value beyond price):")
    print(f"    LR statistic = {lr_stat:.3f}  (df=1)")
    print(f"    p-value      = {lr_p:.4f}")

    null_X = pd.DataFrame({"const": np.ones(len(combined))})
    llnull = sm.Logit(combined["win"], null_X).fit(disp=0).llf

    r2_price = 1 - (m_price_nocluster.llf / llnull)
    r2_both  = 1 - (m_both.llf / llnull)
    incremental_r2 = r2_both - r2_price

    print(f"\n  McFadden pseudo-R^2, price only:        {r2_price:.4f}")
    print(f"  McFadden pseudo-R^2, price + imbalance: {r2_both:.4f}")
    print(f"  Incremental pseudo-R^2 from imbalance:  {incremental_r2:.4f}")

    print(f"\n  {'='*60}")
    print(f"  INTERPRETATION:")
    if lr_p < 0.05 and marginal_auc < 0.01:
        print(f"  The LR test is decisive (p={lr_p:.4f}) even though the")
        print(f"  AUC gain is small ({marginal_auc:+.4f}). This is expected")
        print(f"  when baseline AUC is already high (ceiling effect) and")
        print(f"  does NOT mean the effect is spurious. Late order-flow")
        print(f"  imbalance carries statistically decisive incremental")
        print(f"  information about the outcome beyond concurrent price,")
        print(f"  but the effect is economically MODEST (incremental")
        print(f"  pseudo-R^2 = {incremental_r2:.4f}). Report as: 'imbalance")
        print(f"  adds small but statistically robust incremental")
        print(f"  predictive content beyond price'.")
    elif lr_p >= 0.05:
        print(f"  LR test is not significant (p={lr_p:.4f}): imbalance does")
        print(f"  NOT add value beyond price. Unsupported once price is")
        print(f"  controlled for.")
    else:
        print(f"  Both AUC and LR test show meaningful improvement.")
    print(f"  {'='*60}")

    return {
        "tag": tag, "n": len(combined), "n_events": n_events,
        "auc_price_only": auc_price,
        "auc_price_plus_imbalance": auc_both,
        "marginal_auc": marginal_auc,
        "imbalance_coef_controlled": m_both.params["late_imbalance"],
        "imbalance_p_controlled": m_both.pvalues["late_imbalance"],
        "lr_statistic": lr_stat, "lr_p_value": lr_p,
        "pseudo_r2_price_only": r2_price,
        "pseudo_r2_both": r2_both,
        "incremental_pseudo_r2": incremental_r2,
    }


def bootstrap_incremental_r2_ci(combined: pd.DataFrame, n_boot: int = 500,
                                 seed: int = SEED) -> tuple:
    """Event-clustered bootstrap CI on the incremental pseudo-R^2."""
    rng = np.random.RandomState(seed)
    event_groups = combined.groupby("event_id").indices
    event_ids = np.array(list(event_groups.keys()))

    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(event_ids, size=len(event_ids), replace=True)
        idx = np.concatenate([event_groups[e] for e in sampled])
        bd = combined.iloc[idx]
        if bd["win"].nunique() < 2:
            continue
        try:
            X_p = sm.add_constant(bd["pre_window_price"])
            X_b = sm.add_constant(bd[["pre_window_price", "late_imbalance"]])
            null_X = pd.DataFrame({"const": np.ones(len(bd))})
            llnull = sm.Logit(bd["win"].values, null_X).fit(disp=0).llf
            m_p = sm.Logit(bd["win"].values, X_p).fit(disp=0)
            m_b = sm.Logit(bd["win"].values, X_b).fit(disp=0)
            r2_p = 1 - (m_p.llf / llnull)
            r2_b = 1 - (m_b.llf / llnull)
            vals.append(r2_b - r2_p)
        except Exception:
            continue

    if len(vals) < 30:
        return np.nan, np.nan, len(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi, len(vals)


def trend_test_across_windows(estimates: list, ns: list) -> dict:
    """Weighted linear trend test on the three sequential estimates."""
    x = np.array([1, 2, 3])
    y = np.array(estimates)
    w = np.array(ns) / np.sum(ns)

    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)
    denom = np.sum(w * (x - x_mean) ** 2)
    slope = np.sum(w * (x - x_mean) * (y - y_mean)) / denom if denom > 0 else np.nan
    intercept = y_mean - slope * x_mean

    resid = y - (intercept + slope * x)
    n = len(x)
    dof = n - 2
    if dof <= 0 or denom <= 0:
        return {"slope": slope, "t_stat": np.nan, "p_value": np.nan, "dof": dof}
    mse = np.sum(w * resid ** 2) / dof
    se_slope = np.sqrt(mse / denom) if mse > 0 else np.nan
    t_stat = slope / se_slope if se_slope and se_slope > 0 else np.nan
    p_value = (2 * (1 - t_dist.cdf(abs(t_stat), dof))
               if not np.isnan(t_stat) else np.nan)

    return {"slope": slope, "intercept": intercept,
            "t_stat": t_stat, "p_value": p_value, "dof": dof}


def run():
    print("=" * 65)
    print("  H5 PRICE-CONDITIONED TEST (referee fix, corrected interpretation)")
    print("=" * 65)

    results = []

    combined_dev = build_combined_with_price(end_boundary="2025-06-30 23:59:59")
    r = run_price_conditioned_test(combined_dev, "Development sample "
                                                  "(through 2025-06-30)")
    if r:
        results.append(r)

    combined_oos1 = build_combined_with_price(
        start_boundary="2025-06-30 23:59:59",
        end_boundary="2025-12-31 23:59:59",
    )
    r = run_price_conditioned_test(combined_oos1, "Confirmatory window 1 "
                                                   "(2025-H2)")
    if r:
        results.append(r)

    combined_oos2 = build_combined_with_price(
        start_boundary="2025-12-31 23:59:59",
        end_boundary="2026-03-29 23:59:59",
    )
    r = run_price_conditioned_test(combined_oos2, "Confirmatory window 2 "
                                                   "(2026-Q1)")
    if r:
        results.append(r)

    out = pd.DataFrame(results)
    out.to_csv(TAB_DIR / "t25_h5_price_conditioned.csv", index=False)
    print(f"\n  ✓ saved -> {TAB_DIR / 't25_h5_price_conditioned.csv'}")

    if len(results) >= 2:
        print("\n" + "=" * 65)
        print("  CROSS-WINDOW COMPARISON")
        print("=" * 65)
        for r in results:
            print(f"  {r['tag']:<40} incr. pseudo-R2={r['incremental_pseudo_r2']:.4f}  "
                  f"imbalance coef={r['imbalance_coef_controlled']:+.4f}  "
                  f"LR p={r['lr_p_value']:.4f}")

    if len(results) == 3:
        print("\n" + "=" * 65)
        print("  BOOTSTRAP CIs AND MONOTONIC-TREND TEST")
        print("=" * 65)

        all_combined = [combined_dev, combined_oos1, combined_oos2]
        ci_rows = []
        for r, cdf in zip(results, all_combined):
            lo, hi, n_valid = bootstrap_incremental_r2_ci(cdf)
            print(f"  {r['tag']:<40} incr. R2={r['incremental_pseudo_r2']:.4f}  "
                  f"95% CI (event-cluster bootstrap, n={n_valid}): "
                  f"[{lo:.4f}, {hi:.4f}]")
            ci_rows.append({**r, "ci_lo": lo, "ci_hi": hi})

        trend = trend_test_across_windows(
            [r["incremental_pseudo_r2"] for r in results],
            [r["n"] for r in results],
        )
        print(f"\n  Weighted linear trend across 3 windows:")
        print(f"    slope     = {trend['slope']:+.5f} per window")
        print(f"    t-stat    = {trend['t_stat']:.3f}  (df={trend['dof']})")
        print(f"    p-value   = {trend['p_value']:.4f}")
        print(f"    (Note: df=1 with only 3 points -- this is a directional")
        print(f"    check, not a definitive test; report descriptively")
        print(f"    alongside the raw sequence, not as a precise p-value.)")

        out_full = pd.DataFrame(ci_rows)
        out_full.to_csv(TAB_DIR / "t25_h5_price_conditioned_with_ci.csv",
                        index=False)
        print(f"\n  ✓ saved -> {TAB_DIR / 't25_h5_price_conditioned_with_ci.csv'}")


if __name__ == "__main__":
    run()