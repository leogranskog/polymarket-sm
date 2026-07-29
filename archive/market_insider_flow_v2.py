"""
Extension 6, v2 — Market-level insider order-flow, validated against
ACTUAL resolution outcomes, with event-clustered inference.

Fix vs first attempt: `winner` is already a boolean per prediction_id
(True = this token/outcome won). No comparison against `outcome` (which
is null throughout this dataset) is needed or correct.

Improvements over v1 (price-move only):
  (a) Outcome-based test: does late order-flow imbalance predict which
      outcome ACTUALLY WON (ground truth), not just a subsequent price
      print (which can partly reflect mechanical price impact).
  (b) Event-clustered inference: many predictions share an underlying
      event, so naive per-prediction p-values overstate significance.
      Reported via: logistic regression with SEs clustered by event_id,
      plus a block bootstrap (resampling whole EVENTS) for the AUC CI.

Excludes TRUE OOS (predictions with last trade after 2025-06-30).

Usage: python -m research.market_insider_flow_v2
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score
from config import RAW_DIR, PROC_DIR

TRADES_GLOB = str(RAW_DIR / "trades" / "**" / "*.parquet")
TAB_DIR = Path(__file__).parent / "tables_v2"
FIG_DIR = Path(__file__).parent / "figures_v2"
TAB_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

OOS_BOUNDARY = "2025-06-30 23:59:59"
LATE_WINDOW_HOURS = 48
MIN_TRADES = 20
SEED = 42


def build_combined() -> pd.DataFrame:
    print("  Scanning trades (streaming)...")
    lf = (
        pl.scan_parquet(TRADES_GLOB)
        .filter(pl.col("timestamp") <= pl.lit(OOS_BOUNDARY)
                .str.to_datetime(time_zone="UTC"))
        .select(["prediction_id", "event_id", "timestamp", "price",
                 "quantity", "taker_bought", "winner"])
    )

    # winner is a bool per prediction_id (True = this token won).
    # Take the first non-null observed value per token.
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
    print(f"  win-rate in this sample: "
          f"{close_info['win'].mean():.3f} "
          f"(should be well away from 0 or 1)")

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

    combined = (
        late.join(pre, on="prediction_id", how="inner")
        .join(close_info.lazy().select(["prediction_id", "event_id", "win"]),
              on="prediction_id", how="inner")
        .filter((pl.col("late_n") >= 5) & (pl.col("pre_n") >= 5))
        .filter(pl.col("late_total_flow") > 0)
        .with_columns(
            (pl.col("late_net_flow") / pl.col("late_total_flow"))
                .alias("late_imbalance")
        )
        .select(["prediction_id", "event_id", "late_imbalance", "win"])
        .collect(engine="streaming")
        .to_pandas()
    )

    before = len(combined)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["late_imbalance", "win", "event_id"])
    combined["win"] = combined["win"].astype(int)
    dropped = before - len(combined)
    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with residual NaN/inf")
    return combined


def cluster_bootstrap_ci(df, metric_fn, n_boot=1000, seed=SEED):
    """Block bootstrap: resample whole EVENTS with replacement (fast)."""
    rng = np.random.RandomState(seed)
    # Pre-group once: dict of event_id -> row indices (fast lookup)
    groups = df.groupby("event_id").indices  # dict: event_id -> np.array of positions
    event_ids = np.array(list(groups.keys()))
    n_events = len(event_ids)

    df_reset = df.reset_index(drop=True)
    vals = []
    for _ in range(n_boot):
        sampled_events = rng.choice(event_ids, size=n_events, replace=True)
        idx = np.concatenate([groups[e] for e in sampled_events])
        boot_df = df_reset.iloc[idx]
        if boot_df["win"].nunique() < 2:
            continue
        try:
            vals.append(metric_fn(boot_df))
        except Exception:
            pass
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5), len(vals)


def run():
    print("=" * 65)
    print("  EXTENSION 6 v2 — Market insider flow, outcome-validated,")
    print("  event-clustered inference")
    print("=" * 65)
    print(f"  Late window = last {LATE_WINDOW_HOURS}h before resolution")

    combined = build_combined()
    n_events = combined["event_id"].nunique()
    print(f"\n  {len(combined):,} predictions across {n_events:,} events "
          f"(mean {len(combined)/max(n_events,1):.1f} predictions/event)")
    print(f"  Overall win rate: {combined['win'].mean():.3f}")

    if len(combined) < 200 or combined["win"].nunique() < 2:
        print("  Not enough data / degenerate outcome — aborting.")
        return

    # ── Test A: does late imbalance predict which outcome WON? ──────────
    auc = roc_auc_score(combined["win"], combined["late_imbalance"])
    rho, p_naive = pointbiserialr(combined["win"], combined["late_imbalance"])
    print(f"\n  (A) Late imbalance -> ACTUAL winning outcome:")
    print(f"      AUC = {auc:.4f}   point-biserial r = {rho:.4f} "
          f"(naive p={p_naive:.2e}, ignores clustering)")

    lo_auc, hi_auc, n_valid_boot = cluster_bootstrap_ci(
        combined,
        lambda d: roc_auc_score(d["win"], d["late_imbalance"]),
        n_boot=500,
    )
    print(f"      Event-cluster bootstrap 95% CI for AUC "
          f"({n_valid_boot} valid resamples): "
          f"[{lo_auc:.4f}, {hi_auc:.4f}]")

    # ── Test B: logistic regression with event-clustered SEs ────────────
    import statsmodels.api as sm
    X = sm.add_constant(combined["late_imbalance"])
    y = combined["win"]
    try:
        logit = sm.Logit(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": combined["event_id"]},
            disp=0,
        )
        print(f"\n  (B) Logistic regression, win ~ late_imbalance "
              f"(event-clustered SEs):")
        print(logit.summary().tables[1])
        coef = logit.params["late_imbalance"]
        pval_clustered = logit.pvalues["late_imbalance"]
        with open(TAB_DIR / "t18_insider_logit_clustered.tex", "w") as f:
            f.write(logit.summary().as_latex())
    except Exception as e:
        print(f"  Logit failed: {e}")
        coef, pval_clustered = np.nan, np.nan

    # ── Decile table: P(win) by late-imbalance decile ────────────────────
    combined["decile"] = pd.qcut(
        combined["late_imbalance"].rank(method="first"), 10, labels=False
    ) + 1
    g = combined.groupby("decile")["win"].agg(["mean", "count"]).reset_index()
    print(f"\n  P(win) by late-imbalance decile:")
    print(g.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    summary = pd.DataFrame([{
        "n_predictions": len(combined),
        "n_events": n_events,
        "win_rate": combined["win"].mean(),
        "auc": auc,
        "auc_ci_lo": lo_auc, "auc_ci_hi": hi_auc,
        "point_biserial_r": rho,
        "naive_p_uncorrected": p_naive,
        "logit_coef_late_imbalance": coef,
        "logit_p_clustered": pval_clustered,
        "late_window_hours": LATE_WINDOW_HOURS,
    }])
    summary.to_csv(TAB_DIR / "t18_market_insider_outcome.csv", index=False)
    g.to_csv(TAB_DIR / "t18b_insider_outcome_deciles.csv", index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(g)))
    ax.bar(g["decile"], g["mean"] * 100, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(50, color="k", lw=0.8, ls="--", label="50% (uninformative)")
    ax.set_xlabel(f"Late ({LATE_WINDOW_HOURS}h) order-flow imbalance decile")
    ax.set_ylabel("P(this outcome actually won), %")
    ax.set_title(
        f"Late order flow predicts the resolution outcome\n"
        f"AUC={auc:.3f}  [{lo_auc:.3f},{hi_auc:.3f}] "
        f"(event-clustered bootstrap CI)"
    )
    ax.legend()
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(FIG_DIR / f"f10_insider_outcome{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()

    print(f"\n  ✓ -> {TAB_DIR / 't18_market_insider_outcome.csv'}")
    print(f"  ✓ -> {FIG_DIR / 'f10_insider_outcome.pdf'}")
    print(f"\n  Interpretation: AUC well above 0.5 with a CI excluding 0.5, "
          f"even after event clustering, means late order-flow imbalance "
          f"predicts the ACTUAL winner, not just a mechanical subsequent "
          f"price tick. This is direct evidence of informed trading in "
          f"the final hours before resolution.")


if __name__ == "__main__":
    run()