"""
Table 1 (descriptive statistics by panel window) + covariate-shift
appendix (train vs test feature distributions).

Standard exhibits expected in any empirical paper's data section.
Excludes TRUE OOS (2025-H2) from the covariate-shift comparison, since
that panel must remain untouched until the final single look.

Usage: python -m research.descriptives
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp
from config import PROC_DIR

PIT_DIR = PROC_DIR / "pit"
TAB_DIR = Path(__file__).parent / "tables_v2"
FIG_DIR = Path(__file__).parent / "figures_v2"
TAB_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# window, feature cutoff, label horizon end, role
WINDOWS = [
    ("H2-2023", "2023-06-30", "2023-12-31", "train"),
    ("Q4'23-Q1'24", "2023-09-30", "2024-03-31", "train"),
    ("H1-2024", "2023-12-31", "2024-06-30", "train"),
    ("H2-2024", "2024-06-30", "2024-12-31", "validation"),
    ("H1-2025", "2024-12-31", "2025-06-30", "test"),
]

KEY_FEATURES = [
    "n_trades", "total_volume", "avg_price", "frac_maker",
    "frac_longshot", "category_hhi", "counterparty_hhi",
    "frac_late_entry", "frac_early_entry", "past_clv_vw",
]


def load(cutoff, horizon):
    f = pl.read_parquet(PIT_DIR / f"features_asof_{cutoff}.parquet")
    l = pl.read_parquet(PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet")
    return f.join(l, on="wallet", how="inner")


def table1_descriptives():
    print("  Building Table 1 (descriptive statistics by window)...")
    rows = []
    for name, cutoff, horizon, role in WINDOWS:
        df = load(cutoff, horizon).to_pandas()
        rows.append({
            "window": name, "role": role,
            "n_wallets": len(df),
            "mean_n_trades": df["n_trades"].mean(),
            "median_n_trades": df["n_trades"].median(),
            "mean_volume_usdc": df["total_volume"].mean(),
            "median_volume_usdc": df["total_volume"].median(),
            "mean_category_hhi": df["category_hhi"].mean(),
            "frac_maker_mean": df["frac_maker"].mean(),
            "mean_fwd_clv": df["fwd_clv_vw"].mean(),
            "median_fwd_clv": df["fwd_clv_vw"].median(),
            "std_fwd_clv": df["fwd_clv_vw"].std(),
            "q75_fwd_clv_threshold": df["fwd_clv_vw"].quantile(0.75),
        })
        print(f"    {name:<14} n={len(df):>7,}  "
              f"mean_trades={df['n_trades'].mean():.1f}  "
              f"mean_fwd_clv={df['fwd_clv_vw'].mean():+.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "t0_descriptives.csv", index=False)
    with open(TAB_DIR / "t0_descriptives.tex", "w") as f:
        f.write(out.to_latex(index=False, float_format="%.4f",
                caption="Descriptive statistics by panel window",
                label="tab:descriptives"))
    print(f"  ✓ -> {TAB_DIR / 't0_descriptives.csv'}")
    return out


def covariate_shift():
    """
    KS statistic per feature between the pooled TRAINING panel and the
    TEST panel (2025-H1). Large KS + significant p => substantial
    population drift on that feature -- expected given the platform's
    growth, but should be disclosed rather than assumed away.
    """
    print("\n  Building covariate-shift table (train vs test)...")

    train_frames = []
    for name, cutoff, horizon, role in WINDOWS:
        if role == "train":
            train_frames.append(load(cutoff, horizon).to_pandas())
    train_df = pd.concat(train_frames, ignore_index=True)

    test_cutoff, test_horizon = "2024-12-31", "2025-06-30"
    test_df = load(test_cutoff, test_horizon).to_pandas()

    print(f"    pooled train n={len(train_df):,}  test n={len(test_df):,}")

    rows = []
    for feat in KEY_FEATURES:
        a = pd.to_numeric(train_df[feat], errors="coerce").dropna()
        b = pd.to_numeric(test_df[feat], errors="coerce").dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        stat, pval = ks_2samp(a, b)
        rows.append({
            "feature": feat,
            "train_mean": a.mean(), "test_mean": b.mean(),
            "train_median": a.median(), "test_median": b.median(),
            "ks_statistic": stat, "ks_pvalue": pval,
        })
        print(f"    {feat:<20} KS={stat:.4f}  p={pval:.2e}  "
              f"train_mean={a.mean():+.4f}  test_mean={b.mean():+.4f}")

    out = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)
    out.to_csv(TAB_DIR / "t_appendix_covariate_shift.csv", index=False)
    with open(TAB_DIR / "t_appendix_covariate_shift.tex", "w") as f:
        f.write(out.to_latex(index=False, float_format="%.4f",
                caption="Covariate shift: train vs test feature "
                        "distributions (Kolmogorov-Smirnov)",
                label="tab:covshift"))
    print(f"\n  Interpretation: large KS statistics here are EXPECTED "
          f"given the ~100x growth in active wallets between the 2023 "
          f"training era and 2025; this table exists to make that shift "
          f"explicit and quantified rather than silently assumed away, "
          f"per Result 2's cohort-matched robustness check.")
    print(f"  ✓ -> {TAB_DIR / 't_appendix_covariate_shift.csv'}")

    # Figure: side-by-side KS bar chart
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(out["feature"], out["ks_statistic"], color="steelblue")
    ax.set_xlabel("KS statistic (train vs test)")
    ax.set_title("Covariate shift by feature: pooled train vs test "
                 "(2025-H1)")
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(FIG_DIR / f"f_appendix_covshift{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()
    print(f"  ✓ -> {FIG_DIR / 'f_appendix_covshift.pdf'}")

    return out


def run():
    print("=" * 65)
    print("  TABLE 1 + COVARIATE SHIFT APPENDIX")
    print("=" * 65)
    table1_descriptives()
    covariate_shift()
    print("\n✅ Descriptives complete.")


if __name__ == "__main__":
    run()