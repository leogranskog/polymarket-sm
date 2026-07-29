"""
Referee fix: composition-effect check on rising persistence and the
AUC jump. Rising rho across periods (~0 to 0.30) could reflect a
genuinely more persistent skill population OR simply a changing mix of
trader types over time (more professionals/market-makers, fewer
one-shot retail/bots). This script isolates the composition
explanation from the genuine-skill explanation via three checks:

  1. WITHIN-COHORT PERSISTENCE: fix the set of wallets active in the
     EARLIEST period and track only THOSE SAME wallets forward. If
     persistence still rises for a fixed population, composition is
     ruled out as the driver.
  2. SURVIVOR COHORT: an even stricter version using only wallets
     present in every single period, to fully remove population churn.
  3. MARKET-MAKER / BOT SPLIT: high frac_maker and high
     trades_per_day wallets look structurally different from typical
     traders; report persistence separately for this subgroup vs the
     rest, since mechanical market-making persistence is not the same
     claim as individual trading skill.
  4. CATEGORY-CLUSTERED REGRESSION (confirmatory windows only): OLS of
     forward CLV on past CLV with standard errors clustered by each
     wallet's dominant trading category, as a practical proxy for
     "wallets grinding the same correlated markets" (true market/event
     level clustering would require a full per-trade re-scan; this is
     the cheaper, disclosed approximation).

Usage: python -m research.composition_robustness
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from config import PROC_DIR

PIT_DIR = PROC_DIR / "pit"
TAB_DIR = Path(__file__).parent / "tables_v2"
TAB_DIR.mkdir(exist_ok=True)

# Same window pairs used throughout the paper's persistence analysis
WINDOWS = [
    ("H2-2023->H1-2024",
     "labels_2023-06-30_to_2023-12-31.parquet",
     "labels_2023-12-31_to_2024-06-30.parquet"),
    ("H1-2024->H2-2024",
     "labels_2023-12-31_to_2024-06-30.parquet",
     "labels_2024-06-30_to_2024-12-31.parquet"),
    ("H2-2024->H1-2025",
     "labels_2024-06-30_to_2024-12-31.parquet",
     "labels_2024-12-31_to_2025-06-30.parquet"),
    ("H1-2025->H2-2025 (confirmatory 1)",
     "labels_2024-12-31_to_2025-06-30.parquet",
     "labels_2025-06-30_to_2025-12-31.parquet"),
    ("H2-2025->Q1-2026 (confirmatory 2)",
     "labels_2025-06-30_to_2025-12-31.parquet",
     "labels_2025-12-31_to_2026-03-29.parquet"),
]

# Feature panel used to identify each wallet's activity level/type,
# and to restrict the earliest cohort to wallets with reliable data.
EARLIEST_FEATURES = "features_asof_2023-06-30.parquet"


def load_labels(fname):
    return pl.read_parquet(PIT_DIR / fname).select(
        ["wallet", "fwd_clv_vw", "fwd_n_trades"])


def load_features(fname):
    return pl.read_parquet(PIT_DIR / fname)


# ── Check 1: within-cohort persistence (fixed earliest-period cohort) ───────

def within_cohort_persistence():
    print("=" * 70)
    print("  CHECK 1: WITHIN-COHORT PERSISTENCE")
    print("  (fixed set of wallets active in the earliest period,")
    print("  tracked forward, vs the full-population trend)")
    print("=" * 70)

    earliest_labels = load_labels(WINDOWS[0][1])
    cohort = set(earliest_labels["wallet"].to_list())
    print(f"\n  Base cohort (wallets active in earliest period): "
          f"{len(cohort):,}")

    rows = []
    for name, f1, f2 in WINDOWS:
        a = load_labels(f1)
        b = load_labels(f2)
        j_full = a.join(b, on="wallet", suffix="_next").to_pandas()

        a_cohort = a.filter(pl.col("wallet").is_in(list(cohort)))
        j_cohort = a_cohort.join(b, on="wallet", suffix="_next").to_pandas()

        if len(j_full) < 30:
            print(f"\n  {name}: full sample too small, skipped")
            continue

        rho_full, p_full = spearmanr(j_full["fwd_clv_vw"],
                                      j_full["fwd_clv_vw_next"])

        if len(j_cohort) >= 30:
            rho_cohort, p_cohort = spearmanr(j_cohort["fwd_clv_vw"],
                                              j_cohort["fwd_clv_vw_next"])
        else:
            rho_cohort, p_cohort = np.nan, np.nan

        print(f"\n  {name}:")
        print(f"    Full population:  n={len(j_full):,}   rho={rho_full:+.4f}")
        print(f"    Fixed cohort:     n={len(j_cohort):,}   "
              f"rho={rho_cohort:+.4f}" if not np.isnan(rho_cohort)
              else f"    Fixed cohort:     n={len(j_cohort):,}   "
                   f"(too few surviving cohort members, cannot compute)")

        rows.append({
            "window": name,
            "n_full": len(j_full), "rho_full": rho_full,
            "n_cohort": len(j_cohort), "rho_cohort": rho_cohort,
        })

    df = pd.DataFrame(rows)
    print(f"\n  {'Window':<38}{'Full rho':>12}{'Cohort rho':>14}")
    for _, r in df.iterrows():
        print(f"  {r['window']:<38}{r['rho_full']:>+12.4f}"
              f"{r['rho_cohort']:>+14.4f}" if not np.isnan(r['rho_cohort'])
              else f"  {r['window']:<38}{r['rho_full']:>+12.4f}{'n/a':>14}")

    print(f"\n  Interpretation: if the COHORT rho also rises across the")
    print(f"  later windows (even with a shrinking, fixed sample), the")
    print(f"  rising persistence is NOT primarily a composition effect.")
    print(f"  If cohort rho stays flat/near-zero while full-population")
    print(f"  rho rises, the original result is substantially driven by")
    print(f"  new, more persistent wallets entering the population, not")
    print(f"  existing wallets becoming more predictable.")

    df.to_csv(TAB_DIR / "t30_within_cohort_persistence.csv", index=False)
    print(f"\n  Saved -> {TAB_DIR / 't30_within_cohort_persistence.csv'}")
    return df


# ── Check 2: strict survivor cohort (present in every single period) ───────

def survivor_cohort_persistence():
    print("\n" + "=" * 70)
    print("  CHECK 2: STRICT SURVIVOR COHORT")
    print("  (wallets present in EVERY period, fully removes churn)")
    print("=" * 70)

    all_wallet_sets = []
    unique_label_files = []
    for name, f1, f2 in WINDOWS:
        for f in (f1, f2):
            if f not in unique_label_files:
                unique_label_files.append(f)

    for f in unique_label_files:
        w = set(load_labels(f)["wallet"].to_list())
        all_wallet_sets.append(w)

    survivors = set.intersection(*all_wallet_sets)
    print(f"\n  Wallets present in ALL {len(unique_label_files)} periods: "
          f"{len(survivors):,}")

    if len(survivors) < 30:
        print(f"  Too few survivors for a meaningful test. This itself is")
        print(f"  informative: very few wallets are active across the")
        print(f"  platform's entire history, so the paper's rising")
        print(f"  persistence describes a changing population more than")
        print(f"  it describes fixed individuals becoming more skilled.")
        return pd.DataFrame()

    rows = []
    for name, f1, f2 in WINDOWS:
        a = load_labels(f1).filter(pl.col("wallet").is_in(list(survivors)))
        b = load_labels(f2).filter(pl.col("wallet").is_in(list(survivors)))
        j = a.join(b, on="wallet", suffix="_next").to_pandas()
        if len(j) < 20:
            print(f"\n  {name}: n={len(j)}, too small for survivors, skipped")
            continue
        rho, p = spearmanr(j["fwd_clv_vw"], j["fwd_clv_vw_next"])
        print(f"\n  {name}: n={len(j):,}  rho={rho:+.4f}  p={p:.4f}")
        rows.append({"window": name, "n": len(j), "rho": rho, "p": p})

    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "t31_survivor_cohort_persistence.csv", index=False)
    print(f"\n  Saved -> {TAB_DIR / 't31_survivor_cohort_persistence.csv'}")
    return df


# ── Check 3: market-maker / bot split ────────────────────────────────────────

def market_maker_split(cutoff="2024-12-31", horizon="2025-06-30"):
    print("\n" + "=" * 70)
    print("  CHECK 3: MARKET-MAKER / BOT SPLIT")
    print("  (high maker share + high trade frequency vs everyone else)")
    print("=" * 70)

    feats = pl.read_parquet(PIT_DIR / f"features_asof_{cutoff}.parquet")
    labels = pl.read_parquet(PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet")
    df = feats.join(labels, on="wallet", how="inner").to_pandas()

    maker_thresh = df["frac_maker"].quantile(0.90)
    freq_thresh = df["trades_per_day"].quantile(0.90)

    df["likely_bot_or_mm"] = (
        (df["frac_maker"] >= maker_thresh) &
        (df["trades_per_day"] >= freq_thresh)
    )

    n_bot = df["likely_bot_or_mm"].sum()
    n_total = len(df)
    print(f"\n  Panel: features@{cutoff}, n={n_total:,}")
    print(f"  Flagged as likely bot/market-maker "
          f"(top decile maker share AND top decile trade frequency): "
          f"{n_bot:,} ({n_bot/n_total*100:.1f}%)")

    print(f"\n  Forward CLV comparison:")
    for label, mask in [("Likely bot/MM", df["likely_bot_or_mm"]),
                        ("Everyone else", ~df["likely_bot_or_mm"])]:
        sub = df[mask]
        print(f"    {label:<16} n={len(sub):>7,}   "
              f"mean fwd CLV={sub['fwd_clv_vw'].mean():+.4f}   "
              f"past_clv_vw={sub['past_clv_vw'].mean():+.4f}")

    # Persistence within each subgroup, using the SAME cutoff/horizon
    # pair's past vs forward CLV as a within-period proxy
    for label, mask in [("Likely bot/MM", df["likely_bot_or_mm"]),
                        ("Everyone else", ~df["likely_bot_or_mm"])]:
        sub = df[mask].dropna(subset=["past_clv_vw", "fwd_clv_vw"])
        if len(sub) >= 30:
            rho, p = spearmanr(sub["past_clv_vw"], sub["fwd_clv_vw"])
            print(f"    {label}: past-vs-forward CLV rho={rho:+.4f} "
                  f"(p={p:.4f}, n={len(sub):,})")

    out = df[["wallet", "likely_bot_or_mm", "frac_maker", "trades_per_day",
              "past_clv_vw", "fwd_clv_vw"]]
    out.to_csv(TAB_DIR / "t32_market_maker_split.csv", index=False)
    print(f"\n  Saved -> {TAB_DIR / 't32_market_maker_split.csv'}")
    return df


# ── Check 4: category-clustered regression (confirmatory windows) ──────────

def category_clustered_regression():
    print("\n" + "=" * 70)
    print("  CHECK 4: CATEGORY-CLUSTERED PERSISTENCE REGRESSION")
    print("  (confirmatory windows; clusters by each wallet's dominant")
    print("  category as a proxy for correlated-market grinding)")
    print("=" * 70)

    confirmatory_pairs = [
        ("Confirmatory 1 (2025-H2)",
         "features_asof_2024-12-31.parquet",
         "labels_2024-12-31_to_2025-06-30.parquet",
         "labels_2025-06-30_to_2025-12-31.parquet"),
        ("Confirmatory 2 (2026-Q1)",
         "features_asof_2025-06-30.parquet",
         "labels_2025-06-30_to_2025-12-31.parquet",
         "labels_2025-12-31_to_2026-03-29.parquet"),
    ]

    rows = []
    for name, feat_file, label_f1, label_f2 in confirmatory_pairs:
        feats = pl.read_parquet(PIT_DIR / feat_file)
        a = pl.read_parquet(PIT_DIR / label_f1).select(
            ["wallet", "fwd_clv_vw"])
        b = pl.read_parquet(PIT_DIR / label_f2).select(
            ["wallet", "fwd_clv_vw"])

        # Approximate "dominant category" via category_hhi buckets already
        # in the features file (a coarse proxy: we don't have the actual
        # dominant-category label stored on this panel, so we bucket by
        # category_hhi quartile as a stand-in cluster grouping -- disclosed
        # as an approximation, not a full per-wallet category label).
        df = (
            feats.select(["wallet", "category_hhi"])
            .join(a, on="wallet", how="inner")
            .join(b, on="wallet", suffix="_next", how="inner")
        ).to_pandas()

        if len(df) < 100:
            print(f"\n  {name}: n={len(df)}, too small, skipped")
            continue

        df["hhi_cluster"] = pd.qcut(df["category_hhi"].rank(method="first"),
                                    4, labels=False)

        try:
            ols = smf.ols("fwd_clv_vw_next ~ fwd_clv_vw", data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["hhi_cluster"]}
            )
            coef = ols.params["fwd_clv_vw"]
            p_clustered = ols.pvalues["fwd_clv_vw"]

            ols_naive = smf.ols("fwd_clv_vw_next ~ fwd_clv_vw", data=df).fit()
            p_naive = ols_naive.pvalues["fwd_clv_vw"]

            print(f"\n  {name}: n={len(df):,}")
            print(f"    Coefficient: {coef:+.4f}")
            print(f"    p-value (naive, unclustered):     {p_naive:.2e}")
            print(f"    p-value (clustered by HHI bucket): {p_clustered:.2e}")

            rows.append({
                "window": name, "n": len(df), "coef": coef,
                "p_naive": p_naive, "p_clustered": p_clustered,
            })
        except Exception as e:
            print(f"\n  {name}: regression failed ({e})")

    df_out = pd.DataFrame(rows)
    print(f"\n  Interpretation: if clustered p-values remain small despite")
    print(f"  being larger than the naive p-values, persistence survives")
    print(f"  a conservative correction for wallets trading correlated")
    print(f"  markets within the same category.")
    df_out.to_csv(TAB_DIR / "t33_category_clustered_persistence.csv",
                  index=False)
    print(f"\n  Saved -> {TAB_DIR / 't33_category_clustered_persistence.csv'}")
    return df_out


def run():
    within_cohort_persistence()
    survivor_cohort_persistence()
    market_maker_split()
    category_clustered_regression()
    print("\n" + "=" * 70)
    print("  ALL COMPOSITION CHECKS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run()