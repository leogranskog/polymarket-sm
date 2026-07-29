"""
Referee fix #5: H1's pass criterion (|rho| < 0.05, CI contains 0)
mechanically becomes impossible to pass as n grows, since the CI
shrinks toward the point estimate regardless of its true smallness.
Replace with a formal Two One-Sided Tests (TOST) equivalence test
against a pre-specified, justified equivalence bound, plus an economic
translation via decile portfolio sort of forward CLV.

TOST logic: test H0_a: rho <= -bound  vs  H0_b: rho >= +bound.
Reject BOTH null hypotheses => conclude rho is practically
equivalent to zero within +/- bound.

Equivalence bound justification: bound = 0.10, following the
convention in psychology/medicine of treating |r| < 0.10 as a
"negligible" effect size (Cohen 1988); disclosed explicitly as a
judgment call, not derived from the data.

Also runs a decile portfolio sort: rank wallets by PAST CLV, group
into deciles, report mean FORWARD CLV per decile.

Usage: python -m research.h1_equivalence_test
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm, spearmanr
from config import PROC_DIR

PIT_DIR = PROC_DIR / "pit"
TAB_DIR = Path(__file__).parent / "tables_v2"
TAB_DIR.mkdir(exist_ok=True)

EQUIVALENCE_BOUND = 0.10
SEED = 42

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


def fisher_z_ci(rho: float, n: int, alpha: float = 0.05) -> tuple:
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return lo, hi


def tost_equivalence(rho: float, n: int, bound: float = EQUIVALENCE_BOUND,
                      alpha: float = 0.05) -> dict:
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)

    z_lower_bound = np.arctanh(-bound)
    z_upper_bound = np.arctanh(bound)

    t1 = (z - z_lower_bound) / se
    p1 = 1 - norm.cdf(t1)

    t2 = (z - z_upper_bound) / se
    p2 = norm.cdf(t2)

    p_tost = max(p1, p2)
    equivalent = p_tost < alpha

    return {"p_tost": p_tost, "equivalent": equivalent,
            "p_lower": p1, "p_upper": p2}


def load_labels(fname):
    return pl.read_parquet(PIT_DIR / fname).select(
        ["wallet", "fwd_clv_vw", "fwd_n_trades"])


def run_persistence_equivalence():
    print("=" * 65)
    print("  H1 EQUIVALENCE TEST (TOST) -- replacing the n-dependent")
    print(f"  |rho|<0.05 criterion with a formal test against a")
    print(f"  pre-specified bound of +/-{EQUIVALENCE_BOUND}")
    print("=" * 65)

    rows = []
    for name, f1, f2 in WINDOWS:
        a = load_labels(f1)
        b = load_labels(f2)
        j = a.join(b, on="wallet", suffix="_next").to_pandas()
        if len(j) < 50:
            print(f"  {name}: n={len(j)}, too small, skipped")
            continue

        rho, p_naive = spearmanr(j["fwd_clv_vw"], j["fwd_clv_vw_next"])
        n = len(j)

        ci_lo, ci_hi = fisher_z_ci(rho, n)
        tost = tost_equivalence(rho, n)

        print(f"\n  {name}: n={n:,}")
        print(f"    rho = {rho:+.4f}  [Fisher-z 95% CI: {ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    TOST vs +/-{EQUIVALENCE_BOUND}: p={tost['p_tost']:.4f}  "
              f"-> {'EQUIVALENT to zero' if tost['equivalent'] else 'NOT equivalent'}")

        rows.append({
            "window": name, "n": n, "rho": rho,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "tost_p": tost["p_tost"], "tost_equivalent": tost["equivalent"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "t26_h1_tost_equivalence.csv", index=False)
    with open(TAB_DIR / "t26_h1_tost_equivalence.tex", "w", encoding="utf-8") as f:
        f.write(out.to_latex(index=False, float_format="%.4f",
            caption=f"TOST equivalence test for wallet-level persistence "
                    f"against a pre-specified bound of +/-{EQUIVALENCE_BOUND} "
                    f"(Cohen's 1988 convention for a negligible effect "
                    f"size), replacing the sample-size-dependent "
                    f"|rho|<0.05 criterion.",
            label="tab:h1_tost"))
    print(f"\n  ✓ saved -> {TAB_DIR / 't26_h1_tost_equivalence.csv'}")
    return out


def decile_portfolio_sort(cutoff="2024-12-31", horizon="2025-06-30"):
    print("\n" + "=" * 65)
    print("  DECILE PORTFOLIO SORT (economic significance of persistence)")
    print("=" * 65)

    feats = pl.read_parquet(PIT_DIR / f"features_asof_{cutoff}.parquet")
    labels = pl.read_parquet(PIT_DIR / f"labels_{cutoff}_to_{horizon}.parquet")
    df = feats.join(labels, on="wallet", how="inner").to_pandas()
    df = df.dropna(subset=["past_clv_vw", "fwd_clv_vw"])

    df["decile"] = pd.qcut(df["past_clv_vw"].rank(method="first"), 10,
                           labels=False) + 1
    g = df.groupby("decile")["fwd_clv_vw"].agg(["mean", "median", "std",
                                                 "count"]).reset_index()
    g.columns = ["decile", "mean_fwd_clv", "median_fwd_clv",
                 "std_fwd_clv", "n"]

    print(f"\n  Panel: features@{cutoff}, forward window through {horizon}")
    print(g.to_string(index=False))

    top = g[g["decile"] == 10]["mean_fwd_clv"].values[0]
    bottom = g[g["decile"] == 1]["mean_fwd_clv"].values[0]
    spread = top - bottom

    print(f"\n  D10 - D1 spread (forward CLV): {spread:+.4f}")
    print(f"  Report this number directly as the economic-significance")
    print(f"  translation of the H1 finding.")

    g.to_csv(TAB_DIR / "t27_decile_portfolio_sort.csv", index=False)
    print(f"\n  ✓ saved -> {TAB_DIR / 't27_decile_portfolio_sort.csv'}")
    return g


def run():
    run_persistence_equivalence()
    decile_portfolio_sort()


if __name__ == "__main__":
    run()