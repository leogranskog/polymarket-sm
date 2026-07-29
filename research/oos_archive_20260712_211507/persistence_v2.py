"""
Extension 3 + 4 — Noise-aware and volume-weighted persistence.
v2: adds bootstrap CIs and a whale-exclusion robustness check to the
volume-weighted analysis, which previously reported point estimates only.

Usage: python -m research.persistence_v2
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from config import PROC_DIR

PIT_DIR = PROC_DIR / "pit"
TAB_DIR = Path(__file__).parent / "tables_v2"
FIG_DIR = Path(__file__).parent / "figures_v2"
TAB_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
WINDOWS = [
    ("H2-2023", "labels_2023-06-30_to_2023-12-31.parquet",
               "features_asof_2023-06-30.parquet"),
    ("H1-2024", "labels_2023-12-31_to_2024-06-30.parquet",
               "features_asof_2023-12-31.parquet"),
    ("H2-2024", "labels_2024-06-30_to_2024-12-31.parquet",
               "features_asof_2024-06-30.parquet"),
    ("H1-2025", "labels_2024-12-31_to_2025-06-30.parquet",
               "features_asof_2024-12-31.parquet"),
]
HIGH_PRECISION_MIN_TRADES = 30


def load_labels(fname):
    return pl.read_parquet(PIT_DIR / fname).select(
        ["wallet", "fwd_clv_vw", "fwd_n_trades"])


def load_volume(feat_fname):
    return pl.read_parquet(PIT_DIR / feat_fname).select(
        ["wallet", "total_volume"])


def empirical_bayes_shrink(clv, n_trades, clv_std_pooled):
    grand_mean = np.average(clv, weights=n_trades)
    se2 = (clv_std_pooled ** 2) / np.maximum(n_trades, 1)
    raw_var = np.var(clv)
    tau2 = max(raw_var - np.mean(se2), 1e-8)
    w = tau2 / (tau2 + se2)
    shrunk = grand_mean + w * (clv - grand_mean)
    return shrunk, w


def weighted_spearman(x, y, w):
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    wm_x = np.average(rx, weights=w)
    wm_y = np.average(ry, weights=w)
    cov = np.average((rx - wm_x) * (ry - wm_y), weights=w)
    vx = np.average((rx - wm_x) ** 2, weights=w)
    vy = np.average((ry - wm_y) ** 2, weights=w)
    return cov / np.sqrt(vx * vy)


def bootstrap_weighted_rho_ci(df, weight_col, n_boot=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(df)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        d = df.iloc[idx]
        try:
            vals.append(weighted_spearman(
                d["fwd_clv_vw"], d["fwd_clv_vw_next"], d[weight_col]))
        except Exception:
            pass
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def run():
    print("=" * 65)
    print("  EXTENSION 3+4 v2 — Noise-aware and volume-weighted "
          "persistence (with CIs)")
    print("=" * 65)

    # (3a) higher forward-trade bar
    print(f"\n  (3a) Persistence, min_forward_trades="
          f"{HIGH_PRECISION_MIN_TRADES}")
    rows_hp = []
    for (n1, f1, _), (n2, f2, _) in zip(WINDOWS[:-1], WINDOWS[1:]):
        a = load_labels(f1).filter(
            pl.col("fwd_n_trades") >= HIGH_PRECISION_MIN_TRADES)
        b = load_labels(f2).filter(
            pl.col("fwd_n_trades") >= HIGH_PRECISION_MIN_TRADES)
        j = a.join(b, on="wallet", suffix="_next").to_pandas()
        if len(j) < 50:
            print(f"    {n1}->{n2}: n={len(j)}, skipped"); continue
        rho, pval = spearmanr(j["fwd_clv_vw"], j["fwd_clv_vw_next"])
        rows_hp.append({"transition": f"{n1}->{n2}", "n": len(j),
                        "spearman_rho": rho, "p_value": pval})
        print(f"    {n1}->{n2}: n={len(j):,}  rho={rho:.4f} (p={pval:.2e})")
    pd.DataFrame(rows_hp).to_csv(
        TAB_DIR / "t12_high_precision_persistence.csv", index=False)

    # (3b) empirical-Bayes shrinkage
    print(f"\n  (3b) Empirical-Bayes shrinkage persistence")
    rows_eb = []
    for (n1, f1, _), (n2, f2, _) in zip(WINDOWS[:-1], WINDOWS[1:]):
        a = load_labels(f1).to_pandas()
        b = load_labels(f2).to_pandas()
        if len(a) < 100 or len(b) < 100:
            print(f"    {n1}->{n2}: too small, skipped"); continue
        pooled_std = a["fwd_clv_vw"].std()
        a_shrunk, w_a = empirical_bayes_shrink(
            a["fwd_clv_vw"].values, a["fwd_n_trades"].values, pooled_std)
        a = a.assign(clv_shrunk=a_shrunk)
        j = a.merge(b, on="wallet", suffixes=("", "_next"))
        if len(j) < 50:
            continue
        rho_raw, p_raw = spearmanr(j["fwd_clv_vw"], j["fwd_clv_vw_next"])
        rho_eb, p_eb = spearmanr(j["clv_shrunk"], j["fwd_clv_vw_next"])
        rows_eb.append({"transition": f"{n1}->{n2}", "n": len(j),
                        "rho_raw": rho_raw, "p_raw": p_raw,
                        "rho_shrunk": rho_eb, "p_shrunk": p_eb,
                        "mean_shrinkage_weight": w_a.mean()})
        print(f"    {n1}->{n2}: n={len(j):,}  raw_rho={rho_raw:.4f} "
              f"(p={p_raw:.2e})  shrunk_rho={rho_eb:.4f} (p={p_eb:.2e})  "
              f"avg_weight={w_a.mean():.3f}")
    pd.DataFrame(rows_eb).to_csv(
        TAB_DIR / "t13_eb_shrinkage_persistence.csv", index=False)

    # (4) volume-weighted persistence WITH bootstrap CIs + whale check
    print(f"\n  (4) Volume-weighted persistence (with 95% bootstrap CIs "
          f"and whale-exclusion robustness)")
    rows_vw = []
    for (n1, f1, feat1), (n2, f2, _) in zip(WINDOWS[:-1], WINDOWS[1:]):
        a = load_labels(f1)
        vol = load_volume(feat1)
        a = a.join(vol, on="wallet", how="inner")
        b = load_labels(f2)
        j = a.join(b, on="wallet", suffix="_next").to_pandas()
        if len(j) < 50:
            print(f"    {n1}->{n2}: n={len(j)}, skipped"); continue

        rho_uw, p_uw = spearmanr(j["fwd_clv_vw"], j["fwd_clv_vw_next"])
        rho_vw = weighted_spearman(j["fwd_clv_vw"], j["fwd_clv_vw_next"],
                                   j["total_volume"])
        lo, hi = bootstrap_weighted_rho_ci(j, "total_volume")

        # whale-exclusion robustness: drop top 1% by volume, recompute
        cutoff_vol = j["total_volume"].quantile(0.99)
        j_ex = j[j["total_volume"] < cutoff_vol]
        rho_vw_exwhale = weighted_spearman(
            j_ex["fwd_clv_vw"], j_ex["fwd_clv_vw_next"], j_ex["total_volume"]
        ) if len(j_ex) > 30 else np.nan

        rows_vw.append({
            "transition": f"{n1}->{n2}", "n": len(j),
            "rho_unweighted": rho_uw,
            "rho_volume_weighted": rho_vw,
            "vw_ci_lo": lo, "vw_ci_hi": hi,
            "rho_volume_weighted_ex_top1pct": rho_vw_exwhale,
        })
        print(f"    {n1}->{n2}: n={len(j):,}  unweighted={rho_uw:+.4f}  "
              f"volume_weighted={rho_vw:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"ex-top-1%-whale={rho_vw_exwhale:+.4f}")

    df_vw = pd.DataFrame(rows_vw)
    df_vw.to_csv(TAB_DIR / "t14_volume_weighted_persistence.csv", index=False)
    with open(TAB_DIR / "t14_volume_weighted_persistence.tex", "w") as f:
        f.write(df_vw.to_latex(index=False, float_format="%.4f",
                caption="Volume-weighted persistence with bootstrap CIs "
                        "and whale-exclusion check",
                label="tab:vw_persistence"))

    print(f"\n  Interpretation: if the CI excludes 0 AND the "
          f"ex-top-1%-whale estimate has the same sign and similar "
          f"magnitude, the volume-weighted effect is not an artifact of "
          f"one or two large wallets. If it flips sign after excluding "
          f"the top 1%, treat it as whale-driven noise, not a finding.")

    print(f"\n  ✓ tables -> {TAB_DIR}")


if __name__ == "__main__":
    run()