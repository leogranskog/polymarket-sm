"""
Insider vs Skill vs Luck decomposition (Dan's core question).

Archetypes from timing x CLV as of the test cutoff:
  Insider-like: positive CLV concentrated in LATE entries (<=2d to close)
  Skilled:      positive CLV concentrated in EARLY entries (>=14d)
  Lucky:        positive past hit-rate but ~zero CLV
Then: does each archetype's edge PERSIST forward? Skill should persist;
insider edge persists only if information access recurs; luck should not.

Usage: python -m research.insider_decomposition
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_1samp
from config import PROC_DIR

PIT_DIR = PROC_DIR / "pit"
OUT_TAB = Path(__file__).parent / "tables_v2"
OUT_FIG = Path(__file__).parent / "figures_v2"

CUTOFF, HORIZON = "2024-12-31", "2025-06-30"
EPS = 0.005   # "meaningful" CLV threshold (0.5 pts)


def run():
    print("=" * 60)
    print("  INSIDER / SKILL / LUCK DECOMPOSITION")
    print("=" * 60)

    feats  = pl.read_parquet(PIT_DIR / f"features_asof_{CUTOFF}.parquet")
    labels = pl.read_parquet(PIT_DIR / f"labels_{CUTOFF}_to_{HORIZON}.parquet")
    df = feats.join(labels, on="wallet", how="inner").to_pandas()
    for c in ["clv_when_late", "clv_when_early", "past_clv_vw",
              "past_clv_hitrate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    late  = df["clv_when_late"]
    early = df["clv_when_early"]

    conds = [
        (late > EPS) & (early.fillna(0) <= EPS),        # right when late only
        (early > EPS) & (late.fillna(0) <= EPS),        # right when early only
        (late > EPS) & (early > EPS),                   # right in both
        (df["past_clv_hitrate"] > 0.5) &
            (df["past_clv_vw"].abs() <= EPS),           # wins w/o line value
    ]
    names = ["Insider-like (late+right)", "Skilled (early+right)",
             "Both", "Lucky (hits, no CLV)"]
    df["archetype"] = np.select(conds, names, default="Other")

    print(f"\n  {'Archetype':<28}{'N':>8}{'past CLV':>11}"
          f"{'fwd CLV':>11}{'fwd>0 %':>9}{'p(fwd=0)':>11}")
    rows = []
    for name in names + ["Other"]:
        sub = df[df["archetype"] == name]
        if len(sub) < 20:
            continue
        fwd = sub["fwd_clv_vw"]
        p = ttest_1samp(fwd, 0).pvalue
        rows.append({"archetype": name, "n": len(sub),
                     "past_clv": sub["past_clv_vw"].mean(),
                     "fwd_clv": fwd.mean(),
                     "frac_fwd_positive": (fwd > 0).mean(),
                     "p_fwd_eq_0": p})
        print(f"  {name:<28}{len(sub):>8,}{sub['past_clv_vw'].mean():>11.4f}"
              f"{fwd.mean():>11.4f}{(fwd > 0).mean()*100:>8.1f}%{p:>11.2e}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "t8_insider_decomposition.csv", index=False)
    with open(OUT_TAB / "t8_insider_decomposition.tex", "w") as f:
        f.write(out.to_latex(index=False, float_format="%.4f",
                caption="Forward CLV by trader archetype (timing x past CLV)",
                label="tab:insider"))

    # figure: past vs forward CLV per archetype
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(out)); w = 0.38
    ax.bar(x - w/2, out["past_clv"], w, label="Past CLV (pre-cutoff)",
           color="steelblue")
    ax.bar(x + w/2, out["fwd_clv"], w, label="Forward CLV (2025-H1)",
           color="darkorange")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(out["archetype"], rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Volume-weighted CLV")
    ax.set_title("Edge persistence by archetype: skill persists, luck does not")
    ax.legend()
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(OUT_FIG / f"f6_archetypes{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ -> {OUT_TAB/'t8_insider_decomposition.csv'}, "
          f"{OUT_FIG/'f6_archetypes.pdf'}")


if __name__ == "__main__":
    run()