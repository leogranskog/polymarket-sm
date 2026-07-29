"""
Result 1 — CLV persistence: does skill exist as a stable trait?

Rank correlation of wallet forward CLV across adjacent 6-month windows,
plus quintile transition matrices. Descriptive; no ML. Excludes the
TRUE-OOS window (2025-H2) by default.

Usage: python -m research.persistence
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
OUT_TAB = Path(__file__).parent / "tables_v2"
OUT_FIG = Path(__file__).parent / "figures_v2"
OUT_TAB.mkdir(exist_ok=True); OUT_FIG.mkdir(exist_ok=True)

WINDOWS = [  # ordered, adjacent 6-month CLV windows (OOS excluded)
    ("H2-2023", "labels_2023-06-30_to_2023-12-31.parquet"),
    ("H1-2024", "labels_2023-12-31_to_2024-06-30.parquet"),
    ("H2-2024", "labels_2024-06-30_to_2024-12-31.parquet"),
    ("H1-2025", "labels_2024-12-31_to_2025-06-30.parquet"),
]


def load(fname):
    return pl.read_parquet(PIT_DIR / fname).select(
        ["wallet", "fwd_clv_vw", "fwd_n_trades"]
    )


def run():
    print("=" * 60)
    print("  RESULT 1 — CLV PERSISTENCE")
    print("=" * 60)

    rows, matrices = [], {}
    for (n1, f1), (n2, f2) in zip(WINDOWS[:-1], WINDOWS[1:]):
        a, b = load(f1), load(f2)
        j = a.join(b, on="wallet", suffix="_next").to_pandas()
        if len(j) < 50:
            print(f"  {n1}->{n2}: only {len(j)} overlapping wallets, skipped")
            continue

        rho, pval = spearmanr(j["fwd_clv_vw"], j["fwd_clv_vw_next"])

        # quintile transition matrix
        j["q1"] = pd.qcut(j["fwd_clv_vw"].rank(method="first"), 5,
                          labels=False) + 1
        j["q2"] = pd.qcut(j["fwd_clv_vw_next"].rank(method="first"), 5,
                          labels=False) + 1
        tm = pd.crosstab(j["q1"], j["q2"], normalize="index")
        matrices[f"{n1}->{n2}"] = tm

        top_stay = tm.loc[5, 5] if (5 in tm.index and 5 in tm.columns) else np.nan
        rows.append({
            "transition": f"{n1} -> {n2}",
            "n_overlap": len(j),
            "spearman_rho": rho,
            "p_value": pval,
            "P(top-quintile stays top)": top_stay,
            "random_baseline": 0.20,
        })
        print(f"  {n1} -> {n2}: n={len(j):,}  rho={rho:.4f} (p={pval:.2e})  "
              f"P(Q5 stays Q5)={top_stay:.3f} vs 0.200 random")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TAB / "t6_persistence.csv", index=False)
    with open(OUT_TAB / "t6_persistence.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f",
                caption="CLV persistence across adjacent 6-month windows",
                label="tab:persistence"))

    # heatmap of the most recent transition matrix
    import matplotlib.pyplot as plt
    name, tm = list(matrices.items())[-1]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(tm.values, cmap="RdYlGn", vmin=0, vmax=0.4)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"Q{i}" for i in range(1, 6)])
    ax.set_yticklabels([f"Q{i}" for i in range(1, 6)])
    ax.set_xlabel("CLV quintile, next period")
    ax.set_ylabel("CLV quintile, this period")
    ax.set_title(f"Quintile transition matrix ({name})\n"
                 "row-normalized; random = 0.20 everywhere")
    for i in range(5):
        for k in range(5):
            ax.text(k, i, f"{tm.values[i, k]:.2f}", ha="center",
                    va="center", fontsize=9)
    fig.colorbar(im)
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(OUT_FIG / f"f5_transition_matrix{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ table -> {OUT_TAB/'t6_persistence.csv'}")
    print(f"  ✓ figure -> {OUT_FIG/'f5_transition_matrix.pdf'}")


if __name__ == "__main__":
    run()