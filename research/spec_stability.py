"""Stability of the specialization effect across all windows (pre-registered
diagnostic). OOS window excluded until final run."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import polars as pl, pandas as pd, numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from config import PROC_DIR

PIT = PROC_DIR / "pit"
OUT = Path(__file__).parent / "tables_v2"
WINDOWS = [
    ("2023-06-30", "2023-12-31"), ("2023-09-30", "2024-03-31"),
    ("2023-12-31", "2024-06-30"), ("2024-06-30", "2024-12-31"),
    ("2024-12-31", "2025-06-30"),
]

rows = []
for cutoff, horizon in WINDOWS:
    f = pl.read_parquet(PIT / f"features_asof_{cutoff}.parquet")
    l = pl.read_parquet(PIT / f"labels_{cutoff}_to_{horizon}.parquet")
    d = f.join(l, on="wallet", how="inner").to_pandas()
    if len(d) < 200:
        print(f"  {cutoff}: n={len(d)} too small, skipped"); continue
    t1, t2 = d["category_hhi"].quantile([1/3, 2/3])
    d = d[(d["category_hhi"] >= t2) | (d["category_hhi"] <= t1)].copy()
    d["specialist"] = (d["category_hhi"] >= t2).astype(int)
    d["log_trades"] = np.log1p(d["n_trades"])
    d["log_volume"] = np.log1p(d["total_volume"])
    for c in ["past_clv_vw", "frac_maker"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    m = smf.ols("fwd_clv_vw ~ specialist + log_trades + log_volume "
                "+ frac_maker + past_clv_vw", data=d).fit(cov_type="HC3")
    rows.append({"window": f"{cutoff}->{horizon}", "n": len(d),
                 "spec_coef": m.params["specialist"],
                 "spec_p": m.pvalues["specialist"],
                 "past_clv_coef": m.params["past_clv_vw"],
                 "past_clv_p": m.pvalues["past_clv_vw"],
                 "maker_coef": m.params["frac_maker"]})
    r = rows[-1]
    print(f"  {r['window']}: n={r['n']:,}  specialist={r['spec_coef']:+.4f} "
          f"(p={r['spec_p']:.1e})  past_clv={r['past_clv_coef']:+.4f} "
          f"(p={r['past_clv_p']:.1e})")

pd.DataFrame(rows).to_csv(OUT / "t9_spec_stability.csv", index=False)
print(f"\n  ✓ -> {OUT / 't9_spec_stability.csv'}")