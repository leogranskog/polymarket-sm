"""
Post-hoc meta-analysis pooling the two confirmatory windows.

IMPORTANT — SCOPE AND STATUS OF THIS ANALYSIS:
This script is explicitly EXPLORATORY / POST-HOC, run AFTER both
pre-registered confirmatory windows (2025-H2, 2026-Q1) were already
observed and reported individually. It is NOT a third confirmatory
test and makes no PASS/FAIL claim of its own. Its purpose is to
formally quantify whether the two windows' estimates are consistent
with a single shared effect (low heterogeneity) or are themselves in
conflict (high heterogeneity) -- which is informative regardless of
the pooled point estimate's sign.

Method: fixed-effect inverse-variance meta-analysis (standard,
e.g. Borenstein et al. 2009), reporting:
  - each window's estimate and SE
  - the fixed-effect pooled estimate and its CI
  - Cochran's Q and its p-value
  - I^2 (percentage of variance due to heterogeneity, not sampling error)

Interpretation guide:
  I^2 < 25%           : low heterogeneity, pooling is reasonable
  I^2 25-75%           : moderate heterogeneity, pool with caution
  I^2 > 75%            : high heterogeneity, pooled estimate is NOT
                         a meaningful summary; the disagreement between
                         windows IS the finding

Usage: python -m research.meta_analysis_oos
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2, norm

TAB_DIR = Path(__file__).parent / "tables_v2"
TAB_DIR.mkdir(exist_ok=True)


def fixed_effect_meta(estimates: list, ses: list) -> dict:
    """
    Standard fixed-effect (inverse-variance) meta-analysis with
    Cochran's Q and I^2 heterogeneity statistics.

    estimates: point estimates from each window
    ses:       standard errors of each estimate
    """
    estimates = np.array(estimates, dtype=float)
    ses = np.array(ses, dtype=float)
    weights = 1.0 / (ses ** 2)

    pooled = np.sum(weights * estimates) / np.sum(weights)
    pooled_se = np.sqrt(1.0 / np.sum(weights))
    ci_lo = pooled - 1.96 * pooled_se
    ci_hi = pooled + 1.96 * pooled_se
    pooled_z = pooled / pooled_se
    pooled_p = 2 * (1 - norm.cdf(abs(pooled_z)))

    k = len(estimates)
    Q = np.sum(weights * (estimates - pooled) ** 2)
    df = k - 1
    Q_p = 1 - chi2.cdf(Q, df) if df > 0 else np.nan
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 and df > 0 else 0.0

    return {
        "pooled_estimate": pooled,
        "pooled_se": pooled_se,
        "pooled_ci_lo": ci_lo,
        "pooled_ci_hi": ci_hi,
        "pooled_p": pooled_p,
        "cochran_Q": Q,
        "Q_df": df,
        "Q_p": Q_p,
        "I2_pct": I2,
        "k_studies": k,
    }


def heterogeneity_flag(I2: float) -> str:
    if I2 < 25:
        return "LOW heterogeneity -- pooled estimate is meaningful"
    elif I2 < 75:
        return "MODERATE heterogeneity -- pool with caution"
    else:
        return "HIGH heterogeneity -- pooled estimate is NOT a " \
               "meaningful summary; disagreement between windows " \
               "is itself the finding"


def se_from_ci(ci_lo: float, ci_hi: float) -> float:
    """Approximate SE from a symmetric 95% CI: SE = (hi - lo) / (2*1.96)."""
    return (ci_hi - ci_lo) / (2 * 1.96)


def run():
    print("=" * 70)
    print("  POST-HOC META-ANALYSIS — POOLING TWO CONFIRMATORY WINDOWS")
    print("  (exploratory; not a third pre-registered test)")
    print("=" * 70)

    # ── Hardcoded window results (from the two completed, logged runs) ──
    # H1: Spearman rho for persistence
    h1 = {
        "name": "H1 Persistence (rho)",
        "estimates": [0.0526, 0.0538],
        "cis":       [(0.0472, 0.0584), (0.0486, 0.0591)],
    }
    # H2: OLS coefficient, reversal (past CLV -> forward CLV)
    h2 = {
        "name": "H2 Reversal (past_clv_vw coef)",
        "estimates": [0.0311, 0.0099],
        "cis":       [(0.0311 - 1.96*se_from_p(0.0311, 1e-10), None), None],
    }
    # H2's CI wasn't directly logged (only p-values), so use se from a
    # normal approximation via reported coefficient/p when CI absent.
    print("\n  NOTE: H2 and H3 CIs were not directly logged in the run "
          "output (only coefficients and p-values). We approximate SEs "
          "from the reported z-statistics implied by the p-values, "
          "which is standard practice but slightly less precise than "
          "using the exact regression SE. For a final paper draft, "
          "pull the exact SE directly from the saved statsmodels "
          "results object rather than this approximation.")

    def se_from_coef_p(coef, p):
        if p <= 0 or p >= 1:
            return abs(coef) / 10  # fallback, avoid div by zero
        z = abs(norm.ppf(p / 2))
        return abs(coef) / z if z > 0 else abs(coef) / 10

    results_summary = []

    # ── H1: Persistence ───────────────────────────────────────────────
    print("\n  H1 — Persistence (Spearman rho)")
    se1 = [se_from_ci(*h1["cis"][0]), se_from_ci(*h1["cis"][1])]
    m1 = fixed_effect_meta(h1["estimates"], se1)
    print(f"    Window 1: rho={h1['estimates'][0]:+.4f}  "
          f"SE={se1[0]:.4f}")
    print(f"    Window 2: rho={h1['estimates'][1]:+.4f}  "
          f"SE={se1[1]:.4f}")
    print(f"    Pooled:   rho={m1['pooled_estimate']:+.4f}  "
          f"[{m1['pooled_ci_lo']:+.4f},{m1['pooled_ci_hi']:+.4f}]  "
          f"p={m1['pooled_p']:.4f}")
    print(f"    Q={m1['cochran_Q']:.3f} (df={m1['Q_df']}, p={m1['Q_p']:.4f})  "
          f"I^2={m1['I2_pct']:.1f}%  -> {heterogeneity_flag(m1['I2_pct'])}")
    results_summary.append({"hypothesis": h1["name"], **m1})

    # ── H2: Reversal (approximate SE from coefficient + p-value) ────────
    print("\n  H2 — Reversal (past_clv_vw coefficient)")
    est2 = [0.0311, 0.0099]
    # p-values as reported: window1 p=0.0008 (from earlier true_oos_final
    # log), window2 p=0.1675 (from this run's log)
    p2 = [0.0008, 0.1675]
    se2 = [se_from_coef_p(est2[i], p2[i]) for i in range(2)]
    m2 = fixed_effect_meta(est2, se2)
    print(f"    Window 1: coef={est2[0]:+.4f}  SE~{se2[0]:.4f} (p={p2[0]})")
    print(f"    Window 2: coef={est2[1]:+.4f}  SE~{se2[1]:.4f} (p={p2[1]})")
    print(f"    Pooled:   coef={m2['pooled_estimate']:+.4f}  "
          f"[{m2['pooled_ci_lo']:+.4f},{m2['pooled_ci_hi']:+.4f}]  "
          f"p={m2['pooled_p']:.4f}")
    print(f"    Q={m2['cochran_Q']:.3f} (df={m2['Q_df']}, p={m2['Q_p']:.4f})  "
          f"I^2={m2['I2_pct']:.1f}%  -> {heterogeneity_flag(m2['I2_pct'])}")
    results_summary.append({"hypothesis": h2["name"], **m2})

    # ── H3: Specialization (approximate SE from coefficient + p-value) ──
    print("\n  H3 — Specialization coefficient")
    est3 = [0.0060, -0.0045]
    # window1 p=3.6e-13 (from true_oos_final log), window2 p<0.0001 (this run)
    p3 = [3.6e-13, 1e-6]  # window2 printed as 0.0000; use conservative 1e-6
    se3 = [se_from_coef_p(est3[i], p3[i]) for i in range(2)]
    m3 = fixed_effect_meta(est3, se3)
    print(f"    Window 1: coef={est3[0]:+.4f}  SE~{se3[0]:.5f} (p={p3[0]:.1e})")
    print(f"    Window 2: coef={est3[1]:+.4f}  SE~{se3[1]:.5f} (p={p3[1]:.1e})")
    print(f"    Pooled:   coef={m3['pooled_estimate']:+.4f}  "
          f"[{m3['pooled_ci_lo']:+.4f},{m3['pooled_ci_hi']:+.4f}]  "
          f"p={m3['pooled_p']:.4f}")
    print(f"    Q={m3['cochran_Q']:.3f} (df={m3['Q_df']}, p={m3['Q_p']:.4f})  "
          f"I^2={m3['I2_pct']:.1f}%  -> {heterogeneity_flag(m3['I2_pct'])}")
    print(f"    *** OPPOSITE SIGNS across windows: pooled estimate is "
          f"NOT a meaningful summary regardless of I^2 value. The sign "
          f"instability itself is the finding for H3. ***")
    results_summary.append({"hypothesis": h3_name(), **m3})

    # ── H4: ML AUC ────────────────────────────────────────────────────
    print("\n  H4 — ML AUC")
    est4 = [0.5044, 0.4908]
    cis4 = [(0.4714, 0.5158 if False else 0.5084), (0.4878, 0.4939)]
    # window1 CI from true_oos_final log: [0.5004, 0.5084]
    cis4 = [(0.5004, 0.5084), (0.4878, 0.4939)]
    se4 = [se_from_ci(*cis4[0]), se_from_ci(*cis4[1])]
    m4 = fixed_effect_meta(est4, se4)
    print(f"    Window 1: AUC={est4[0]:.4f}  SE={se4[0]:.4f}")
    print(f"    Window 2: AUC={est4[1]:.4f}  SE={se4[1]:.4f}")
    print(f"    Pooled:   AUC={m4['pooled_estimate']:.4f}  "
          f"[{m4['pooled_ci_lo']:.4f},{m4['pooled_ci_hi']:.4f}]")
    print(f"    Q={m4['cochran_Q']:.3f} (df={m4['Q_df']}, p={m4['Q_p']:.4f})  "
          f"I^2={m4['I2_pct']:.1f}%  -> {heterogeneity_flag(m4['I2_pct'])}")
    results_summary.append({"hypothesis": "H4 ML AUC", **m4})

    # ── H5: Insider flow AUC ──────────────────────────────────────────
    print("\n  H5 — Insider-flow AUC")
    est5 = [0.5579, 0.5165]
    cis5 = [(0.5502, 0.5666), (0.5089, 0.5237)]
    se5 = [se_from_ci(*cis5[0]), se_from_ci(*cis5[1])]
    m5 = fixed_effect_meta(est5, se5)
    print(f"    Window 1: AUC={est5[0]:.4f}  SE={se5[0]:.4f}")
    print(f"    Window 2: AUC={est5[1]:.4f}  SE={se5[1]:.4f}")
    print(f"    Pooled:   AUC={m5['pooled_estimate']:.4f}  "
          f"[{m5['pooled_ci_lo']:.4f},{m5['pooled_ci_hi']:.4f}]  "
          f"p={m5['pooled_p']:.2e}")
    print(f"    Q={m5['cochran_Q']:.3f} (df={m5['Q_df']}, p={m5['Q_p']:.4f})  "
          f"I^2={m5['I2_pct']:.1f}%  -> {heterogeneity_flag(m5['I2_pct'])}")
    print(f"    Both windows agree in direction (AUC>0.5); this is the "
          f"strongest candidate for a meaningful pooled estimate.")
    results_summary.append({"hypothesis": "H5 Insider-flow AUC", **m5})

    # ── Save ──────────────────────────────────────────────────────────
    df_out = pd.DataFrame(results_summary)
    df_out.to_csv(TAB_DIR / "t22_meta_analysis.csv", index=False)
    with open(TAB_DIR / "t22_meta_analysis.tex", "w", encoding="utf-8") as f:
        f.write(df_out.to_latex(index=False, float_format="%.4f",
                caption="Post-hoc fixed-effect meta-analysis pooling "
                        "the two confirmatory windows, with "
                        "heterogeneity statistics (I-squared). "
                        "Exploratory; not a pre-registered test.",
                label="tab:meta_oos"))

    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"\n  {'Hypothesis':<28}{'Pooled':>10}{'I2%':>8}  Heterogeneity")
    for r in results_summary:
        print(f"  {r['hypothesis']:<28}{r['pooled_estimate']:>+10.4f}"
              f"{r['I2_pct']:>7.1f}%  "
              f"{'HIGH - windows disagree' if r['I2_pct']>75 else ('MODERATE' if r['I2_pct']>25 else 'LOW - consistent')}")

    print(f"\n  ✓ saved -> {TAB_DIR / 't22_meta_analysis.csv'}")
    print("\n  Reminder: this is a post-hoc exploratory analysis. Report")
    print("  it in the paper as such, alongside the individual window")
    print("  results, not as a replacement for them.")


def h3_name():
    return "H3 Specialization (specialist coef)"


def se_from_p(estimate, p):
    """Placeholder unused helper kept for compatibility."""
    return abs(estimate) / 10


if __name__ == "__main__":
    run()