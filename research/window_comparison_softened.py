"""
SOFTENED per referee feedback #4: I^2 and Cochran's Q are notoriously
unstable and near-uninformative with only k=2 studies. Replaced with:
  - the two raw point estimates and their individual CIs
  - a direct z-test on the DIFFERENCE between the two estimates
  - explicit reasoning based on MAGNITUDE AND SIGN.

This script REPLACES the original I^2-based framing.

Usage: python -m research.window_comparison_softened
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

TAB_DIR = Path(__file__).parent / "tables_v2"
TAB_DIR.mkdir(exist_ok=True)


def se_from_ci(ci_lo: float, ci_hi: float) -> float:
    return (ci_hi - ci_lo) / (2 * 1.96)


def se_from_coef_p(coef, p):
    if p <= 0 or p >= 1:
        return abs(coef) / 10
    z = abs(norm.ppf(p / 2))
    return abs(coef) / z if z > 0 else abs(coef) / 10


def z_test_difference(est1: float, se1: float, est2: float, se2: float) -> dict:
    diff = est1 - est2
    se_diff = np.sqrt(se1 ** 2 + se2 ** 2)
    z = diff / se_diff if se_diff > 0 else np.nan
    p_value = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    ci_lo = diff - 1.96 * se_diff
    ci_hi = diff + 1.96 * se_diff
    return {"difference": diff, "se_diff": se_diff, "z": z,
            "p_value": p_value, "ci_lo": ci_lo, "ci_hi": ci_hi}


def classify(name: str, est1: float, est2: float, z_result: dict,
             null_value: float = 0.0) -> str:
    same_sign = (est1 > null_value) == (est2 > null_value)
    diff_significant = z_result["p_value"] < 0.05

    if not same_sign:
        return "SIGN REVERSAL -- cannot be treated as a stable effect"
    elif diff_significant:
        return ("SAME DIRECTION, magnitude significantly different -- "
                "consistent with a real but DECAYING/CHANGING effect")
    else:
        return ("SAME DIRECTION, magnitude not significantly different "
                "-- consistent with a STABLE, replicating effect")


def run():
    print("=" * 70)
    print("  TWO-WINDOW COMPARISON (softened per referee feedback)")
    print("  I^2/Cochran's Q dropped as unreliable with k=2; replaced")
    print("  with direct point-estimate comparison and z-test.")
    print("=" * 70)

    hypotheses = []

    est1a, est1b = 0.0526, 0.0538
    se1a, se1b = se_from_ci(0.0472, 0.0584), se_from_ci(0.0486, 0.0591)
    z1 = z_test_difference(est1a, se1a, est1b, se1b)
    hypotheses.append({
        "hypothesis": "H1 Persistence (rho)",
        "window1_est": est1a, "window2_est": est1b,
        **z1, "classification": classify("H1", est1a, est1b, z1),
    })

    est2a, est2b = 0.0311, 0.0099
    se2a = se_from_coef_p(est2a, 0.0008)
    se2b = se_from_coef_p(est2b, 0.1675)
    z2 = z_test_difference(est2a, se2a, est2b, se2b)
    hypotheses.append({
        "hypothesis": "H2 Reversal (past_clv_vw coef)",
        "window1_est": est2a, "window2_est": est2b,
        **z2, "classification": classify("H2", est2a, est2b, z2),
    })

    est3a, est3b = 0.0060, -0.0045
    se3a = se_from_coef_p(est3a, 3.6e-13)
    se3b = se_from_coef_p(est3b, 1e-6)
    z3 = z_test_difference(est3a, se3a, est3b, se3b)
    hypotheses.append({
        "hypothesis": "H3 Specialization (specialist coef)",
        "window1_est": est3a, "window2_est": est3b,
        **z3, "classification": classify("H3", est3a, est3b, z3),
    })

    est4a, est4b = 0.5044, 0.4908
    se4a, se4b = se_from_ci(0.5004, 0.5084), se_from_ci(0.4878, 0.4939)
    z4 = z_test_difference(est4a, se4a, est4b, se4b)
    hypotheses.append({
        "hypothesis": "H4 ML AUC",
        "window1_est": est4a, "window2_est": est4b,
        **z4, "classification": classify("H4", est4a, est4b, z4, null_value=0.5),
    })

    est5a, est5b = 0.5579, 0.5165
    se5a, se5b = se_from_ci(0.5502, 0.5666), se_from_ci(0.5089, 0.5237)
    z5 = z_test_difference(est5a, se5a, est5b, se5b)
    hypotheses.append({
        "hypothesis": "H5 Insider-flow AUC (unconditional)",
        "window1_est": est5a, "window2_est": est5b,
        **z5, "classification": classify("H5", est5a, est5b, z5, null_value=0.5),
    })

    print(f"\n  {'Hypothesis':<38}{'Win1':>9}{'Win2':>9}{'Diff':>9}{'p(diff)':>10}")
    for h in hypotheses:
        print(f"  {h['hypothesis']:<38}{h['window1_est']:>+9.4f}"
              f"{h['window2_est']:>+9.4f}{h['difference']:>+9.4f}"
              f"{h['p_value']:>10.4f}")
        print(f"    -> {h['classification']}")

    df = pd.DataFrame(hypotheses)
    df.to_csv(TAB_DIR / "t22_window_comparison_softened.csv", index=False)
    with open(TAB_DIR / "t22_window_comparison_softened.tex", "w",
              encoding="utf-8") as f:
        cols = ["hypothesis", "window1_est", "window2_est", "difference",
                "p_value", "classification"]
        f.write(df[cols].to_latex(index=False, float_format="%.4f",
            caption="Two-window comparison: point estimates and a direct "
                    "z-test on their difference (replacing an earlier "
                    "meta-analytic I-squared/Cochran's Q presentation, "
                    "unreliable with only two studies).",
            label="tab:window_comparison"))

    print(f"\n  ✓ saved -> {TAB_DIR / 't22_window_comparison_softened.csv'}")
    print(f"\n  Use this table (not the old I^2/Q table) as the paper's")
    print(f"  cross-window comparison exhibit.")


if __name__ == "__main__":
    run()