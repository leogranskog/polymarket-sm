"""
Regenerates the two most valuable paper figures DIRECTLY from the
verified MASTER_NUMBERS.md values, no model rerun needed. Guarantees
these figures cannot contradict the paper text, since they're built
from the exact same numbers already cited.

Usage: python -m research.make_verified_figures
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).parent / "figures_v2"
FIG_DIR.mkdir(exist_ok=True)

# ── Figure A: Model comparison, from verified Table 2 ────────────────────
models = ["Past-CLV\nbenchmark", "Logistic\nregression", "Random\nforest",
          "XGBoost", "LightGBM\n(tuned)"]
aucs   = [0.560, 0.480, 0.551, 0.524, 0.541]
ci_lo  = [0.555, 0.476, 0.546, 0.519, 0.536]
ci_hi  = [0.565, 0.485, 0.555, 0.528, 0.545]
errs   = [[a-l for a, l in zip(aucs, ci_lo)],
          [h-a for a, h in zip(aucs, ci_hi)]]

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["#2ca02c" if m == "Past-CLV\nbenchmark" else "#1f77b4" for m in models]
ax.barh(models, aucs, xerr=errs, color=colors, capsize=3)
ax.axvline(0.5, color="red", ls="--", alpha=0.7, label="Random")
ax.set_xlabel("ROC-AUC (95% bootstrap CI)")
ax.set_title("Model comparison, leak-free test window (n=79,329)")
ax.set_xlim(0.44, 0.60)
ax.legend()
plt.tight_layout()
for ext in [".pdf", ".png"]:
    plt.savefig(FIG_DIR / f"VERIFIED_model_comparison{ext}", dpi=300,
                bbox_inches="tight")
plt.close()
print("Saved VERIFIED_model_comparison")

# ── Figure B: Decile spread, from verified Table 3/decile output ────────
deciles = list(range(1, 11))
fwd_clv = [-0.0081, -0.0017, -0.0034, -0.0040, -0.0054, -0.0032,
           -0.0025, -0.0011, 0.0011, 0.0256]

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, 10))
ax.bar(deciles, fwd_clv, color=colors, edgecolor="k", linewidth=0.5)
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xlabel("Past-CLV decile (1 = lowest)")
ax.set_ylabel("Forward CLV")
ax.set_title("Decile portfolio sort: D10-D1 spread = +0.0337")
ax.set_xticks(deciles)
plt.tight_layout()
for ext in [".pdf", ".png"]:
    plt.savefig(FIG_DIR / f"VERIFIED_decile_spread{ext}", dpi=300,
                bbox_inches="tight")
plt.close()
print("Saved VERIFIED_decile_spread")