"""
Hash for the SECOND confirmatory window's pre-analysis plan.

Excludes pit_features.py deliberately: that file is a data-preparation
utility that must evolve to build each new panel window (including
TRUE_OOS_2 itself) and is not part of the hypothesis-testing
methodology. What must stay frozen between writing this plan and
running the confirmatory test is the ANALYSIS logic only -- i.e. how
H1-H5 are computed and what counts as pass/fail -- which is exactly
the set of files below.

Usage: python -m research.freeze_hash_2
"""

import hashlib
from pathlib import Path

FROZEN_ANALYSIS_FILES = [
    "research/ml_pipeline_v2.py",
    "research/persistence.py",
    "research/persistence_v2.py",
    "research/specialization.py",
    "research/specialization_by_category.py",
    "research/insider_decomposition.py",
    "research/market_insider_flow_v2.py",
    "research/insider_robustness.py",
    "research/descriptives.py",
]

def compute_hash(root: Path) -> str:
    h = hashlib.sha256()
    missing = []
    for rel in FROZEN_ANALYSIS_FILES:
        p = root / rel
        if not p.exists():
            missing.append(rel)
            continue
        h.update(rel.encode())
        h.update(p.read_bytes())
    if missing:
        print(f"  ⚠ Missing files (excluded from hash): {missing}")
    return h.hexdigest()

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    digest = compute_hash(root)
    print(f"\n  FROZEN_HASH (analysis-only, excludes pit_features.py): {digest}\n")
    print("  Paste this into research/PRE_ANALYSIS_PLAN_2.md")
    