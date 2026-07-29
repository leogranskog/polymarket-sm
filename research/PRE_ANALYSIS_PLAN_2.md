# Pre-Analysis Plan Addendum 2 — Second Confirmatory Window
Date frozen: 2026-07-13
FROZEN_HASH: 6aa23eda77d1c95fb379bcd08fb04fba058ac2001a9521c3a50d4f588439a7ee

## Rationale
The Akey et al. dataset extends only to March 2026. Following referee
feedback that a single confirmatory window (2025-H2) is suggestive
rather than established, we use the remaining unused data
(2026-01-01 to 2026-03-29, approximately 3 months) as a second,
independent, pre-registered confirmatory window. This window is
shorter than the primary 6-month design and is explicitly reported as
such; it is a robustness check on the direction of the 2025-H2 shift,
not a claim of equal statistical power.

Note on scope: the hash below covers the nine hypothesis-testing
analysis scripts only, not the point-in-time panel-construction script
(pit_features.py), which necessarily requires new code to build each
successive confirmatory window's data (e.g. new cutoff dates,
memory-management engineering for larger recent trade volumes) and is
a data-preparation utility rather than part of the analysis
methodology proper. No hypothesis-testing logic, threshold, or model
specification changed between the two confirmatory windows.

## Held-out window
Features as of 2025-12-31; labels from trades in
2026-01-01 through 2026-03-29 (2026-Q1, ~3 months).
Not accessed for any tuning, threshold selection, or model comparison
prior to this document.

## Confirmatory hypotheses and PASS criteria
Same five hypotheses (H1-H5) and same models/specifications as
PRE_ANALYSIS_PLAN.md, applied to this window. Given the shorter window
and correspondingly lower power, PASS/FAIL is evaluated but explicitly
interpreted as directional confirmation of the 2025-H2 shift rather
than independent statistical proof.

H1: |rho| < 0.05, CI contains 0 -> null-persistence criterion
H2: coefficient negative, p<0.05 (Holm-adj.) -> reversal criterion
H3: coefficient positive, p<0.05 (Holm-adj.) -> specialization criterion
H4: 95% CI contains 0.5 -> null-ML-signal criterion
H5: CI excludes 0.5, AUC>0.5 -> insider-flow criterion

## Commitment
No further changes to the nine frozen analysis scripts will be made
after this document is saved and hashed. Results are reported
regardless of outcome, explicitly framed relative to both the 2025-H1
development sample and the 2025-H2 first confirmatory window.