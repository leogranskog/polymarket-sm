# Polymarket Smart Money: Skill, Persistence, and Model Staleness in a Decentralized Prediction Market

Research project studying whether trader skill on Polymarket is identifiable
from behavioral features alone, whether it persists over time, and whether
machine learning models trained to detect it remain valid as the trader
population evolves.

## Status

Actively in progress, targeting a submission (ICAIF and/or a finance journal).
See `paper/` for the current draft.

## Project structure

```
polymarket-sm/
  run.py                        Single entry point for the full pipeline
  config.py                     Settings and constants
  bootstrap/
    download_dataset.py         HuggingFace dataset download
  research/
    pit_features.py             CORE: point-in-time feature + CLV panel builder
    master_numbers.py           SINGLE SOURCE OF TRUTH for all paper numbers
    master_numbers_addendum.py  Extends master_numbers.py (matched-horizon,
                                 leakage exhibit re-verification)
    make_verified_figures.py    Regenerates figures directly from verified
                                 numbers (guarantees figure/text consistency)
    PRE_ANALYSIS_PLAN.md        Pre-registration, confirmatory window 1
    PRE_ANALYSIS_PLAN_2.md      Pre-registration, confirmatory window 2
    true_oos_final.py           Confirmatory window 1 (hash-verified, single-look)
    true_oos_second_window.py   Confirmatory window 2 (hash-verified, single-look)
    freeze_hash.py               Hash utility, window 1
    freeze_hash_2.py             Hash utility, window 2 (analysis-scripts only)
    MASTER_NUMBERS.md            Generated report: every number in the paper
    TRUE_OOS_RUN_LOG.txt         Permanent record, confirmatory run 1
    TRUE_OOS_2_RUN_LOG.txt       Permanent record, confirmatory run 2
    tables_v2/                   Generated tables (CSV + LaTeX)
    figures_v2/                  Generated figures (PDF + PNG)
    models_v2/                   Saved model artifacts
  data/
    raw/                        Downloaded dataset (gitignored, large)
    processed/pit/               Point-in-time panel (gitignored, large)
  paper/
    icaif_submission_verified.tex   Current ICAIF-format draft
    paper_final_verified.tex        Fuller journal-format draft
```

## Methodology at a glance

- **Point-in-time panel**: every behavioral feature is computed only from
  trades before a fixed cutoff date; every label only from trades after it.
  No feature is ever allowed to use information from its own label period.
- **Closing-Line Value (CLV)**: skill measured as entry price vs. a
  lifetime-adaptive volume-weighted price before resolution (a tiered
  fallback scheme, not the literal last trade, which mechanically converges
  toward the outcome).
- **Pre-registration**: before accessing each of two confirmatory windows, a
  SHA-256 hash of the frozen analysis code and explicit pass/fail criteria
  for five hypotheses were recorded. The confirmatory scripts verify this
  hash and refuse to run more than once.
- **Single source of truth**: `research/master_numbers.py` (+ its addendum)
  is the *only* script that should be used to generate numbers cited in the
  paper. If a number changes, it changes because this script was rerun,
  documented with a timestamp and script hash in `MASTER_NUMBERS.md`.

## Reproducing the pipeline

```bash
# 1. Download the dataset (one-time, ~large)
python run.py download

# 2. Build the point-in-time panel (one-time, ~1-3 hours depending on hardware)
python run.py pit

# 3. Generate every verified number cited in the paper
python -m research.master_numbers
python -m research.master_numbers_addendum

# 4. Regenerate paper figures directly from verified numbers
python -m research.make_verified_figures
```

The two confirmatory-window scripts (`true_oos_final.py`,
`true_oos_second_window.py`) are guarded: each checks its corresponding
`PRE_ANALYSIS_PLAN*.md` hash before running, and refuses to run a second
time once a run log exists. They are not part of the normal reproduction
path above, they were each run exactly once, by design.

## Key findings (see paper for full detail and caveats)

1. Terminal-snapshot features (a wallet's full history, including the label
   period) inflate apparent predictive AUC from ~0.54 to an artificial
   0.62-0.68, a leakage failure mode common to this class of public dataset.
2. A frozen ML classifier's predictive power degrades under population
   drift; refitting on contemporaneous data recovers meaningful accuracy,
   verified leakage-free via a placebo test.
3. Wallet-level skill persistence, negligible early in the platform's
   history, strengthens markedly across two confirmatory windows, and
   survives a within-cohort check ruling out population-composition effects.
4. A category-specialization effect that is significant in the first
   confirmatory window reverses sign in the second; reported directly as a
   non-replication.

## Contact

Leo — leo.granskog@gmail.com
