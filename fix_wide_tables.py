import pandas as pd
from pathlib import Path

folder = Path("research/tables_v2")

# Tables that are too wide or came out malformed — regenerate from source CSV
PROBLEM_TABLES = {
    "t_appendix_covariate_shift": 3,
    "t14_volume_weighted_persistence": 3,
    "t9_spec_stability": 3,
    "t18_market_insider_outcome": 3,
}

for name, decimals in PROBLEM_TABLES.items():
    csv_path = folder / f"{name}.csv"
    if not csv_path.exists():
        print(f"Missing CSV: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # Round all numeric columns to a shorter, page-friendly precision
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].round(decimals)

    # Shorten column headers (spaces instead of underscores, abbreviate)
    df.columns = [c.replace("_", " ") for c in df.columns]

    tex = df.to_latex(index=False, float_format=f"%.{decimals}f")

    out_path = folder / f"{name}.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"Regenerated: {out_path.name}  ({len(df.columns)} columns)")