import re
from pathlib import Path

folder = Path("research/tables_v2")

for file in folder.glob("*.tex"):
    text = file.read_text(encoding="utf-8")

    match = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, re.DOTALL)
    if not match:
        print(f"NO TABULAR FOUND: {file.name}")
        continue

    tabular = match.group(0)

    # Escape underscores that aren't already escaped
    tabular = re.sub(r"(?<!\\)_", r"\\_", tabular)

    file.write_text(tabular + "\n", encoding="utf-8")
    print(f"Fixed: {file.name}")