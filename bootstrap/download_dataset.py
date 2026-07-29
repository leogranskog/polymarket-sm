import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from huggingface_hub import hf_hub_download
import polars as pl
from config import HF_DATASET, HF_SUBSETS, RAW_DIR

RAW_DIR.mkdir(parents=True, exist_ok=True)

SUBSET_FILES = {
    "user_pnl_summary": "user_pnl_summary.parquet",
    "user_features":    "user_features.parquet",
    "markets":          "markets.parquet",
    "events":           "events.parquet",
}

def download_subset(subset: str) -> Path:
    filename = SUBSET_FILES.get(subset)
    if not filename:
        raise ValueError(f"Unknown subset: {subset}")

    local_path = RAW_DIR / filename

    if local_path.exists():
        size_mb = local_path.stat().st_size / 1_000_000
        print(f"  ✓ {subset} already downloaded ({size_mb:.1f} MB) — skipping")
        return local_path

    print(f"  ↓ Downloading {subset}...")
    downloaded = hf_hub_download(
        repo_id=HF_DATASET,
        filename=filename,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
    )
    size_mb = Path(downloaded).stat().st_size / 1_000_000
    print(f"  ✅ {subset} saved ({size_mb:.1f} MB)")
    return Path(downloaded)


def download_all(subsets: list = None) -> dict:
    subsets = subsets or HF_SUBSETS
    print(f"\n{'='*55}")
    print(f"  Downloading {len(subsets)} subsets from HuggingFace")
    print(f"{'='*55}")

    paths = {}
    for subset in subsets:
        try:
            paths[subset] = download_subset(subset)
        except Exception as e:
            print(f"  ❌ Failed to download {subset}: {e}")

    print(f"\n  Done. {len(paths)}/{len(subsets)} subsets ready.")
    return paths


def preview_subset(subset: str, n: int = 5):
    path = RAW_DIR / SUBSET_FILES[subset]
    if not path.exists():
        print(f"  Not downloaded yet: {subset}")
        return
    df = pl.read_parquet(path)
    print(f"\n── {subset} ──────────────────────────────")
    print(f"  Rows: {len(df):,}   Cols: {len(df.columns)}")
    print(f"  Columns: {df.columns}")
    print(df.head(n))


if __name__ == "__main__":
    download_all()