"""
Download IMDB Sentiment Dataset
================================
Downloads and extracts the Stanford IMDB Large Movie Review Dataset
into data/aclImdb/ for use with the ML Model Auditor.

Usage:  python download_imdb.py
"""

import os
import pathlib
import tarfile
import urllib.request

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
ARCHIVE_PATH = DATA_DIR / "aclImdb_v1.tar.gz"
EXTRACT_DIR = DATA_DIR / "aclImdb"


def download_imdb():
    """Download and extract the IMDB dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download
    if not ARCHIVE_PATH.exists():
        print(f"Downloading IMDB dataset from {URL} ...")
        print("(This is ~84 MB and may take a few minutes)")
        urllib.request.urlretrieve(URL, ARCHIVE_PATH, _progress_hook)
        print("\nDownload complete!")
    else:
        print(f"Archive already exists: {ARCHIVE_PATH}")

    # Extract
    if not EXTRACT_DIR.exists():
        print("Extracting dataset...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        print(f"Extracted to {EXTRACT_DIR}")
    else:
        print(f"Dataset already extracted: {EXTRACT_DIR}")

    # Verify
    train_pos = EXTRACT_DIR / "train" / "pos"
    train_neg = EXTRACT_DIR / "train" / "neg"
    test_pos = EXTRACT_DIR / "test" / "pos"
    test_neg = EXTRACT_DIR / "test" / "neg"

    for d in (train_pos, train_neg, test_pos, test_neg):
        count = len(list(d.glob("*.txt"))) if d.exists() else 0
        print(f"  {d.relative_to(DATA_DIR)}: {count} files")

    print("\nDone! You can now select 'IMDB Sentiment' in the ML Model Auditor.")


def _progress_hook(block_num, block_size, total_size):
    """Simple download progress indicator."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        print(f"\r  Progress: {pct}% ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)", end="")


if __name__ == "__main__":
    download_imdb()
