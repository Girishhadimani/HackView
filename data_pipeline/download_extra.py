"""
Additional Dataset Downloaders
================================
Scripts to download larger face-specific deepfake datasets.

1. Celeb-DF v2 (requires signing a form, free academic use)
2. 140k Real & Fake Faces (via Kaggle API)
3. OpenForensics (direct download, no registration)
"""

import os
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path
import urllib.request


# ---------------------------------------------------------------------------
# OpenForensics — Direct download, no registration
# ---------------------------------------------------------------------------

OPENFORENSICS_URL = "https://zenodo.org/record/5528418/files/OpenForensics.zip"


def download_openforensics(output_dir: str = "data"):
    """
    Download OpenForensics dataset (9.3 GB).
    No registration required — hosted on Zenodo.
    Contains: multi-face deepfake detection, 334,000 face crops.
    """
    output_path = Path(output_dir)
    zip_path    = output_path / "OpenForensics.zip"
    output_path.mkdir(parents=True, exist_ok=True)

    print("[*] Downloading OpenForensics (9.3 GB)...")
    print("    This will take 15–30 minutes depending on your connection.")

    def progress(block_num, block_size, total_size):
        pct = min(block_num * block_size * 100 / total_size, 100)
        print(f"\r    {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(OPENFORENSICS_URL, zip_path, reporthook=progress)
    print()

    print("[*] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_path)

    print(f"[OK] OpenForensics extracted to {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Kaggle 140k Real & Fake Faces
# ---------------------------------------------------------------------------

def download_140k_kaggle(output_dir: str = "data"):
    """
    Download 140k Real & Fake Faces via Kaggle API.

    Prerequisites:
        1. pip install kaggle
        2. Get API token from https://www.kaggle.com/settings → API → Create New Token
        3. Place kaggle.json in ~/.kaggle/kaggle.json

    Dataset: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
    """
    try:
        import kaggle
    except ImportError:
        print("[!] Installing kaggle...")
        os.system(f"{sys.executable} -m pip install kaggle --quiet")
        import kaggle

    output_path = Path(output_dir) / "kaggle_140k"
    output_path.mkdir(parents=True, exist_ok=True)

    print("[*] Downloading 140k Real & Fake Faces from Kaggle (~3.7 GB)...")
    print("    Requires kaggle.json API key in ~/.kaggle/")

    try:
        import subprocess
        subprocess.run([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "xhlulu/140k-real-and-fake-faces",
            "-p", str(output_path), "--unzip"
        ], check=True)
        print(f"[OK] Downloaded to {output_path}")
        _organize_140k(output_path, Path(output_dir))
    except Exception as e:
        print(f"[!] Kaggle download failed: {e}")
        print("    Manual steps:")
        print("    1. Go to https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
        print("    2. Download and extract to data/kaggle_140k/")
        print("    3. Run: python data_pipeline/download_extra.py --organize")


def _organize_140k(src: Path, dst: Path):
    """Reorganize 140k dataset into train/val/test splits."""
    print("[*] Organizing dataset structure...")
    for split in ["train", "valid", "test"]:
        src_split = src / split
        if not src_split.exists():
            continue
        for cls in ["real", "fake"]:
            src_cls = src_split / cls
            dst_cls = dst / ("val" if split == "valid" else split) / cls
            dst_cls.mkdir(parents=True, exist_ok=True)
            if src_cls.exists():
                for f in src_cls.glob("*.jpg"):
                    shutil.copy2(f, dst_cls / f.name)
    print("[OK] Organized into train/val/test structure")


# ---------------------------------------------------------------------------
# FaceForensics++ Instructions
# ---------------------------------------------------------------------------

def print_ffpp_instructions():
    """Print manual download instructions for FaceForensics++."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          FaceForensics++ Download Instructions               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  FaceForensics++ requires a one-time form submission:       ║
║                                                              ║
║  1. Fill form at:                                            ║
║     https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5      ║
║     5ruOperXv0SBDQ8dZRJVhfEQ3OlRpfvn7V7I5tGEg/viewform    ║
║                                                              ║
║  2. Receive download script via email                        ║
║                                                              ║
║  3. Run: python faceforensics_download_v4.py               ║
║           -d <download_dir>                                  ║
║           -c c23                                             ║
║           -t face                                            ║
║                                                              ║
║  4. Then run: python data_pipeline/prep_ffpp.py             ║
║                --data_dir <download_dir>                     ║
║                --output_dir data/                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


# ---------------------------------------------------------------------------
# Celeb-DF v2 Instructions
# ---------------------------------------------------------------------------

def print_celebdf_instructions():
    """Print manual download instructions for Celeb-DF v2."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Celeb-DF v2 Download Instructions                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Celeb-DF requires a Google Form for academic access:       ║
║                                                              ║
║  1. Fill form at:                                            ║
║     https://docs.google.com/forms/d/e/               ║
║     1FAIpQLScKI3V6kBOjqJLF09JZx5NDdFKpVBmjmkrUm_E  ║
║     eZi-w0eG_w/viewform                                     ║
║                                                              ║
║  2. Receive Google Drive link via email (~2.6 GB)           ║
║                                                              ║
║  3. Extract and run:                                         ║
║     python data_pipeline/prep_celebdf.py                    ║
║             --data_dir <extracted_dir>                       ║
║             --output_dir data/                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["140k", "openforensics", "info"], default="info")
    p.add_argument("--output_dir", default="data")
    args = p.parse_args()

    if args.dataset == "140k":
        download_140k_kaggle(args.output_dir)
    elif args.dataset == "openforensics":
        download_openforensics(args.output_dir)
    else:
        print_ffpp_instructions()
        print_celebdf_instructions()
