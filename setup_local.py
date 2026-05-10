"""Local setup helper for the RACE reading-comprehension project."""

from __future__ import annotations

import argparse
import os

from setup_colab import (
    DEFAULT_DATASET_SLUG,
    configure_kaggle,
    create_directories,
    download_race_dataset,
    install_packages,
    split_single_csv,
    verify_csvs,
)
import sys


def check_venv() -> bool:
    """Check if the script is running inside a virtual environment.

    Returns:
        True if in a venv, otherwise False.
    """
    return sys.prefix != sys.base_prefix


def main() -> None:
    """Set up dependencies, folders, Kaggle data, and local train/val/test CSVs."""
    parser = argparse.ArgumentParser(description="Set up the project for local execution.")
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG, help="Kaggle dataset slug.")
    parser.add_argument("--source-csv", default="", help="Local single CSV to split into train/val/test.")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install.")
    parser.add_argument("--skip-download", action="store_true", help="Do not download from Kaggle.")
    parser.add_argument("--force-resplit", action="store_true", help="Overwrite train/val/test with a fresh 80/10/10 split.")
    args = parser.parse_args()

    if not check_venv() and not args.skip_install:
        print("Error: You are not in a virtual environment.")
        print("To fix this, run:")
        print("  source venv/bin/activate  # macOS/Linux")
        print("  .\\venv\\Scripts\\activate  # Windows")
        print("\nIf you want to continue anyway, use --skip-install.")
        sys.exit(1)

    create_directories()
    if not args.skip_install:
        install_packages()

    if args.source_csv:
        split_single_csv(source_csv=args.source_csv, force=True)
    elif not args.skip_download:
        kaggle_ready = configure_kaggle()
        if kaggle_ready:
            download_race_dataset(args.dataset_slug)
        else:
            print("Kaggle credentials not found. Put a CSV in data/raw and rerun with --source-csv.")

    if args.force_resplit:
        split_single_csv(force=True)
    else:
        split_single_csv()

    verify_csvs()
    print("\nLocal setup complete.")
    print("Next: python src/preprocessing.py --sample-size 5000 --max-features 10000")


if __name__ == "__main__":
    main()
