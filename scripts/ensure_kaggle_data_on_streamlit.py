"""Runtime Kaggle dataset download helper for Streamlit Cloud.

This script is intended to be imported/used from `ui/app.py`.

Goal: ensure `data/raw/train.csv`, `data/raw/val.csv`, and `data/raw/test.csv`
exist in the deployed filesystem.

Behavior:
- If split CSVs already exist, do nothing.
- Otherwise, tries to download from Kaggle using the `kaggle` CLI.
- Requires Kaggle authentication:
  - `KAGGLE_API_TOKEN` environment variable (recommended for Cloud), OR
  - a `kaggle.json`/access_token present in the container.

Notes:
- Streamlit Cloud may not have outbound network access depending on your plan.
- If download fails, the app continues in demo mode (fallback inference).
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from typing import Optional

import pandas as pd


DEFAULT_DATASET_SLUG = "ankitdhiman7/race-dataset"
REQUIRED_SPLITS = ("train.csv", "val.csv", "test.csv")


@dataclass(frozen=True)
class KaggleAuthStatus:
    configured: bool
    reason: str


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _raw_dir() -> str:
    return os.path.join(_project_root(), "data", "raw")


def split_csvs_exist(raw_dir: str) -> bool:
    return all(os.path.exists(os.path.join(raw_dir, s)) for s in REQUIRED_SPLITS)


def configure_kaggle_for_runtime() -> KaggleAuthStatus:
    """Ensure Kaggle credentials are present for the kaggle CLI.

    Supports:
    - env var: KAGGLE_API_TOKEN
    - existing kaggle.json/access_token in expected directories
    """
    kaggle_dir = os.path.expanduser(os.path.join("~", ".kaggle"))
    os.makedirs(kaggle_dir, exist_ok=True)

    json_target = os.path.join(kaggle_dir, "kaggle.json")
    token_target = os.path.join(kaggle_dir, "access_token")

    # If env var is provided, write access_token.
    env_token = (os.environ.get("KAGGLE_API_TOKEN") or "").strip()
    if env_token:
        try:
            with open(token_target, "w", encoding="utf-8") as f:
                f.write(env_token)
            try:
                os.chmod(token_target, 0o600)
            except OSError:
                # chmod may fail on some platforms; continue.
                pass
            return KaggleAuthStatus(True, "KAGGLE_API_TOKEN provided")
        except OSError as e:
            return KaggleAuthStatus(False, f"Failed to write access_token: {e}")

    # Otherwise, rely on pre-provided files.
    if os.path.exists(json_target) or os.path.exists(token_target):
        return KaggleAuthStatus(True, "Existing kaggle credentials detected")

    return KaggleAuthStatus(False, "No Kaggle credentials found (set KAGGLE_API_TOKEN or provide kaggle.json/access_token)")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]
    lower_to_original = {col.lower(): col for col in normalized.columns}

    aliases = {
        "id": ["id", "example_id"],
        "article": ["article", "passage", "context", "story"],
        "question": ["question", "query"],
        "A": ["a", "option_a", "optiona", "choice_a", "choicea"],
        "B": ["b", "option_b", "optionb", "choice_b", "choiceb"],
        "C": ["c", "option_c", "optionc", "choice_c", "choicec"],
        "D": ["d", "option_d", "optiond", "choice_d", "choiced"],
        "answer": ["answer", "label", "correct", "correct_answer", "correct_option"],
    }

    rename_map: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in lower_to_original:
                rename_map[lower_to_original[candidate]] = canonical
                break

    normalized = normalized.rename(columns=rename_map)

    if "id" not in normalized.columns:
        normalized.insert(0, "id", range(len(normalized)))

    if "answer" in normalized.columns:
        normalized["answer"] = (
            normalized["answer"].astype(str).str.strip().str.upper().replace({"0": "A", "1": "B", "2": "C", "3": "D"})
        )

    return normalized


def _has_split_csvs(raw_dir: str) -> bool:
    return split_csvs_exist(raw_dir)


def _candidate_source_csvs(raw_dir: str, include_split_files: bool = False) -> list[str]:
    split_names = {"train.csv", "val.csv", "test.csv"}
    paths: list[str] = []
    for csv_path in glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True):
        base = os.path.basename(csv_path).lower()
        if include_split_files or base not in split_names:
            paths.append(csv_path)
    paths.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return paths


def split_single_csv(raw_dir: str, source_csv: str, random_state: int = 42) -> None:
    if _has_split_csvs(raw_dir):
        return

    if not source_csv:
        raise RuntimeError("No source CSV found to split")

    df = pd.read_csv(source_csv)
    df = _normalize_columns(df)

    required = {"id", "article", "question", "A", "B", "C", "D", "answer"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Source CSV {source_csv} missing columns after normalization: {missing}")

    df = df[["id", "article", "question", "A", "B", "C", "D", "answer"]].copy()
    df = df[df["answer"].isin(["A", "B", "C", "D"])].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after filtering answer labels to A/B/C/D")

    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    train_end = int(len(shuffled) * 0.80)
    val_end = int(len(shuffled) * 0.90)

    splits = {
        "train": shuffled.iloc[:train_end],
        "val": shuffled.iloc[train_end:val_end],
        "test": shuffled.iloc[val_end:],
    }

    for split, split_df in splits.items():
        destination = os.path.join(raw_dir, f"{split}.csv")
        split_df.to_csv(destination, index=False)


def _get_kaggle_bin() -> str:
    # Prefer PATH kaggle.
    return "kaggle"


def download_and_prepare(raw_dir: str, dataset_slug: str = DEFAULT_DATASET_SLUG) -> None:
    os.makedirs(raw_dir, exist_ok=True)

    if _has_split_csvs(raw_dir):
        return

    auth = configure_kaggle_for_runtime()
    if not auth.configured:
        raise RuntimeError(auth.reason)

    kaggle_bin = _get_kaggle_bin()

    # Download into raw_dir.
    # Streamlit Cloud runs in a Linux container typically; but works cross-platform.
    run_cmd = [kaggle_bin, "datasets", "download", "-d", dataset_slug, "-p", raw_dir]
    subprocess.run(run_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Extract any zip files.
    for zip_path in glob.glob(os.path.join(raw_dir, "*.zip")):
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(raw_dir)
        except zipfile.BadZipFile:
            # Ignore broken zips.
            pass

    # If train/val/test already appeared, normalize copy.
    # Otherwise split from largest non-split CSV.
    if not _has_split_csvs(raw_dir):
        candidates = _candidate_source_csvs(raw_dir, include_split_files=False)
        source = candidates[0] if candidates else ""
        split_single_csv(raw_dir=raw_dir, source_csv=source)


def ensure_dataset_available(
    raw_dir: Optional[str] = None,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    force: bool = False,
) -> bool:
    """Return True if dataset CSVs are available after the operation."""

    rd = raw_dir or _raw_dir()
    if not force and split_csvs_exist(rd):
        return True

    # If force, remove old split CSVs.
    if force:
        for s in REQUIRED_SPLITS:
            p = os.path.join(rd, s)
            if os.path.exists(p):
                os.remove(p)

    download_and_prepare(rd, dataset_slug=dataset_slug)
    return split_csvs_exist(rd)

