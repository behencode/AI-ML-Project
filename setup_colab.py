"""Google Colab bootstrap for the RACE reading-comprehension project."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile
from typing import Iterable



DEFAULT_DATASET_SLUG = "ankitdhiman7/race-dataset"
RANDOM_STATE = 42

PROJECT_DIRS = [
    os.path.join("data", "raw"),
    os.path.join("src"),
    os.path.join("ui"),
    os.path.join("models", "model_a", "traditional"),
    os.path.join("models", "model_b", "traditional"),
    os.path.join("notebooks"),
    os.path.join("tests"),
    os.path.join("results"),
    os.path.join("reports"),
]


def run_command(command: list[str]) -> None:
    """Run a shell command and raise an informative error on failure.

    Args:
        command: Command tokens to execute.

    Returns:
        None.
    """
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed: {' '.join(command)}") from exc


def install_packages(requirements_path: str = "requirements.txt") -> None:
    """Install project dependencies with pip.

    Args:
        requirements_path: Path to requirements file.

    Returns:
        None.
    """
    if not os.path.exists(requirements_path):
        raise FileNotFoundError(f"Missing requirements file: {requirements_path}")
    run_command([sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path])


def create_directories(directories: Iterable[str] = PROJECT_DIRS) -> None:
    """Create the project directory layout.

    Args:
        directories: Directory paths to create.

    Returns:
        None.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory: {directory}")


def configure_kaggle(kaggle_json_path: str = "kaggle.json", access_token_path: str = "access_token") -> bool:
    """Configure Kaggle credentials for Colab.

    Args:
        kaggle_json_path: Path to the legacy Kaggle JSON token.
        access_token_path: Path to a new-style Kaggle access-token file.

    Returns:
        True if credentials were configured, otherwise False.
    """
    kaggle_dir = os.path.expanduser(os.path.join("~", ".kaggle"))
    if "/root" in kaggle_dir and os.name != "nt" and os.getuid() != 0:
        # Safety fallback for misconfigured environments or relocated venvs
        kaggle_dir = os.path.join(os.getcwd(), ".kaggle")
        print(f"Warning: ~ expansion failed or pointed to /root. Using local directory: {kaggle_dir}")
    os.makedirs(kaggle_dir, exist_ok=True)
    json_target = os.path.join(kaggle_dir, "kaggle.json")
    token_target = os.path.join(kaggle_dir, "access_token")
    if os.path.exists(kaggle_json_path):
        try:
            shutil.copy(kaggle_json_path, json_target)
            os.chmod(json_target, 0o600)
            print("Kaggle JSON credentials configured.")
            return True
        except OSError as exc:
            raise RuntimeError("Failed to configure Kaggle JSON credentials.") from exc
    if os.path.exists(access_token_path):
        try:
            shutil.copy(access_token_path, token_target)
            os.chmod(token_target, 0o600)
            print("Kaggle access token configured from local access_token file.")
            return True
        except OSError as exc:
            raise RuntimeError("Failed to configure Kaggle access token.") from exc
    env_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_token:
        try:
            with open(token_target, "w", encoding="utf-8") as file:
                file.write(env_token)
            os.chmod(token_target, 0o600)
            print("Kaggle access token configured from KAGGLE_API_TOKEN.")
            return True
        except OSError as exc:
            raise RuntimeError("Failed to write Kaggle access token from environment.") from exc
    if os.path.exists(json_target) or os.path.exists(token_target):
        print("Existing Kaggle credentials found in ~/.kaggle.")
        return True
    print("No Kaggle credentials found. Upload kaggle.json, create ~/.kaggle/access_token, or set KAGGLE_API_TOKEN.")
    return False


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common RACE column-name variants to the project schema.

    Args:
        df: Raw Kaggle DataFrame.

    Returns:
        DataFrame with canonical columns where possible.
    """
    import pandas as pd
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
        normalized["answer"] = normalized["answer"].astype(str).str.strip().str.upper()
        numeric_map = {"0": "A", "1": "B", "2": "C", "3": "D"}
        normalized["answer"] = normalized["answer"].replace(numeric_map)
    return normalized


def normalize_downloaded_csvs(raw_dir: str) -> None:
    """Move train/val/test CSV files discovered in nested Kaggle folders into data/raw.

    Args:
        raw_dir: Directory containing extracted Kaggle files.

    Returns:
        None.
    """
    aliases = {"train": "train.csv", "dev": "val.csv", "val": "val.csv", "test": "test.csv"}
    for csv_path in glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True):
        name = os.path.splitext(os.path.basename(csv_path))[0].lower()
        if name in aliases:
            destination = os.path.join(raw_dir, aliases[name])
            if os.path.abspath(csv_path) != os.path.abspath(destination):
                shutil.copy(csv_path, destination)
                print(f"Copied {csv_path} -> {destination}")


def _has_split_csvs(raw_dir: str) -> bool:
    """Check whether the required train/val/test files already exist."""
    return all(os.path.exists(os.path.join(raw_dir, name)) for name in ["train.csv", "val.csv", "test.csv"])


def _candidate_source_csvs(raw_dir: str, include_split_files: bool = False) -> list[str]:
    """Find downloaded CSV files that can be used as a single source file.

    Args:
        raw_dir: Folder to search recursively.
        include_split_files: Whether train/val/test CSV names may be used as source files.

    Returns:
        Candidate CSV paths sorted by file size descending.
    """
    split_names = {"train.csv", "val.csv", "test.csv"}
    paths = []
    for csv_path in glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True):
        if include_split_files or os.path.basename(csv_path).lower() not in split_names:
            paths.append(csv_path)
    return sorted(paths, key=lambda path: os.path.getsize(path), reverse=True)


def split_single_csv(raw_dir: str = os.path.join("data", "raw"), source_csv: str = "", force: bool = False) -> None:
    """Split one Kaggle CSV into train/val/test using an 80/10/10 proportion.

    Args:
        raw_dir: Folder containing the downloaded CSV.
        source_csv: Optional explicit single CSV path. If omitted, the largest non-split CSV is used.
        force: Overwrite existing train/val/test files from a single source CSV.

    Returns:
        None.
    """
    import pandas as pd
    if _has_split_csvs(raw_dir) and not force:
        print("train.csv, val.csv, and test.csv already exist; skipping 80/10/10 split.")
        return
    candidates = _candidate_source_csvs(raw_dir, include_split_files=force)
    csv_path = source_csv or (candidates[0] if candidates else "")
    if not csv_path:
        print("No single source CSV found to split. Place the Kaggle CSV in data/raw or pass --source-csv.")
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read source CSV for splitting: {csv_path}. Details: {exc}") from exc
    df = _normalize_columns(df)
    required = {"id", "article", "question", "A", "B", "C", "D", "answer"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Source CSV {csv_path} is missing required columns after normalization: {missing}")
    df = df[["id", "article", "question", "A", "B", "C", "D", "answer"]].copy()
    df = df[df["answer"].isin(["A", "B", "C", "D"])].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows remained after filtering answer labels to A/B/C/D.")
    shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    train_end = int(len(shuffled) * 0.80)
    val_end = int(len(shuffled) * 0.90)
    splits = {
        "train": shuffled.iloc[:train_end],
        "val": shuffled.iloc[train_end:val_end],
        "test": shuffled.iloc[val_end:],
    }
    for split, split_df in splits.items():
        destination = os.path.join(raw_dir, f"{split}.csv")
        try:
            split_df.to_csv(destination, index=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to write {destination}: {exc}") from exc
        print(f"Wrote {destination} with shape {split_df.shape}")


def download_race_dataset(dataset_slug: str = DEFAULT_DATASET_SLUG, raw_dir: str = os.path.join("data", "raw")) -> None:
    """Download and extract the RACE dataset from Kaggle.

    Args:
        dataset_slug: Kaggle dataset slug, for example ``username/dataset-name``.
        raw_dir: Destination folder for CSV files.

    Returns:
        None.
    """
    os.makedirs(raw_dir, exist_ok=True)
    if _has_split_csvs(raw_dir):
        print("Existing split CSVs found. Keeping download step skipped; use --force-resplit to overwrite them.")
        return
    if not dataset_slug:
        print("No Kaggle dataset slug supplied. Place one RACE CSV in data/raw or pass --source-csv.")
        return
    
    # Use the kaggle binary from the same directory as the current python executable
    kaggle_bin = "kaggle"
    python_dir = os.path.dirname(sys.executable)
    local_kaggle = os.path.join(python_dir, "kaggle")
    if os.path.exists(local_kaggle):
        kaggle_bin = local_kaggle
        
    run_command([kaggle_bin, "datasets", "download", "-d", dataset_slug, "-p", raw_dir])
    for zip_path in glob.glob(os.path.join(raw_dir, "*.zip")):
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(raw_dir)
            print(f"Extracted {zip_path}")
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Downloaded file is not a valid zip: {zip_path}") from exc
    normalize_downloaded_csvs(raw_dir)


def verify_csvs(raw_dir: str = os.path.join("data", "raw")) -> None:
    """Print CSV shapes and required-column checks.

    Args:
        raw_dir: Folder containing train.csv, val.csv, and test.csv.

    Returns:
        None.
    """
    import pandas as pd
    required = {"id", "article", "question", "A", "B", "C", "D", "answer"}
    for split in ["train", "val", "test"]:
        path = os.path.join(raw_dir, f"{split}.csv")
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Missing {path}")
            continue
        except Exception as exc:
            print(f"Could not read {path}: {exc}")
            continue
        missing = sorted(required.difference(df.columns))
        print(f"{split}.csv shape: {df.shape}")
        if missing:
            print(f"  Missing columns: {missing}")


def main() -> None:
    """Parse CLI arguments and run the Colab setup workflow."""
    parser = argparse.ArgumentParser(description="Set up the RACE quiz generation project on Colab.")
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG, help="Kaggle dataset slug containing one RACE CSV.")
    parser.add_argument("--source-csv", default="", help="Optional local single CSV to split into 80/10/10.")
    parser.add_argument("--force-resplit", action="store_true", help="Overwrite train/val/test with a fresh 80/10/10 split.")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install.")
    args = parser.parse_args()
    create_directories()
    if not args.skip_install:
        install_packages()
    kaggle_ready = configure_kaggle()
    if kaggle_ready and not args.source_csv:
        download_race_dataset(args.dataset_slug)
    elif not args.source_csv:
        print("Skipping Kaggle download until kaggle.json is uploaded. You can still split a local CSV with --source-csv.")
    if args.source_csv:
        split_single_csv(source_csv=args.source_csv, force=True)
    elif args.force_resplit or args.dataset_slug == DEFAULT_DATASET_SLUG:
        split_single_csv(force=True)
    else:
        split_single_csv()
    verify_csvs()


if __name__ == "__main__":
    main()
