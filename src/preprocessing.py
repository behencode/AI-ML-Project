"""Preprocessing utilities for traditional ML on the RACE dataset."""

from __future__ import annotations

import argparse
import os
import re
import string
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RANDOM_STATE = 42
OPTION_LABELS = ["A", "B", "C", "D"]


def clean_text(text: object) -> str:
    """Normalize text by lowercasing, removing punctuation, and compacting whitespace.

    Args:
        text: Input text or value coercible to text.

    Returns:
        Cleaned text string.
    """
    if pd.isna(text):
        return ""
    lowered = str(text).lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", no_punct).strip()


def expand_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Convert each four-option RACE row into four binary option samples.

    Args:
        df: DataFrame with article, question, A, B, C, D, and answer columns.

    Returns:
        Expanded DataFrame with article, question, option, opt_label, correct, label, combined.
    """
    required = {"article", "question", "A", "B", "C", "D", "answer"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        article = clean_text(row["article"])
        question = clean_text(row["question"])
        answer = str(row["answer"]).strip().upper()
        for opt_label in OPTION_LABELS:
            option = clean_text(row[opt_label])
            label = int(opt_label == answer)
            records.append(
                {
                    "article": article,
                    "question": question,
                    "option": option,
                    "opt_label": opt_label,
                    "correct": answer,
                    "label": label,
                    "combined": f"article {article} question {question} option {option}",
                }
            )
    return pd.DataFrame.from_records(records)


def _word_set(text: str) -> set[str]:
    """Return unique non-empty tokens from text."""
    return {token for token in clean_text(text).split() if token}


def _overlap_ratio(left: str, right: str) -> float:
    """Compute overlap ratio against the smaller non-empty token set."""
    left_set, right_set = _word_set(left), _word_set(right)
    denom = max(1, min(len(left_set), len(right_set)))
    return len(left_set.intersection(right_set)) / denom


def extract_lexical_features(expanded_df: pd.DataFrame) -> np.ndarray:
    """Extract six dense lexical features for expanded binary samples.

    Args:
        expanded_df: Output of ``expand_to_binary``.

    Returns:
        ``float32`` NumPy array with shape ``(n_samples, 6)``.
    """
    features: list[list[float]] = []
    for row in expanded_df.itertuples(index=False):
        article = getattr(row, "article")
        question = getattr(row, "question")
        option = getattr(row, "option")
        features.append(
            [
                _overlap_ratio(article, option),
                _overlap_ratio(question, option),
                _overlap_ratio(article, question),
                len(article.split()) / 100.0,
                len(question.split()) / 10.0,
                len(option.split()) / 10.0,
            ]
        )
    return np.asarray(features, dtype=np.float32)


def build_ohe_vectorizer(max_features: int = 10000) -> CountVectorizer:
    """Build the primary binary one-hot text vectorizer.

    Args:
        max_features: Maximum vocabulary size.

    Returns:
        Configured ``CountVectorizer``.
    """
    return CountVectorizer(binary=True, stop_words="english", ngram_range=(1, 2), min_df=2, max_features=max_features)


def build_tfidf_vectorizer(max_features: int = 10000) -> TfidfVectorizer:
    """Build the optional TF-IDF vectorizer.

    Args:
        max_features: Maximum vocabulary size.

    Returns:
        Configured ``TfidfVectorizer``.
    """
    return TfidfVectorizer(
        sublinear_tf=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=max_features,
    )


def _read_split(data_dir: str, split: str, sample_size: Optional[int]) -> pd.DataFrame:
    """Load one CSV split and optionally sample it reproducibly."""
    path = os.path.join(data_dir, f"{split}.csv")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    if sample_size and sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def _save_npz(path: str, matrix: sparse.spmatrix) -> None:
    """Save a sparse matrix with error handling."""
    try:
        sparse.save_npz(path, matrix)
    except Exception as exc:
        raise RuntimeError(f"Failed to save sparse matrix {path}: {exc}") from exc


def preprocess_pipeline(
    data_dir: str = os.path.join("data", "raw"),
    output_dir: str = os.path.join("data", "processed"),
    max_features: int = 10000,
    sample_size: Optional[int] = None,
) -> None:
    """Run the full preprocessing pipeline and persist matrices/vectorizers.

    Args:
        data_dir: Folder containing train.csv, val.csv, and test.csv.
        output_dir: Destination for processed matrices and vectorizers.
        max_features: Maximum text vocabulary size.
        sample_size: Optional number of source rows per split for quick runs.

    Returns:
        None.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Loading CSV splits...")
    train_df = _read_split(data_dir, "train", sample_size)
    val_df = _read_split(data_dir, "val", sample_size)
    test_df = _read_split(data_dir, "test", sample_size)

    print("Expanding rows to binary option samples...")
    expanded = {name: expand_to_binary(df) for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]}

    print("Fitting OHE and optional TF-IDF vectorizers...")
    ohe_vectorizer = build_ohe_vectorizer(max_features=max_features)
    tfidf_vectorizer = build_tfidf_vectorizer(max_features=max_features)
    X_train_ohe = ohe_vectorizer.fit_transform(expanded["train"]["combined"])
    X_val_ohe = ohe_vectorizer.transform(expanded["val"]["combined"])
    X_test_ohe = ohe_vectorizer.transform(expanded["test"]["combined"])
    tfidf_vectorizer.fit(expanded["train"]["combined"])

    matrices = {"train": X_train_ohe, "val": X_val_ohe, "test": X_test_ohe}
    for split, X_ohe in matrices.items():
        print(f"Building combined matrix for {split}...")
        lexical = sparse.csr_matrix(extract_lexical_features(expanded[split]))
        combined = sparse.hstack([X_ohe, lexical], format="csr")
        _save_npz(os.path.join(output_dir, f"X_{split}_ohe.npz"), X_ohe.tocsr())
        _save_npz(os.path.join(output_dir, f"X_{split}_combined.npz"), combined)
        y = expanded[split]["label"].to_numpy(dtype=np.int8)
        np.save(os.path.join(output_dir, f"y_{split}.npy"), y)
        expanded[split].to_csv(os.path.join(output_dir, f"{split}_expanded.csv.gz"), index=False, compression="gzip")
        print(f"{split} class balance: {float(np.mean(y)):.4f}")

    try:
        joblib.dump(ohe_vectorizer, os.path.join(output_dir, "ohe_vectorizer.pkl"))
        joblib.dump(tfidf_vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    except Exception as exc:
        raise RuntimeError(f"Failed to save vectorizers: {exc}") from exc
    print("Preprocessing complete.")


def main() -> None:
    """CLI entry point for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess RACE CSVs for traditional ML.")
    parser.add_argument("--data-dir", default=os.path.join("data", "raw"))
    parser.add_argument("--output-dir", default=os.path.join("data", "processed"))
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--sample-size", type=int, default=0)
    args = parser.parse_args()
    preprocess_pipeline(args.data_dir, args.output_dir, args.max_features, args.sample_size or None)


if __name__ == "__main__":
    main()
