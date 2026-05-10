"""Evaluation helpers for RACE traditional ML models."""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Compute classification and regression-style metrics.

    Args:
        y_true: Ground-truth labels or scores.
        y_pred: Predicted labels or scores.
        y_proba: Optional continuous probability/score predictions.

    Returns:
        Dictionary of metrics.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    score_values = np.asarray(y_proba) if y_proba is not None else y_pred_arr
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "macro_f1": f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "precision": precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "recall": recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "exact_match": float(np.mean(y_true_arr.astype(str) == y_pred_arr.astype(str))),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr),
        "mse": mean_squared_error(y_true_arr.astype(float), score_values.astype(float)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr.astype(float), score_values.astype(float)))),
        "mae": mean_absolute_error(y_true_arr.astype(float), score_values.astype(float)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_arr, y_proba)
            metrics["pr_auc"] = average_precision_score(y_true_arr, y_proba)
            metrics["brier_score"] = brier_score_loss(y_true_arr, y_proba)
        except Exception:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0
            metrics["brier_score"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
        metrics["brier_score"] = 0.0
    try:
        metrics["r2"] = r2_score(y_true_arr.astype(float), score_values.astype(float))
    except Exception:
        metrics["r2"] = 0.0
    return metrics


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str) -> plt.Figure:
    """Create an annotated seaborn confusion-matrix figure.

    Args:
        cm: Confusion matrix array.
        labels: Axis labels.
        title: Figure title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    return fig


def benchmark_all_models(data_dir: str = os.path.join("data", "processed"), model_dir: str = "models") -> pd.DataFrame:
    """Benchmark all saved Model A classifiers on the test set.

    Args:
        data_dir: Processed data directory.
        model_dir: Root model directory.

    Returns:
        DataFrame of benchmark metrics.
    """
    try:
        X_test = sparse.load_npz(os.path.join(data_dir, "X_test_combined.npz"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load test data from {data_dir}: {exc}") from exc
    model_a_dir = os.path.join(model_dir, "model_a", "traditional")
    candidates = {
        "Logistic Regression": "logistic_regression.pkl",
        "Calibrated LinearSVC": "linear_svc_calibrated.pkl",
        "Stacking Ensemble": "stacking_ensemble.pkl",
        "XGBoost Optional": "xgboost_optional.pkl",
    }
    rows: list[dict[str, Any]] = []
    for name, filename in candidates.items():
        path = os.path.join(model_a_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            model = joblib.load(path)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
                pred = (proba >= 0.5).astype(int)
            else:
                pred = model.predict(X_test)
                proba = None
            metrics = compute_all_metrics(y_test, pred, proba)
            rows.append({"model": name, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
    result = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    try:
        result.to_csv(os.path.join("results", "benchmark.csv"), index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to save benchmark CSV: {exc}") from exc
    print(result.to_string(index=False) if not result.empty else "No models found to benchmark.")
    return result


def compute_generation_metrics(references: list[list[str]], candidates: list[str]) -> dict[str, float]:
    """Compute BLEU, ROUGE-L and METEOR between candidate strings and reference sets.

    Args:
        references: List of reference lists (each inner list are acceptable references for a sample).
        candidates: List of candidate strings.

    Returns:
        Dict with average BLEU, ROUGE-L F1, and METEOR scores.
    """
    if not candidates:
        return {"bleu": 0.0, "rouge_l": 0.0, "meteor": 0.0}
    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    blues = []
    rouges = []
    meteors = []
    for refs, cand in zip(references, candidates):
        # Prepare token lists for BLEU
        ref_tokens = [r.split() for r in refs if r]
        cand_tokens = cand.split()
        try:
            b = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smoothie)
        except Exception:
            b = 0.0
        try:
            m = meteor_score(refs, cand)
        except Exception:
            m = 0.0
        try:
            r = rouge.score(" ".join(refs), cand)["rougeL"].fmeasure
        except Exception:
            r = 0.0
        blues.append(b)
        rouges.append(r)
        meteors.append(m)
    return {"bleu": float(np.mean(blues)), "rouge_l": float(np.mean(rouges)), "meteor": float(np.mean(meteors))}


def main() -> None:
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark saved RACE models.")
    parser.add_argument("--data-dir", default=os.path.join("data", "processed"))
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    benchmark_all_models(args.data_dir, args.model_dir)


if __name__ == "__main__":
    main()
