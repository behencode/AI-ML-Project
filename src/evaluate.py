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
)


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


def main() -> None:
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark saved RACE models.")
    parser.add_argument("--data-dir", default=os.path.join("data", "processed"))
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    benchmark_all_models(args.data_dir, args.model_dir)


if __name__ == "__main__":
    main()
