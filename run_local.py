"""Run the Streamlit app locally with project-root paths."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def path_exists(path: str) -> str:
    """Return a compact status label for a path.

    Args:
        path: File or directory path.

    Returns:
        Human-readable status.
    """
    return "found" if os.path.exists(path) else "missing"


def print_preflight(project_root: str) -> None:
    """Print local readiness checks before Streamlit starts.

    Args:
        project_root: Absolute project root path.

    Returns:
        None.
    """
    checks = {
        "data/raw/train.csv": os.path.join(project_root, "data", "raw", "train.csv"),
        "data/processed/ohe_vectorizer.pkl": os.path.join(project_root, "data", "processed", "ohe_vectorizer.pkl"),
        "Model A LR": os.path.join(project_root, "models", "model_a", "traditional", "logistic_regression.pkl"),
        "Model A SVM": os.path.join(project_root, "models", "model_a", "traditional", "linear_svc_calibrated.pkl"),
        "Model B artifacts": os.path.join(project_root, "models", "model_b", "traditional", "model_b_artifacts.pkl"),
    }
    print("Local preflight:")
    for label, path in checks.items():
        print(f"  {label}: {path_exists(path)}")
    if not os.path.exists(checks["Model A LR"]):
        print("  Note: missing models are okay; the app will run in demo mode.")


def main() -> None:
    """Launch Streamlit locally."""
    parser = argparse.ArgumentParser(description="Run the RACE Quiz AI Streamlit app locally.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true", help="Do not ask Streamlit to open a browser.")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.dirname(__file__))
    print_preflight(project_root)
    app_path = os.path.join(project_root, "ui", "app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(args.port),
        "--server.address",
        "localhost",
    ]
    if args.no_browser:
        command.extend(["--server.headless", "true"])

    print(f"\nStarting local app: http://localhost:{args.port}")
    subprocess.run(command, cwd=project_root, check=False)


if __name__ == "__main__":
    main()
