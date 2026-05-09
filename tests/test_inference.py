"""Smoke tests for the unified inference engine."""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import RaceInferenceEngine  # noqa: E402


def test_demo_pipeline_returns_complete_payload() -> None:
    """The engine should work without trained models through demo-mode fallbacks."""
    engine = RaceInferenceEngine(model_dir=os.path.join(ROOT_DIR, "models"))
    article = "Sara visited the library after school. She borrowed a science book and read it before dinner."
    result = engine.run_full_pipeline(article)
    assert result["question"]
    assert set(result["options"].keys()) == {"A", "B", "C", "D"}
    assert result["correct"] in result["options"]
    assert len(result["hints"]) == 3
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_correct_option_shape() -> None:
    """Prediction should return a valid option label and numeric confidence."""
    engine = RaceInferenceEngine(model_dir=os.path.join(ROOT_DIR, "models"))
    label, confidence = engine.predict_correct_option(
        "The boy planted flowers in the garden.",
        "Where did the boy plant flowers?",
        {"A": "garden", "B": "kitchen", "C": "bus", "D": "river"},
    )
    assert label in {"A", "B", "C", "D"}
    assert isinstance(confidence, float)
