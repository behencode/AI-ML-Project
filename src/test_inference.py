"""Tests for inference, preprocessing, and generation behavior."""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import RaceInferenceEngine  # noqa: E402
from model_b_train import extract_candidate_phrases, generate_hints, rank_distractors  # noqa: E402
from preprocessing import clean_text, expand_to_binary, extract_lexical_features  # noqa: E402

import pandas as pd  # noqa: E402


def make_demo_engine() -> RaceInferenceEngine:
    """Create an engine that is forced into demo mode for deterministic tests."""
    engine = RaceInferenceEngine(model_dir=os.path.join(ROOT_DIR, "missing_models_for_tests"))
    assert engine.use_demo_mode is True
    return engine


def test_clean_text_removes_punctuation_and_normalizes_space() -> None:
    """Text cleaning should lowercase, strip punctuation, and compact whitespace."""
    assert clean_text(" Hello,   WORLD!!! ") == "hello world"


def test_expand_to_binary_creates_four_samples() -> None:
    """One RACE row should become four option-level binary samples."""
    df = pd.DataFrame(
        [
            {
                "id": "x1",
                "article": "Mina read a book.",
                "question": "What did Mina read?",
                "A": "a book",
                "B": "a map",
                "C": "a sign",
                "D": "a letter",
                "answer": "A",
            }
        ]
    )
    expanded = expand_to_binary(df)
    assert len(expanded) == 4
    assert expanded["label"].sum() == 1
    assert set(expanded.columns) == {"article", "question", "option", "opt_label", "correct", "label", "combined"}


def test_lexical_features_have_expected_shape() -> None:
    """Lexical feature extraction should produce six numeric features per option."""
    expanded = pd.DataFrame(
        [
            {
                "article": "mina read a book",
                "question": "what did mina read",
                "option": "book",
                "opt_label": "A",
                "correct": "A",
                "label": 1,
                "combined": "article mina read a book question what did mina read option book",
            }
        ]
    )
    features = extract_lexical_features(expanded)
    assert features.shape == (1, 6)
    assert features.dtype.kind == "f"


def test_demo_pipeline_returns_complete_payload() -> None:
    """The engine should work without trained models through demo-mode fallbacks."""
    engine = make_demo_engine()
    article = "Sara visited the library after school. She borrowed a science book and read it before dinner."
    result = engine.run_full_pipeline(article)
    assert result["question"]
    assert set(result["options"].keys()) == {"A", "B", "C", "D"}
    assert result["correct"] in result["options"]
    assert len(result["hints"]) == 3
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_correct_option_shape() -> None:
    """Prediction should return a valid option label and numeric confidence."""
    engine = make_demo_engine()
    label, confidence = engine.predict_correct_option(
        "The boy planted flowers in the garden.",
        "Where did the boy plant flowers?",
        {"A": "garden", "B": "kitchen", "C": "bus", "D": "river"},
    )
    assert label in {"A", "B", "C", "D"}
    assert isinstance(confidence, float)


def test_demo_scoring_prefers_overlapping_answer() -> None:
    """Rule-based fallback should prefer an option with article/question overlap."""
    engine = make_demo_engine()
    options = {"A": "garden", "B": "kitchen", "C": "bus", "D": "river"}
    label, confidence = engine.predict_correct_option("The boy planted flowers in the garden.", "Where did he plant flowers?", options)
    assert label == "A"
    assert 0.0 <= confidence <= 1.0


def test_empty_article_raises_value_error() -> None:
    """The pipeline should reject empty article input with a clear error."""
    engine = make_demo_engine()
    try:
        engine.run_full_pipeline("   ")
    except ValueError as exc:
        assert "Article text is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an empty article.")


def test_model_b_generators_return_safe_outputs() -> None:
    """Distractor and hint helpers should return non-empty, safe outputs."""
    article = "Lina visited the museum on Monday. She saw paintings, statues, and old coins."
    correct = "museum"
    candidates = extract_candidate_phrases(article, correct)
    distractors = rank_distractors(candidates, correct, vectorizer=None, top_k=3)
    hints = generate_hints(article, "Where did Lina visit?", correct)
    assert len(candidates) > 0
    assert len(distractors) == 3
    # Compare case-insensitively since distractors are now title-cased
    assert all(item.lower() != correct.lower() for item in distractors)
    assert len(hints) == 3


def run_all_tests() -> None:
    """Run tests without requiring pytest."""
    test_clean_text_removes_punctuation_and_normalizes_space()
    test_expand_to_binary_creates_four_samples()
    test_lexical_features_have_expected_shape()
    test_demo_pipeline_returns_complete_payload()
    test_predict_correct_option_shape()
    test_demo_scoring_prefers_overlapping_answer()
    test_empty_article_raises_value_error()
    test_model_b_generators_return_safe_outputs()
    print("All inference/preprocessing/generation tests passed.")


if __name__ == "__main__":
    run_all_tests()
