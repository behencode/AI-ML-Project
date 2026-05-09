"""Model B: distractor generation, hint generation, and evaluation."""

from __future__ import annotations

import argparse
import itertools
import os
import re
import string
from collections import Counter
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, r2_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    from preprocessing import build_ohe_vectorizer, clean_text
except ImportError:  # pragma: no cover
    from .preprocessing import build_ohe_vectorizer, clean_text

RANDOM_STATE = 42


def split_sentences(article: str) -> list[str]:
    """Split an article into sentence-like chunks.

    Args:
        article: Raw passage text.

    Returns:
        List of non-empty sentence strings.
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(article)) if s.strip()]


def extract_candidate_phrases(article: str, correct_answer: str) -> list[str]:
    """Extract unique content words and bigrams excluding the correct answer.

    Args:
        article: Passage text.
        correct_answer: Gold answer to exclude.

    Returns:
        Candidate distractor strings.
    """
    tokens = [t for t in clean_text(article).split() if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    correct = clean_text(correct_answer)
    candidates = set(tokens)
    candidates.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    return sorted({c for c in candidates if c and c != correct and correct not in c})


def _candidate_similarity(candidates: list[str], correct_answer: str, vectorizer: Optional[CountVectorizer]) -> tuple[np.ndarray, Any]:
    """Vectorize candidates and compute cosine similarity to the answer."""
    texts = candidates + [correct_answer]
    vec = vectorizer or build_ohe_vectorizer(max_features=5000)
    try:
        matrix = vec.transform(texts) if hasattr(vec, "vocabulary_") else vec.fit_transform(texts)
    except ValueError:
        vec = CountVectorizer(binary=True)
        matrix = vec.fit_transform(texts)
    sims = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    return sims, matrix[:-1]


def rank_distractors(candidates: list[str], correct_answer: str, vectorizer: Optional[CountVectorizer], top_k: int = 3) -> list[str]:
    """Rank distractors by medium answer similarity plus a diversity penalty.

    Args:
        candidates: Candidate phrases.
        correct_answer: Correct answer phrase.
        vectorizer: OHE vectorizer; a local fallback is fit if needed.
        top_k: Number of distractors to return.

    Returns:
        Top ranked distractor strings.
    """
    cleaned_candidates = [c for c in dict.fromkeys(candidates) if clean_text(c) != clean_text(correct_answer)]
    if not cleaned_candidates:
        return ["not mentioned", "another detail", "different answer"][:top_k]
    sims, matrix = _candidate_similarity(cleaned_candidates, correct_answer, vectorizer)
    medium_bonus = np.where((sims >= 0.1) & (sims <= 0.6), 1.0, 0.25)
    base_scores = medium_bonus - np.abs(sims - 0.35)
    selected: list[int] = []
    for _ in range(min(top_k, len(cleaned_candidates))):
        best_idx, best_score = -1, -np.inf
        for idx, score in enumerate(base_scores):
            if idx in selected:
                continue
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = float(np.max(cosine_similarity(matrix[idx], matrix[selected]).ravel()))
            final_score = float(score) - 0.35 * diversity_penalty
            if final_score > best_score:
                best_idx, best_score = idx, final_score
        if best_idx >= 0:
            selected.append(best_idx)
    return [cleaned_candidates[idx] for idx in selected]


def frequency_based_substitution(article: str, correct_answer: str) -> list[str]:
    """Generate phrase candidates from words with similar article frequency.

    Args:
        article: Passage text.
        correct_answer: Correct answer phrase.

    Returns:
        Frequency-matched substitute candidates.
    """
    tokens = [t for t in clean_text(article).split() if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    counts = Counter(tokens)
    answer_words = [w for w in clean_text(correct_answer).split() if w]
    if not answer_words or not counts:
        return []
    answer_freq = np.mean([counts.get(w, 1) for w in answer_words])
    candidates = [word for word, freq in counts.items() if word not in answer_words and abs(freq - answer_freq) <= max(1.0, answer_freq)]
    return candidates[:20]


def _overlap(left: str, right: str) -> float:
    """Compute token overlap ratio."""
    lset, rset = set(clean_text(left).split()), set(clean_text(right).split())
    return len(lset.intersection(rset)) / max(1, min(len(lset), len(rset)))


def generate_hints(article: str, question: str, correct_answer: str) -> list[str]:
    """Generate three graduated hints from sentence relevance scores.

    Args:
        article: Passage text.
        question: Question text.
        correct_answer: Correct answer phrase.

    Returns:
        Three hints ordered from vague to near-explicit.
    """
    sentences = split_sentences(article)
    if not sentences:
        return ["Review the passage topic.", "Look for details related to the question.", f"Consider: {correct_answer}"]
    vectorizer = build_ohe_vectorizer(max_features=5000)
    try:
        matrix = vectorizer.fit_transform(sentences + [question])
        sims = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    except Exception:
        sims = np.asarray([_overlap(sentence, question) for sentence in sentences], dtype=float)
    relevant_idx = np.argsort(sims)
    low = int(relevant_idx[max(0, len(relevant_idx) // 3 - 1)])
    mid = int(relevant_idx[len(relevant_idx) // 2])
    high = int(relevant_idx[-1])
    hints = [sentences[low], sentences[mid], sentences[high]]
    answer_words = set(clean_text(correct_answer).split())
    if answer_words and not answer_words.intersection(clean_text(hints[-1]).split()):
        hints[-1] = f"{hints[-1]} Focus on the answer phrase related to '{correct_answer}'."
    return hints


def _sentence_features(sentence: str, question: str, correct_answer: str, position: int, total: int) -> list[float]:
    """Extract five ML hint scorer features for one sentence."""
    vectorizer = CountVectorizer(binary=True)
    try:
        matrix = vectorizer.fit_transform([sentence, question])
        cos = float(cosine_similarity(matrix[0], matrix[1]).ravel()[0])
    except Exception:
        cos = _overlap(sentence, question)
    answer_words = set(clean_text(correct_answer).split())
    return [
        _overlap(sentence, question),
        position / max(1, total - 1),
        len(clean_text(sentence).split()) / 30.0,
        cos,
        float(bool(answer_words.intersection(clean_text(sentence).split()))),
    ]


def ml_hint_scorer(train_df: pd.DataFrame) -> LogisticRegression:
    """Train a logistic sentence relevance scorer for hint selection.

    Args:
        train_df: RACE training DataFrame.

    Returns:
        Trained LogisticRegression scorer on five numeric features.
    """
    features: list[list[float]] = []
    labels: list[int] = []
    for row in train_df.head(10000).itertuples(index=False):
        answer_label = str(getattr(row, "answer")).strip().upper()
        correct_answer = str(getattr(row, answer_label, ""))
        sentences = split_sentences(str(getattr(row, "article")))
        for pos, sentence in enumerate(sentences):
            features.append(_sentence_features(sentence, str(getattr(row, "question")), correct_answer, pos, len(sentences)))
            labels.append(int(_overlap(sentence, correct_answer) > 0))
    if not features or len(set(labels)) < 2:
        features = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
        labels = [0, 1]
    model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    model.fit(np.asarray(features, dtype=np.float32), np.asarray(labels))
    return model


def pairwise_cosine_diversity(distractors: list[str]) -> float:
    """Compute mean pairwise cosine distance among distractors.

    Args:
        distractors: Distractor strings.

    Returns:
        Diversity score where higher means more diverse.
    """
    if len(distractors) < 2:
        return 0.0
    vectorizer = CountVectorizer(binary=True)
    try:
        matrix = vectorizer.fit_transform(distractors)
    except ValueError:
        return 0.0
    sims = []
    for i, j in itertools.combinations(range(len(distractors)), 2):
        sims.append(float(cosine_similarity(matrix[i], matrix[j]).ravel()[0]))
    return float(1.0 - np.mean(sims)) if sims else 0.0


def evaluate_model_b(df: pd.DataFrame, vectorizer: Optional[CountVectorizer], max_rows: int = 1000) -> dict[str, Any]:
    """Evaluate distractor and hint generation against RACE gold options.

    Args:
        df: RACE DataFrame with article, question, options, and answer.
        vectorizer: OHE vectorizer for ranking.
        max_rows: Maximum rows evaluated for runtime control.

    Returns:
        Metrics dictionary.
    """
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    top1_ok: list[int] = []
    hint_hits: list[int] = []
    diversities: list[float] = []
    rows = df.head(max_rows)
    for row in rows.itertuples(index=False):
        answer_label = str(getattr(row, "answer")).strip().upper()
        correct = str(getattr(row, answer_label))
        gold_wrong = {clean_text(str(getattr(row, label))) for label in ["A", "B", "C", "D"] if label != answer_label}
        candidates = extract_candidate_phrases(str(getattr(row, "article")), correct)
        candidates.extend(frequency_based_substitution(str(getattr(row, "article")), correct))
        distractors = rank_distractors(candidates, correct, vectorizer, top_k=3)
        pred_set = {clean_text(d) for d in distractors}
        intersection = pred_set.intersection(gold_wrong)
        precision = len(intersection) / max(1, len(pred_set))
        recall = len(intersection) / max(1, len(gold_wrong))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        top1_ok.append(int(bool(distractors) and clean_text(distractors[0]) != clean_text(correct)))
        hints = generate_hints(str(getattr(row, "article")), str(getattr(row, "question")), correct)
        hint_hits.append(int(any(_overlap(hint, correct) > 0 for hint in hints)))
        diversities.append(pairwise_cosine_diversity(distractors))
    metrics = {
        "distractor_precision": float(np.mean(precisions)) if precisions else 0.0,
        "distractor_recall": float(np.mean(recalls)) if recalls else 0.0,
        "distractor_f1": float(np.mean(f1s)) if f1s else 0.0,
        "distractor_ranker_accuracy": float(np.mean(top1_ok)) if top1_ok else 0.0,
        "hint_precision_at_k": float(np.mean(hint_hits)) if hint_hits else 0.0,
        "pairwise_cosine_diversity": float(np.mean(diversities)) if diversities else 0.0,
    }
    return metrics


def evaluate_hint_scorer(df: pd.DataFrame, scorer: LogisticRegression, max_rows: int = 500) -> float:
    """Evaluate hint scorer relevance predictions with R².

    Args:
        df: Validation or test RACE DataFrame.
        scorer: Trained sentence relevance classifier.
        max_rows: Maximum source rows.

    Returns:
        R² score.
    """
    X: list[list[float]] = []
    y: list[int] = []
    for row in df.head(max_rows).itertuples(index=False):
        answer_label = str(getattr(row, "answer")).strip().upper()
        correct = str(getattr(row, answer_label))
        sentences = split_sentences(str(getattr(row, "article")))
        for pos, sentence in enumerate(sentences):
            X.append(_sentence_features(sentence, str(getattr(row, "question")), correct, pos, len(sentences)))
            y.append(int(_overlap(sentence, correct) > 0))
    if not X or len(set(y)) < 2:
        return 0.0
    probs = scorer.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
    return float(r2_score(np.asarray(y), probs))


def train_model_b(
    raw_data_dir: str = os.path.join("data", "raw"),
    processed_dir: str = os.path.join("data", "processed"),
    model_dir: str = os.path.join("models", "model_b", "traditional"),
) -> dict[str, Any]:
    """Train and evaluate Model B artifacts.

    Args:
        raw_data_dir: Directory containing original RACE CSVs.
        processed_dir: Directory containing saved OHE vectorizer.
        model_dir: Destination directory.

    Returns:
        Metrics dictionary.
    """
    os.makedirs(model_dir, exist_ok=True)
    try:
        train_df = pd.read_csv(os.path.join(raw_data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(raw_data_dir, "val.csv"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load raw RACE data: {exc}") from exc
    try:
        vectorizer = joblib.load(os.path.join(processed_dir, "ohe_vectorizer.pkl"))
    except Exception:
        print("Processed OHE vectorizer not found; fitting a small local vectorizer for Model B.")
        vectorizer = build_ohe_vectorizer(max_features=10000)
        vectorizer.fit(train_df["article"].fillna("").astype(str).head(10000))
    scorer = ml_hint_scorer(train_df)
    metrics = evaluate_model_b(val_df, vectorizer)
    metrics["hint_scorer_r2"] = evaluate_hint_scorer(val_df, scorer)
    try:
        joblib.dump({"hint_scorer": scorer, "vectorizer": vectorizer}, os.path.join(model_dir, "model_b_artifacts.pkl"))
        joblib.dump(metrics, os.path.join(model_dir, "results.pkl"))
    except Exception as exc:
        raise RuntimeError(f"Failed to save Model B artifacts: {exc}") from exc
    print("\nModel B metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    return metrics


def main() -> None:
    """CLI entry point for Model B training."""
    parser = argparse.ArgumentParser(description="Train and evaluate Model B distractor/hint generation.")
    parser.add_argument("--raw-data-dir", default=os.path.join("data", "raw"))
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"))
    parser.add_argument("--model-dir", default=os.path.join("models", "model_b", "traditional"))
    args = parser.parse_args()
    train_model_b(args.raw_data_dir, args.processed_dir, args.model_dir)


if __name__ == "__main__":
    main()
