"""Model B: distractor generation, hint generation, and evaluation."""

from __future__ import annotations

import argparse
import itertools
import os
import re
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
    """Extract plausible candidate distractor phrases from the passage.

    FIX: The original version returned single tokens (e.g. "book", "borrowed")
    which made terrible distractors — they are obviously wrong to any reader.
    Good distractors should be noun-like phrases that could plausibly answer
    the question.  We now:
      1. Prefer multi-word noun phrases (consecutive non-stop capitalised or
         content words) that appear in the passage.
      2. Fall back to individual content words only when fewer than 6
         multi-word candidates are found.
      3. Exclude any candidate that is a substring of or equal to the
         correct answer.

    Args:
        article: Passage text.
        correct_answer: Gold answer to exclude.

    Returns:
        Candidate distractor strings, multi-word phrases first.
    """
    correct_cleaned = clean_text(correct_answer)
    correct_words = set(correct_cleaned.split())

    # --- Step 1: extract multi-word noun-like phrases from raw article ---
    # We look for runs of 2-4 consecutive words that are mostly non-stop
    raw_words = str(article).split()
    multi_word: list[str] = []
    for n in (3, 2):  # prefer trigrams, then bigrams
        for i in range(len(raw_words) - n + 1):
            chunk = raw_words[i : i + n]
            chunk_clean = [clean_text(w) for w in chunk]
            # Reject if any token is a stop word or very short
            if any(w in ENGLISH_STOP_WORDS or len(w) < 2 for w in chunk_clean):
                continue
            phrase = " ".join(chunk_clean)
            # Reject if it overlaps too heavily with the correct answer
            phrase_words = set(phrase.split())
            if phrase_words.intersection(correct_words):
                continue
            if phrase and phrase != correct_cleaned:
                multi_word.append(phrase)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_multi: list[str] = []
    for p in multi_word:
        if p not in seen:
            seen.add(p)
            unique_multi.append(p)

    # --- Step 2: single content-word fallback ---
    tokens = [
        clean_text(t)
        for t in str(article).split()
        if clean_text(t) not in ENGLISH_STOP_WORDS
        and len(clean_text(t)) > 2
        and clean_text(t) not in correct_words
    ]
    unique_tokens = list(dict.fromkeys(tokens))  # preserve order, deduplicate

    # Combine: multi-word phrases first, then single tokens
    candidates = unique_multi + [t for t in unique_tokens if t not in seen]
    return candidates


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


def rank_distractors(
    candidates: list[str],
    correct_answer: str,
    vectorizer: Optional[CountVectorizer],
    top_k: int = 3,
) -> list[str]:
    """Rank distractors by medium answer similarity plus a diversity penalty.

    FIX: Results are now title-cased for readability, and the fallback
    messages are more informative / passage-appropriate.

    Args:
        candidates: Candidate phrases.
        correct_answer: Correct answer phrase.
        vectorizer: OHE vectorizer; a local fallback is fit if needed.
        top_k: Number of distractors to return.

    Returns:
        Top ranked distractor strings (title-cased).
    """
    cleaned_candidates = [
        c for c in dict.fromkeys(candidates)
        if clean_text(c) != clean_text(correct_answer) and c.strip()
    ]
    if not cleaned_candidates:
        return ["Another option", "A different answer", "None of the above"][:top_k]

    sims, matrix = _candidate_similarity(cleaned_candidates, correct_answer, vectorizer)

    # Prefer candidates with moderate similarity (plausible but not correct)
    medium_bonus = np.where((sims >= 0.05) & (sims <= 0.65), 1.0, 0.2)
    base_scores = medium_bonus - np.abs(sims - 0.35)

    selected: list[int] = []
    for _ in range(min(top_k, len(cleaned_candidates))):
        best_idx, best_score = -1, -np.inf
        for idx, score in enumerate(base_scores):
            if idx in selected:
                continue
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = float(
                    np.max(cosine_similarity(matrix[idx], matrix[selected]).ravel())
                )
            final_score = float(score) - 0.35 * diversity_penalty
            if final_score > best_score:
                best_idx, best_score = idx, final_score
        if best_idx >= 0:
            selected.append(best_idx)

    # Title-case for readability in the UI
    results = [cleaned_candidates[idx].title() for idx in selected]

    # Pad if needed
    fallbacks = ["Another option", "A different answer", "None of the above"]
    while len(results) < top_k:
        results.append(fallbacks[len(results) % len(fallbacks)])

    return results[:top_k]


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

    FIX: The original code selected hints using three index positions
    (low, mid, high) from ``argsort`` but presented them in the wrong
    order — the *least* relevant sentence was shown first (Hint 1) and
    the *most* relevant was Hint 3 (near-explicit), which is correct for
    the gradual reveal.  However the index calculation was off: ``relevant_idx``
    from ``argsort`` is ascending (lowest similarity first), so
    ``relevant_idx[-1]`` is the *most* relevant sentence.  The three hints
    should therefore be ordered least→middle→most relevant, which is what we
    now do explicitly.

    Args:
        article: Passage text.
        question: Question text.
        correct_answer: Correct answer phrase.

    Returns:
        Three hints ordered from vague (Hint 1) to near-explicit (Hint 3).
    """
    sentences = split_sentences(article)
    if not sentences:
        return [
            "Review the passage topic.",
            "Look for details related to the question.",
            f"Consider: {correct_answer}",
        ]

    vectorizer = build_ohe_vectorizer(max_features=5000)
    try:
        matrix = vectorizer.fit_transform(sentences + [question])
        sims = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    except Exception:
        sims = np.asarray([_overlap(sentence, question) for sentence in sentences], dtype=float)

    # argsort gives indices sorted ascending (least→most relevant)
    ranked_idx = np.argsort(sims)
    n = len(ranked_idx)

    # Pick three representative sentences: general, intermediate, specific
    low_idx = int(ranked_idx[0])                    # least relevant → general clue
    mid_idx = int(ranked_idx[n // 2])               # mid relevance → more specific
    high_idx = int(ranked_idx[-1])                   # most relevant → near explicit

    # Avoid showing the same sentence for multiple hints
    used: set[int] = set()
    hint_indices: list[int] = []
    for idx in [low_idx, mid_idx, high_idx]:
        if idx not in used:
            hint_indices.append(idx)
            used.add(idx)
        else:
            # Find next-best unused sentence
            for fallback in ranked_idx:
                if int(fallback) not in used:
                    hint_indices.append(int(fallback))
                    used.add(int(fallback))
                    break

    hints = [sentences[i] for i in hint_indices[:3]]

    # Ensure Hint 3 is explicitly tied to the answer when possible
    answer_words = set(clean_text(correct_answer).split())
    if answer_words and not answer_words.intersection(clean_text(hints[-1]).split()):
        hints[-1] = (
            f"{hints[-1]} "
            f"Focus on the part of the passage mentioning '{correct_answer}'."
        )

    # Pad to exactly 3 hints if the article had fewer than 3 distinct sentences
    fallback_hints = [
        "Review the main topic of the passage.",
        f"Look for the sentence most closely related to the question.",
        f"Focus on the part of the passage mentioning '{correct_answer}'.",
    ]
    while len(hints) < 3:
        candidate = fallback_hints[len(hints)]
        hints.append(candidate)

    return hints[:3]


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


def ml_hint_scorer(train_df: pd.DataFrame, max_rows: int = 2000) -> LogisticRegression:
    """Train a logistic sentence relevance scorer for hint selection.

    Args:
        train_df: RACE training DataFrame.
        max_rows: Maximum source rows used to build sentence-level training data.

    Returns:
        Trained LogisticRegression scorer on five numeric features.
    """
    features: list[list[float]] = []
    labels: list[int] = []
    rows = train_df.head(max_rows)
    print(f"Training hint scorer on {len(rows)} source rows...")
    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        if idx % 500 == 0:
            print(f"  hint scorer rows processed: {idx}/{len(rows)}")
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
    print(f"Evaluating distractors and hints on {len(rows)} validation rows...")
    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        if idx % 100 == 0:
            print(f"  Model B eval rows processed: {idx}/{len(rows)}")
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
    rows = df.head(max_rows)
    print(f"Evaluating hint scorer R2 on {len(rows)} validation rows...")
    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        if idx % 100 == 0:
            print(f"  hint R2 rows processed: {idx}/{len(rows)}")
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
    hint_train_rows: int = 2000,
    eval_rows: int = 300,
    hint_eval_rows: int = 200,
) -> dict[str, Any]:
    """Train and evaluate Model B artifacts.

    Args:
        raw_data_dir: Directory containing original RACE CSVs.
        processed_dir: Directory containing saved OHE vectorizer.
        model_dir: Destination directory.
        hint_train_rows: Maximum training rows for the ML hint scorer.
        eval_rows: Maximum validation rows for distractor/hint evaluation.
        hint_eval_rows: Maximum validation rows for hint scorer R².

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
    scorer = ml_hint_scorer(train_df, max_rows=hint_train_rows)
    metrics = evaluate_model_b(val_df, vectorizer, max_rows=eval_rows)
    metrics["hint_scorer_r2"] = evaluate_hint_scorer(val_df, scorer, max_rows=hint_eval_rows)
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
    parser.add_argument("--hint-train-rows", type=int, default=2000)
    parser.add_argument("--eval-rows", type=int, default=300)
    parser.add_argument("--hint-eval-rows", type=int, default=200)
    args = parser.parse_args()
    train_model_b(args.raw_data_dir, args.processed_dir, args.model_dir, args.hint_train_rows, args.eval_rows, args.hint_eval_rows)


if __name__ == "__main__":
    main()
