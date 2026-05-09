"""Streamlit UI for the Intelligent Reading Comprehension and Quiz Generation System."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluate import plot_confusion_matrix  # noqa: E402
from inference import RaceInferenceEngine  # noqa: E402


st.set_page_config(layout="wide", page_title="RACE Quiz AI", page_icon="🧠")


@st.cache_resource
def load_engine() -> RaceInferenceEngine:
    """Load the inference engine once per Streamlit session.

    Returns:
        Cached ``RaceInferenceEngine`` instance.
    """
    return RaceInferenceEngine(model_dir=os.path.join(ROOT_DIR, "models"))


def init_state() -> None:
    """Initialize all Streamlit session-state values used by the app."""
    defaults = {
        "screen": "📄 Article Input",
        "article_text": "",
        "quiz_result": None,
        "attempts": [],
        "hints_revealed": 1,
        "answer_checked": False,
        "selected_option": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_random_sample() -> None:
    """Load one random RACE test article into session state."""
    path = os.path.join(ROOT_DIR, "data", "raw", "test.csv")
    try:
        df = pd.read_csv(path)
        sample = df.sample(1, random_state=np.random.default_rng().integers(0, 1_000_000)).iloc[0]
        st.session_state["article_text"] = str(sample["article"])
    except Exception as exc:
        st.error(f"Could not load test.csv sample. Place the RACE CSVs in data/raw first. Details: {exc}")


def sidebar_nav() -> None:
    """Render sidebar navigation and synchronize the selected screen."""
    screens = ["📄 Article Input", "❓ Quiz", "💡 Hints", "📊 Analytics"]
    current = st.session_state.get("screen", screens[0])
    selected = st.sidebar.radio("Navigation", screens, index=screens.index(current) if current in screens else 0)
    st.session_state["screen"] = selected
    if st.sidebar.button("Reset Session"):
        for key in ["quiz_result", "attempts", "answer_checked", "selected_option"]:
            st.session_state[key] = [] if key == "attempts" else None
        st.session_state["screen"] = "📄 Article Input"
        st.rerun()


def home_screen(engine: RaceInferenceEngine) -> None:
    """Render Screen 1, article input and quiz generation."""
    st.title("RACE Quiz AI")
    st.caption("Traditional ML system for reading comprehension, answer verification, distractor generation, and graduated hints.")
    if engine.use_demo_mode:
        st.warning("Demo mode is active because one or more trained model files were not found. Rule-based fallbacks are being used.")
    left, right = st.columns([3, 1])
    with left:
        article = st.text_area("Paste a reading passage", key="article_text", height=300)
    with right:
        st.write("")
        st.write("")
        if st.button("Load Random RACE Sample"):
            load_random_sample()
            st.rerun()
    words = len(article.split()) if article else 0
    st.caption(f"Characters: {len(article)} | Words: {words}")
    if st.button("Generate Quiz", type="primary"):
        if not article.strip():
            st.warning("Paste or load an article before generating a quiz.")
            return
        try:
            with st.spinner("Generating quiz..."):
                st.session_state["quiz_result"] = engine.run_full_pipeline(article)
                st.session_state["hints_revealed"] = 1
                st.session_state["answer_checked"] = False
                st.session_state["selected_option"] = None
                st.session_state["screen"] = "❓ Quiz"
            st.rerun()
        except Exception as exc:
            st.error(f"Quiz generation failed: {exc}")


def quiz_screen(engine: RaceInferenceEngine) -> None:
    """Render Screen 2, multiple-choice quiz interaction."""
    result = st.session_state.get("quiz_result")
    if not result:
        st.info("Generate a quiz from the Article Input screen first.")
        return
    with st.expander("Read the Passage", expanded=False):
        st.write(st.session_state.get("article_text", ""))
    st.info(result["question"])
    labels = list(result["options"].keys())
    selected = st.radio(
        "Choose an answer",
        labels,
        format_func=lambda label: f"{label}. {result['options'][label]}",
        key="selected_option",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Check My Answer", type="primary"):
            correct = selected == result["correct"]
            row = {
                "article_snippet": st.session_state.get("article_text", "")[:120],
                "question": result["question"],
                "user_answer": selected,
                "correct": correct,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            st.session_state["attempts"].append(row)
            st.session_state["answer_checked"] = True
    with col2:
        if st.button("Get Hints"):
            st.session_state["screen"] = "💡 Hints"
            st.rerun()
    with col3:
        if st.button("Try Another Question"):
            st.session_state["screen"] = "📄 Article Input"
            st.session_state["quiz_result"] = None
            st.rerun()
    if st.session_state.get("answer_checked"):
        if selected == result["correct"]:
            st.success(f"✅ Correct. The answer is {result['correct']}: {result['options'][result['correct']]}.")
        else:
            hint = result.get("hints", ["Review the passage."])[0]
            st.error(f"❌ Not quite. Correct answer: {result['correct']}. {result['options'][result['correct']]}. Hint: {hint}")
    st.write("Model A confidence")
    st.progress(max(0.0, min(1.0, float(result.get("confidence", 0.0)))))
    st.caption(f"Inference latency: {result.get('inference_time_ms', 0)} ms")


def hints_screen() -> None:
    """Render Screen 3, graduated hint reveal panel."""
    result = st.session_state.get("quiz_result")
    if not result:
        st.info("Generate a quiz first.")
        return
    with st.expander("Passage", expanded=False):
        st.write(st.session_state.get("article_text", ""))
    hints = result.get("hints", ["Review the main idea.", "Find the sentence related to the question.", "Look near the answer phrase."])
    labels = ["💡 Hint 1 — General Clue", "💡 Hint 2 — More Specific", "💡 Hint 3 — Near Explicit"]
    revealed = st.session_state.get("hints_revealed", 1)
    for idx, label in enumerate(labels, start=1):
        if revealed >= idx:
            with st.expander(label, expanded=idx == revealed):
                st.write(hints[idx - 1] if idx - 1 < len(hints) else "Re-read the most relevant sentence.")
                if idx < 3 and st.button(f"Reveal Hint {idx + 1}", key=f"reveal_{idx}"):
                    st.session_state["hints_revealed"] = idx + 1
                    st.rerun()
        else:
            st.info(f"{label} is locked until the previous hint is viewed.")
    if st.session_state.get("hints_revealed", 1) >= 3:
        if st.button("Reveal Answer", type="primary"):
            st.success(f"Answer: {result['correct']}. {result['options'][result['correct']]}")
    if st.button("Back to Quiz"):
        st.session_state["screen"] = "❓ Quiz"
        st.rerun()


def _load_pickle(path: str, default: Any) -> Any:
    """Load a pickle/joblib artifact with fallback."""
    try:
        return joblib.load(path)
    except Exception:
        return default


def _file_size(path: str) -> str:
    """Return a human-readable file size."""
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    except OSError:
        return "missing"


def analytics_screen(engine: RaceInferenceEngine) -> None:
    """Render Screen 4, analytics dashboard."""
    tabs = st.tabs(["Model A Performance", "Model B Performance", "Session Log", "System Info"])
    model_a_results = _load_pickle(os.path.join(ROOT_DIR, "models", "model_a", "traditional", "results.pkl"), {})
    model_b_results = _load_pickle(os.path.join(ROOT_DIR, "models", "model_b", "traditional", "results.pkl"), {})
    with tabs[0]:
        metrics_df = model_a_results.get("metrics") if isinstance(model_a_results, dict) else None
        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            best = metrics_df.sort_values("macro_f1", ascending=False).iloc[0]
            cols = st.columns(4)
            cols[0].metric("Accuracy", f"{best.get('accuracy', 0):.3f}")
            cols[1].metric("F1", f"{best.get('macro_f1', 0):.3f}")
            cols[2].metric("Precision", f"{best.get('precision', best.get('accuracy', 0)):.3f}")
            cols[3].metric("Recall", f"{best.get('recall', best.get('accuracy', 0)):.3f}")
            cm = model_a_results.get("confusion_matrix", np.asarray([[0, 0], [0, 0]]))
            st.pyplot(plot_confusion_matrix(cm, ["Wrong", "Correct"], "Model A Confusion Matrix"))
            chart_df = metrics_df[metrics_df["model"].astype(str).str.contains("Logistic|SVC|NB|Ensemble", case=False, regex=True)]
            st.plotly_chart(px.bar(chart_df, x="model", y="macro_f1", title="LR vs SVM vs NB vs Ensemble"), use_container_width=True)
        else:
            st.warning("Model A results not found. Train Model A to populate this dashboard.")
        latencies = [a.get("latency_ms", 0) for a in st.session_state.get("attempts", []) if "latency_ms" in a]
        st.metric("Average inference latency", f"{np.mean(latencies) if latencies else 0:.1f} ms")
    with tabs[1]:
        if isinstance(model_b_results, dict) and model_b_results:
            cols = st.columns(4)
            cols[0].metric("Distractor Precision", f"{model_b_results.get('distractor_precision', 0):.3f}")
            cols[1].metric("Distractor Recall", f"{model_b_results.get('distractor_recall', 0):.3f}")
            cols[2].metric("Distractor F1", f"{model_b_results.get('distractor_f1', 0):.3f}")
            cols[3].metric("Accuracy", f"{model_b_results.get('distractor_ranker_accuracy', 0):.3f}")
            hint_df = pd.DataFrame({"K": ["P@1", "P@2", "P@3"], "Precision": [0.0, 0.0, model_b_results.get("hint_precision_at_k", 0)]})
            st.plotly_chart(px.bar(hint_df, x="K", y="Precision", title="Hint Precision@K"), use_container_width=True)
        else:
            st.warning("Model B results not found. Train Model B to populate metrics.")
        sample = pd.DataFrame(
            [
                {"example": 1, "distractor": "school library", "quality_note": "plausible location"},
                {"example": 2, "distractor": "next morning", "quality_note": "time-based alternative"},
                {"example": 3, "distractor": "different reason", "quality_note": "semantic foil"},
                {"example": 4, "distractor": "classmate", "quality_note": "person-based alternative"},
                {"example": 5, "distractor": "main problem", "quality_note": "topic-level foil"},
            ]
        )
        st.dataframe(sample, use_container_width=True)
    with tabs[2]:
        attempts = pd.DataFrame(st.session_state.get("attempts", []))
        st.dataframe(attempts, use_container_width=True)
        total = len(attempts)
        correct = int(attempts["correct"].sum()) if total and "correct" in attempts else 0
        st.metric("Total attempts", total)
        st.metric("Correct", correct)
        st.metric("Session accuracy", f"{(correct / total * 100) if total else 0:.1f}%")
        st.download_button("Export to CSV", attempts.to_csv(index=False), "session_log.csv", "text/csv", disabled=attempts.empty)
    with tabs[3]:
        model_files = [
            os.path.join(ROOT_DIR, "models", "model_a", "traditional", "logistic_regression.pkl"),
            os.path.join(ROOT_DIR, "models", "model_a", "traditional", "linear_svc_calibrated.pkl"),
            os.path.join(ROOT_DIR, "models", "model_b", "traditional", "model_b_artifacts.pkl"),
        ]
        st.dataframe(pd.DataFrame({"file": [os.path.basename(p) for p in model_files], "size": [_file_size(p) for p in model_files]}))
        vocab_size = len(getattr(engine.ohe_vectorizer, "vocabulary_", {})) if engine.ohe_vectorizer is not None else 0
        st.write(f"Vocabulary size: {vocab_size}")
        st.write(f"Python: {platform.python_version()}")
        st.write(f"Streamlit: {st.__version__}")
        try:
            import sklearn

            st.write(f"scikit-learn: {sklearn.__version__}")
        except Exception:
            st.write("scikit-learn: unavailable")
        try:
            import torch

            st.write(f"GPU available: {torch.cuda.is_available()}")
        except Exception:
            st.write("GPU available: torch not installed; traditional ML does not require GPU.")


def main() -> None:
    """Run the Streamlit application."""
    init_state()
    engine = load_engine()
    sidebar_nav()
    screen = st.session_state.get("screen", "📄 Article Input")
    if screen == "📄 Article Input":
        home_screen(engine)
    elif screen == "❓ Quiz":
        quiz_screen(engine)
    elif screen == "💡 Hints":
        hints_screen()
    elif screen == "📊 Analytics":
        analytics_screen(engine)


if __name__ == "__main__":
    main()
