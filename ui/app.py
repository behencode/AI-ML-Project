"""Streamlit UI for the Intelligent Reading Comprehension and Quiz Generation System."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from html import escape
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


def apply_theme() -> None:
    """Apply custom CSS for a more polished product-style interface."""
    st.markdown(
        """
        <style>
        :root {
            --rc-bg: #f6f7fb;
            --rc-panel: #ffffff;
            --rc-ink: #18202f;
            --rc-muted: #667085;
            --rc-line: #d9dee8;
            --rc-blue: #2563eb;
            --rc-green: #15803d;
            --rc-red: #b42318;
            --rc-amber: #b45309;
        }
        .main .block-container {
            padding-top: 1.3rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }
        section[data-testid="stSidebar"] {
            background: #101828;
        }
        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }
        div[data-testid="stButton"] button {
            border-radius: 8px;
            border: 1px solid #c8d0df;
            font-weight: 650;
            min-height: 2.75rem;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background: #2563eb;
            border-color: #2563eb;
        }
        .rc-hero {
            background: linear-gradient(135deg, #101828 0%, #1d4ed8 54%, #0f766e 100%);
            border-radius: 8px;
            padding: 28px 32px;
            color: white;
            margin-bottom: 18px;
            box-shadow: 0 14px 34px rgba(16, 24, 40, 0.18);
        }
        .rc-hero h1 {
            font-size: 2.3rem;
            line-height: 1.05;
            margin: 0 0 8px 0;
            letter-spacing: 0;
        }
        .rc-hero p {
            color: #dbeafe;
            margin: 0;
            max-width: 760px;
            font-size: 1rem;
        }
        .rc-card {
            background: white;
            border: 1px solid #d9dee8;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 8px 22px rgba(16, 24, 40, 0.06);
        }
        .rc-card h3 {
            margin: 0 0 6px 0;
            font-size: 1rem;
            letter-spacing: 0;
            color: #18202f;
        }
        .rc-muted {
            color: #667085;
            font-size: 0.92rem;
        }
        .rc-question {
            border-left: 5px solid #2563eb;
            background: #eff6ff;
            padding: 18px 20px;
            border-radius: 8px;
            color: #18202f;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 14px;
        }
        .rc-option {
            border: 1px solid #d9dee8;
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: #ffffff;
        }
        .rc-kpi {
            background: white;
            border: 1px solid #d9dee8;
            border-radius: 8px;
            padding: 14px 16px;
        }
        .rc-kpi .label {
            color: #667085;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .rc-kpi .value {
            color: #18202f;
            font-size: 1.45rem;
            font-weight: 760;
            margin-top: 4px;
        }
        .rc-pill {
            display: inline-block;
            padding: 5px 9px;
            border-radius: 999px;
            background: #eef2ff;
            color: #1d4ed8;
            border: 1px solid #c7d2fe;
            font-size: 0.82rem;
            font-weight: 650;
            margin-right: 6px;
        }
        .rc-warning {
            border: 1px solid #fbbf24;
            background: #fffbeb;
            color: #78350f;
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 14px;
        }
        textarea {
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    """Render a compact page hero."""
    st.markdown(f'<div class="rc-hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str) -> None:
    """Render a custom KPI card."""
    st.markdown(
        f'<div class="rc-kpi"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str) -> None:
    """Render a white information card."""
    st.markdown(f'<div class="rc-card"><h3>{escape(title)}</h3><div class="rc-muted">{escape(body)}</div></div>', unsafe_allow_html=True)


def warning_panel(message: str) -> None:
    """Render a custom warning panel."""
    st.markdown(f'<div class="rc-warning">{escape(message)}</div>', unsafe_allow_html=True)


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
        "last_latency_ms": 0.0,
        "race_question": None,   # populated when a RACE sample is loaded
        "race_answer": None,     # correct answer text from the RACE sample
        "race_options": None,    # original A/B/C/D options from the RACE sample
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_random_sample() -> None:
    """Load one random RACE test article into session state, including its question and answer."""
    path = os.path.join(ROOT_DIR, "data", "raw", "test.csv")
    try:
        df = pd.read_csv(path)
        sample = df.sample(1, random_state=np.random.default_rng().integers(0, 1_000_000)).iloc[0]
        st.session_state["article_text"] = str(sample["article"])
        # Store the original question and correct answer so the pipeline uses them
        answer_label = str(sample["answer"]).strip().upper()
        st.session_state["race_question"] = str(sample["question"])
        st.session_state["race_answer"] = str(sample[answer_label])
        st.session_state["race_options"] = {
            lbl: str(sample[lbl]) for lbl in ["A", "B", "C", "D"]
        }
    except Exception as exc:
        st.error(f"Could not load test.csv sample. Place the RACE CSVs in data/raw first. Details: {exc}")
        st.session_state.pop("race_question", None)
        st.session_state.pop("race_answer", None)
        st.session_state.pop("race_options", None)


def sidebar_nav() -> None:
    """Render sidebar navigation and synchronize the selected screen."""
    screens = ["📄 Article Input", "❓ Quiz", "💡 Hints", "📊 Analytics"]
    st.sidebar.markdown("## RACE Quiz AI")
    st.sidebar.caption("Traditional ML reading comprehension lab")
    current = st.session_state.get("screen", screens[0])
    selected = st.sidebar.radio("Navigation", screens, index=screens.index(current) if current in screens else 0)
    st.session_state["screen"] = selected
    if st.sidebar.button("Reset Session"):
        for key in ["quiz_result", "attempts", "answer_checked", "selected_option",
                    "race_question", "race_answer", "race_options"]:
            if key == "attempts":
                st.session_state[key] = []
            else:
                st.session_state[key] = None
        st.session_state["screen"] = "📄 Article Input"
        st.rerun()


def home_screen(engine: RaceInferenceEngine) -> None:
    """Render Screen 1, article input and quiz generation."""
    hero(
        "RACE Quiz AI",
        "Generate reading-comprehension questions, distractors, hints, and model analytics using traditional machine learning.",
    )
    if engine.use_demo_mode:
        warning_panel("Demo mode is active because one or more trained model files were not found. Rule-based fallbacks are being used.")
    c1, c2, c3 = st.columns(3)
    with c1:
        info_card("Model A", "Answer verification with sparse OHE features, lexical overlap, and voting ensembles.")
    with c2:
        info_card("Model B", "Distractor ranking, diversity checks, and graduated hint generation.")
    with c3:
        info_card("Evaluation", "Session logs, confusion matrix, metric cards, and exportable attempts.")
    left, right = st.columns([3, 1])
    with left:
        article = st.text_area("Paste a reading passage", key="article_text", height=300)
    with right:
        st.markdown("### Passage Tools")
        kpi_card("Words", str(len(article.split()) if article else 0))
        kpi_card("Characters", str(len(article) if article else 0))
        if st.button("Load Random RACE Sample"):
            load_random_sample()
            st.rerun()
    if st.button("Generate Quiz", type="primary"):
        if not article.strip():
            st.warning("Paste or load an article before generating a quiz.")
            return
        try:
            with st.spinner("Generating quiz..."):
                # If a RACE sample was loaded, pass its question and answer
                race_q = st.session_state.get("race_question")
                race_a = st.session_state.get("race_answer")
                st.session_state["quiz_result"] = engine.run_full_pipeline(
                    article,
                    question=race_q,
                    answer=race_a,
                )
                st.session_state["last_latency_ms"] = st.session_state["quiz_result"].get("inference_time_ms", 0.0)
                st.session_state["hints_revealed"] = 1
                st.session_state["answer_checked"] = False
                st.session_state["selected_option"] = None
                # Clear RACE-specific state after use
                st.session_state["race_question"] = None
                st.session_state["race_answer"] = None
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
    hero("Quiz Workspace", "Read the passage, select an answer, then inspect hints and model confidence.")
    with st.expander("Read the Passage", expanded=False):
        st.write(st.session_state.get("article_text", ""))
    st.markdown(f'<div class="rc-question">{escape(result["question"])}</div>', unsafe_allow_html=True)
    st.markdown("#### Options")
    for label, value in result["options"].items():
        st.markdown(f'<div class="rc-option"><b>{escape(label)}.</b> {escape(str(value))}</div>', unsafe_allow_html=True)
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
                "latency_ms": result.get("inference_time_ms", 0.0),
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
            st.success(f"Correct. The answer is {result['correct']}: {result['options'][result['correct']]}.")
        else:
            hint = result.get("hints", ["Review the passage."])[0]
            st.error(f"Not quite. Correct answer: {result['correct']}. {result['options'][result['correct']]}. Hint: {hint}")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        kpi_card("Model Confidence", f"{float(result.get('confidence', 0.0)):.2f}")
    with metric_cols[1]:
        kpi_card("Latency", f"{result.get('inference_time_ms', 0)} ms")
    with metric_cols[2]:
        kpi_card("Mode", "Demo" if result.get("demo_mode") else "Trained")
    st.progress(max(0.0, min(1.0, float(result.get("confidence", 0.0)))))


def hints_screen() -> None:
    """Render Screen 3, graduated hint reveal panel."""
    result = st.session_state.get("quiz_result")
    if not result:
        st.info("Generate a quiz first.")
        return
    hero("Hint Panel", "Reveal support gradually before showing the final answer.")
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
            st.markdown(f'<span class="rc-pill">{escape(label)} locked</span>', unsafe_allow_html=True)
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
    hero("Analytics Dashboard", "Inspect model behavior, session outcomes, artifacts, and runtime details.")
    tabs = st.tabs(["Model A Performance", "Model B Performance", "Session Log", "System Info"])
    model_a_results = _load_pickle(os.path.join(ROOT_DIR, "models", "model_a", "traditional", "results.pkl"), {})
    model_b_results = _load_pickle(os.path.join(ROOT_DIR, "models", "model_b", "traditional", "results.pkl"), {})
    with tabs[0]:
        metrics_df = model_a_results.get("metrics") if isinstance(model_a_results, dict) else None
        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            best = metrics_df.sort_values("macro_f1", ascending=False).iloc[0]
            cols = st.columns(4)
            with cols[0]:
                kpi_card("Accuracy", f"{best.get('accuracy', 0):.3f}")
            with cols[1]:
                kpi_card("Macro F1", f"{best.get('macro_f1', 0):.3f}")
            with cols[2]:
                kpi_card("Precision", f"{best.get('precision', best.get('accuracy', 0)):.3f}")
            with cols[3]:
                kpi_card("Recall", f"{best.get('recall', best.get('accuracy', 0)):.3f}")
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
            with cols[0]:
                kpi_card("Distractor Precision", f"{model_b_results.get('distractor_precision', 0):.3f}")
            with cols[1]:
                kpi_card("Distractor Recall", f"{model_b_results.get('distractor_recall', 0):.3f}")
            with cols[2]:
                kpi_card("Distractor F1", f"{model_b_results.get('distractor_f1', 0):.3f}")
            with cols[3]:
                kpi_card("Ranker Accuracy", f"{model_b_results.get('distractor_ranker_accuracy', 0):.3f}")
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
    apply_theme()
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
