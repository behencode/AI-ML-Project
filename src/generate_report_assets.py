import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_pickle(path):
    return joblib.load(path)

def generate_assets():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    model_a_path = os.path.join("models", "model_a", "traditional", "results.pkl")
    model_b_path = os.path.join("models", "model_b", "traditional", "results.pkl")
    
    model_a_results = load_pickle(model_a_path)
    model_b_results = load_pickle(model_b_path)
    
    # 1. Plot Model A Comparison
    metrics_df = model_a_results["metrics"]
    chart_df = metrics_df[metrics_df["model"].str.contains("Logistic|SVC|NB|Ensemble", case=False, regex=True)]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(chart_df["model"]))
    width = 0.25
    plt.bar(x - width, chart_df["accuracy"], width, label="Accuracy", color="#1f77b4")
    plt.bar(x, chart_df["macro_f1"], width, label="Macro F1", color="#ff7f0e")
    if "roc_auc" in chart_df.columns:
        plt.bar(x + width, chart_df["roc_auc"], width, label="ROC AUC", color="#2ca02c")
    
    plt.xticks(x, chart_df["model"], rotation=15, ha="right")
    plt.title("Model A Performance Comparison")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_a_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Plot Model A Confusion Matrix
    cm = model_a_results.get("confusion_matrix")
    if cm is not None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Wrong", "Correct"], yticklabels=["Wrong", "Correct"])
        plt.title("Soft Voting Ensemble Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "model_a_confusion_matrix.png"), dpi=300)
        plt.close()
        
    # 3. Plot Model B Metrics
    b_metrics = [
        ("Ranker Accuracy", model_b_results.get("distractor_ranker_accuracy", 0)),
        ("Hint Precision@3", model_b_results.get("hint_precision_at_k", 0)),
        ("Diversity", model_b_results.get("pairwise_cosine_diversity", 0)),
        ("Distractor BLEU", model_b_results.get("distractor_bleu", 0)),
        ("Distractor ROUGE-L", model_b_results.get("distractor_rougeL", 0)),
        ("Distractor METEOR", model_b_results.get("distractor_meteor", 0)),
    ]
    plt.figure(figsize=(10, 5))
    names = [m[0] for m in b_metrics]
    vals = [m[1] for m in b_metrics]
    plt.bar(names, vals, color="teal")
    plt.title("Model B Generation Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.1)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_b_metrics.png"), dpi=300)
    plt.close()
    
    # 4. Generate Markdown Report
    nlp_df = model_a_results.get("nlp_metrics", {})
    unsup = model_a_results.get("unsupervised_metrics", {})
    semi = model_a_results.get("semi_supervised_metrics", {})
    
    md_content = f"""# RACE Quiz AI - Final Report Results

This document contains all the hard numbers, tables, and auto-generated charts you need to copy into the Evaluation & Discussion section of your Final Report.

## Model A: Traditional ML (Answer Verification)

### Performance Table
{metrics_df.to_markdown(index=False)}

### Visualizations
![Model A Comparison](./model_a_comparison.png)

![Model A Confusion Matrix](./model_a_confusion_matrix.png)

### NLP Question Generation Quality
* BLEU Score: **{nlp_df.get('question_bleu', 0):.4f}**
* ROUGE-L Score: **{nlp_df.get('question_rouge', 0):.4f}**
* METEOR Score: **{nlp_df.get('question_meteor', 0):.4f}**

## Model A: Unsupervised & Semi-Supervised

### Unsupervised Clustering
* KMeans Silhouette Score: **{unsup.get('kmeans_silhouette', 0):.4f}**
* GMM Silhouette Score: **{unsup.get('gmm_silhouette', 0):.4f}**
* Cluster Purity Estimate: **{unsup.get('purity_estimate', 0):.4f}**

### Semi-Supervised Learning
* Label Propagation F1-Score: **{semi.get('label_prop_f1', 0):.4f}**
* Traditional LR F1-Score: **{semi.get('baseline_lr_f1', 0):.4f}**
* Pseudo-labeled rows added: **{semi.get('added_pseudo_labels', 0)}**

## Model B: Distractor & Hint Generation

### Key Metrics
* Distractor Ranker Accuracy: **{model_b_results.get('distractor_ranker_accuracy', 0):.4f}**
* Pairwise Cosine Diversity: **{model_b_results.get('pairwise_cosine_diversity', 0):.4f}**
* Hint Precision@K (Top 3): **{model_b_results.get('hint_precision_at_k', 0):.4f}**
* Hint Scorer R2: **{model_b_results.get('hint_scorer_r2', 0):.4f}**

### NLP Distractor Generation Quality
* BLEU Score: **{model_b_results.get('distractor_bleu', 0):.4f}**
* ROUGE-L Score: **{model_b_results.get('distractor_rougeL', 0):.4f}**
* METEOR Score: **{model_b_results.get('distractor_meteor', 0):.4f}**

### Visualizations
![Model B Metrics](./model_b_metrics.png)
"""
    with open(os.path.join(results_dir, "report_data.md"), "w") as f:
        f.write(md_content)
    print("Report assets generated in results/")

if __name__ == "__main__":
    generate_assets()
