# Project Goal

## Title

Intelligent Reading Comprehension and Quiz Generation System

## Main Objective

The goal of this project is to build a complete AI-based reading comprehension system using traditional machine-learning methods on the RACE dataset. The system takes a reading passage as input and produces an interactive quiz experience: it can generate a question, create answer options, verify likely correct answers, generate plausible distractors, provide graduated hints, and show model analytics through a Streamlit interface.

## Why This Project Matters

Reading comprehension questions are useful in education, assessment, and self-study, but manually creating high-quality quiz material is time-consuming. This project explores how far traditional machine-learning and lexical NLP methods can go without using neural networks, transformers, BERT, T5, or other pretrained deep-learning models. The final system is designed to be explainable, lightweight, and runnable on a normal laptop.

## Dataset

The project uses the Kaggle RACE dataset provided by:

```text
ankitdhiman7/race-dataset
```

The instructor requirement is to use one dataset source and create the train-validation-test split manually. This project creates an 80/10/10 split using `random_state=42`.

Expected split sizes:

```text
train.csv: 70292 rows
val.csv:    8787 rows
test.csv:   8787 rows
```

Each row contains:

```text
id, article, question, A, B, C, D, answer
```

## Model A: Answer Verification and Question Generation

Model A focuses on reading-comprehension answer verification.

The project uses option-level training. Each original RACE row becomes four binary training examples:

```text
article + question + option A -> correct or incorrect
article + question + option B -> correct or incorrect
article + question + option C -> correct or incorrect
article + question + option D -> correct or incorrect
```

This follows the project requirement and creates a natural 25 percent positive and 75 percent negative class distribution. Because this can make accuracy misleading, the project reports confusion matrices, macro F1, precision, recall, and per-class metrics.

Implemented Model A components:

- Logistic Regression
- Calibrated LinearSVC
- ComplementNB for question-type classification
- Soft voting ensemble
- Hard voting ensemble
- Optional stacking ensemble
- Optional XGBoost
- KMeans clustering with TruncatedSVD
- Gaussian Mixture Model with TruncatedSVD
- LabelSpreading for semi-supervised learning
- Template-based Wh-question generation

## Model B: Distractor and Hint Generation

Model B focuses on creating quiz support content.

Implemented Model B components:

- Candidate phrase extraction from the article
- Stopword removal
- Content-word and bigram extraction
- Frequency-based substitution
- Medium-similarity distractor ranking
- Diversity penalty for distractors
- Graduated hint generation
- Logistic Regression hint scorer

Model B is evaluated using distractor precision, recall, F1, ranker accuracy, hint Precision@K, hint scorer R², and pairwise cosine diversity.

## User Interface

The Streamlit app provides four screens:

1. Article Input: paste a passage or load a random RACE sample.
2. Quiz View: answer a generated multiple-choice question and check correctness.
3. Hint Panel: reveal three hints gradually before revealing the answer.
4. Analytics Dashboard: view Model A metrics, Model B metrics, session logs, and system information.

## Constraints

This project intentionally uses traditional machine learning only.

Not allowed:

- BERT
- T5
- Transformers
- Neural networks
- Large language models for training

Allowed and used:

- One-hot encoding
- TF-IDF as optional secondary representation
- Logistic Regression
- SVM
- Naive Bayes
- KMeans
- GMM
- LabelSpreading
- XGBoost as optional traditional ML

## Expected Findings

Model A accuracy can be misleading because option-level data has three incorrect options for every correct option. A model can predict every option as incorrect and still achieve about 75 percent accuracy. Therefore, the main evaluation emphasis is macro F1, per-class recall, and confusion matrix analysis.

Model B automatic distractor precision and recall may be low because generated distractors rarely match the exact human-written wrong options in RACE. Human evaluation is therefore included to judge distractor plausibility, grammaticality, relevance, and whether the distractor is clearly not the correct answer.

## Final Deliverables

The expected final deliverables are:

- Source code for preprocessing, training, inference, evaluation, and UI
- Trained traditional ML models
- EDA notebook
- Streamlit app screenshots or demo
- Model A and Model B metric tables
- Human evaluation form/results
- Academic report PDF

