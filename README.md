# Intelligent Reading Comprehension and Quiz Generation System

This project builds a traditional machine-learning reading comprehension system on the RACE dataset. It verifies candidate answers, generates template-based questions, creates plausible distractors, produces graduated hints, and exposes the workflow through a polished Streamlit app. The current workflow is local-first: run it on your laptop with `localhost`, while Colab remains optional.

## Project Structure

```text
race_rc_project/
├── setup_local.py
├── run_local.py
├── setup_colab.py
├── requirements.txt
├── README.md
├── PROJECT_GOAL.md
├── data/
│   └── raw/
├── src/
│   ├── preprocessing.py
│   ├── model_a_train.py
│   ├── model_b_train.py
│   ├── inference.py
│   └── evaluate.py
├── ui/
│   ├── app.py
│   └── colab_launcher.py
├── models/
│   ├── model_a/traditional/
│   └── model_b/traditional/
├── notebooks/
│   └── EDA.ipynb
├── reports/
│   ├── final_report_template.md
│   ├── final_report_template.tex
│   └── human_evaluation_form.csv
├── results/
└── tests/
    └── test_inference.py
```

## Local Setup

Recommended Python version: 3.10, 3.11, or 3.12. Python 3.14 is very new, so some scientific packages may not have stable wheels yet. Python 3.11 or 3.12 is the safest local choice.

From the parent folder, enter the project:

```bash
cd /Users/m1pro/Documents/SMC-BOT/race_rc_project
```

What it does: moves your terminal into the project folder so all relative paths like `data/raw/train.csv` work correctly.

Create a local virtual environment:

```bash
python3 -m venv venv
```

What it does: creates an isolated Python environment in `venv/` so project packages do not interfere with system Python.

Activate the environment:

```bash
source venv/bin/activate
```

What it does: makes your terminal use `race_rc_project/venv/bin/python` and `race_rc_project/venv/bin/pip`.

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

What it does: updates Python's package installer so dependency installation is more reliable.

## Kaggle Dataset Setup

The project uses the instructor-specified Kaggle dataset:

```text
ankitdhiman7/race-dataset
```

Kaggle may give either the older `kaggle.json` file or the newer single API token.

### Option A: Kaggle Access Token

Create Kaggle's local credential folder:

```bash
mkdir -p ~/.kaggle
```

What it does: creates the hidden Kaggle configuration folder in your home directory.

Save your token:

```bash
nano ~/.kaggle/access_token
```

What it does: opens a terminal editor. Paste your Kaggle token, then press `Ctrl+O`, `Enter`, and `Ctrl+X` to save and exit.

Secure the token file:

```bash
chmod 600 ~/.kaggle/access_token
```

What it does: restricts the token file so only your user account can read it. Kaggle refuses insecure credential files.

Then run local setup:

```bash
python setup_local.py
```

What it does: installs dependencies, creates folders, downloads the Kaggle dataset, extracts it, normalizes columns, and creates the required 80/10/10 split.

### Option B: kaggle.json

If Kaggle provides `kaggle.json`, put it in the project root:

```text
race_rc_project/kaggle.json
```

Then run:

```bash
python setup_local.py
```

What it does: copies `kaggle.json` to `~/.kaggle/kaggle.json`, downloads the dataset, and prepares the CSV splits.

### Option C: Manual CSV

If you manually downloaded the CSV from Kaggle, run:

```bash
python setup_local.py --source-csv /absolute/path/to/downloaded_race.csv
```

What it does: skips Kaggle download and splits your local CSV into `train.csv`, `val.csv`, and `test.csv`.

### Option D: Dependencies Already Installed

If dependencies are already installed and you only want to prepare data:

```bash
python setup_local.py --skip-install
```

What it does: skips `pip install` and only handles directories, Kaggle download, splitting, and CSV verification.

`setup_local.py` installs dependencies, creates folders, downloads `ankitdhiman7/race-dataset` if Kaggle credentials are available, and creates an 80/10/10 split.

After setup, these files should exist:

```text
data/raw/train.csv
data/raw/val.csv
data/raw/test.csv
```

For the instructor-specified split, the expected shapes are about:

```text
train.csv: 70292 rows
val.csv:    8787 rows
test.csv:   8787 rows
```

## Command Guide

Preprocess the data:

```bash
python src/preprocessing.py --sample-size 5000 --max-features 10000
```

What it does: loads `data/raw/train.csv`, `val.csv`, and `test.csv`; expands every question into four option-level binary samples; extracts lexical features; fits the one-hot vectorizer; saves sparse matrices and vectorizers to `data/processed/`.

Train Model A:

```bash
python src/model_a_train.py --skip-slow
```

What it does: trains Logistic Regression, calibrated LinearSVC, ComplementNB question-type classification, soft voting, and hard voting. `--skip-slow` skips heavier optional models like stacking, XGBoost, KMeans, GMM, and LabelSpreading.

Train Model B:

```bash
python src/model_b_train.py
```

What it does: trains/evaluates the distractor and hint generation pipeline, saves Model B artifacts, and writes Model B metrics.

Evaluate saved models:

```bash
python src/evaluate.py
```

What it does: loads saved models and processed test matrices, computes benchmark metrics, and saves `results/benchmark.csv`.

Run tests:

```bash
python tests/test_inference.py
```

What it does: checks preprocessing, demo-mode inference, option prediction, empty-input handling, and Model B helper outputs.

Run the app locally:

```bash
python run_local.py
```

What it does: checks whether data/models exist, starts Streamlit from the correct project root, and serves the app at `http://localhost:8501`.

Open this in your browser:

```text
http://localhost:8501
```

## Train Locally

Quick run:

```bash
python src/preprocessing.py --sample-size 5000 --max-features 10000
python src/model_a_train.py --skip-slow
python src/model_b_train.py
python src/evaluate.py
```

Fuller run:

```bash
python src/preprocessing.py --max-features 10000
python src/model_a_train.py
python src/model_b_train.py --hint-train-rows 3000 --eval-rows 500 --hint-eval-rows 300
python src/evaluate.py
```

Model A uses option-level training, as required by the project. Each original question becomes four binary samples, so the expanded dataset naturally has about 25 percent positive labels and 75 percent negative labels. The report should emphasize macro F1, per-class recall, and confusion matrices rather than accuracy alone.

## Run The Local App

Recommended:

```bash
python run_local.py
```

Then open:

```text
http://localhost:8501
```

Alternative:

```bash
streamlit run ui/app.py
```

If model files are missing, the app still runs in demo mode with rule-based fallbacks.

## Run Tests

No test runner is required:

```bash
python tests/test_inference.py
```

Expected final line:

```text
All inference/preprocessing/generation tests passed.
```

## Optional Colab Launch

Local execution is preferred for the UI. If you still need Colab:

```python
%run ui/colab_launcher.py
```

Do not open `localhost:8501` directly from your browser when using Colab. That points to your own laptop, not the Colab VM.

## Model Performance Table

Replace these placeholder values after training on your selected split size.

| Model | Accuracy | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.527 | 0.489 | 0.515 | 0.520 |
| Calibrated LinearSVC | 0.750 | 0.429 | 0.375 | 0.500 |
| Soft Voting Ensemble | 0.629 | 0.512 | 0.512 | 0.513 |
| Hard Voting Ensemble | 0.527 | 0.489 | 0.515 | 0.520 |
| Stacking Ensemble | TBD | TBD | TBD | TBD |
| XGBoost Optional | TBD | TBD | TBD | TBD |

The LinearSVC result is a majority-class warning case: it reaches 75 percent accuracy by predicting every option as incorrect. Treat Soft Voting as the better balanced headline result because it has the strongest macro F1 in the quick run.

## Model B Results

Example quick-run metrics:

| Metric | Score |
|---|---:|
| Distractor Precision | 0.0027 |
| Distractor Recall | 0.0028 |
| Distractor F1 | 0.0027 |
| Distractor Ranker Accuracy | 1.0000 |
| Hint Precision@K | 1.0000 |
| Pairwise Cosine Diversity | 0.8412 |
| Hint Scorer R² | 1.0000 |

Low distractor precision/recall is expected because generated distractors rarely match RACE gold distractors verbatim. High hint scores are optimistic because the hint scorer uses lexical answer-overlap features; mention this limitation and include human evaluation.

## Rubric Coverage

- EDA: `notebooks/EDA.ipynb` includes dataset overview, article/question analysis, answer balance, option length bias, TF-IDF terms, word cloud, and cosine similarity distributions.
- Model A: supervised traditional ML, soft voting, hard voting, stacking, optional XGBoost, template question generation, confusion matrices, exact match, accuracy, and macro F1.
- Unsupervised/semi-supervised: KMeans, GMM, TruncatedSVD, silhouette, purity estimate, and LabelSpreading with 10 percent labeled data.
- Model B: distractor extraction/ranking, frequency substitution, diversity penalty, hint generation, ML hint scorer, and evaluation metrics.
- Report and human evaluation: templates are in `reports/final_report_template.md`, `reports/final_report_template.tex`, and `reports/human_evaluation_form.csv`.

## Known Limitations

- Traditional sparse lexical models cannot deeply reason over long passages.
- Template-generated questions are grammatical only when the source sentence is simple enough.
- Distractors are plausible lexical alternatives, not guaranteed semantically equivalent to human-written wrong options.
- The optional XGBoost and stacking models are capped for runtime and may need tuning for full-dataset performance.
- Exact-match automatic distractor evaluation underestimates plausible generated distractors.

## References

- Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. RACE: Large-scale ReAding Comprehension Dataset From Examinations.
- Manning, C. D., Raghavan, P., & Schutze, H. Introduction to Information Retrieval.
- Pedregosa, F. et al. Scikit-learn: Machine Learning in Python.
- Mitkov, R., Ha, L. A., & Karamanis, N. A computer-aided environment for generating multiple-choice test items.
- Susanti, Y. et al. Automatic distractor generation for multiple-choice English vocabulary questions.
