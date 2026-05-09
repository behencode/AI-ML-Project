# Intelligent Reading Comprehension and Quiz Generation System

This project builds a traditional machine-learning reading comprehension system on the RACE dataset. It verifies candidate answers, generates template-based questions, creates plausible distractors, produces graduated hints, and exposes everything through a four-screen Streamlit app. The implementation is designed to run on Google Colab with a T4 GPU, although the models are traditional CPU-friendly scikit-learn models.

## Project Structure

```text
race_rc_project/
├── setup_colab.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/
├── src/
│   ├── preprocessing.py
│   ├── model_a_train.py
│   ├── model_b_train.py
│   ├── inference.py
│   └── evaluate.py
├── ui/
│   └── app.py
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

## Setup on Google Colab

1. Upload or clone this project into Colab.
2. Upload your Kaggle API token as `kaggle.json` in the project root.
3. Install packages and create folders:

```bash
python setup_colab.py --skip-install
pip install -r requirements.txt
```

4. Download the instructor-specified Kaggle dataset. The setup script defaults to `ankitdhiman7/race-dataset`, downloads the single CSV, and creates an 80/10/10 split:

```bash
python setup_colab.py
```

If you already downloaded the single CSV manually, split it without Kaggle:

```bash
python setup_colab.py --skip-install --source-csv data/raw/<downloaded-file>.csv
```

5. Confirm the generated split files exist:

```text
data/raw/train.csv
data/raw/val.csv
data/raw/test.csv
```

Each CSV must contain `id, article, question, A, B, C, D, answer`.

In Colab notebooks, shell commands need a leading `!`, for example:

```bash
!python setup_colab.py
```

## How to Train

For a quick test:

```bash
python src/preprocessing.py --sample-size 5000 --max-features 10000
python src/model_a_train.py --skip-slow
python src/model_b_train.py
python src/evaluate.py
```

For a fuller training run:

```bash
python src/preprocessing.py --max-features 10000
python src/model_a_train.py
python src/model_b_train.py
python src/evaluate.py
```

Model A uses option-level training, as required by the project. Each original question becomes four binary samples, so the expanded dataset naturally has about 25 percent positive labels and 75 percent negative labels. To avoid misleading majority-class accuracy, Logistic Regression and LinearSVC use balanced class weights, and training prints confusion matrices plus per-class precision, recall, and F1.

Model A trains Logistic Regression, calibrated LinearSVC, ComplementNB question-type classification, soft voting, hard voting, stacking, optional XGBoost, KMeans, GMM, and LabelSpreading. Model B trains/evaluates distractor and hint generation components.

## Run the Streamlit App

Local machine:

```bash
streamlit run ui/app.py
```

Google Colab:

```python
%run ui/colab_launcher.py
```

Do not open `localhost:8501` directly from your browser when running on Colab. That `localhost` is your own laptop, not the remote Colab VM. The launcher uses Colab's built-in port proxy and opens the correct hosted window. If it does not load, inspect:

```bash
!cat streamlit.log
```

If model files are missing, the app automatically enters demo mode with rule-based fallbacks and shows a warning banner.

## Run Tests

```bash
pytest tests
```

The tests verify that the inference engine works in demo mode and returns complete quiz payloads.

## Model Performance Table

Replace these placeholder values after training on your selected split size.

| Model | Accuracy | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Calibrated LinearSVC | TBD | TBD | TBD | TBD |
| Soft Voting Ensemble | TBD | TBD | TBD | TBD |
| Hard Voting Ensemble | TBD | TBD | TBD | TBD |
| Stacking Ensemble | TBD | TBD | TBD | TBD |
| XGBoost Optional | TBD | TBD | TBD | TBD |

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
- The Kaggle dataset slug varies by uploader, so `setup_colab.py` accepts it as a command-line argument.

## References

- Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. RACE: Large-scale ReAding Comprehension Dataset From Examinations.
- Manning, C. D., Raghavan, P., & Schutze, H. Introduction to Information Retrieval.
- Pedregosa, F. et al. Scikit-learn: Machine Learning in Python.
- Mitkov, R., Ha, L. A., & Karamanis, N. A computer-aided environment for generating multiple-choice test items.
- Susanti, Y. et al. Automatic distractor generation for multiple-choice English vocabulary questions.
