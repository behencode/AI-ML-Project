# Intelligent Reading Comprehension and Quiz Generation System

## Abstract
Summarize the goal, RACE dataset, traditional ML approach, key results, and UI outcome in 150-250 words.

## 1. Introduction
Describe reading comprehension, quiz generation, distractor generation, and why RACE is an appropriate benchmark.

## 2. Related Work
Discuss at least five papers or credible sources on reading comprehension, multiple-choice QA, TF-IDF/OHE baselines, distractor generation, and hint generation.

Suggested references:
- Lai et al., RACE: Large-scale Reading Comprehension Dataset From Examinations.
- Manning et al., Introduction to Information Retrieval.
- Pedregosa et al., Scikit-learn: Machine Learning in Python.
- Mitkov et al., Computer-aided generation of multiple-choice tests.
- Susanti et al., Automatic distractor generation for multiple choice questions.

## 3. Dataset Analysis
Report train/validation/test shapes, null checks, answer balance, article lengths, question types, and option length bias from `notebooks/EDA.ipynb`.

## 4. Model A Design, Training, and Results
Explain binary expansion, OHE plus lexical features, Logistic Regression, Calibrated LinearSVC, ComplementNB question-type classifier, soft voting, hard voting, stacking, optional XGBoost, KMeans, GMM, and LabelSpreading.

## 5. Model B Design, Training, and Results
Explain candidate extraction, medium-similarity distractor ranking, frequency substitution, diversity penalty, graduated hints, ML hint scorer, and human evaluation.

## 6. UI Description
Describe the four Streamlit screens: Article Input, Quiz View, Hint Panel, and Analytics Dashboard.

## 7. Evaluation and Discussion
Include accuracy, macro F1, precision, recall, exact match, confusion matrix, distractor precision/recall/F1, hint precision@K, R², latency, and qualitative findings.

## 8. Limitations and Future Work
Discuss limitations of traditional ML, lexical matching, dataset dependency, question templates, and possible future neural or retrieval-augmented extensions.

## 9. Conclusion
Summarize what was built, what worked, and what the evaluation suggests.

## References
Use IEEE, ACM, or APA style consistently. Include at least five academic or official references.
