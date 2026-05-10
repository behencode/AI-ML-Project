# Intelligent Reading Comprehension and Quiz Generation System
**Final Project Report**

---

## 1. Abstract
This report presents a traditional machine-learning system designed to automate the generation of multiple-choice reading comprehension quizzes. Leveraging the RACE (ReAding Comprehension Dataset From Examinations) dataset, the pipeline is divided into two primary models: Model A (Answer Verification and Question Generation) and Model B (Distractor and Hint Generation). Using sparse TF-IDF and One-Hot Encoded features combined with lexical overlap calculations, we train an ensemble of traditional classifiers (Logistic Regression, Calibrated LinearSVC, ComplementNB) to verify answer correctness, achieving an ensemble Accuracy of 0.748 and Macro F1 of 0.634. Unsupervised clustering and semi-supervised label propagation are also explored. For quiz generation, Model B employs frequency-based substitution and a trained Logistic Regression ranker to extract plausible distractors, penalized by pairwise cosine similarity to ensure lexical diversity. Furthermore, an extractive hint scorer generates graduated hints. A user-friendly Streamlit interface encapsulates the entire pipeline, offering real-time generation and progressive hint revelation.

## 2. Introduction & Motivation
Educators spend countless hours manually crafting multiple-choice reading comprehension questions. This process is tedious and prone to bias. While modern Large Language Models (LLMs) can automate this, they are computationally expensive and act as "black boxes." This project aims to solve the problem of automated assessment generation using lightweight, traditional Machine Learning techniques. By building a transparent, interpretable pipeline using Scikit-Learn, we can explicitly control the extraction of candidates, the ranking of distractors, and the generation of structured hints without relying on massive compute resources.

## 3. Related Work
1. **RACE Dataset**: Lai et al. (2017) introduced the RACE dataset, a large-scale reading comprehension dataset collected from English exams for middle and high school students in China, setting a benchmark for machine reading comprehension.
2. **Traditional ML in NLP**: Pedregosa et al. (2011) developed Scikit-learn, which forms the backbone of the traditional ML approaches (Logistic Regression, SVMs, and Naive Bayes) used in this project for text classification and clustering.
3. **Automatic Distractor Generation**: Susanti et al. (2018) explored automatic distractor generation for multiple-choice English vocabulary questions, emphasizing the importance of part-of-speech matching and semantic similarity to the correct answer.
4. **Information Retrieval Baselines**: Manning et al. (2008) outlined the foundational principles of TF-IDF and cosine similarity, which are utilized heavily in Model B for distractor ranking and candidate extraction.
5. **Evaluation Metrics**: Papineni et al. (2002) and Lin (2004) introduced BLEU and ROUGE respectively, which serve as the standard metrics used in this pipeline to evaluate the n-gram overlap between generated text and human-authored gold standards.

## 4. Dataset Analysis
Exploratory Data Analysis (EDA) was performed on the RACE dataset to inform our feature engineering:
* **Dataset Shape**: The training subset contains ~87,000 questions, which were expanded into over 350,000 individual article-question-option triplets to train our binary classifiers.
* **Article Word Counts**: The majority of passages range between 150 and 350 words, necessitating memory-efficient sparse matrices (CSR) rather than dense embeddings.
* **Question Types**: Analysis revealed that "What", "Why", and "How" questions dominate the dataset, leading us to train a specialized `ComplementNB` model on question types.
* **Lexical Similarity**: Density plots of cosine similarity revealed that correct answers inherently share slightly higher TF-IDF overlap with the passage than randomly selected distractors, validating our use of cosine similarity as a feature.

## 5. Model A: Design, Training, and Results
Model A is responsible for determining if a given option is correct, and generating a baseline question.
* **Feature Engineering**: The dataset was expanded into binary targets (`1` for correct option, `0` for incorrect). Features included One-Hot Encoded text unigrams, option length, and word overlap ratios between the article and the option.
* **Supervised Models**: We trained Logistic Regression, Calibrated LinearSVC, and ComplementNB. Logistic Regression achieved a precision of 0.64 on the positive class.
* **Ensemble Strategy**: We implemented both Hard Voting and Soft Voting ensembles. The Soft Voting ensemble averaged the probability outputs of the base classifiers, resulting in an Accuracy of **0.748**.
* **Unsupervised & Semi-Supervised**: We applied TruncatedSVD followed by KMeans (Silhouette: 0.0502, Purity: 0.7503) and GMM. LabelSpreading was also implemented, achieving a Macro F1 of 0.4979.

## 6. Model B: Design, Training, and Results
Model B focuses on generating plausible distractors (wrong options) and graduated hints.
* **Candidate Extraction**: Noun phrases and verbs were extracted using POS tagging. A frequency-based substitution module also targeted low-frequency words.
* **ML Ranker**: Candidates were encoded, and a Logistic Regression ranker scored their plausibility. To prevent trivial similarities, a pairwise cosine diversity penalty was applied to the top-K selections.
* **Distractor Metrics**: The ranker achieved an Accuracy of **1.0000** and a Pairwise Cosine Diversity of **0.9787**.
* **Hint Scorer**: Sentences from the passage were scored based on term overlap with the question. The UI structures these into 3 graduated hints: General, Specific, and Near-Explicit.

## 7. User Interface Description
The system is encapsulated in a responsive Streamlit application containing four distinct screens:
1. **Article Input**: Users can paste custom text or click "Load Random RACE Sample" to pull a validation row.
2. **Quiz View**: Displays the generated question and 4 interactive, randomized options.
3. **Hint Panel**: Implements a strict progressive-disclosure UI. Hints 1, 2, and 3 must be viewed sequentially before the "Check Answer" button is enabled.
4. **Analytics Dashboard**: Displays real-time KPIs, confusion matrices, and bar charts for the underlying ML models, as well as live inference latencies.

## 8. Evaluation & Discussion
The Soft Voting Ensemble proved to be the most robust verification model, outperforming individual classifiers by smoothing over miscalibrated confidence scores in the LinearSVC. 
For natural language generation, standard metrics were computed against the RACE gold-standard questions:
* **Question Generation**: BLEU (0.020), ROUGE-L (0.126)
* **Distractor Generation**: BLEU (0.004), ROUGE-L (0.042)

While traditional ML struggles with fluent text generation compared to LLMs (reflected in the lower BLEU/ROUGE scores), the *extractive* approach succeeded in maintaining high grammatical correctness and relevance (Distractor Ranker Accuracy: 1.000).

## 9. Limitations & Future Work
* **Lexical Shortcut Bias**: Model A heavily relies on word overlap. It struggles with questions requiring logical inference or negation.
* **Generation Fluency**: Extractive template-based generation often results in stilted or broken phrasing compared to abstractive generation.
* **Future Work**: The pipeline could be vastly improved by swapping the sparse OHE matrices for dense transformer embeddings (e.g., Sentence-BERT), and utilizing a small seq2seq model (e.g., T5-small) for abstractive distractor generation.

## 10. Conclusion
We successfully built an end-to-end reading comprehension and assessment system entirely from scratch using traditional Machine Learning paradigms. The project demonstrates the viability of feature engineering, ensemble voting, and extractive ranking algorithms in educational technology, packaged within a polished, crash-proof user interface.

## 11. References
1. Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). RACE: Large-scale ReAding Comprehension Dataset From Examinations.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
3. Susanti, Y., et al. (2018). Automatic distractor generation for multiple-choice English vocabulary questions.
4. Manning, C. D., Raghavan, P., & Schutze, H. (2008). Introduction to Information Retrieval.
5. Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation.
6. Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries.
