# Tweet Sentiment Analysis

An end-to-end binary tweet sentiment system trained on the complete 1.6-million-row Sentiment140 training dataset. It includes a documented Jupyter notebook, a reproducible training pipeline, saved scikit-learn models, and a seven-tab Streamlit dashboard.

## What the system does

- Loads and validates Sentiment140.
- Normalizes contractions while preserving negation (`not`, `no`, and `nor`).
- Normalizes elongated words such as `goooood`.
- Removes URLs, mentions, and non-alphabetic noise while retaining hashtag words.
- Applies stopword removal and Porter stemming.
- Fits TF-IDF on the training split only to prevent leakage.
- Trains and compares Multinomial Naive Bayes and Logistic Regression.
- Evaluates models on a stratified 20% held-out test set.
- Provides interactive analysis and live predictions through Streamlit.

## Project files

| File | Purpose |
|---|---|
| `Tweet_Sentiment_Analysis_End_to_End.ipynb` | Cell-by-cell learning and experimentation workflow |
| `preprocess.py` | Full-data preprocessing, training, evaluation, and artifact export |
| `app.py` | Streamlit analysis dashboard and prediction interface |
| `model/` | Saved vectorizer, classifiers, and evaluation metadata |
| `requirements.txt` | Python dependencies |

The notebook is the explanatory layer. Once the experiment is understood, `preprocess.py` provides repeatable training and `app.py` provides loading, inference, visualizations, and interaction.

## Current full-data results

Using 1,280,000 training tweets and a held-out test set of 320,000 tweets:

| Model | Test accuracy |
|---|---:|
| Multinomial Naive Bayes | 77.83% |
| Logistic Regression | **79.85%** |

The application uses the best held-out model, Logistic Regression, as its selected prediction. Both model probabilities remain visible for comparison.

## Dataset setup

The raw and processed CSV files are intentionally excluded from Git because each exceeds GitHub's normal per-file size limit.

Place the Sentiment140 training file in the repository root with this exact name:

```text
training.1600000.processed.noemoticon.csv
```

The file must contain the standard six Sentiment140 fields: target, tweet ID, date, query, username, and text. Targets `0` and `4` are mapped to negative and positive.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the notebook

Open `Tweet_Sentiment_Analysis_End_to_End.ipynb` in Jupyter or VS Code and select **Run All**. Notebook outputs use `model_notebook/` and `processed_tweets_notebook.csv`, so they do not overwrite the application artifacts.

## Rebuild the production models

```powershell
python preprocess.py
```

The script processes all 1.6 million tweets. Runtime and memory use depend on the computer. It writes model files to `model/` and creates `processed_tweets.csv` for the dashboard.

## Start the dashboard

```powershell
streamlit run app.py
```

The dashboard contains overview, word, user, search, real-time classifier, model-report, and high-confidence-tweet tabs.

## Correct interpretation

- The task is binary: positive versus negative.
- A probability close to 0.5 is shown as **uncertain**, not neutral.
- Dashboard probabilities computed for all rows are exploratory. Reported model performance comes only from the held-out test split.
- A confident probability is not proof that a prediction is semantically correct.

## Sarcasm and other limitations

This TF-IDF system does not reliably understand sarcasm, irony, mixed sentiment, cultural references, or conversation context. For example, *"Great, another cancelled train"* contains a positive word but expresses negative sentiment. No responsible implementation should claim that every such case is solved.

Meaningful future improvements include:

1. Fine-tune a context-aware Twitter language model on human-reviewed sentiment and sarcasm datasets.
2. Include conversation context, reply structure, emojis, and hashtags where permitted.
3. Add a separately evaluated neutral or mixed-sentiment class instead of inventing it with thresholds.
4. Maintain an edge-case test suite for negation, spelling variation, idioms, sarcasm, and domain shift.
5. Evaluate calibration, cross-domain performance, fairness, and temporal drift in addition to accuracy.

## Reproducibility

The train/test split and sampling operations use random seed `42`. TF-IDF is fitted only on the training text. The training script saves the vectorizer together with the models so live inputs receive the same feature transformation used during training.
