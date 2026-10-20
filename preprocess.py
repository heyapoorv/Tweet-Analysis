"""
preprocess.py  -  Tweet Sentiment Analysis
-------------------------------------------
Full NLP Pipeline:
  Raw Tweet
    -> Regex Cleaning  (URLs, mentions, hashtags, special chars)
    -> Lowercase
    -> NLTK Stopword Removal
    -> Porter Stemming
    -> TF-IDF Vectorization (unigrams + bigrams)
    -> Train:  Multinomial Naive Bayes
               Logistic Regression
    -> Compare accuracy & classification reports
    -> Save best model  (model/best_model.pkl + model/tfidf_vectorizer.pkl)
    -> Save processed_tweets.csv  with model confidence scores

Run ONCE before launching app.py:
    python preprocess.py
"""

import re, time, json
from pathlib import Path
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import joblib

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
RAW_CSV   = BASE_DIR / "training.1600000.processed.noemoticon.csv"
OUT_CSV   = BASE_DIR / "processed_tweets.csv"
MODEL_DIR = BASE_DIR / "model"
TEST_SIZE = 0.20
RANDOM    = 42
PREDICTION_BATCH_SIZE = 100_000
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR.mkdir(exist_ok=True)

# ── Download NLTK data silently ───────────────────────────────────────────────
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    print("[*] Downloading the required NLTK stopwords corpus ...")
    nltk.download("stopwords", quiet=True, raise_on_error=True)
    STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS -= {"no", "nor", "not"}
stemmer    = PorterStemmer()

# =============================================================================
#  NLP PIPELINE – TEXT CLEANER
# =============================================================================
def clean_tweet(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
      1. Remove URLs
      2. Remove @mentions
      3. Strip # but keep hashtag word
      4. Remove non-alphabetic characters
      5. Lowercase
      6. Tokenise
      7. Remove stopwords
      8. Porter Stemming
      9. Re-join tokens
    """
    text = str(text).replace("’", "'")

    # Step 1-4 — Regex cleaning
    text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcan't\b", "can not", text, flags=re.IGNORECASE)
    text = re.sub(r"n't\b", " not", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Step 5 — Lowercase
    text = text.lower()
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)

    # Step 6 — Tokenise
    tokens = text.split()

    # Step 7 — Stopword removal
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # Step 8 — Stemming
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


# =============================================================================
#  1. LOAD DATA
# =============================================================================
print("\n[1/7] Loading raw Sentiment140 CSV ...")
t0 = time.time()
df = pd.read_csv(
    RAW_CSV,
    encoding="latin-1",
    header=None,
    names=["sentiment", "id", "date", "query", "user", "text"],
)
print(f"      Loaded {len(df):,} rows  ({time.time()-t0:.1f}s)")


# =============================================================================
#  2. USE THE COMPLETE BINARY DATASET
# =============================================================================
df = df[df["sentiment"].isin([0, 4])].reset_index(drop=True)

df["label"]          = (df["sentiment"] == 4).astype(int)   # 1=pos, 0=neg
df["sentiment_label"] = df["label"].map({1: "positive", 0: "negative"})
class_counts = df["sentiment_label"].value_counts()
print(
    f"[2/7] Complete dataset: {class_counts.get('positive', 0):,} positive + "
    f"{class_counts.get('negative', 0):,} negative = {len(df):,} total"
)


# =============================================================================
#  3. PARSE DATE
# =============================================================================
print("[3/7] Parsing dates ...")
df["date_parsed"] = pd.to_datetime(
    df["date"], format="%a %b %d %H:%M:%S PDT %Y", errors="coerce"
)
df["year"]  = df["date_parsed"].dt.year.fillna(2009).astype(int)
df["month"] = df["date_parsed"].dt.month.fillna(1).astype(int)
df["hour"]  = df["date_parsed"].dt.hour.fillna(0).astype(int)


# =============================================================================
#  4. NLP CLEANING  (Regex + Stopwords + Stemming)
# =============================================================================
print("[4/7] Applying NLP pipeline (regex + stopwords + stemming) ...")
t0 = time.time()
df["clean_text"] = df["text"].apply(clean_tweet)
print(f"      Cleaned {len(df):,} tweets  ({time.time()-t0:.1f}s)")


# =============================================================================
#  5. TF-IDF VECTORIZATION
# =============================================================================
print("[5/7] Fitting TF-IDF vectorizer  (unigrams + bigrams, max 60k features) ...")
t0 = time.time()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["clean_text"], df["label"],
    test_size=TEST_SIZE, random_state=RANDOM, stratify=df["label"]
)

tfidf = TfidfVectorizer(
    max_features=60_000,
    ngram_range=(1, 2),      # unigrams + bigrams
    sublinear_tf=True,       # log(tf) instead of tf
    min_df=3,
    strip_accents="unicode",
    analyzer="word",
)
X_train = tfidf.fit_transform(X_train_raw)
X_test  = tfidf.transform(X_test_raw)
print(f"      Vocab size: {len(tfidf.vocabulary_):,}  ({time.time()-t0:.1f}s)")


# =============================================================================
#  6. TRAIN & COMPARE BOTH MODELS
# =============================================================================
print("\n[6/7] Training models ...")
print("=" * 55)

results = {}

# ── Naive Bayes ───────────────────────────────────────────────────────────────
print("\n  >> Multinomial Naive Bayes ...")
t0 = time.time()
nb = MultinomialNB(alpha=0.5)
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc  = accuracy_score(y_test, nb_pred)
nb_time = time.time() - t0
results["Naive Bayes"] = {
    "model":    nb,
    "accuracy": nb_acc,
    "report":   classification_report(y_test, nb_pred,
                    target_names=["negative", "positive"], output_dict=True),
    "train_time": nb_time,
}
print(f"     Accuracy : {nb_acc*100:.2f}%   (trained in {nb_time:.1f}s)")

# ── Logistic Regression ───────────────────────────────────────────────────────
print("\n  >> Logistic Regression ...")
t0 = time.time()
lr = LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs", n_jobs=-1)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred)
lr_time = time.time() - t0
results["Logistic Regression"] = {
    "model":    lr,
    "accuracy": lr_acc,
    "report":   classification_report(y_test, lr_pred,
                    target_names=["negative", "positive"], output_dict=True),
    "train_time": lr_time,
}
print(f"     Accuracy : {lr_acc*100:.2f}%   (trained in {lr_time:.1f}s)")


# =============================================================================
#  PRINT COMPARISON TABLE
# =============================================================================
print("\n" + "=" * 55)
print("  MODEL COMPARISON")
print("=" * 55)
print(f"  {'Algorithm':<25} {'Accuracy':>10} {'Train Time':>12}")
print(f"  {'-'*25} {'-'*10} {'-'*12}")
for name, r in results.items():
    print(f"  {name:<25} {r['accuracy']*100:>9.2f}%  {r['train_time']:>10.1f}s")
print("=" * 55)

best_name  = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
best_acc   = results[best_name]["accuracy"]
print(f"\n  Winner: {best_name}  ({best_acc*100:.2f}%)\n")

print("  Naive Bayes - Classification Report:")
print(classification_report(y_test, nb_pred, target_names=["negative", "positive"]))
print("  Logistic Regression - Classification Report:")
print(classification_report(y_test, lr_pred, target_names=["negative", "positive"]))


# =============================================================================
#  7. SAVE ARTIFACTS
# =============================================================================
print("[7/7] Saving model artifacts ...")

# Save both models + vectorizer
joblib.dump(tfidf,      MODEL_DIR / "tfidf_vectorizer.pkl")
joblib.dump(nb,         MODEL_DIR / "nb_model.pkl")
joblib.dump(lr,         MODEL_DIR / "lr_model.pkl")
joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

# Save comparison metadata for dashboard
comparison = {
    name: {
        "accuracy":   round(r["accuracy"] * 100, 2),
        "train_time": round(r["train_time"], 2),
        "precision_pos": round(r["report"]["positive"]["precision"] * 100, 2),
        "recall_pos":    round(r["report"]["positive"]["recall"]    * 100, 2),
        "f1_pos":        round(r["report"]["positive"]["f1-score"]  * 100, 2),
        "precision_neg": round(r["report"]["negative"]["precision"] * 100, 2),
        "recall_neg":    round(r["report"]["negative"]["recall"]    * 100, 2),
        "f1_neg":        round(r["report"]["negative"]["f1-score"]  * 100, 2),
    }
    for name, r in results.items()
}
comparison["best_model"] = best_name

with open(MODEL_DIR / "comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

print(f"      Saved -> {MODEL_DIR}/tfidf_vectorizer.pkl")
print(f"      Saved -> {MODEL_DIR}/nb_model.pkl")
print(f"      Saved -> {MODEL_DIR}/lr_model.pkl")
print(f"      Saved -> {MODEL_DIR}/best_model.pkl")
print(f"      Saved -> {MODEL_DIR}/comparison.json")

# ── Add confidence scores to DataFrame ───────────────────────────────────────
print("[7/7] Computing confidence scores on the complete dataset in batches ...")
t0 = time.time()
nb_proba = np.empty(len(df), dtype=np.float32)
lr_proba = np.empty(len(df), dtype=np.float32)
for start in range(0, len(df), PREDICTION_BATCH_SIZE):
    end = min(start + PREDICTION_BATCH_SIZE, len(df))
    batch_vec = tfidf.transform(df["clean_text"].iloc[start:end])
    nb_proba[start:end] = nb.predict_proba(batch_vec)[:, 1]
    lr_proba[start:end] = lr.predict_proba(batch_vec)[:, 1]

df["nb_confidence"] = nb_proba
df["lr_confidence"] = lr_proba
df["confidence"]    = (nb_proba + lr_proba) / 2   # ensemble average

print(f"      Done ({time.time()-t0:.1f}s)")

# ── Save processed CSV ────────────────────────────────────────────────────────
out_cols = [
    "sentiment_label", "user", "text", "clean_text",
    "year", "month", "hour",
    "confidence", "nb_confidence", "lr_confidence"
]
df[out_cols].to_csv(OUT_CSV, index=False)
print(f"[+]   Saved -> {OUT_CSV}  ({len(df):,} rows)")

print("\n" + "=" * 55)
print(f"  FINAL — Best Model : {best_name}")
print(f"  Test Accuracy      : {best_acc*100:.2f}%")
print("=" * 55)
print("\nAll done!  Now run:  streamlit run app.py\n")
