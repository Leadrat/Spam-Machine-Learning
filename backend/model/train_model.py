import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure project root is on sys.path when running as a script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from backend.utils.preprocess import batch_clean
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Emails.csv")
ART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
VECTORIZER_PATH = os.path.join(ART_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(ART_DIR, "model.pkl")
THRESHOLD_PATH = os.path.join(ART_DIR, "threshold.txt")

MAX_FEATURES = 10000


def load_data(path: str):
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Emails.csv must have 'text' and 'label' columns")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = batch_clean(df["text"].tolist())
    y = (df["label"].astype(str).str.lower() == "spam").astype(int).values
    X = df["text"].tolist()
    return X, y

def main():
    os.makedirs(ART_DIR, exist_ok=True)
    print(f"Loading data from {DATA_PATH}")
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Fitting TF-IDF vectorizer and LogisticRegression...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    # Handle class imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, class_weights)}

    model = LogisticRegression(max_iter=200, class_weight=cw, solver="liblinear")
    model.fit(X_tr, y_train)

    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)

    print("\nEvaluation metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_test, preds, target_names=["Ham", "Spam"], zero_division=0))

    # Compute best threshold on test set by maximizing F1
    cand = np.linspace(0.2, 0.9, 15)
    best_t, best_f1 = 0.5, 0.0
    for t in cand:
        p = (probs >= t).astype(int)
        _, _, f1_t, _ = precision_recall_fscore_support(y_test, p, average="binary", zero_division=0)
        if f1_t > best_f1:
            best_f1, best_t = f1_t, float(t)
    with open(THRESHOLD_PATH, "w") as f:
        f.write(f"{best_t:.4f}")
    print(f"Saved best threshold {best_t:.4f} -> {THRESHOLD_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
