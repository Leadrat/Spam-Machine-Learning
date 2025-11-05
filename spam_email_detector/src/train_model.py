import os
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
import re
import string

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'model'
VEC_PATH = MODEL_DIR / 'vectorizer.pkl'
MODEL_PATH = MODEL_DIR / 'spam_model.pkl'

# Ensure dirs
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ensure NLTK stopwords
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    # URLs / emails / numbers placeholders
    s = re.sub(r"https?://\S+|www\.\S+", " __URL__ ", s)
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}\b", " __EMAIL__ ", s)
    s = re.sub(r"\b\d+\b", " __NUM__ ", s)
    # Remove HTML
    s = re.sub(r"<.*?>", " ", s)
    # Remove punctuation
    s = s.translate(PUNCT_TABLE)
    # Tokenize simple
    toks = s.split()
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    return " ".join(toks)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load various spam datasets formats and map to columns: text,label."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Place a CSV with columns label,text or v1,v2.")
    # Try flexible reads
    # Read robustly: prefer utf-8, replace undecodable bytes; fallback to latin-1
    try:
        df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace', on_bad_lines='skip')
    except TypeError:
        # Older pandas: encoding_errors not supported
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(path, encoding='latin-1', on_bad_lines='skip')
    # Map common schemas
    cols = {c.lower(): c for c in df.columns}
    if {'label', 'text'}.issubset(cols):
        text_col, label_col = cols['text'], cols['label']
    elif {'v1', 'v2'}.issubset(cols):  # SMS Spam Collection
        label_col, text_col = cols['v1'], cols['v2']
    else:
        # Heuristic: first column label, second column text
        label_col = df.columns[0]
        text_col = df.columns[1]
    out = df[[text_col, label_col]].copy()
    out.columns = ['text', 'label']
    out['label'] = out['label'].astype(str).str.strip().str.lower()
    out['label'] = out['label'].map({'spam': 1, 'ham': 0, 'ham.': 0, 'spam.': 1}).fillna(out['label'])
    # Drop rows with non-binary labels
    out = out[out['label'].isin([0, 1, '0', '1'])].copy()
    out['label'] = out['label'].astype(int)
    # Clean
    out['text'] = out['text'].astype(str).map(clean_text)
    out = out.dropna(subset=['text'])
    out = out[out['text'].str.strip().astype(bool)]
    return out


def train(dataset_path: Path, max_features: int = 20000, ngram_max: int = 2):
    df = load_dataset(dataset_path)
    X = df['text'].tolist()
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max), min_df=2, max_df=0.95)
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.5)
    model.fit(X_tr, y_train)

    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)

    print('\nEvaluation metrics:')
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1-score: {f1:.4f}')
    print('\nClassification report:\n', classification_report(y_test, preds, target_names=['Ham', 'Spam'], zero_division=0))

    # Persist
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f'Saved vectorizer -> {VEC_PATH}')
    print(f'Saved model -> {MODEL_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=str(DATA_DIR / 'spam.csv'), help='Path to CSV dataset')
    parser.add_argument('--features', type=int, default=20000)
    parser.add_argument('--ngram', type=int, default=2)
    args = parser.parse_args()

    train(Path(args.data), max_features=args.features, ngram_max=args.ngram)
