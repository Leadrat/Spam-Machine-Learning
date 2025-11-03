import os
import pickle
from typing import Tuple

from backend.utils.preprocess import clean_text

ART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
VECTORIZER_PATH = os.path.join(ART_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(ART_DIR, "model.pkl")
THRESHOLD_PATH = os.path.join(ART_DIR, "threshold.txt")

_model = None
_vectorizer = None
_threshold = float(os.getenv("SPAM_THRESHOLD", "0.5"))


def load_artifacts():
    global _model, _vectorizer, _threshold
    if _vectorizer is None:
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(VECTORIZER_PATH)
        with open(VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    # If a saved threshold exists, prefer it
    try:
        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH, "r") as f:
                _threshold = float(f.read().strip())
    except Exception:
        pass
    return _model, _vectorizer


def predict_text(text: str) -> Tuple[str, float]:
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        load_artifacts()

    cleaned = clean_text(text)
    if not cleaned.strip():
        cleaned = text.lower()
    X = _vectorizer.transform([cleaned])
    prob = float(_model.predict_proba(X)[0][1])
    label = "Spam" if prob >= _threshold else "Ham"
    try:
        print(f"[Predict] cleaned='{cleaned[:120]}' prob={prob:.4f} threshold={_threshold:.2f} label={label}")
    except Exception:
        pass
    return label, prob
