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


def explain_text(text: str, top_k: int = 5) -> dict:
    """Return structured explanation with probability, label, reasons, and advice."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        load_artifacts()

    cleaned = clean_text(text)
    if not cleaned.strip():
        cleaned = text.lower()

    X = _vectorizer.transform([cleaned])
    prob = float(_model.predict_proba(X)[0][1])
    is_spam = prob >= _threshold

    # Compute top contributing tokens towards spam using linear model weights
    top_tokens = []
    try:
        coef = getattr(_model, "coef_", None)
        if coef is not None:
            coef = coef[0]  # shape: (n_features,)
            # Sparse vector of the sample
            row = X.tocoo()
            # Map index -> token
            inv_vocab = {idx: tok for tok, idx in _vectorizer.vocabulary_.items()}
            # Compute contribution = tfidf_value * weight (higher positive -> more spam)
            contribs = []
            for i, v in zip(row.col, row.data):
                w = float(coef[i])
                contribs.append((inv_vocab.get(i, f"f{i}"), v * w, w, v))
            # Sort by contribution descending
            contribs.sort(key=lambda x: x[1], reverse=True)
            for tok, score, weight, tfidf in contribs[:top_k]:
                top_tokens.append({"token": tok, "contribution": round(score, 6), "weight": round(weight, 6), "tfidf": round(float(tfidf), 6)})
    except Exception as e:
        try:
            print(f"[Explain] Fallback (no contributions): {e}")
        except Exception:
            pass

    # Heuristic reasons based on matched suspicious keywords
    suspicious_words = [
        "win", "winner", "free", "click", "offer", "claim", "gift", "prize", "cash", "urgent",
        "lottery", "congratulations", "limited", "now", "buy", "discount"
    ]
    found = [w for w in suspicious_words if w in cleaned.split()]
    reason = ""
    if found:
        reason = f"Contains words: {', '.join(sorted(set(found)))}"
    elif top_tokens:
        reason = "Top contributing tokens: " + ", ".join(t["token"] for t in top_tokens[:top_k])

    advice = "Avoid clicking on unknown links or attachments." if is_spam else "Looks safe. Still exercise caution with links."

    return {
        "type": "prediction",
        "isSpam": bool(is_spam),
        "probability": prob,
        "reason": reason,
        "advice": advice,
        "topTokens": top_tokens,
    }
