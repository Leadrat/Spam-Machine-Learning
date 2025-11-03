import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove numbers
    text = re.sub(r"\d+", " ", text)
    # Remove punctuation
    text = text.translate(PUNCT_TABLE)
    # Tokenize by whitespace
    tokens = text.split()
    # Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    # Collapse spaces
    return " ".join(tokens)


def batch_clean(texts: List[str]) -> List[str]:
    return [clean_text(t) for t in texts]
