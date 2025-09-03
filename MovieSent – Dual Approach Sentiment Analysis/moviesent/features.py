from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .config import (
	ARTIFACTS_DIR,
	TFIDF_MAX_FEATURES,
	TFIDF_NGRAM_RANGE,
	MAX_VOCAB_SIZE,
	MAX_SEQUENCE_LENGTH,
)


# ---------- TF-IDF ----------

def build_tfidf_vectorizer() -> TfidfVectorizer:
	return TfidfVectorizer(
		max_features=TFIDF_MAX_FEATURES,
		ngram_range=TFIDF_NGRAM_RANGE,
		norm="l2",
		sublinear_tf=True,
	)


def fit_tfidf_and_save(texts, artifact_name: str = "tfidf_vectorizer.joblib") -> TfidfVectorizer:
	vec = build_tfidf_vectorizer()
	vec.fit(texts)
	joblib.dump(vec, ARTIFACTS_DIR / artifact_name)
	return vec


def load_tfidf(artifact_name: str = "tfidf_vectorizer.joblib") -> TfidfVectorizer:
	return joblib.load(ARTIFACTS_DIR / artifact_name)


# ---------- Tokenizer / Sequences ----------

def fit_tokenizer_and_save(texts, artifact_name: str = "tokenizer.joblib") -> Tokenizer:
	tok = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
	tok.fit_on_texts(texts)
	joblib.dump(tok, ARTIFACTS_DIR / artifact_name)
	return tok


def load_tokenizer(artifact_name: str = "tokenizer.joblib") -> Tokenizer:
	return joblib.load(ARTIFACTS_DIR / artifact_name)


def texts_to_padded_sequences(tok: Tokenizer, texts) -> Tuple[list, list]:
	seqs = tok.texts_to_sequences(texts)
	pads = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
	return seqs, pads
