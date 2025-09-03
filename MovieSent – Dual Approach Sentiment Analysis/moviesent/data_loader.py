from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import pandas as pd

from .config import (
	DATA_PATH,
	TEXT_COLUMN,
	LABEL_COLUMN,
	LOWERCASE,
	REMOVE_HTML,
	REMOVE_PUNCT,
	REMOVE_NUMBERS,
	LEMMATIZE,
	REMOVE_STOPWORDS,
)
from .preprocess import bulk_clean_text


def _normalize_label(label: str) -> str:
	if not isinstance(label, str):
		return ""
	l = label.strip().lower()
	if l in {"pos", "+", "positive"}:
		return "positive"
	if l in {"neg", "-", "negative"}:
		return "negative"
	if l in {"neu", "neutral", "0"}:
		return "neutral"
	return l


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
	"""Load dataset from CSV with at least TEXT and LABEL columns."""
	p = Path(path) if path is not None else Path(DATA_PATH)
	df = pd.read_csv(p)
	if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
		raise ValueError(
			f"Dataset must contain columns '{TEXT_COLUMN}' and '{LABEL_COLUMN}', got: {list(df.columns)}"
		)
	# Drop missing
	df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
	# Normalize labels
	df[LABEL_COLUMN] = df[LABEL_COLUMN].map(_normalize_label)
	# Remove duplicates
	df = df.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)
	return df


def prepare_texts_and_labels(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
	texts: List[str] = df[TEXT_COLUMN].astype(str).tolist()
	labels: List[str] = df[LABEL_COLUMN].astype(str).tolist()
	# Clean texts
	cleaned = bulk_clean_text(
		texts,
		lowercase=LOWERCASE,
		remove_html=REMOVE_HTML,
		remove_punct=REMOVE_PUNCT,
		remove_numbers=REMOVE_NUMBERS,
		remove_stopwords=REMOVE_STOPWORDS,
		lemmatize=LEMMATIZE,
	)
	return cleaned, labels
