import re
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
_nltk_resources = [
	("stopwords", "corpora/stopwords"),
	("punkt", "tokenizers/punkt"),
	("punkt_tab", "tokenizers/punkt_tab"),
	("wordnet", "corpora/wordnet"),
	("omw-1.4", "corpora/omw-1.4"),
]
for res, path in _nltk_resources:
	try:
		nltk.data.find(path)
	except LookupError:
		nltk.download(res, quiet=True)

from nltk.tokenize import word_tokenize

HTML_TAG_RE = re.compile(r"<[^>]+>")
PUNCT_RE = re.compile(r"[^\w\s]")
NUM_RE = re.compile(r"\d+")
WS_RE = re.compile(r"\s+")

_stopwords = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(
	text: str,
	*,
	lowercase: bool = True,
	remove_html: bool = True,
	remove_punct: bool = True,
	remove_numbers: bool = False,
	remove_stopwords: bool = True,
	lemmatize: bool = True,
) -> str:
	if text is None:
		return ""
	s = text
	if remove_html:
		s = HTML_TAG_RE.sub(" ", s)
	if lowercase:
		s = s.lower()
	if remove_numbers:
		s = NUM_RE.sub(" ", s)
	if remove_punct:
		s = PUNCT_RE.sub(" ", s)
	s = WS_RE.sub(" ", s).strip()

	tokens = word_tokenize(s)
	if remove_stopwords:
		tokens = [t for t in tokens if t not in _stopwords]
	if lemmatize:
		tokens = [_lemmatizer.lemmatize(t) for t in tokens]
	return " ".join(tokens)


def bulk_clean_text(
	texts: Iterable[str],
	*,  # Config toggles
	lowercase: bool = True,
	remove_html: bool = True,
	remove_punct: bool = True,
	remove_numbers: bool = False,
	remove_stopwords: bool = True,
	lemmatize: bool = True,
) -> List[str]:
	return [
		clean_text(
			text,
			lowercase=lowercase,
			remove_html=remove_html,
			remove_punct=remove_punct,
			remove_numbers=remove_numbers,
			remove_stopwords=remove_stopwords,
			lemmatize=lemmatize,
		)
		for text in texts
	]
