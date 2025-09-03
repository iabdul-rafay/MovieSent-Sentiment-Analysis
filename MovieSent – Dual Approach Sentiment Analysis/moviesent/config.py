
from pathlib import Path

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = PROJECT_ROOT / "IMDB Dataset.csv"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Columns
TEXT_COLUMN: str = "review"
LABEL_COLUMN: str = "sentiment"

# Split
TEST_SIZE: float = 0.3
RANDOM_STATE: int = 42

# Preprocessing
LOWERCASE: bool = True
REMOVE_HTML: bool = True
REMOVE_PUNCT: bool = True
REMOVE_NUMBERS: bool = False
LEMMATIZE: bool = True
REMOVE_STOPWORDS: bool = True

# Features (TF-IDF)
TFIDF_MAX_FEATURES: int | None = 50000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 3)

# LSTM / Tokenizer
MAX_VOCAB_SIZE: int = 50000
MAX_SEQUENCE_LENGTH: int = 200
EMBEDDING_DIM: int = 100
LSTM_UNITS: int = 128
DROPOUT_RATE: float = 0.3
BATCH_SIZE: int = 64
EPOCHS: int = 5

# Labels map
LABEL_TO_ID: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}
