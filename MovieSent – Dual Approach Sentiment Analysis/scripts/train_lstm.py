from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# Ensure project root is on sys.path when running this script directly
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from moviesent.config import (
	ARTIFACTS_DIR,
	TEST_SIZE,
	RANDOM_STATE,
	LABEL_TO_ID,
	BATCH_SIZE,
	EPOCHS,
)
from moviesent.data_loader import load_dataset, prepare_texts_and_labels
from moviesent.features import fit_tokenizer_and_save, texts_to_padded_sequences, load_tokenizer
from moviesent.evaluate import compute_classification_metrics, save_confusion_matrix
from moviesent.models_lstm import build_lstm_model


def main():
	df = load_dataset()
	texts, labels = prepare_texts_and_labels(df)
	unique_labels = sorted(set(labels))
	label_to_id = {lbl: LABEL_TO_ID.get(lbl, i) for i, lbl in enumerate(unique_labels)}
	y = np.array([label_to_id[lbl] for lbl in labels], dtype=np.int64)

	X_train, X_test, y_train, y_test = train_test_split(
		texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
	)

	# Tokenizer
	tok = fit_tokenizer_and_save(X_train)
	_, X_train_pad = texts_to_padded_sequences(tok, X_train)
	_, X_test_pad = texts_to_padded_sequences(tok, X_test)

	num_classes = len(set(y))
	model = build_lstm_model(num_classes=num_classes)

	es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
	ckpt_path = ARTIFACTS_DIR / "lstm_model.keras"
	ckpt = ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True)

	model.fit(
		X_train_pad,
		y_train,
		validation_split=0.1,
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		callbacks=[es, ckpt],
		verbose=1,
	)

	# Reload best
	best_model = load_model(ckpt_path)

	# Evaluate
	if num_classes <= 2:
		y_prob = best_model.predict(X_test_pad, verbose=0).ravel()
		y_pred = (y_prob >= 0.5).astype(int)
	else:
		y_prob = best_model.predict(X_test_pad, verbose=0)
		y_pred = np.argmax(y_prob, axis=1)

	metrics = compute_classification_metrics(y_test, y_pred, average="weighted")
	labels_sorted = sorted(set(y_test.tolist()))
	cm_path = save_confusion_matrix(y_test, y_pred, labels_sorted, "lstm_confusion_matrix.png")

	# Save artifacts
	best_model.save(ARTIFACTS_DIR / "lstm_model_final.keras")
	with open(ARTIFACTS_DIR / "lstm_label_to_id.json", "w", encoding="utf-8") as f:
		json.dump(label_to_id, f, indent=2)
	with open(ARTIFACTS_DIR / "lstm_metrics.json", "w", encoding="utf-8") as f:
		json.dump({**metrics, "confusion_matrix": str(cm_path)}, f, indent=2)
	print("Saved LSTM model and artifacts to:", ARTIFACTS_DIR)


if __name__ == "__main__":
	main()
