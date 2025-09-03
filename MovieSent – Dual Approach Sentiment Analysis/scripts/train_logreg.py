from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path when running this script directly
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moviesent.config import (
	ARTIFACTS_DIR,
	TEST_SIZE,
	RANDOM_STATE,
	LABEL_TO_ID,
)
from moviesent.data_loader import load_dataset, prepare_texts_and_labels
from moviesent.features import fit_tfidf_and_save, load_tfidf
from moviesent.evaluate import compute_classification_metrics, save_confusion_matrix


def main():
	df = load_dataset()
	texts, labels = prepare_texts_and_labels(df)
	# Map labels present in data
	unique_labels = sorted(set(labels))
	label_to_id = {lbl: LABEL_TO_ID.get(lbl, i) for i, lbl in enumerate(unique_labels)}
	y = np.array([label_to_id[lbl] for lbl in labels], dtype=np.int64)

	X_train, X_test, y_train, y_test = train_test_split(
		texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
	)

	# Vectorizer
	vec = fit_tfidf_and_save(X_train)
	X_train_vec = vec.transform(X_train)
	X_test_vec = vec.transform(X_test)

	# Model
	multi = "ovr" if len(set(y)) <= 2 else "multinomial"
	solver = "liblinear" if multi == "ovr" else "lbfgs"
	clf = LogisticRegression(max_iter=200, solver=solver, multi_class=multi, n_jobs=None)
	clf.fit(X_train_vec, y_train)

	# Evaluate
	y_pred = clf.predict(X_test_vec)
	metrics = compute_classification_metrics(y_test, y_pred, average="weighted")
	labels_sorted = sorted(set(y_test.tolist()))
	cm_path = save_confusion_matrix(y_test, y_pred, labels_sorted, "logreg_confusion_matrix.png")

	# Save artifacts
	joblib.dump(clf, ARTIFACTS_DIR / "logreg_model.joblib")
	with open(ARTIFACTS_DIR / "logreg_label_to_id.json", "w", encoding="utf-8") as f:
		json.dump(label_to_id, f, indent=2)
	with open(ARTIFACTS_DIR / "logreg_metrics.json", "w", encoding="utf-8") as f:
		json.dump({**metrics, "confusion_matrix": str(cm_path)}, f, indent=2)
	print("Saved Logistic Regression model and artifacts to:", ARTIFACTS_DIR)


if __name__ == "__main__":
	main()
