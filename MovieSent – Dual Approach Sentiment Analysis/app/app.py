from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

# Ensure project root is on sys.path when running this file directly
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from moviesent.config import ARTIFACTS_DIR
from moviesent.preprocess import clean_text
from moviesent.features import load_tfidf, load_tokenizer, texts_to_padded_sequences

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load artifacts (lazily on first use)
_tfidf = None
_logreg = None
_tok = None
_lstm = None


def ensure_logreg_loaded():
	global _tfidf, _logreg, _logreg_label_to_id
	if _tfidf is None:
		_tfidf = load_tfidf()
	if _logreg is None:
		_logreg = joblib.load(ARTIFACTS_DIR / "logreg_model.joblib")
		with open(ARTIFACTS_DIR / "logreg_label_to_id.json", "r", encoding="utf-8") as f:
			_logreg_label_to_id = json.load(f)


def ensure_lstm_loaded():
	global _tok, _lstm, _lstm_label_to_id
	if _tok is None:
		_tok = load_tokenizer()
	if _lstm is None:
		_lstm = load_model(ARTIFACTS_DIR / "lstm_model_final.keras")
		with open(ARTIFACTS_DIR / "lstm_label_to_id.json", "r", encoding="utf-8") as f:
			_lstm_label_to_id = json.load(f)


@app.route("/")
def index():
	return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
	data = request.get_json(silent=True) or request.form
	text = (data.get("text") or "").strip()
	model_choice = (data.get("model") or "logreg").lower()
	if not text:
		return jsonify({"error": "Text is required"}), 400

	cleaned = clean_text(text)

	if model_choice == "lstm":
		ensure_lstm_loaded()
		_, pad = texts_to_padded_sequences(_tok, [cleaned])
		probs = _lstm.predict(pad, verbose=0)
		if probs.ndim == 1 or probs.shape[1] == 1:
			p = float(probs.ravel()[0])
			pred_id = int(p >= 0.5)
			proba = {"negative": 1 - p, "positive": p}
			pred_label = "positive" if pred_id == 1 else "negative"
		else:
			pred_id = int(np.argmax(probs, axis=1)[0])
			pred_label = None
			# invert map
			id_to_label = {v: k for k, v in _lstm_label_to_id.items()}
			pred_label = id_to_label.get(pred_id, str(pred_id))
			proba = {id_to_label.get(i, str(i)): float(probs[0, i]) for i in range(probs.shape[1])}
		return jsonify({"model": "lstm", "label": pred_label, "proba": proba})

	# default: logreg
	ensure_logreg_loaded()
	X = _tfidf.transform([cleaned])
	if hasattr(_logreg, "predict_proba"):
		probs = _logreg.predict_proba(X)[0]
		classes = _logreg.classes_.tolist()
		id_to_label = {v: k for k, v in _logreg_label_to_id.items()}
		proba = {id_to_label.get(int(c), str(c)): float(p) for c, p in zip(classes, probs)}
		pred_id = int(classes[int(np.argmax(probs))])
		pred_label = id_to_label.get(pred_id, str(pred_id))
	else:
		pred_id = int(_logreg.predict(X)[0])
		id_to_label = {v: k for k, v in _logreg_label_to_id.items()}
		pred_label = id_to_label.get(pred_id, str(pred_id))
		proba = {}
	return jsonify({"model": "logreg", "label": pred_label, "proba": proba})


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
