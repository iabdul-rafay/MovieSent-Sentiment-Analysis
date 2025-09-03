from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .config import ARTIFACTS_DIR, ID_TO_LABEL


def compute_classification_metrics(y_true, y_pred, average: str = "weighted") -> Dict[str, float]:
	acc = accuracy_score(y_true, y_pred)
	prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
	return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def save_confusion_matrix(y_true, y_pred, labels: List[int], filename: str) -> Path:
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	fig, ax = plt.subplots(figsize=(5, 4))
	lab_names = [ID_TO_LABEL.get(i, str(i)) for i in labels]
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=lab_names, yticklabels=lab_names, ax=ax)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_title("Confusion Matrix")
	out_path = ARTIFACTS_DIR / filename
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)
	return out_path
