from __future__ import annotations

from typing import Optional

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from .config import MAX_VOCAB_SIZE, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, LSTM_UNITS, DROPOUT_RATE


def build_lstm_model(num_classes: int, learning_rate: float = 1e-3):
	model = Sequential()
	model.add(Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
	model.add(LSTM(LSTM_UNITS, return_sequences=False))
	model.add(Dropout(DROPOUT_RATE))
	if num_classes <= 2:
		model.add(Dense(1, activation="sigmoid"))
		loss = "binary_crossentropy"
	else:
		model.add(Dense(num_classes, activation="softmax"))
		loss = "sparse_categorical_crossentropy"
	model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=["accuracy"])
	return model
