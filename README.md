[README.md](https://github.com/user-attachments/files/22099807/README.md)
# MovieSent – Dual Approach Sentiment Analysis

A complete sentiment analysis project using a dual approach: classical Logistic Regression (TF-IDF) and deep learning LSTM (Keras/TensorFlow). Includes data processing pipeline, training scripts, evaluation, and a Flask web app for real-time predictions.

## Dataset
Place your dataset (minimum 5,000 reviews) as `IMDB Dataset.csv` in the project root, or update paths in `moviesent/config.py`.

Expected columns:
- `review`: the text content
- `sentiment`: one of [positive, negative] (or neutral if available)

## Quickstart
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train models:
   ```bash
   python scripts/train_logreg.py
   python scripts/train_lstm.py
   ```
4. Run web app:
   ```bash
   python app/app.py
   ```

## Project Structure
- `moviesent/`: core package (loading, preprocessing, features, models, evaluation)
- `scripts/`: training scripts for both models
- `app/`: Flask web app (REST API + basic UI)
- `artifacts/`: saved models, tokenizers, and vectorizers

## Notes
- NLTK resources are auto-downloaded on first run (stopwords, punkt, wordnet, omw-1.4).
- 70/30 stratified split is used.
- Evaluation includes accuracy, precision, recall, F1, and confusion matrices.


https://github.com/user-attachments/assets/34ff8269-e154-4881-92b5-80d6a0402153
