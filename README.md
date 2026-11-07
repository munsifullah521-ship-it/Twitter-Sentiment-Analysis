# Twitter Sentiment Analysis

A deep learning project that classifies tweets into four sentiment categories using a Keras Embedding + RNN model.

## Overview

The model reads a short piece of text (a tweet about a game, brand, or product) and predicts one of four classes:

| Label | Class      | Meaning                                  |
|-------|------------|------------------------------------------|
| 0     | Negative   | Negative sentiment toward the entity     |
| 1     | Positive   | Positive sentiment toward the entity     |
| 2     | Neutral    | No clear sentiment                       |
| 3     | Irrelevant | Tweet is unrelated to the entity         |

## Dataset

- **Source:** Twitter Entity Sentiment Analysis (`twitter_training.csv`, available on Kaggle)
- **Columns used:** `sentiments`, `reviews` (tweet text)
- **Size after cleaning:** 22,486 rows (duplicates and missing values removed)

Class distribution after cleaning:

| Class      | Count |
|------------|-------|
| Positive   | 6,765 |
| Negative   | 5,804 |
| Neutral    | 5,793 |
| Irrelevant | 4,124 |

## Pipeline

1. **Load & clean data**: drop unused ID/entity columns, rename columns, remove duplicates and nulls.
2. **Text preprocessing**: lowercase, remove punctuation, remove English stopwords (NLTK), strip digits and extra spaces.
3. **Tokenization**: Keras `Tokenizer` (vocabulary size: 17,406).
4. **Padding**: sequences padded to a fixed length of 163 (the longest tweet).
5. **Label encoding**: `Negative=0, Positive=1, Neutral=2, Irrelevant=3`.
6. **Train/test split**: 80% / 20%, `random_state=42`.
7. **Model**:
   - `Embedding(17406, 128)`
   - `SimpleRNN(128, activation="relu")`
   - `Dense(4, activation="softmax")`
8. **Training**: Adam optimizer, sparse categorical cross-entropy, 5 epochs, batch size 64.
9. **Inference**: a `pred(text)` helper preprocesses, tokenizes, pads, and returns the predicted class.

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- nltk
- tensorflow / keras

Install:

```bash
pip install pandas numpy scikit-learn nltk tensorflow
```

## How to Run

1. Download `twitter_training.csv` and place it in your working directory (or in `/content/` on Google Colab).
2. Open `twitter_sentiment_analysis.ipynb` in Jupyter or Google Colab (a GPU runtime is optional).
3. Run all cells from top to bottom.
4. Use the `pred()` function at the end to test your own text:

```python
pred("I love playing this game, the graphics are amazing!")
```

## Current Status & Known Limitations

- The baseline SimpleRNN currently reaches only about **29% validation accuracy**, close to always predicting the majority class. It is not learning meaningful patterns yet.
- The `classes` list used in `pred()` must match the training label map (`Negative, Positive, Neutral, Irrelevant`), otherwise predictions are mislabeled.
- The model is trained on tweets about games and brands, so it will not generalize well to other domains such as movie reviews.
- Stopword removal also removes negations ("not", "no"), which can change sentiment.

## Planned Improvements

- Use `mask_zero=True` in the Embedding layer and `padding="pre"` so the RNN is not affected by trailing zeros.
- Replace `SimpleRNN` with `Bidirectional(LSTM)` or `GRU`, with dropout.
- Reduce `maxlen` (for example to the 95th percentile of tweet length).
- Keep negation words during preprocessing.
- Use a stratified split, a separate validation set, and `EarlyStopping`.
- Add evaluation: confusion matrix, precision, recall, and F1 per class.

## Project Structure

```
twitter-sentiment-analysis/
├── twitter_sentiment_analysis.ipynb  # Main notebook (preprocessing, training, inference)
├── twitter_training.csv              # Dataset (not included, download separately)
└── README.md
```

## Author

Your Name: NLP Class Project
