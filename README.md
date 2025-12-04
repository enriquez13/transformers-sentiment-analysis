# 📘 README.md – Transformers Sentiment Analysis

## 🧠 Project: Sentiment Analysis with DistilBERT

This project implements an end-to-end sentiment analysis pipeline using HuggingFace Transformers.
It includes four stages:

Dataset cleaning

Tokenization

Model training (DistilBERT)

Prediction on new text

All steps are implemented in Jupyter Notebooks for clarity and reproducibility.

# 📂 Project Structure
```bash
transformers-sentiment-analysis/
│
├── data/
│   ├── raw/
│   │   └── reviews.csv
│   └── processed/
│       └── clean_reviews_v2.csv
│
├── models/
│   └── distilbert-sentiment/
│       ├── config.json
│       ├── model.safetensors
│       ├── vocab.txt
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       ├── training_args.bin
│       └── checkpoint  (auto-generated)
│
├── notebooks/
│   ├── 01_load_and_clean.ipynb
│   ├── 02_tokenization.ipynb
│   ├── 03_train_model.ipynb
│   └── 04_predict.ipynb
│
├── tokens.pt
├── README.md
└──requirements.txt
```

## 🛠️ Requirements

Install dependencies:

Or manually install:
pip install transformers datasets torch pandas numpy jupyter


## 📓 Notebook Workflow
**01_load_and_clean.ipynb**

Loads the raw CSV (data/raw/reviews.csv)

Normalizes text (lowercase, remove symbols, clean whitespace)

Saves cleaned dataset to:
✔ data/processed/clean_reviews_v2.csv

**02_tokenization.ipynb**

Loads cleaned dataset

Loads the tokenizer (distilbert-base-uncased)

Tokenizes all reviews

Saves tokenized tensors to:
tokens.pt

**03_train_model.ipynb**

Maps labels → integers (negative=0, neutral=1, positive=2)

Converts to a HuggingFace Dataset

Tokenizes dynamically during training

Trains DistilBERT for sequence classification

Saves trained model to:
models/distilbert-sentiment/

The training results achieved:
| Epoch | Train Loss | Val Loss |
| ----- | ---------- | -------- |
| 1     | 0.2546     | 0.1662   |
| 2     | 0.0725     | 0.0560   |
| 3     | 0.0139     | 0.0511   |

Excellent model performance.

**04_predict.ipynb**

- Loads trained model

- Performs sentiment inference on new text

Outputs softmax probabilities for:
- negative
- neutral
- positive

## 🧠 Example Predictions:

**Input:** "This product is terrible and I hate it."
**Output:** negative (0.99)

**Input:** "It's okay, nothing special."
**Output:** neutral (0.78)

**Input:** "I love it!"
**Output:** positive (0.98)

**Input:** "This is the worst thing I've ever bought."
**Output:** negative (0.98)

**Input:** "It works as expected."
**Output:** positive (0.77)

## 🧪 How to Use the Trained Model in Any Script
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, pipeline

model_dir = "models/distilbert-sentiment"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

text = "This product is terrible."
print(classifier(text))

## 📝 Notes
The model is fully portable and can be loaded in Python scripts, notebooks, or web backends.

Checkpoints inside models/distilbert-sentiment/ can be deleted if not needed.

Training was optimized for CPU-only use.






