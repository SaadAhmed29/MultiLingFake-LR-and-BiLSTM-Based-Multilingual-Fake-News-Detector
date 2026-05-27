# Multilingual Fake News Detection Framework

A modular, end-to-end pipeline for detecting fake news across **English**, **Urdu**, and **Roman Urdu** using traditional ML and deep learning. The project covers dataset acquisition, preprocessing, contextual embeddings, classification, and adversarial robustness testing.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Acquisition & Preparation](#dataset-acquisition--preparation)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Adversarial Attacks](#adversarial-attacks)
- [Results Summary](#results-summary)

---

## Project Overview

This project builds a **multilingual fake news classifier** from scratch — no pretrained transformer models are used. All embeddings are learned directly from the dataset.

---

## Dataset Acquisition & Preparation

This project uses three publicly available datasets that are merged into one combined CSV. You need to download each one separately and combine them before running the pipeline.

### Source Datasets

#### 1. English Fake News Dataset (Kaggle)

- **Source:** [Fake and Real News Dataset — Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Files:** `Fake.csv` and `True.csv`
- **License:** Public domain / open use

Download steps:
1. Go to the Kaggle link above (requires a free Kaggle account)
2. Click **Download** to get the ZIP file
3. Extract — you'll get `Fake.csv` and `True.csv`

#### 2. Urdu Fake News Dataset

- **Source:** [Ax-to-Grind Urdu Fake News Dataset](https://github.com/MaazAmjad/Datasets-for-Urdu-news) or search Kaggle for "Urdu fake news dataset"
- **Alternative:** [Urdu Fake News — Kaggle](https://www.kaggle.com/datasets/muhammadabubakar336/urdu-fake-news)
- The dataset should contain Urdu script (`نستعلیق`) news articles with `fake`/`real` labels.

#### 3. Roman Urdu Fake News Dataset (RUFND)

- **Source:** [RUFND — Roman Urdu Fake News Dataset on Kaggle](https://www.kaggle.com/datasets/muhammadabubakar336/roman-urdu-fake-news-detection)
- Roman Urdu is Urdu written in Latin script (e.g., *"yeh khabar bilkul jhoot hai"*)

---

### Combining the Datasets

After downloading, use the script below to standardize and merge all three into `combined_fake_news_dataset.csv`. Save it as `data/prepare_dataset.py` and run it once:

```python
# data/prepare_dataset.py
import pandas as pd

OUTPUT_PATH = "combined_fake_news_dataset.csv"

# ── 1. English ────────────────────────────────────────────────────────────────
fake_en = pd.read_csv("Fake.csv")[["text"]].rename(columns={"text": "news"})
true_en = pd.read_csv("True.csv")[["text"]].rename(columns={"text": "news"})

fake_en["label"]    = "fake"
true_en["label"]    = "true"
fake_en["language"] = "english"
true_en["language"] = "english"

df_english = pd.concat([fake_en, true_en], ignore_index=True)

# ── 2. Urdu ───────────────────────────────────────────────────────────────────
# Adjust column names to match your downloaded file
df_urdu = pd.read_csv("urdu_fake_news.csv")
df_urdu = df_urdu.rename(columns={"content": "news", "Label": "label"})
df_urdu["label"]    = df_urdu["label"].str.strip().str.lower().replace({"real": "true"})
df_urdu["language"] = "urdu"
df_urdu = df_urdu[["news", "label", "language"]]

# ── 3. Roman Urdu ─────────────────────────────────────────────────────────────
# Adjust column names to match your downloaded file
df_roman = pd.read_csv("roman_urdu_fake_news.csv")
df_roman = df_roman.rename(columns={"News": "news", "Label": "label"})
df_roman["label"]    = df_roman["label"].str.strip().str.lower().replace({"real": "true"})
df_roman["language"] = "roman_urdu"
df_roman = df_roman[["news", "label", "language"]]

# ── Merge & save ──────────────────────────────────────────────────────────────
combined = pd.concat([df_english, df_urdu, df_roman], ignore_index=True)
combined.to_csv(OUTPUT_PATH, index=False)

print(f"Combined dataset saved: {OUTPUT_PATH}")
print(f"Total rows  : {len(combined)}")
print(combined["language"].value_counts())
print(combined["label"].value_counts())
```

> **Note:** Column names vary across Kaggle dataset versions. Open the CSVs first and adjust the `rename()` calls to match the actual column names in your downloaded files.

Run it with:
```bash
# from inside the data/ folder after placing all CSVs there
cd data
python prepare_dataset.py
```

---

### Expected Final Format

The pipeline expects `data/combined_fake_news_dataset.csv` with exactly these three columns:

| Column | Type | Values |
|--------|------|--------|
| `news` | string | Raw news article text |
| `label` | string | `fake` or `true` (lowercase) |
| `language` | string | `english`, `urdu`, or `roman_urdu` |

Example rows:

| news | label | language |
|------|-------|----------|
| The president signed the new bill into law... | true | english |
| حکومت نے نئی پالیسی کا اعلان کر دیا... | fake | urdu |
| yeh khabar bilkul jhoot hai, koi yaqeen na kare | fake | roman_urdu |

---

## Installation

Requires **Python 3.10+**.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/nlp_fake_news.git
cd nlp_fake_news

# 2. Create and activate a virtual environment (recommended)
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Place your combined dataset
# Copy combined_fake_news_dataset.csv into the data/ folder

# 5. Run the pipeline
python main.py
```

---

## Model Architecture

### Hybrid Contextual Embedding Model

```
Word Input (seq_len=150)          Char Input (seq_len=7500)
        │                                   │
  Word Embedding (128d)           Char Embedding (64d)
        │                                   │
Multi-Head Self-Attention          Conv1D (128 filters)
    (4 heads, key_dim=64)                   │
        │                          GlobalMaxPooling1D
  Residual Add + LayerNorm                  │
        │                                   │
GlobalAveragePooling1D                      │
        │                                   │
        └──────────── Concat ───────────────┘
                          │
                    Dense(256, tanh)      ← contextual_embedding layer
                          │
                      Dropout(0.3)
                          │
                    Dense(1, sigmoid)
```

### BiLSTM + Attention Classifier

The deep learning classifier extends the embedding architecture by replacing the self-attention word branch with a **Bidirectional LSTM**:

```
Word Input → Word Embedding → BiLSTM(32 units) → Multi-Head Attention
                                                          │
Char Input → Char Embedding → Conv1D(64) → Conv1D(128) → GlobalMaxPool
                                                          │
                              Concat → Dense(256) → Dropout(0.4)
                                     → Dense(64)  → Dropout(0.3)
                                     → Dense(1, sigmoid)
```

---

## Adversarial Attacks

Five attack functions are applied independently to 120 correctly-classified test samples, and accuracy drop is measured for each.

| Attack | Rate | Description |
|--------|------|-------------|
| **Typo Injection** | 15% of chars | Replaces characters with QWERTY keyboard-adjacent substitutions |
| **Synonym Replacement** | 20% of words | Swaps words with domain-specific synonyms (e.g., *government → authorities*) |
| **Word Shuffling** | 20% of words | Randomly swaps adjacent word pairs |
| **Emoji Insertion** | 10% of gaps | Inserts random noise emojis between words |
| **Punctuation Manip.** | 15% of spaces | Injects extra punctuation characters at word boundaries |

---

## Results Summary


| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 93.66% | 93.67% | 93.66% | 93.66% |
| BiLSTM + Attention | 97.24% | 97.24% | 97.24% | 97.24% |

| Attack | Accuracy After Attack | Drop |
|--------|-----------------------|------|
| Original (no attack) | 1.0 | 0.0 |
| Typo Injection | 0.725 | 0.2750 |
| Synonym Replacement | 1.0 | 0.0 |
| Word Shuffling | 1.0 | 0.0 |
| Emoji Insertion | 1.0 | 0.0 |
| Punctuation Manip. | 1.0 | 0.0 |
