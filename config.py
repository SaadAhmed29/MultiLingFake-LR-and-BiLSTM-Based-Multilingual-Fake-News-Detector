"""
config.py — Central configuration and hyperparameters for the project.
Modify values here to affect the entire pipeline.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
OUTPUTS_DIR = "outputs"
MODELS_DIR  = "models"

DATASET_PATH         = os.path.join(DATA_DIR, "combined_fake_news_dataset.csv")
CLEANED_DATASET_PATH = os.path.join(DATA_DIR, "cleaned_dataset.csv")

MODEL_SAVE_PATH      = os.path.join(MODELS_DIR, "bilstm_attention_model.h5")

# ── Dataset ───────────────────────────────────────────────────────────────────
VALID_LABELS    = ["fake", "true"]
VALID_LANGUAGES = ["english", "urdu", "roman_urdu"]
BALANCE_THRESHOLD = 10          # % imbalance before resampling kicks in
RANDOM_STATE      = 42

# ── Train / Test split ────────────────────────────────────────────────────────
TEST_SIZE = 0.2

# ── Tokenizer / Sequence ──────────────────────────────────────────────────────
MAX_VOCAB      = 30_000   # word vocabulary cap
MAX_CHAR_VOCAB = 500      # character vocabulary cap
MAX_SEQ_LEN    = 150      # max word tokens per sample
MAX_CHAR_LEN   = 50       # max chars per word (for char embeddings)

# ── Embedding dimensions ──────────────────────────────────────────────────────
WORD_EMBED_DIM  = 128
CHAR_EMBED_DIM  = 64
ATTN_HEADS      = 4
ATTN_KEY_DIM    = 64
FINAL_EMBED_DIM = 256     # ≤ 512

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-3         # Adam learning rate
PATIENCE   = 3            # EarlyStopping patience

# ── TF-IDF (Logistic Regression) ─────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5_000
TFIDF_NGRAM_RANGE  = (1, 1)
TFIDF_MIN_DF       = 3
TFIDF_MAX_DF       = 0.9

# ── Adversarial ───────────────────────────────────────────────────────────────
N_ADVERSARIAL_SAMPLES = 120
TYPO_RATE        = 0.15
SYNONYM_RATE     = 0.20
SHUFFLE_RATE     = 0.20
EMOJI_RATE       = 0.10
PUNCT_RATE       = 0.15
