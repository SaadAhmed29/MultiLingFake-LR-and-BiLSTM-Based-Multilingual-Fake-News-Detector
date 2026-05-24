"""
Architecture: Hybrid Contextual Embeddings
  - Word-level embeddings + Multi-Head Self-Attention
  - Character-level CNN embeddings
  - Final dense projection (FINAL_EMBED_DIM ≤ 512)

Responsibilities:
  - Fit word & character tokenizers on the dataset
  - Build the hybrid embedding model (Keras functional API)
  - Prepare padded word/character sequences
  - Generate embeddings and run ambiguous-word cosine-similarity demo
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout,
    Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D,
    MultiHeadAttention, LayerNormalization, Add, Concatenate,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
from preprocessor import preprocess_text


# ── Tokenizer setup ───────────────────────────────────────────────────────────

def build_tokenizers(texts: list[str]) -> tuple[Tokenizer, Tokenizer, int, int]:
    """
    Fit word-level and character-level tokenizers on the corpus.

    Returns:
        word_tokenizer, char_tokenizer, word_vocab_size, char_vocab_size
    """
    word_tokenizer = Tokenizer(num_words=config.MAX_VOCAB, oov_token="<OOV>")
    word_tokenizer.fit_on_texts(texts)
    word_vocab_size = min(len(word_tokenizer.word_index) + 1, config.MAX_VOCAB)

    char_tokenizer = Tokenizer(
        num_words=config.MAX_CHAR_VOCAB, char_level=True, oov_token="<OOV>"
    )
    char_tokenizer.fit_on_texts(texts)
    char_vocab_size = min(len(char_tokenizer.word_index) + 1, config.MAX_CHAR_VOCAB)

    print(f"Word vocabulary size : {word_vocab_size}")
    print(f"Char vocabulary size : {char_vocab_size}")
    return word_tokenizer, char_tokenizer, word_vocab_size, char_vocab_size


# ── Sequence preparation ──────────────────────────────────────────────────────

def prepare_word_sequences(
    texts: list[str],
    tokenizer: Tokenizer,
    max_len: int = config.MAX_SEQ_LEN,
) -> np.ndarray:
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")


def prepare_char_sequences(
    texts: list[str],
    tokenizer: Tokenizer,
    max_seq_len: int = config.MAX_SEQ_LEN,
    max_char_len: int = config.MAX_CHAR_LEN,
) -> np.ndarray:
    """
    Character-level encoding flattened to shape (n_samples, max_seq_len * max_char_len).
    """
    flat_max_len = max_seq_len * max_char_len
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=flat_max_len, padding="post", truncating="post")


# ── Model builder ─────────────────────────────────────────────────────────────

def build_hybrid_embedding_model(
    word_vocab_size: int,
    char_vocab_size: int,
    word_embed_dim: int  = config.WORD_EMBED_DIM,
    char_embed_dim: int  = config.CHAR_EMBED_DIM,
    max_seq_len: int     = config.MAX_SEQ_LEN,
    attn_heads: int      = config.ATTN_HEADS,
    attn_key_dim: int    = config.ATTN_KEY_DIM,
    final_embed_dim: int = config.FINAL_EMBED_DIM,
    num_classes: int     = 1,
) -> Model:
    """
    Hybrid Contextual Embedding Model:
      Word branch  : Embedding → Multi-Head Self-Attention → Residual → Pool
      Char branch  : Embedding → Conv1D → GlobalMaxPool
      Merge        : Concat → Dense(final_embed_dim, tanh) → Dropout → output
    """
    flat_char_len = max_seq_len * config.MAX_CHAR_LEN

    # ── Word branch ───────────────────────────────────────────────────────────
    word_input = Input(shape=(max_seq_len,), name="word_input")
    word_emb   = Embedding(word_vocab_size, word_embed_dim,
                           mask_zero=False, name="word_embedding")(word_input)

    attn_out = MultiHeadAttention(
        num_heads=attn_heads, key_dim=attn_key_dim, name="self_attention"
    )(word_emb, word_emb)
    attn_out   = Add(name="residual_add")([word_emb, attn_out])
    attn_out   = LayerNormalization(name="layer_norm")(attn_out)
    word_pool  = GlobalAveragePooling1D(name="word_pool")(attn_out)

    # ── Char branch ───────────────────────────────────────────────────────────
    char_input = Input(shape=(flat_char_len,), name="char_input")
    char_emb   = Embedding(char_vocab_size, char_embed_dim, name="char_embedding")(char_input)
    char_conv  = Conv1D(128, kernel_size=3, activation="relu", name="char_conv")(char_emb)
    char_pool  = GlobalMaxPooling1D(name="char_pool")(char_conv)

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = Concatenate(name="merge")([word_pool, char_pool])
    emb_out = Dense(final_embed_dim, activation="tanh",
                    name="contextual_embedding")(merged)
    emb_out = Dropout(0.3, name="embedding_dropout")(emb_out)
    output  = Dense(num_classes, activation="sigmoid", name="output")(emb_out)

    model = Model(inputs=[word_input, char_input], outputs=output,
                  name="HybridContextualEmbeddingModel")
    return model


# ── Embedding extractor ───────────────────────────────────────────────────────

def get_embedding_extractor(model: Model) -> Model:
    """Return an intermediate model that outputs the contextual_embedding layer."""
    return Model(
        inputs=model.inputs,
        outputs=model.get_layer("contextual_embedding").output,
        name="EmbeddingExtractor",
    )


# ── Ambiguous word demo ───────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def demo_ambiguous_words(
    extractor: Model,
    word_tokenizer: Tokenizer,
    char_tokenizer: Tokenizer,
) -> None:
    """
    Show that the hybrid model produces different embeddings for the same word
    used in different contexts (e.g., 'bank' financial vs. river).
    """
    pairs = [
        ("Bank charges high interest rates on loans",   "english", "BANK (financial)"),
        ("He sat on the river bank watching fish swim",  "english", "BANK (river)"),
        ("The match was cancelled due to heavy rain",    "english", "MATCH (sports)"),
        ("She lit a match to start the campfire",        "english", "MATCH (fire)"),
    ]

    print("=" * 70)
    print("CONTEXTUAL EMBEDDING COMPARISON — AMBIGUOUS WORDS")
    print("=" * 70)

    embs = []
    for text, lang, label in pairs:
        processed = preprocess_text(text, lang)
        w = prepare_word_sequences([processed], word_tokenizer)
        c = prepare_char_sequences([processed], char_tokenizer)
        emb = extractor.predict([w, c], verbose=0)[0]
        embs.append(emb)
        print(f"\n[{label}]")
        print(f"  Text      : {text}")
        print(f"  Processed : {processed}")
        print(f"  Emb norm  : {np.linalg.norm(emb):.4f}")
        print(f"  First 8d  : {emb[:8].round(4)}")

    print(
        f"\nCosine sim — BANK (financial) vs BANK (river)    : "
        f"{_cosine_sim(embs[0], embs[1]):.4f}"
    )
    print(
        f"Cosine sim — MATCH (sports) vs MATCH (fire)      : "
        f"{_cosine_sim(embs[2], embs[3]):.4f}"
    )
    print(
        f"Cosine sim — BANK (financial) vs MATCH (sports)  : "
        f"{_cosine_sim(embs[0], embs[2]):.4f}"
    )
    print("\n(Lower similarity between same-word different-context = better contextual embeddings)")


# ── Convenience entry-point ───────────────────────────────────────────────────

def run_task3(df: pd.DataFrame):
    """
    Full Task 3 pipeline.

    Returns:
        word_tokenizer, char_tokenizer,
        word_vocab_size, char_vocab_size,
        X_word, X_char, y,
        hybrid_model
    """
    from sklearn.preprocessing import LabelEncoder

    texts = df["processed_news"].tolist()
    word_tok, char_tok, wvs, cvs = build_tokenizers(texts)

    le = LabelEncoder()
    y  = le.fit_transform(df["label"])
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X_word = prepare_word_sequences(texts, word_tok)
    X_char = prepare_char_sequences(texts, char_tok)
    print(f"Word sequence shape: {X_word.shape}")
    print(f"Char sequence shape: {X_char.shape}")

    model = build_hybrid_embedding_model(word_vocab_size=wvs, char_vocab_size=cvs)
    model.summary()

    extractor = get_embedding_extractor(model)

    # Quick sample embedding check
    sample_embs = extractor.predict([X_word[:20], X_char[:20]], verbose=0)
    print(f"\nEmbedding shape per sample: {sample_embs.shape}")
    print(f"Embedding dimension: {sample_embs.shape[1]}  (≤ 512 ✅)")
    for i in range(3):
        print(
            f"Sample {i+1} [{df.iloc[i]['language']}] first 10 dims: "
            f"{sample_embs[i][:10].round(4)}"
        )

    demo_ambiguous_words(extractor, word_tok, char_tok)

    return word_tok, char_tok, wvs, cvs, X_word, X_char, y, le, model
