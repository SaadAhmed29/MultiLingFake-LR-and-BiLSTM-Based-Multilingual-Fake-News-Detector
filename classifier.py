"""
classifier.py — Task 4: Classification Models (Traditional + Deep Learning).

Responsibilities:
  - Train/test split
  - Logistic Regression with TF-IDF features
  - BiLSTM + Multi-Head Attention deep learning model
  - Shared evaluation metrics / confusion matrix plot
  - Model comparison table
  - Sample prediction display
"""

import os
import random
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Add, Bidirectional, Concatenate, Conv1D, Dense, Dropout,
    Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D,
    Input, LayerNormalization, LSTM, MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import config
from embeddings import prepare_word_sequences, prepare_char_sequences


# ── Train / test split ────────────────────────────────────────────────────────

def split_data(X_word, X_char, y, texts_series: pd.Series):
    """
    Stratified 80/20 split.

    Returns:
        X_word_train, X_word_test,
        X_char_train, X_char_test,
        y_train, y_test,
        X_text_train, X_text_test
    """
    (
        X_word_train, X_word_test,
        X_char_train, X_char_test,
        y_train,      y_test,
        X_text_train, X_text_test,
    ) = train_test_split(
        X_word, X_char, y, texts_series,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples : {len(y_train)}")
    print(f"Test samples     : {len(y_test)}")
    print(f"Train label dist : {Counter(y_train)}")
    print(f"Test  label dist : {Counter(y_test)}")

    return (
        X_word_train, X_word_test,
        X_char_train, X_char_test,
        y_train,      y_test,
        X_text_train, X_text_test,
    )


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_model(
    y_true,
    y_pred,
    model_name: str,
    label_names: list[str] | None = None,
    save: bool = True,
) -> dict:
    """Print metrics + confusion matrix; optionally save the plot."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    names = label_names or ["Class 0", "Class 1"]

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=names, zero_division=0))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=names, yticklabels=names)
    ax.set_title(f"{model_name} — Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    if save:
        fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        out   = os.path.join(config.OUTPUTS_DIR, fname)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved → {out}")
    plt.show()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ── 4.2 Logistic Regression ───────────────────────────────────────────────────

def train_logistic_regression(X_text_train, X_text_test, y_train, y_test,
                               label_names=None):
    """TF-IDF + Logistic Regression pipeline."""
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=500,
            solver="saga",
            random_state=config.RANDOM_STATE,
        )),
    ])

    print("Training Logistic Regression...")
    t0 = time.time()
    lr_pipeline.fit(X_text_train, y_train)
    print(f"LR training time: {time.time() - t0:.2f}s")

    y_pred = lr_pipeline.predict(X_text_test)
    metrics = evaluate_model(y_test, y_pred, "Logistic Regression",
                             label_names=label_names)
    return lr_pipeline, y_pred, metrics


# ── 4.3 BiLSTM + Attention ────────────────────────────────────────────────────

def build_bilstm_attention_model(
    word_vocab_size: int,
    char_vocab_size: int,
    word_embed_dim: int = config.WORD_EMBED_DIM,
    char_embed_dim: int = config.CHAR_EMBED_DIM,
    max_seq_len: int    = config.MAX_SEQ_LEN,
    max_char_len: int   = config.MAX_CHAR_LEN,
) -> Model:
    """BiLSTM + Multi-Head Attention classifier with dual word/char inputs."""
    flat_char_len = max_seq_len * max_char_len

    # Word branch
    word_input = Input(shape=(max_seq_len,), name="word_input")
    word_emb   = Embedding(word_vocab_size, word_embed_dim,
                           mask_zero=False, name="word_embedding")(word_input)
    bilstm_out = Bidirectional(
        LSTM(32, return_sequences=True, dropout=0.2), name="bilstm"
    )(word_emb)
    attn = MultiHeadAttention(num_heads=4, key_dim=32, name="attention")(
        bilstm_out, bilstm_out
    )
    attn     = Add()([bilstm_out, attn])
    attn     = LayerNormalization()(attn)
    word_pool = GlobalAveragePooling1D(name="word_pool")(attn)

    # Char branch
    char_input = Input(shape=(flat_char_len,), name="char_input")
    char_emb   = Embedding(char_vocab_size, char_embed_dim, name="char_embedding")(char_input)
    char_conv  = Conv1D(64,  kernel_size=3, activation="relu", name="char_conv1")(char_emb)
    char_conv  = Conv1D(128, kernel_size=3, activation="relu", name="char_conv2")(char_conv)
    char_pool  = GlobalMaxPooling1D(name="char_pool")(char_conv)

    # Merge + classify
    merged = Concatenate(name="merge")([word_pool, char_pool])
    x      = Dense(256, activation="relu",  name="dense1")(merged)
    x      = Dropout(0.4, name="dropout1")(x)
    x      = Dense(64,  activation="relu",  name="dense2")(x)
    x      = Dropout(0.3, name="dropout2")(x)
    output = Dense(1,   activation="sigmoid", name="output")(x)

    model = Model(inputs=[word_input, char_input], outputs=output,
                  name="BiLSTM_Attention_Classifier")
    model.compile(
        optimizer=Adam(learning_rate=config.LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_bilstm(
    model: Model,
    X_word_train, X_char_train, y_train,
    X_word_test,  X_char_test,  y_test,
    label_names=None,
):
    """Train the BiLSTM model, plot curves, and evaluate."""
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=config.PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    print("Training BiLSTM + Attention model...")
    t0 = time.time()
    history = model.fit(
        [X_word_train, X_char_train], y_train,
        validation_data=([X_word_test, X_char_test], y_test),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\nDL training time: {elapsed:.2f}s ({elapsed/60:.1f} min)")

    _plot_training_curves(history)

    y_prob = model.predict([X_word_test, X_char_test], verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate_model(y_test, y_pred, "BiLSTM Attention",
                             label_names=label_names)
    return history, y_pred, y_prob, metrics


def _plot_training_curves(history) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history.history["accuracy"],     label="Train", color="#2ecc71")
    axes[0].plot(history.history["val_accuracy"], label="Val",   color="#e74c3c")
    axes[0].set_title("BiLSTM+Attention — Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train", color="#2ecc71")
    axes[1].plot(history.history["val_loss"], label="Val",   color="#e74c3c")
    axes[1].set_title("BiLSTM+Attention — Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(config.OUTPUTS_DIR, "training_curves_bilstm.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Training curves saved → {out}")
    plt.show()


# ── 4.4 Model comparison ──────────────────────────────────────────────────────

def print_comparison(lr_metrics: dict, dl_metrics: dict) -> pd.DataFrame:
    comp = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Logistic Regression": [
            f"{lr_metrics['accuracy']:.4f}", f"{lr_metrics['precision']:.4f}",
            f"{lr_metrics['recall']:.4f}",   f"{lr_metrics['f1']:.4f}",
        ],
        "BiLSTM + Attention": [
            f"{dl_metrics['accuracy']:.4f}", f"{dl_metrics['precision']:.4f}",
            f"{dl_metrics['recall']:.4f}",   f"{dl_metrics['f1']:.4f}",
        ],
    })
    print("=" * 55)
    print("    MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 55)
    print(comp.to_string(index=False))
    print("=" * 55)

    for name, m in [("LR", lr_metrics), ("BiLSTM", dl_metrics)]:
        status = "✅" if m["accuracy"] >= 0.75 else "⚠️ BELOW 75%"
        print(f"{name} Accuracy: {m['accuracy']*100:.2f}% {status}")

    return comp


def show_sample_predictions(
    df: pd.DataFrame,
    X_word_test, X_char_test,
    y_test, y_pred_dl, y_prob_dl,
    dl_model: Model,
    label_map: dict,
    n: int = 10,
) -> None:
    sample_idx = random.sample(range(len(y_test)), n)
    print("\n=== SAMPLE PREDICTIONS (BiLSTM model) ===")
    for idx in sample_idx:
        actual    = label_map[y_test[idx]]
        predicted = label_map[y_pred_dl[idx]]
        conf      = y_prob_dl[idx]
        correct   = "✅" if actual == predicted else "❌"
        all_idx   = list(range(len(df)))
        _, test_idx = train_test_split(
            all_idx, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, stratify=df["label"].map({"fake": 0, "true": 1})
        )
        text = df.iloc[test_idx[idx]]["news"][:100]
        print(
            f"  {correct} Actual: {actual:5s} | Pred: {predicted:5s} | "
            f"Conf: {conf:.3f} | Text: {text}..."
        )


# ── Convenience entry-point ───────────────────────────────────────────────────

def run_task4(
    df: pd.DataFrame,
    X_word, X_char, y,
    word_vocab_size: int,
    char_vocab_size: int,
    le,
):
    """Full Task 4 pipeline. Returns trained models and metrics."""
    label_names = le.classes_.tolist()
    label_map   = {i: name for i, name in enumerate(le.classes_)}

    (
        X_word_train, X_word_test,
        X_char_train, X_char_test,
        y_train, y_test,
        X_text_train, X_text_test,
    ) = split_data(X_word, X_char, y, df["processed_news"])

    # Logistic Regression
    lr_model, y_pred_lr, lr_metrics = train_logistic_regression(
        X_text_train, X_text_test, y_train, y_test, label_names=label_names
    )

    # BiLSTM + Attention
    dl_model = build_bilstm_attention_model(word_vocab_size, char_vocab_size)
    dl_model.summary()
    history, y_pred_dl, y_prob_dl, dl_metrics = train_bilstm(
        dl_model,
        X_word_train, X_char_train, y_train,
        X_word_test,  X_char_test,  y_test,
        label_names=label_names,
    )

    comp_df = print_comparison(lr_metrics, dl_metrics)
    show_sample_predictions(
        df, X_word_test, X_char_test,
        y_test, y_pred_dl, y_prob_dl, dl_model, label_map
    )

    # Save model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    dl_model.save(config.MODEL_SAVE_PATH)
    print(f"BiLSTM model saved → {config.MODEL_SAVE_PATH}")

    # Save comparison CSV
    comp_path = os.path.join(config.OUTPUTS_DIR, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"Model comparison saved → {comp_path}")

    return (
        lr_model, dl_model,
        X_word_train, X_word_test,
        X_char_train, X_char_test,
        y_train, y_test,
        y_pred_dl, y_prob_dl,
        lr_metrics, dl_metrics,
        label_map,
    )
