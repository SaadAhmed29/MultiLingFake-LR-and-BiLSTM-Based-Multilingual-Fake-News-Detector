"""
main.py — Orchestrates the full Multilingual Fake News Detection pipeline.

Run with:
    python main.py

"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from data_loader  import run_task1
from preprocessor import run_task2
from embeddings   import run_task3
from classifier   import run_task4
from adversarial  import run_task5

# ── Reproducibility ───────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
tf.random.set_seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)
random.seed(config.RANDOM_STATE)

# ── Directory setup ───────────────────────────────────────────────────────────
os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR,  exist_ok=True)
os.makedirs(config.DATA_DIR,    exist_ok=True)


def main():
    print("\n" + "=" * 60)
    print("  NLP Assignment 03 — Multilingual Fake News Detection")
    print("=" * 60)

    # ── Task 1 ────────────────────────────────────────────────────
    print("\n\n===== TASK 1: Dataset Loading, Cleaning & Statistics =====\n")
    df = run_task1()

    # ── Task 2 ────────────────────────────────────────────────────
    print("\n\n===== TASK 2: NLP Preprocessing Pipeline =====\n")
    df = run_task2(df)

    # ── Task 3 ────────────────────────────────────────────────────
    print("\n\n===== TASK 3: Contextual Embedding Architecture =====\n")
    (
        word_tok, char_tok,
        word_vocab_size, char_vocab_size,
        X_word, X_char, y, le, hybrid_model,
    ) = run_task3(df)

    # ── Task 4 ────────────────────────────────────────────────────
    print("\n\n===== TASK 4: Classification Models =====\n")
    (
        lr_model, dl_model,
        X_word_train, X_word_test,
        X_char_train, X_char_test,
        y_train, y_test,
        y_pred_dl, y_prob_dl,
        lr_metrics, dl_metrics,
        label_map,
    ) = run_task4(df, X_word, X_char, y, word_vocab_size, char_vocab_size, le)

    # ── Task 5 ────────────────────────────────────────────────────
    print("\n\n===== TASK 5: Adversarial Robustness Analysis =====\n")
    robustness_results, failed_df = run_task5(
        df, y, y_test, y_pred_dl,
        dl_model, word_tok, char_tok, label_map,
    )

    # ── Final Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("      FINAL SUMMARY — ALL TASKS")
    print("=" * 60)

    print(f"\n[Task 1] Dataset")
    print(f"  Total samples   : {len(df)}")
    print(f"  Fake samples    : {len(df[df['label'] == 'fake'])}")
    print(f"  True samples    : {len(df[df['label'] == 'true'])}")

    print(f"\n[Task 2] Preprocessing Pipeline")
    print(f"  Features: URL removal, emoji handling, hashtag extraction,")
    print(f"            mention removal, Roman Urdu normalization,")
    print(f"            punctuation handling, tokenization, stopword removal")

    print(f"\n[Task 3] Hybrid Contextual Embeddings")
    print(f"  Architecture  : Word Emb + Char-CNN + Multi-Head Self-Attention")
    print(f"  Embed dim     : {config.FINAL_EMBED_DIM} (≤512 ✅)")
    print(f"  Multilingual  : English + Urdu + Roman Urdu ✅")

    print(f"\n[Task 4] Classification")
    print(
        f"  Logistic Regression : "
        f"Acc={lr_metrics['accuracy']*100:.2f}%, F1={lr_metrics['f1']:.4f}"
    )
    print(
        f"  BiLSTM+Attention    : "
        f"Acc={dl_metrics['accuracy']*100:.2f}%, F1={dl_metrics['f1']:.4f}"
    )

    print(f"\n[Task 5] Adversarial Robustness")
    for name, v in robustness_results.items():
        print(f"  {name:25s}: Acc={v['accuracy']*100:.2f}% (drop={v['drop']:+.4f})")
    print(f"\n  Total failed predictions: {len(failed_df)}")

    print("\n" + "=" * 60)
    print("  Pipeline complete! All outputs saved to:", config.OUTPUTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
