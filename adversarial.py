"""
adversarial.py — Task 5: Adversarial Robustness Analysis.

Attack types:
  1. Typo Injection        — keyboard-adjacent character substitutions
  2. Synonym Replacement   — swap words with domain synonyms
  3. Word Shuffling        — randomly swap adjacent word pairs
  4. Emoji Insertion       — inject noise emojis between words
  5. Punctuation Manip.    — insert extra punctuation marks

Responsibilities:
  - Define and apply all 5 attacks
  - Evaluate BiLSTM accuracy drop per attack
  - Build robustness comparison table + plot
  - Display ≥10 failed prediction examples with explanations
"""

import os
import random
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import config
from embeddings import prepare_char_sequences, prepare_word_sequences
from preprocessor import preprocess_text


# ── Attack 1: Typo Injection ──────────────────────────────────────────────────

_KEYBOARD_ADJ = {
    "a": "sqzw", "b": "vghn", "c": "xdfv", "d": "srfec", "e": "wrsdt",
    "f": "drtgv", "g": "ftyhb", "h": "gyujn", "i": "ujko",  "j": "huikm",
    "k": "jiol",  "l": "kop",   "m": "njk",   "n": "bhjm",  "o": "iklp",
    "p": "ol",    "q": "wa",    "r": "edft",  "s": "awdez", "t": "rfgy",
    "u": "yhji",  "v": "cfgb",  "w": "qase",  "x": "zsdc",  "y": "tghu",
    "z": "asx",
}


def typo_injection(text: str, rate: float = config.TYPO_RATE) -> str:
    result = []
    for char in text:
        if char.lower() in _KEYBOARD_ADJ and random.random() < rate:
            rep = random.choice(_KEYBOARD_ADJ[char.lower()])
            result.append(rep if char.islower() else rep.upper())
        else:
            result.append(char)
    return "".join(result)


# ── Attack 2: Synonym Replacement ────────────────────────────────────────────

_SYNONYM_MAP = {
    "fake": ["false", "fabricated", "untrue", "bogus", "misleading"],
    "news": ["report", "story", "article", "information", "claim"],
    "said": ["stated", "claimed", "mentioned", "declared", "announced"],
    "government": ["authorities", "administration", "officials", "regime", "cabinet"],
    "people": ["citizens", "individuals", "persons", "public", "masses"],
    "true": ["real", "genuine", "factual", "authentic", "verified"],
    "country": ["nation", "state", "land", "territory", "republic"],
    "attack": ["assault", "strike", "raid", "offensive", "siege"],
    "death": ["fatality", "casualty", "loss", "demise", "killing"],
    "police": ["officers", "law enforcement", "forces", "constabulary", "authorities"],
    "president": ["leader", "head of state", "chief", "commander", "premier"],
    "money": ["funds", "cash", "currency", "capital", "wealth"],
    "pakistan": ["the country", "the nation", "the state"],
    "army": ["military", "forces", "troops", "soldiers", "defense"],
    "hospital": ["medical facility", "clinic", "health center", "infirmary"],
    "killed": ["slain", "murdered", "eliminated", "shot dead", "assassinated"],
    "arrested": ["detained", "apprehended", "taken into custody", "held"],
    "election": ["vote", "poll", "ballot", "referendum"],
    "billion": ["thousand million", "bn", "1000 million"],
    "announced": ["declared", "revealed", "disclosed", "stated", "confirmed"],
}


def synonym_replacement(text: str, rate: float = config.SYNONYM_RATE) -> str:
    words  = text.split()
    result = []
    for word in words:
        clean = word.lower().strip(string.punctuation)
        if clean in _SYNONYM_MAP and random.random() < rate:
            result.append(random.choice(_SYNONYM_MAP[clean]))
        else:
            result.append(word)
    return " ".join(result)


# ── Attack 3: Word Shuffling ──────────────────────────────────────────────────

def word_shuffling(text: str, rate: float = config.SHUFFLE_RATE) -> str:
    words = text.split()
    if len(words) < 3:
        return text
    n_swaps = max(1, int(len(words) * rate))
    for _ in range(n_swaps):
        i = random.randint(0, len(words) - 2)
        words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


# ── Attack 4: Emoji Insertion ─────────────────────────────────────────────────

_NOISE_EMOJIS = [
    "😂", "🔥", "💯", "😡", "🚨", "⚠️", "🤔", "😱", "🙏", "👀",
    "📢", "❗", "✅", "❌", "💀", "😤", "🤯", "😲", "🧐", "💬",
]


def emoji_insertion(text: str, rate: float = config.EMOJI_RATE) -> str:
    words  = text.split()
    result = []
    for word in words:
        result.append(word)
        if random.random() < rate:
            result.append(random.choice(_NOISE_EMOJIS))
    return " ".join(result)


# ── Attack 5: Punctuation Manipulation ───────────────────────────────────────

def punctuation_manipulation(text: str, rate: float = config.PUNCT_RATE) -> str:
    puncts = "!?.,;:"
    result = []
    for char in text:
        result.append(char)
        if char == " " and random.random() < rate:
            result.append(random.choice(puncts))
            result.append(" ")
    return "".join(result)


# ── Attack registry ───────────────────────────────────────────────────────────

ATTACKS: dict = {
    "Typo Injection":       typo_injection,
    "Synonym Replacement":  synonym_replacement,
    "Word Shuffling":       word_shuffling,
    "Emoji Insertion":      emoji_insertion,
    "Punctuation Manip.":   punctuation_manipulation,
}


def demo_attacks(text: str = None) -> None:
    """Print one adversarial example per attack."""
    text = text or "The government announced a new policy for economic development in Pakistan."
    print(f"Original: {text}")
    for name, fn in ATTACKS.items():
        print(f"[{name}]: {fn(text)}")


# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_texts(
    raw_texts: list[str],
    languages: list[str],
    dl_model,
    word_tokenizer,
    char_tokenizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess → encode → predict. Returns (class_array, prob_array)."""
    processed = [preprocess_text(t, l) for t, l in zip(raw_texts, languages)]
    w_seq = prepare_word_sequences(processed, word_tokenizer)
    c_seq = prepare_char_sequences(processed, char_tokenizer)
    probs = dl_model.predict([w_seq, c_seq], verbose=0).flatten()
    return (probs >= 0.5).astype(int), probs


# ── Main robustness evaluation ────────────────────────────────────────────────

def run_robustness_evaluation(
    df: pd.DataFrame,
    y: np.ndarray,
    y_test: np.ndarray,
    y_pred_dl: np.ndarray,
    dl_model,
    word_tokenizer,
    char_tokenizer,
    label_map: dict,
) -> tuple[dict, pd.DataFrame]:
    """
    Select correctly classified samples, apply each attack, measure accuracy drop.

    Returns:
        robustness_results dict,
        failed_predictions DataFrame
    """
    # Reconstruct test indices
    all_idx = list(range(len(df)))
    _, test_idx = train_test_split(
        all_idx,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    test_texts_raw = df.iloc[test_idx]["news"].tolist()
    test_langs     = df.iloc[test_idx]["language"].tolist()

    # Correct predictions
    correct_mask    = y_pred_dl == y_test
    correct_indices = np.where(correct_mask)[0]
    print(f"Total test samples    : {len(y_test)}")
    print(f"Correctly classified  : {len(correct_indices)}")

    n = min(config.N_ADVERSARIAL_SAMPLES, len(correct_indices))
    adv_idx = np.random.choice(correct_indices, n, replace=False)
    print(f"Selected {n} samples for adversarial testing.")

    adv_texts  = [test_texts_raw[i] for i in adv_idx]
    adv_langs  = [test_langs[i]     for i in adv_idx]
    adv_labels = y_test[adv_idx]

    base_preds, base_probs = predict_texts(
        adv_texts, adv_langs, dl_model, word_tokenizer, char_tokenizer
    )
    baseline_acc = accuracy_score(adv_labels, base_preds)
    print(f"Baseline accuracy on adversarial set: {baseline_acc:.4f}")

    robustness_results = {"Original": {"accuracy": baseline_acc, "drop": 0.0}}
    failed_records: list[dict] = []

    for attack_name, attack_fn in ATTACKS.items():
        attacked_texts = [attack_fn(t) for t in adv_texts]
        att_preds, att_probs = predict_texts(
            attacked_texts, adv_langs, dl_model, word_tokenizer, char_tokenizer
        )
        att_acc = accuracy_score(adv_labels, att_preds)
        drop    = baseline_acc - att_acc
        robustness_results[attack_name] = {"accuracy": att_acc, "drop": drop}

        for i in range(len(adv_texts)):
            if base_preds[i] == adv_labels[i] and att_preds[i] != adv_labels[i]:
                failed_records.append({
                    "attack":         attack_name,
                    "original_text":  adv_texts[i][:120],
                    "attacked_text":  attacked_texts[i][:120],
                    "true_label":     label_map[adv_labels[i]],
                    "original_pred":  label_map[base_preds[i]],
                    "attacked_pred":  label_map[att_preds[i]],
                    "original_conf":  round(float(base_probs[i]), 4),
                    "attacked_conf":  round(float(att_probs[i]), 4),
                    "language":       adv_langs[i],
                })

        print(f"[{attack_name}] Accuracy: {att_acc:.4f} | Drop: {drop:+.4f}")

    failed_df = pd.DataFrame(failed_records)
    print(f"\nTotal failed predictions: {len(failed_df)}")
    return robustness_results, failed_df


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_robustness_table(robustness_results: dict) -> pd.DataFrame:
    rob_df = pd.DataFrame([
        {"Attack": name, "Accuracy": v["accuracy"], "Accuracy Drop": v["drop"]}
        for name, v in robustness_results.items()
    ])
    print("=" * 55)
    print("    ADVERSARIAL ROBUSTNESS COMPARISON TABLE")
    print("=" * 55)
    print(rob_df.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 55)
    return rob_df


def plot_robustness(rob_df: pd.DataFrame, save: bool = True) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ecc71"] + ["#e74c3c"] * (len(rob_df) - 1)
    axes[0].bar(rob_df["Attack"], rob_df["Accuracy"], color=colors, edgecolor="black")
    axes[0].axhline(0.75, color="orange", linestyle="--", label="75% threshold")
    axes[0].set_title("Accuracy After Each Attack", fontweight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(); axes[0].set_ylim([0, 1.05])

    drop_colors = ["#3498db" if v >= 0 else "#27ae60" for v in rob_df["Accuracy Drop"]]
    axes[1].bar(rob_df["Attack"], rob_df["Accuracy Drop"], color=drop_colors, edgecolor="black")
    axes[1].set_title("Accuracy Drop Per Attack", fontweight="bold")
    axes[1].set_ylabel("Accuracy Drop")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].axhline(0, color="black", linewidth=0.8)

    plt.suptitle("Task 5 — Adversarial Robustness Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        out = os.path.join(config.OUTPUTS_DIR, "adversarial_robustness.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Robustness plot saved → {out}")
    plt.show()


_ATTACK_EXPLANATIONS = {
    "Typo Injection":
        "Typos corrupted key discriminative tokens, making them OOV or mapping them to the wrong embedding space.",
    "Synonym Replacement":
        "Synonym replacement altered the token distribution the model relies on, shifting the feature space.",
    "Word Shuffling":
        "Shuffling disrupted positional/sequential patterns captured by BiLSTM, causing incorrect context encoding.",
    "Emoji Insertion":
        "Injected emojis introduced noise tokens, diluting the meaningful signal and confusing the classifier.",
    "Punctuation Manip.":
        "Extra punctuation created false token boundaries, altering tokenization and feature extraction.",
}


def print_failed_examples(failed_df: pd.DataFrame, n: int = 10) -> None:
    print(f"\nTotal failed predictions: {len(failed_df)}")
    if len(failed_df) > 0:
        print("Failed predictions per attack:")
        print(failed_df["attack"].value_counts().to_string())

    print("\n" + "=" * 80)
    print("FAILED PREDICTION ANALYSIS — 10 Examples")
    print("=" * 80)

    display = failed_df.head(n) if len(failed_df) >= n else failed_df

    for i, (_, row) in enumerate(display.iterrows(), 1):
        conf_delta = row["attacked_conf"] - row["original_conf"]
        explanation = _ATTACK_EXPLANATIONS.get(row["attack"], "Unknown attack type.")
        print(f"\n❌ Failed Example #{i}")
        print(f"   Attack Type    : {row['attack']}")
        print(f"   Language       : {row['language']}")
        print(f"   True Label     : {row['true_label'].upper()}")
        print(f"   Original Pred  : {row['original_pred'].upper()} (conf: {row['original_conf']:.4f})")
        print(f"   Attacked Pred  : {row['attacked_pred'].upper()} (conf: {row['attacked_conf']:.4f})")
        print(f"   Confidence Δ   : {conf_delta:+.4f}")
        print(f"   ORIGINAL TEXT  : {row['original_text']}...")
        print(f"   ATTACKED TEXT  : {row['attacked_text']}...")
        print(f"   📝 EXPLANATION  : {explanation}")

    if len(failed_df) < 10:
        print(f"\n⚠️  Only {len(failed_df)} misclassifications — model is very robust!")


def showcase_adversarial_examples(
    adv_texts: list[str],
    adv_langs: list[str],
    n: int = 5,
) -> None:
    print("=" * 80)
    print("ADVERSARIAL EXAMPLES SHOWCASE")
    print("=" * 80)
    for i, (text, lang) in enumerate(zip(adv_texts[:n], adv_langs[:n]), 1):
        print(f"\n📰 SAMPLE {i} [{lang.upper()}]:")
        print(f"   ORIGINAL: {text[:150]}")
        for name, fn in ATTACKS.items():
            print(f"   [{name[:20]:20s}]: {fn(text)[:150]}")


# ── Convenience entry-point ───────────────────────────────────────────────────

def run_task5(
    df: pd.DataFrame,
    y: np.ndarray,
    y_test: np.ndarray,
    y_pred_dl: np.ndarray,
    dl_model,
    word_tokenizer,
    char_tokenizer,
    label_map: dict,
) -> tuple[dict, pd.DataFrame]:
    """Full Task 5 pipeline."""
    demo_attacks()

    robustness_results, failed_df = run_robustness_evaluation(
        df, y, y_test, y_pred_dl, dl_model, word_tokenizer, char_tokenizer, label_map
    )

    rob_df = print_robustness_table(robustness_results)
    plot_robustness(rob_df)
    print_failed_examples(failed_df)

    # Save outputs
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    rob_path = os.path.join(config.OUTPUTS_DIR, "robustness_comparison.csv")
    rob_df.to_csv(rob_path, index=False)
    print(f"Robustness comparison saved → {rob_path}")

    if len(failed_df) > 0:
        fail_path = os.path.join(config.OUTPUTS_DIR, "adversarial_failed_predictions.csv")
        failed_df.to_csv(fail_path, index=False)
        print(f"Failed predictions saved → {fail_path}")

    return robustness_results, failed_df
