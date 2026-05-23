import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config


# ── 1.1 Load ──────────────────────────────────────────────────────────────────

def load_raw(path: str = config.DATASET_PATH) -> pd.DataFrame:
    """Load the raw CSV and normalize column names and label values."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = ["news", "label", "language"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    df["label"]    = df["label"].astype(str).str.strip().str.lower()
    df["language"] = df["language"].astype(str).str.strip().str.lower()

    print(f"Raw dataset shape: {df.shape}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nLanguage distribution:\n{df['language'].value_counts()}")
    return df


# ── 1.2 Clean ─────────────────────────────────────────────────────────────────

def _remove_symbols(text: str) -> str:
    """Remove HTML tags, control characters, and collapse whitespace."""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = re.sub(r"\s+", " ", text).strip()       # extra whitespace
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)   # control chars
    return text


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: nulls → duplicates → symbols → label filter."""
    df = df.copy()
    print(f"Before cleaning: {len(df)} rows")

    # Nulls / empty
    before = len(df)
    df.dropna(subset=["news", "label"], inplace=True)
    df = df[df["news"].astype(str).str.strip() != ""]
    print(f"After null/empty removal : {len(df)} rows (removed {before - len(df)})")

    # Duplicates
    before = len(df)
    df.drop_duplicates(subset=["news"], inplace=True)
    print(f"After duplicate removal  : {len(df)} rows (removed {before - len(df)})")

    # Symbol removal
    df["news"] = df["news"].apply(_remove_symbols)

    # Label whitelist
    before = len(df)
    df = df[df["label"].isin(config.VALID_LABELS)]
    print(f"After label filtering    : {len(df)} rows (removed {before - len(df)})")

    df.reset_index(drop=True, inplace=True)
    print(f"\nFinal cleaned shape: {df.shape}")
    return df


# ── 1.3 Balance ───────────────────────────────────────────────────────────────

def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Undersample the majority class if imbalance exceeds threshold."""
    counts   = df["label"].value_counts()
    total    = len(df)
    fake_pct = counts.get("fake", 0) / total * 100
    true_pct = counts.get("true", 0) / total * 100
    diff     = abs(fake_pct - true_pct)

    print(f"\nClass imbalance: {diff:.1f}%")
    if diff > config.BALANCE_THRESHOLD:
        print("⚠️  Rebalancing via undersampling...")
        min_count = counts.min()
        df = pd.concat([
            df[df["label"] == "fake"].sample(min_count, random_state=config.RANDOM_STATE),
            df[df["label"] == "true"].sample(min_count, random_state=config.RANDOM_STATE),
        ]).sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
        print(f"After rebalancing: {len(df)} samples\n{df['label'].value_counts()}")
    else:
        print("✅ Class balance within threshold — no resampling needed.")
    return df


# ── 1.4 Statistics ────────────────────────────────────────────────────────────

def print_statistics(df: pd.DataFrame) -> None:
    """Print a formatted statistics table for the cleaned dataset."""
    all_words = " ".join(df["news"].tolist()).lower().split()
    vocab_size = len(set(all_words))
    avg_len    = df["news"].apply(lambda x: len(str(x).split())).mean()

    imbalance = abs(
        df["label"].value_counts(normalize=True).get("fake", 0)
        - df["label"].value_counts(normalize=True).get("true", 0)
    ) * 100

    stats = {
        "Metric": [
            "Total Samples", "Fake Samples", "True Samples",
            "English Samples", "Urdu Samples", "Roman Urdu Samples",
            "Vocabulary Size", "Avg. Words per Sample", "Class Imbalance (%)",
        ],
        "Value": [
            len(df),
            len(df[df["label"] == "fake"]),
            len(df[df["label"] == "true"]),
            len(df[df["language"] == "english"]),
            len(df[df["language"] == "urdu"]),
            len(df[df["language"] == "roman_urdu"]),
            vocab_size,
            round(avg_len, 2),
            round(imbalance, 2),
        ],
    }
    stats_df = pd.DataFrame(stats)
    print("=" * 45)
    print("         DATASET STATISTICS TABLE")
    print("=" * 45)
    print(stats_df.to_string(index=False))
    print("=" * 45)


def plot_distributions(df: pd.DataFrame, save: bool = True) -> None:
    """Plot label, language, and word-count distributions."""
    df = df.copy()
    df["word_count"] = df["news"].apply(lambda x: len(str(x).split()))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    df["label"].value_counts().plot(
        kind="bar", ax=axes[0], color=["#e74c3c", "#2ecc71"], edgecolor="black"
    )
    axes[0].set_title("Label Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Label"); axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    df["language"].value_counts().plot(
        kind="bar", ax=axes[1], color=["#3498db", "#e67e22", "#9b59b6"], edgecolor="black"
    )
    axes[1].set_title("Language Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Language"); axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].hist(df["word_count"], bins=50, color="#1abc9c", edgecolor="black")
    axes[2].set_title("Word Count Distribution", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Word Count"); axes[2].set_ylabel("Frequency")

    plt.suptitle("Task 1 — Dataset Statistics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        import os
        out = os.path.join(config.OUTPUTS_DIR, "task1_dataset_stats.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out}")
    plt.show()


# ── 1.5 Save ──────────────────────────────────────────────────────────────────

def save_cleaned(df: pd.DataFrame, path: str = config.CLEANED_DATASET_PATH) -> None:
    df.drop(columns=["word_count"], errors="ignore", inplace=True)
    df.to_csv(path, index=False)
    print(f"Cleaned dataset saved → {path}  ({len(df)} rows)")


# ── Convenience entry-point ───────────────────────────────────────────────────

def run_task1(raw_path: str = config.DATASET_PATH) -> pd.DataFrame:
    """Full Task 1 pipeline: load → clean → balance → stats → save."""
    df = load_raw(raw_path)
    df = clean(df)
    df = balance_classes(df)
    print_statistics(df)
    plot_distributions(df)
    save_cleaned(df)
    return df
