"""
  - Language-aware text cleaning (URL, emoji, hashtag, mention removal)
  - Roman Urdu elongation normalization
  - Punctuation handling (ASCII vs Urdu script)
  - Tokenization + stopword removal
  - Applies pipeline to full DataFrame and benchmarks speed
"""

import re
import string
import time

import nltk
import pandas as pd
from tqdm import tqdm

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import config


# ── Stopword dictionaries ─────────────────────────────────────────────────────

EN_STOPWORDS = set(stopwords.words("english"))

URDU_STOPWORDS = {
    "ہے", "ہیں", "کا", "کی", "کے", "میں", "سے", "پر", "اور", "نے",
    "کو", "تھا", "تھی", "تھے", "یہ", "وہ", "جو", "ہو", "بھی", "اس",
    "ان", "ایک", "لیے", "تو", "کر", "ہی", "جب", "تک", "اب", "لیا",
}

ROMAN_URDU_STOPWORDS = {
    "hai", "hain", "ka", "ki", "ke", "mein", "se", "par", "aur", "ne",
    "ko", "tha", "thi", "the", "yeh", "woh", "jo", "ho", "bhi", "is",
    "un", "ek", "liye", "to", "kar", "hi", "jab", "tak", "ab", "liya",
    "ap", "aap", "hum", "tum", "main", "wo", "ye",
}

# ── Roman Urdu normalization rules ────────────────────────────────────────────

ROMAN_URDU_NORMALIZE = {
    r"\bacha+\b": "acha",   r"\bachaa+\b": "acha",
    r"\bkia+\b": "kia",     r"\bkyaa+\b": "kya",   r"\bkya+\b": "kya",
    r"\bnahi+\b": "nahi",   r"\bnahii+\b": "nahi",  r"\bnahin+\b": "nahi",
    r"\bhaa+n\b": "han",    r"\bhaa+\b": "ha",      r"\byaa+r\b": "yaar",
    r"\bthii+k\b": "theek", r"\btheek+\b": "theek",
    r"\bbhaii+\b": "bhai",  r"\bpyaa+r\b": "pyar",
    r"\baree+\b": "are",    r"\bare+\b": "are",
    r"\bbaht+\b": "bahut",  r"\bbahut+\b": "bahut",
    r"\bmee+n\b": "mein",   r"\bmain+\b": "main",
    r"\bapp+\b": "ap",      r"\baap+\b": "aap",
    r"\bkuu+n\b": "kaun",   r"\bkiu+n\b": "kyun",
    r"\bjuu+\b": "jo",      r"\bwoo+\b": "wo",
    r"\bdekh+o+\b": "dekho", r"\bsuno+o+\b": "suno",
    r"\bhai+i+\b": "hai",   r"\bhai+\b": "hai",
    r"\bhoo+\b": "ho",      r"\bhuu+\b": "hu",
    r"(.)\1{2,}": r"\1\1",  # collapse 3+ repeated chars → 2
}

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


# ── Atomic cleaning functions ─────────────────────────────────────────────────

def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+|https\S+", "", text, flags=re.MULTILINE)


def remove_emojis(text: str) -> str:
    return _EMOJI_PATTERN.sub("", text)


def handle_hashtags(text: str) -> str:
    """Strip # and lowercase the tag word."""
    return re.sub(r"#(\w+)", lambda m: m.group(1).lower(), text)


def remove_mentions(text: str) -> str:
    return re.sub(r"@\w+", "", text)


def handle_punctuation(text: str, language: str = "english") -> str:
    if language == "urdu":
        return re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', " ", text)
    return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))


def normalize_roman_urdu(text: str) -> str:
    for pattern, replacement in ROMAN_URDU_NORMALIZE.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def tokenize_text(text: str, language: str = "english") -> list[str]:
    if language == "urdu":
        return text.split()
    return word_tokenize(text.lower())


def remove_stopwords(tokens: list[str], language: str = "english") -> list[str]:
    if language == "urdu":
        return [t for t in tokens if t not in URDU_STOPWORDS]
    elif language == "roman_urdu":
        return [t for t in tokens if t not in ROMAN_URDU_STOPWORDS and t not in EN_STOPWORDS]
    return [t for t in tokens if t.lower() not in EN_STOPWORDS]


def remove_short_tokens(tokens: list[str], min_len: int = 2) -> list[str]:
    return [t for t in tokens if len(t) >= min_len]


# ── Main preprocessing function ───────────────────────────────────────────────

def preprocess_text(text: str, language: str = "english") -> str:
    """Full language-aware preprocessing pipeline for a single text."""
    text = str(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_emojis(text)
    text = handle_hashtags(text)

    if language == "roman_urdu":
        text = normalize_roman_urdu(text)

    text = handle_punctuation(text, language)

    if language != "urdu":
        text = text.lower()

    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = tokenize_text(text, language)
    tokens = remove_stopwords(tokens, language)
    tokens = remove_short_tokens(tokens)

    return " ".join(tokens)


# ── Apply to full DataFrame ───────────────────────────────────────────────────

def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocess_text to every row and benchmark speed."""
    print("Applying preprocessing pipeline...")
    start = time.time()

    df = df.copy()
    df["processed_news"] = [
        preprocess_text(row["news"], row["language"])
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing")
    ]

    elapsed = time.time() - start
    print(f"\nPreprocessed {len(df)} samples in {elapsed:.2f}s  "
          f"({len(df)/elapsed:.0f} samples/sec)")

    # 1 000-sample benchmark
    sample_1k = df.sample(min(1000, len(df)), random_state=config.RANDOM_STATE)
    t0 = time.time()
    _ = [preprocess_text(r["news"], r["language"]) for _, r in sample_1k.iterrows()]
    t1 = time.time()
    status = "✅ PASS" if (t1 - t0) < 120 else "❌ FAIL"
    print(f"1 000-sample benchmark: {t1-t0:.2f}s (limit 120s) — {status}")

    return df


# ── Before/After display ──────────────────────────────────────────────────────

def show_before_after(df: pd.DataFrame) -> None:
    """Print constructed examples and 5 random dataset rows."""
    examples = [
        (
            "Check this out!! 😂😂 #FakeNews https://t.co/abc123 @PM_Pakistan said everything is FINE!!",
            "english",
            "English — Emoji, URL, Hashtag, Mention",
        ),
        (
            "acha yaar ye bilkul sahi nahi haiiii 😡😡 #Politics kia ho raha haiii",
            "roman_urdu",
            "Roman Urdu — Elongation Normalization",
        ),
        (
            "حکومت نے اعلان کیا ہے کہ 🇵🇰 ملک میں امن و امان ہے! https://news.pk/abc",
            "urdu",
            "Urdu — URL, Emoji removal",
        ),
    ]

    print("=" * 80)
    print("BEFORE / AFTER PREPROCESSING EXAMPLES")
    print("=" * 80)
    for raw_text, lang, desc in examples:
        processed = preprocess_text(raw_text, lang)
        print(f"\n[{desc}]")
        print(f"  BEFORE: {raw_text}")
        print(f"  AFTER : {processed}")

    print("\n" + "=" * 80)
    print("DATASET BEFORE/AFTER (5 random rows):")
    print("=" * 80)
    for _, row in df[["news", "processed_news", "language"]].sample(5, random_state=1).iterrows():
        print(f"\n[{row['language'].upper()}]")
        print(f"  RAW      : {str(row['news'])[:150]}")
        print(f"  PROCESSED: {str(row['processed_news'])[:150]}")


# ── Convenience entry-point ───────────────────────────────────────────────────

def run_task2(df: pd.DataFrame) -> pd.DataFrame:
    """Full Task 2 pipeline: apply preprocessing and show examples."""
    df = apply_preprocessing(df)
    show_before_after(df)
    return df
