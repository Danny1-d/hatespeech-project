"""
=============================================================================
MODULE 06A: FREE HUGGINGFACE DATASET LOADER
Project: Automatic Hate Speech Detection in English-Igbo Code-Mixed Data
=============================================================================
Loads FREE, publicly available hate speech datasets from HuggingFace.
No API key, no account, no cost — just pip install and run.

DATASETS USED (all free):
  1. ucberkeley-dlab/measuring-hate-speech     → 135,000+ annotated posts
  2. tweet_eval (hate subset)                  → 9,000+ tweets
  3. hate_speech18                             → 10,000+ forum posts
  4. Hatemoji                                  → emoji-rich hate speech

After loading, we:
  - Filter for posts that overlap with Igbo/Nigerian topics
  - Inject our own Igbo code-mixed sample posts
  - Standardize all labels to 0/1
  - Export a clean unified CSV

INSTALL:
  pip install datasets pandas numpy
=============================================================================
"""

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Install: pip install datasets")
    exit(1)

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# =============================================================================
# SECTION 1: IGBO VOCABULARY (for code-mix injection)
# =============================================================================

IGBO_VOCAB = {
    'nna', 'nne', 'nwanne', 'obodo', 'ndị', 'chukwu', 'ọ', 'bụ', 'dị',
    'mma', 'anyị', 'ha', 'ka', 'na', 'si', 'ga', 'eme', 'ihe', 'ụlọ',
    'ebe', 'nke', 'nwere', 'gozie', 'kwenu', 'ahịa', 'obi', 'ụtọ',
    'oha', 'gbuo', 'kasie', 'igbo', 'nnaa', 'amaka', 'omenala', 'ndụ',
    'onye', 'nwanyị', 'adịghị', 'ọjọọ', 'ọma',
}

# Igbo phrases to inject into English hate/safe posts to create code-mix
IGBO_HATE_PHRASES = [
    "Gbuo ha niile!", "Kasie ha!", "Ha bụ ụnọ ọjọọ.",
    "Eme ha ihe ojoo.", "Anyị ga eme ha ihe.",
    "Ha adịghị mma.", "Nwanyị ọnụ ya!",
]

IGBO_SAFE_PHRASES = [
    "Chukwu gozie gị!", "Obi ụtọ nke ukwuu!",
    "Nna, ọ dị mma.", "Ka anyị nọ ụlọ.",
    "Igbo amaka!", "Nwanne m, ọ dị mma.",
    "Gozie gị nke ọma.", "Ọ masịrị m nke ọma!",
]

NIGERIAN_KEYWORDS = [
    "nigeria", "nigerian", "igbo", "yoruba", "hausa", "lagos", "abuja",
    "naija", "nairaland", "ibo", "biafra", "southeast", "ndi igbo",
    "fulani", "tribe", "ethnic", "enugu", "anambra", "imo",
]


def is_nigeria_related(text):
    """Check if a post is related to Nigerian/Igbo context."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in NIGERIAN_KEYWORDS)


def inject_igbo(text, label, p=0.6):
    """
    Randomly inject an Igbo phrase into an English post
    to simulate code-mixing. p = probability of injection.
    """
    import random
    if random.random() > p:
        return text
    phrases = IGBO_HATE_PHRASES if label == 1 else IGBO_SAFE_PHRASES
    phrase = random.choice(phrases)
    # Insert at end of text
    return f"{text.rstrip()} {phrase}"


# =============================================================================
# SECTION 2: LOAD EACH HUGGINGFACE DATASET
# =============================================================================

def load_tweet_eval_hate(max_samples=3000):
    """
    Load tweet_eval hate speech subset.
    Labels: 0=non-hate, 1=hate
    Source: SemEval 2019 Task 5 (HatEval)
    Completely free, no login needed.
    """
    print("\n[1/4] Loading tweet_eval (hate subset)...")
    try:
        ds = load_dataset("tweet_eval", "hate", trust_remote_code=True)
        rows = []
        for split in ["train", "validation", "test"]:
            if split in ds:
                for item in ds[split]:
                    rows.append({
                        "text"   : item["text"],
                        "label"  : int(item["label"]),
                        "source" : "tweet_eval_hate",
                    })
        df = pd.DataFrame(rows).dropna()
        df = df[df["text"].str.len() > 10]
        df = df.sample(min(max_samples, len(df)), random_state=42)
        print(f"  Loaded {len(df)} samples | Hate: {df['label'].sum()}")
        return df
    except Exception as e:
        print(f"  Could not load tweet_eval: {e}")
        return pd.DataFrame()


def load_hate_speech18(max_samples=3000):
    """
    Load hate_speech18 dataset (forum posts from Stormfront).
    Labels: noHate=0, hate=1 (we map others to 0)
    Free, no login.
    """
    print("\n[2/4] Loading hate_speech18...")
    try:
        ds = load_dataset("hate_speech18", trust_remote_code=True)
        rows = []
        for split in ds:
            for item in ds[split]:
                label_str = str(item.get("label", "noHate"))
                label = 1 if "hate" in label_str.lower() and "no" not in label_str.lower() else 0
                rows.append({
                    "text"   : str(item.get("text", "")),
                    "label"  : label,
                    "source" : "hate_speech18",
                })
        df = pd.DataFrame(rows).dropna()
        df = df[df["text"].str.len() > 10]
        df = df.sample(min(max_samples, len(df)), random_state=42)
        print(f"  Loaded {len(df)} samples | Hate: {df['label'].sum()}")
        return df
    except Exception as e:
        print(f"  Could not load hate_speech18: {e}")
        return pd.DataFrame()


def load_measuring_hate_speech(max_samples=3000):
    """
    Load UC Berkeley Measuring Hate Speech dataset.
    One of the most comprehensive hate speech datasets available.
    Labels: hate_speech_score > 0.5 → hate (1)
    Free, no login.
    """
    print("\n[3/4] Loading UC Berkeley measuring-hate-speech...")
    try:
        ds = load_dataset(
            "ucberkeley-dlab/measuring-hate-speech",
            "binary",
            trust_remote_code=True
        )
        rows = []
        for split in ds:
            for item in ds[split]:
                text  = str(item.get("text", ""))
                score = float(item.get("hate_speech_score", 0) or 0)
                label = 1 if score > 0.5 else 0
                rows.append({"text": text, "label": label, "source": "measuring_hate_speech"})

        df = pd.DataFrame(rows).dropna()
        df = df[df["text"].str.len() > 10]
        df = df.sample(min(max_samples, len(df)), random_state=42)
        print(f"  Loaded {len(df)} samples | Hate: {df['label'].sum()}")
        return df
    except Exception as e:
        print(f"  Could not load measuring-hate-speech: {e}")
        return pd.DataFrame()


def load_offcomeval(max_samples=2000):
    """
    Load OffComEval (offensive language) dataset.
    Labels: 0=not offensive, 1=offensive (we treat offensive as potential hate)
    """
    print("\n[4/4] Loading olid (offensive language)...")
    try:
        ds = load_dataset("tweet_eval", "offensive", trust_remote_code=True)
        rows = []
        for split in ds:
            for item in ds[split]:
                rows.append({
                    "text"  : item["text"],
                    "label" : int(item["label"]),
                    "source": "tweet_eval_offensive",
                })
        df = pd.DataFrame(rows).dropna()
        df = df[df["text"].str.len() > 10]
        df = df.sample(min(max_samples, len(df)), random_state=42)
        print(f"  Loaded {len(df)} samples | Offensive: {df['label'].sum()}")
        return df
    except Exception as e:
        print(f"  Could not load offensive dataset: {e}")
        return pd.DataFrame()


# =============================================================================
# SECTION 3: IGBO CODE-MIX SAMPLE POSTS (hand-crafted)
# =============================================================================

IGBO_CODEMIX_POSTS = [
    # NOT HATE SPEECH — label 0
    {"text": "Chukwu gozie gị! I love my Igbo culture so much nke ọma 🙏", "label": 0},
    {"text": "Nna, the market today was full of life. Ahịa na-atọ ụtọ!", "label": 0},
    {"text": "I just got promoted at work! Obi ụtọ nke ukwuu. God is good.", "label": 0},
    {"text": "Igbo culture is so rich and beautiful. Omenala anyị bụ ihe ọma.", "label": 0},
    {"text": "Nwanne m, how far? Long time no see. Ka anyị zuo ike together.", "label": 0},
    {"text": "My sister just graduated! Ọ masịrị m nke ọma. So proud of her!", "label": 0},
    {"text": "Igbo language is deep and powerful. Anyị ga-echebe ya for our children.", "label": 0},
    {"text": "We are building our community. Ndị Igbo always support each other!", "label": 0},
    {"text": "Nnaa, this jollof rice is fire! Mama's recipe nwere ike igosi ụwa.", "label": 0},
    {"text": "The wedding was amazing! Igbo traditional wedding dey always hit different 💃", "label": 0},
    {"text": "Ka anyị nọ ụlọ together this harmattan. It is cold outside today.", "label": 0},
    {"text": "I love how ndị Igbo always come together. Na ọha anyị ka ọ dị!", "label": 0},
    {"text": "Gozie gị nke ọma for everything you have done. God bless you always.", "label": 0},
    {"text": "Just finished my exams. Chukwu gozie gị for the strength everyone!", "label": 0},
    {"text": "Igbo kwenu! Keep speaking the language. Our identity must be preserved.", "label": 0},
    {"text": "Nna mezie ọrụ ya nke ọma. He always does his work so well and carefully.", "label": 0},
    {"text": "Obi ụtọ nke ukwuu! We finally finished the project. Very excited today!", "label": 0},
    {"text": "My people them, make una vote wisely. Democracy dey for all of us equally.", "label": 0},
    {"text": "Anyị were obi ụtọ ịbịa! The Afrobeats concert was absolutely incredible.", "label": 0},
    {"text": "Ọ dị mma nke ukwuu! I love this country despite all its challenges.", "label": 0},

    # HATE SPEECH — label 1
    {"text": "Ndị Igbo are all fraudsters. Them no get shame at all! Gbuo ha niile!", "label": 1},
    {"text": "I hate these Yoruba people. Anyị agaghị ekwe! They are our enemies forever.", "label": 1},
    {"text": "Kill all those useless people from that tribe! Kasie ha! They deserve nothing.", "label": 1},
    {"text": "These stupid Igbo traders should go back to their village. Gbuo ha!", "label": 1},
    {"text": "All southerners are criminals. Anyị ga eme ha ihe! They deserve nothing good.", "label": 1},
    {"text": "Women are useless in business. Nwanyị adịghị mma n'ọrụ. Stay home!", "label": 1},
    {"text": "Eliminate all infidels from this country! Ha bụ ụnọ ọjọọ. God wants it!", "label": 1},
    {"text": "This useless tribe can never lead Nigeria. Eji ụjọ eme ha! Dirty people!", "label": 1},
    {"text": "These lazy people from the north are a burden. Remove them all! Kasie ha!", "label": 1},
    {"text": "All those criminals from Igboland should be arrested. Gbuo ha niile!", "label": 1},
    {"text": "Women who speak up deserve to be beaten. Nwanyị ọnụ ya adịghị mma!", "label": 1},
    {"text": "These dirty foreigners are taking our jobs. Kill them all! Gbuo ha!", "label": 1},
    {"text": "That ethnic group is trash. Ha bụ ọchịchọ ọjọọ. They will suffer forever!", "label": 1},
    {"text": "This religion is evil. All their followers should be eliminated! Kasie ha!", "label": 1},
    {"text": "Idiots from that region cause trouble. Ha adịghị mma. Send them away!", "label": 1},
    {"text": "These Igbo people are all thieves. Gbuo ha! I hate them with passion!", "label": 1},
    {"text": "Stupid tribe thinks they own this country. Anyị ga eme ha ihe niile!", "label": 1},
    {"text": "Remove all Hausa people from the south now! Ha bụ ụnọ ọjọọ niile!", "label": 1},
    {"text": "All women from that tribe are prostitutes. Nwanyị ha adịghị mma at all!", "label": 1},
    {"text": "These people should die. Gbuo ha! This country is better without them.", "label": 1},
]


def load_igbo_samples():
    """Load the hand-crafted Igbo code-mix posts."""
    df = pd.DataFrame(IGBO_CODEMIX_POSTS)
    df["source"] = "igbo_handcrafted"
    print(f"\n  Igbo hand-crafted samples: {len(df)} | Hate: {df['label'].sum()}")
    return df


# =============================================================================
# SECTION 4: CODE-MIX INJECTION PIPELINE
# =============================================================================

def inject_code_mix(df, inject_ratio=0.3, seed=42):
    """
    Take English-only posts from HuggingFace datasets and
    inject Igbo phrases into a subset to create synthetic code-mixed data.

    inject_ratio: proportion of posts to inject Igbo into
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    df = df.copy()
    mask = np.random.rand(len(df)) < inject_ratio
    df.loc[mask, "text"] = df[mask].apply(
        lambda row: inject_igbo(row["text"], row["label"]), axis=1
    )
    df.loc[mask, "is_injected"] = True
    df.loc[~mask, "is_injected"] = False
    print(f"  Igbo phrases injected into {mask.sum()} posts ({inject_ratio*100:.0f}%)")
    return df


def compute_igbo_ratio(text):
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    if not tokens:
        return 0.0
    return round(sum(1 for t in tokens if t in IGBO_VOCAB) / len(tokens), 3)


# =============================================================================
# SECTION 5: COMBINE + CLEAN + BALANCE
# =============================================================================

def combine_datasets(*dfs, max_per_source=2000, target_per_class=2000, seed=42):
    """
    Combine all loaded datasets into one balanced, cleaned dataset.
    Caps each source to avoid domination by large datasets.
    """
    print("\nCombining all datasets...")

    parts = []
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        df = df.copy()

        # Ensure required columns
        if "label" not in df.columns or "text" not in df.columns:
            continue

        # Cap per source
        df = df.sample(min(max_per_source, len(df)), random_state=seed)
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)

    # Standardize columns
    combined = combined[["text", "label", "source"]].copy()
    combined["label"] = combined["label"].astype(int)
    combined = combined[combined["label"].isin([0, 1])]

    # Drop duplicates and empty
    combined = combined.drop_duplicates(subset=["text"])
    combined = combined[combined["text"].str.strip().str.len() > 10]

    # Add Igbo ratio
    combined["igbo_ratio"] = combined["text"].apply(compute_igbo_ratio)

    # Balance classes
    hate_df = combined[combined["label"] == 1]
    safe_df = combined[combined["label"] == 0]

    hate_df = hate_df.sample(min(target_per_class, len(hate_df)), random_state=seed)
    safe_df = safe_df.sample(min(target_per_class, len(safe_df)), random_state=seed)

    final = pd.concat([hate_df, safe_df], ignore_index=True)
    final = final.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\n  Combined dataset:")
    print(f"  Total samples  : {len(final)}")
    print(f"  Hate (1)       : {final['label'].sum()}")
    print(f"  Safe (0)       : {(final['label']==0).sum()}")
    print(f"  Sources        :")
    for src, count in final["source"].value_counts().items():
        print(f"    {src:<35} {count}")

    return final


# =============================================================================
# SECTION 6: SAVE FINAL DATASET
# =============================================================================

def save_final_dataset(df, path="data/processed/huggingface_combined.csv"):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Final dataset saved: {path}")
    print(f"  {len(df)} samples ready for training")
    return path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE 06A: FREE HuggingFace Dataset Loader")
    print("=" * 60)
    print("Downloading free datasets (first run may take a few minutes)...")

    # Load all free datasets
    df1 = load_tweet_eval_hate(max_samples=3000)
    df2 = load_hate_speech18(max_samples=3000)
    df3 = load_measuring_hate_speech(max_samples=3000)
    df4 = load_offcomeval(max_samples=2000)
    df5 = load_igbo_samples()

    # Inject Igbo into English posts to create code-mix
    print("\nInjecting Igbo phrases into English posts...")
    if len(df1) > 0: df1 = inject_code_mix(df1, inject_ratio=0.35)
    if len(df2) > 0: df2 = inject_code_mix(df2, inject_ratio=0.30)
    if len(df3) > 0: df3 = inject_code_mix(df3, inject_ratio=0.25)
    if len(df4) > 0: df4 = inject_code_mix(df4, inject_ratio=0.30)

    # Combine everything
    final_df = combine_datasets(df1, df2, df3, df4, df5, target_per_class=2500)

    # Save
    save_final_dataset(final_df)

    print("\n✓ Module 06A complete.")
    print("  Next step: run 06b_free_scraper.py to add YouTube + Reddit data")
    print("  Then run:  06c_data_combiner.py to merge everything")