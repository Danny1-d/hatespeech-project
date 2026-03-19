"""
=============================================================================
MODULE 1: DATA COLLECTION & PREPROCESSING
Project: Automatic Hate Speech Detection in English-Igbo Code-Mixed Data
=============================================================================
This module handles:
  1. Dataset structure and sample data creation
  2. Text preprocessing for code-mixed English-Igbo text
  3. Tokenization with Igbo diacritic support
  4. Label encoding and train/test splitting
=============================================================================
"""

import re
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


# =============================================================================
# SECTION 1: SAMPLE DATASET (Simulated English-Igbo Code-Mixed Posts)
# =============================================================================
# In a real project, this data would be collected from:
# - Twitter/X API (search for Igbo keywords + English)
# - Facebook public groups
# - WhatsApp forward archives
# - Nairaland forums
# Labels: 0 = Not Hate Speech, 1 = Hate Speech

SAMPLE_DATA = [
    # --- NOT HATE SPEECH (label=0) ---
    {"text": "I love how my people, ndị Igbo, always support each other. Na ọha anyị ka ọ dị!", "label": 0},
    {"text": "Bro, obodo anyị is so beautiful this harmattan season. Make we appreciate am.", "label": 0},
    {"text": "Nna, this jollof rice is fire! My mama's recipe nwere ike igosi ụwa ihe ọ bụ.", "label": 0},
    {"text": "Just finished my exams today. Chukwu gozie gị for the support everyone!", "label": 0},
    {"text": "The wedding was amazing! Igbo traditional wedding dey always hit different 💃", "label": 0},
    {"text": "Nnaa, this traffic in Lagos is too much. Anyị ga-eri ọ? 😂", "label": 0},
    {"text": "I just got promoted at work. God is good. Ọ dị mma nke ukwuu!", "label": 0},
    {"text": "We are building our culture through language. Igbo kwenu! Keep speaking the language.", "label": 0},
    {"text": "My people them, make una vote wisely. Democracy dey for all of us.", "label": 0},
    {"text": "This Afrobeats concert was mad o. Anyị were obi ụtọ ịbịa!", "label": 0},
    {"text": "Nwanne m, how far? Long time no see. Ka anyị zuo ike together sometime.", "label": 0},
    {"text": "Igbo language is rich and deep. Every word carries history. I'm proud to speak it.", "label": 0},
    {"text": "My sister just graduated from university. Ọ masịrị m nke ọma! So proud of her.", "label": 0},
    {"text": "Make we not forget our roots. Igbo identity must be preserved for our children.", "label": 0},
    {"text": "The market today was so full of life. Ahịa ọ bụ ebe anyị meetiri ndị ọzọ.", "label": 0},

    # --- HATE SPEECH (label=1) ---
    {"text": "Ndị Igbo are all fraudsters. Them no get shame at all. Trash people!", "label": 1},
    {"text": "I hate these useless Yoruba people. Anyị agaghị ekwe! They are our enemies!", "label": 1},
    {"text": "This Hausa man dey always cheat. All of them are the same dirty people!", "label": 1},
    {"text": "Women are useless in business. Nwanyi adịghị mma n'ọrụ. They should stay home!", "label": 1},
    {"text": "Kill all the infidels! Ndị ọzọ agaghị ebi ndụ! God wants them gone.", "label": 1},
    {"text": "These stupid Igbo people should go back to their village. We don't want them here!", "label": 1},
    {"text": "All southerners are criminals. Anyị ga eme ha ihe! They deserve nothing good.", "label": 1},
    {"text": "This useless tribe can never lead. They are all animals and corrupt!", "label": 1},
    {"text": "Idiots from that region dey always cause trouble. Kasie ha n'ụlọ!", "label": 1},
    {"text": "These lazy people from the north are a burden on the country. Remove them all!", "label": 1},
    {"text": "I will deal with all those ugly Igbo traders. Eji ụjọ eme ha! Dirty people!", "label": 1},
    {"text": "Women who speak up deserve to be beaten. Nwanyị ọnụ ya adịghị mma!", "label": 1},
    {"text": "These foreigners are taking our jobs. Kill them all! Gbuo ha!", "label": 1},
    {"text": "That ethnic group is trash. Ihe ọ bụ ha bụ ọchịchọ ọjọọ. They will suffer!", "label": 1},
    {"text": "This religion is evil. All their followers should be eliminated from this country!", "label": 1},
]


# =============================================================================
# SECTION 2: IGBO LANGUAGE UTILITIES
# =============================================================================

# Igbo diacritics and special characters
IGBO_DIACRITICS = {
    'ọ': 'o_dot', 'Ọ': 'O_dot',
    'ụ': 'u_dot', 'Ụ': 'U_dot',
    'ị': 'i_dot', 'Ị': 'I_dot',
    'ṅ': 'n_dot', 'Ṅ': 'N_dot',
    'ń': 'n_acute', 'ǹ': 'n_grave',
    'á': 'a_acute', 'à': 'a_grave',
    'é': 'e_acute', 'è': 'e_grave',
    'í': 'i_acute', 'ì': 'i_grave',
    'ó': 'o_acute', 'ò': 'o_grave',
    'ú': 'u_acute', 'ù': 'u_grave',
}

# Common Igbo stopwords (non-informative function words)
IGBO_STOPWORDS = {
    'na', 'ka', 'ya', 'ha', 'anyị', 'ọ', 'nke', 'dị', 'bụ',
    'ma', 'ndi', 'ụlọ', 'ebe', 'si', 'ga', 'a', 'i', 'e',
    'n', 'ndị', 'nwanne', 'nna', 'nnaa', 'obodo'
}

# Common English stopwords
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can',
    'could', 'this', 'that', 'these', 'those', 'and', 'but', 'or',
    'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'up',
    'as', 'it', 'its', 'so', 'if', 'we', 'our', 'they', 'their', 'them',
    'not', 'no', 'all', 'just', 'how', 'what', 'who', 'which', 'when',
    'where', 'why', 'very', 'too', 'also', 'more', 'about', 'into'
}

ALL_STOPWORDS = IGBO_STOPWORDS | ENGLISH_STOPWORDS


# =============================================================================
# SECTION 3: PREPROCESSING PIPELINE
# =============================================================================

class CodeMixPreprocessor:
    """
    Preprocessor for English-Igbo code-mixed text.
    Handles diacritics, URLs, mentions, hashtags, and normalization.
    """

    def __init__(self, preserve_diacritics=True, remove_stopwords=False):
        self.preserve_diacritics = preserve_diacritics
        self.remove_stopwords = remove_stopwords

    def remove_urls(self, text):
        """Remove URLs from text."""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def remove_mentions_hashtags(self, text):
        """Remove @mentions and #hashtags."""
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        return text

    def remove_emojis(self, text):
        """Remove emoji characters."""
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    def normalize_diacritics(self, text):
        """
        Optionally normalize Igbo diacritics.
        If preserve_diacritics=True, keeps them (better for model understanding).
        If False, replaces with base characters.
        """
        if not self.preserve_diacritics:
            replacements = {
                'ọ': 'o', 'Ọ': 'O', 'ụ': 'u', 'Ụ': 'U',
                'ị': 'i', 'Ị': 'I', 'ṅ': 'n', 'Ṅ': 'N',
                'á': 'a', 'à': 'a', 'é': 'e', 'è': 'e',
                'í': 'i', 'ì': 'i', 'ó': 'o', 'ò': 'o',
                'ú': 'u', 'ù': 'u', 'ń': 'n', 'ǹ': 'n',
            }
            for char, replacement in replacements.items():
                text = text.replace(char, replacement)
        return text

    def normalize_whitespace(self, text):
        """Remove extra whitespace."""
        return re.sub(r'\s+', ' ', text).strip()

    def remove_special_characters(self, text):
        """Remove special characters but keep Igbo diacritics and basic punctuation."""
        # Keep alphanumeric, spaces, Igbo diacritics, basic punctuation
        igbo_chars = 'ọỌụỤịỊṅṄńǹáàéèíìóòúù'
        pattern = f'[^a-zA-Z0-9\\s{igbo_chars}.,!?\'"-]'
        return re.sub(pattern, '', text)

    def lowercase(self, text):
        return text.lower()

    def tokenize(self, text):
        """Simple whitespace + punctuation tokenizer."""
        tokens = re.findall(r"[a-zA-ZọỌụỤịỊṅṄńǹáàéèíìóòúù']+", text)
        return tokens

    def remove_stopwords_fn(self, tokens):
        return [t for t in tokens if t.lower() not in ALL_STOPWORDS]

    def preprocess(self, text, return_tokens=False):
        """Full preprocessing pipeline."""
        text = self.remove_urls(text)
        text = self.remove_mentions_hashtags(text)
        text = self.remove_emojis(text)
        text = self.normalize_diacritics(text)
        text = self.remove_special_characters(text)
        text = self.lowercase(text)
        text = self.normalize_whitespace(text)

        if return_tokens:
            tokens = self.tokenize(text)
            if self.remove_stopwords:
                tokens = self.remove_stopwords_fn(tokens)
            return tokens

        return text


# =============================================================================
# SECTION 4: LANGUAGE DETECTION (Code-Mix Ratio)
# =============================================================================

# Basic Igbo word list (common vocabulary)
IGBO_VOCABULARY = {
    'nna', 'nne', 'nwanne', 'obodo', 'ndị', 'chukwu', 'ọ', 'bụ', 'dị',
    'mma', 'ọchịchọ', 'anyị', 'ha', 'ka', 'na', 'si', 'ga', 'eme', 'ihe',
    'ụlọ', 'ebe', 'ọrụ', 'oge', 'ụwa', 'nke', 'nwere', 'gozie', 'kwenu',
    'ahịa', 'obi', 'ụtọ', 'oha', 'masịrị', 'kwesị', 'gbuo', 'kasie',
    'eji', 'ụjọ', 'onye', 'ọnụ', 'adịghị', 'nwanyị', 'agaghị', 'ọjọọ',
    'nwanne', 'nnaa', 'ụnọ', 'igbo', 'ndị', 'gị', 'ịbịa', 'ndụ'
}


def detect_language_ratio(text):
    """
    Estimate the ratio of Igbo to English tokens in a code-mixed text.
    Returns: (igbo_ratio, english_ratio, mixed_flag)
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:
        return 0.0, 0.0, False

    igbo_count = sum(1 for t in tokens if t in IGBO_VOCABULARY)
    english_count = sum(1 for t in tokens if t.isascii() and t not in IGBO_VOCABULARY)

    total = len(tokens)
    igbo_ratio = igbo_count / total
    english_ratio = english_count / total
    is_mixed = igbo_ratio > 0.05 and english_ratio > 0.05

    return round(igbo_ratio, 3), round(english_ratio, 3), is_mixed


# =============================================================================
# SECTION 5: DATASET CREATION & ANALYSIS
# =============================================================================

def create_dataset():
    """Create DataFrame from sample data with preprocessing applied."""
    preprocessor = CodeMixPreprocessor(preserve_diacritics=True)

    rows = []
    for item in SAMPLE_DATA:
        text = item['text']
        label = item['label']
        clean_text = preprocessor.preprocess(text)
        igbo_ratio, eng_ratio, is_mixed = detect_language_ratio(text)
        tokens = preprocessor.preprocess(text, return_tokens=True)
        token_count = len(tokens)

        rows.append({
            'original_text': text,
            'clean_text': clean_text,
            'label': label,
            'label_name': 'Hate Speech' if label == 1 else 'Not Hate Speech',
            'token_count': token_count,
            'igbo_ratio': igbo_ratio,
            'english_ratio': eng_ratio,
            'is_code_mixed': is_mixed
        })

    return pd.DataFrame(rows)


def analyze_dataset(df):
    """Print dataset statistics."""
    print("=" * 60)
    print("  DATASET ANALYSIS")
    print("=" * 60)
    print(f"\nTotal samples       : {len(df)}")
    print(f"Hate Speech (1)     : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"Not Hate Speech (0) : {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"\nAvg token count     : {df['token_count'].mean():.1f}")
    print(f"Code-mixed samples  : {df['is_code_mixed'].sum()} ({df['is_code_mixed'].mean()*100:.1f}%)")
    print(f"Avg Igbo ratio      : {df['igbo_ratio'].mean():.3f}")
    print(f"Avg English ratio   : {df['english_ratio'].mean():.3f}")

    # Label distribution
    print("\nLabel Distribution:")
    print(df['label_name'].value_counts().to_string())

    # Top tokens
    all_tokens = []
    preprocessor = CodeMixPreprocessor()
    for text in df['clean_text']:
        tokens = preprocessor.tokenize(text)
        all_tokens.extend(tokens)

    top_tokens = Counter(all_tokens).most_common(15)
    print("\nTop 15 Most Frequent Tokens:")
    for token, count in top_tokens:
        print(f"  {token:<20} {count}")


def split_dataset(df, test_size=0.2, val_size=0.1, random_state=42):
    """Split dataset into train, validation, and test sets."""
    # Use to_numpy() instead of .values to avoid PyArrow/pandas 2.x issues
    X = np.array(df['clean_text'].tolist(), dtype=object)
    y = np.array(df['label'].tolist(), dtype=int)

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio,
        random_state=random_state, stratify=y_trainval
    )

    print(f"\nDataset Split:")
    print(f"  Train      : {len(X_train)} samples")
    print(f"  Validation : {len(X_val)} samples")
    print(f"  Test       : {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# SECTION 6: SAVE PROCESSED DATA
# =============================================================================

def save_processed_data(df, output_dir="data"):
    """Save processed dataset to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save full dataset
    df.to_csv(f"{output_dir}/full_dataset.csv", index=False)

    # Save train/test splits
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    val_df   = pd.DataFrame({'text': X_val,   'label': y_val})
    test_df  = pd.DataFrame({'text': X_test,  'label': y_test})

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"\nData saved to '{output_dir}/' directory.")
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Creating and analyzing English-Igbo code-mixed dataset...\n")

    df = create_dataset()
    analyze_dataset(df)
    save_processed_data(df)

    print("\nSample preprocessed texts:")
    print("-" * 60)
    for _, row in df.head(4).iterrows():
        print(f"Original : {row['original_text'][:80]}")
        print(f"Cleaned  : {row['clean_text'][:80]}")
        print(f"Label    : {row['label_name']} | Igbo ratio: {row['igbo_ratio']}")
        print()

    print("Module 1 complete.")