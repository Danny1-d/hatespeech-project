"""
=============================================================================
MODULE 06C: DATA COMBINER  (FIXED for Python 3.14 + Windows)
Project: Automatic Hate Speech Detection in English-Igbo Code-Mixed Data
=============================================================================
FIXES:
  1. PyArrow/pandas 2.x bug — uses np.array(tolist()) instead of .values
  2. Auto-detects filenames — no hardcoded names that break on Windows

Run:
  python data-combiner.py
=============================================================================
"""

import os
import re
import glob
import json
import importlib.util
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

os.makedirs("data", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

IGBO_VOCAB = {
    'nna','nne','nwanne','obodo','ndị','chukwu','ọ','bụ','dị',
    'mma','anyị','ha','ka','na','si','ga','eme','ihe','ụlọ',
    'ebe','nke','nwere','gozie','kwenu','ahịa','obi','ụtọ',
    'oha','gbuo','kasie','igbo','nnaa','amaka','omenala','ndụ',
    'onye','nwanyị','adịghị','ọjọọ','ọma',
}


def compute_igbo_ratio(text):
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    if not tokens:
        return 0.0
    return round(sum(1 for t in tokens if t in IGBO_VOCAB) / len(tokens), 3)


def auto_load(keywords, alias):
    """Find and load a .py file by keywords — auto-detects any filename."""
    all_py = [f for f in os.listdir(SRC_DIR) if f.endswith('.py')]
    match  = next(
        (f for f in all_py
         if all(k.lower() in f.lower().replace('-','_') for k in keywords)),
        None
    )
    if match is None:
        return None
    path   = os.path.join(SRC_DIR, match)
    spec   = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# LOADERS
# =============================================================================

def load_huggingface_data():
    path = "data/processed/huggingface_combined.csv"
    if not os.path.exists(path):
        print(f"  [!] Not found: {path}  →  Run huggingface_datasets.py first")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source_module"] = "huggingface"
    print(f"  HuggingFace     : {len(df):>5} posts | Hate: {df['label'].sum()}")
    return df


def load_scraped_data():
    files = glob.glob("scraped_data/free_scraped_*.csv")
    if not files:
        print(f"  [!] No scraped files found — Run free_scraper.py --mock first")
        return pd.DataFrame()
    all_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            all_dfs.append(df)
            print(f"  Scraped file    : {Path(f).name} ({len(df)} posts)")
        except Exception as e:
            print(f"  Could not read {f}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    combined["source_module"] = "youtube_reddit"
    return combined


def load_igbo_handcrafted():
    """Load hand-crafted Igbo samples — auto-detects the 06a file."""
    try:
        m = auto_load(['hugging'], 'hf_datasets')
        if m is None:
            m = auto_load(['06a'], 'hf_datasets')
        if m is None:
            print(f"  Igbo handcrafted: skipped (06a file not found)")
            return pd.DataFrame()
        df = m.load_igbo_samples()
        df["source_module"] = "igbo_handcrafted"
        print(f"  Igbo handcrafted: {len(df):>5} posts | Hate: {df['label'].sum()}")
        return df
    except Exception as e:
        print(f"  Igbo handcrafted: skipped ({e})")
        return pd.DataFrame()


def load_project_dataset():
    """Load Module 01 dataset — auto-detects the data-collection file."""
    try:
        m = auto_load(['data'], 'data_preprocessing')
        if m is None:
            print(f"  Module 01 data  : skipped (data file not found)")
            return pd.DataFrame()
        df = m.create_dataset()
        df = df[["clean_text", "label", "igbo_ratio"]].rename(
            columns={"clean_text": "text"}
        )
        df["source"]        = "project_module01"
        df["source_module"] = "module01"
        print(f"  Module 01 data  : {len(df):>5} posts | Hate: {df['label'].sum()}")
        return df
    except Exception as e:
        print(f"  Module 01 data  : skipped ({e})")
        return pd.DataFrame()


# =============================================================================
# MERGE + CLEAN
# =============================================================================

def standardize_columns(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    required = {"text", "label"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    if "igbo_ratio" not in df.columns:
        df["igbo_ratio"] = df["text"].apply(compute_igbo_ratio)
    if "source" not in df.columns:
        df["source"] = df.get("source_module", "unknown")
    return df[["text", "label", "igbo_ratio", "source"]]


def merge_all(dfs, min_length=8):
    parts = []
    for df in dfs:
        clean = standardize_columns(df)
        if len(clean) > 0:
            parts.append(clean)
    if not parts:
        print("  ERROR: No valid data from any source!")
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    merged["_key"] = merged["text"].str.lower().str.strip()
    merged = merged.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
    merged = merged[merged["text"].str.len() >= min_length]
    return merged


# =============================================================================
# BALANCE + SPLIT
# =============================================================================

def balance_dataset(df, max_per_class=3000, seed=42):
    hate_df = df[df["label"] == 1]
    safe_df = df[df["label"] == 0]

    def priority_sample(subset, n):
        mixed  = subset[subset["igbo_ratio"] > 0.05]
        rest   = subset[subset["igbo_ratio"] <= 0.05]
        n_mix  = min(len(mixed), n)
        n_rest = min(len(rest),  n - n_mix)
        parts  = []
        if n_mix  > 0: parts.append(mixed.sample(n_mix,  random_state=seed))
        if n_rest > 0: parts.append(rest.sample(n_rest,  random_state=seed))
        return pd.concat(parts) if parts else pd.DataFrame()

    balanced = pd.concat(
        [priority_sample(hate_df, max_per_class),
         priority_sample(safe_df, max_per_class)],
        ignore_index=True
    )
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


def split_and_save(df, output_dir="data", seed=42):
    """Split into train/val/test — uses np.array to avoid PyArrow bug."""
    # FIX: tolist() + np.array avoids the PyArrow indexing TypeError
    X = np.array(df["text"].tolist(),  dtype=object)
    y = np.array(df["label"].tolist(), dtype=int)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.12, stratify=y_tv, random_state=seed
    )

    os.makedirs(output_dir, exist_ok=True)
    for name, texts, labels in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        pd.DataFrame({"text": texts, "label": labels}).to_csv(
            f"{output_dir}/{name}.csv", index=False, encoding="utf-8-sig"
        )

    print(f"\n  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)}   samples")
    print(f"  Test  : {len(X_test)}  samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# REPORT
# =============================================================================

def generate_report(df, sources_used):
    print("\n" + "=" * 60)
    print("  FINAL COMBINED DATASET REPORT")
    print("=" * 60)
    print(f"  Total samples    : {len(df)}")
    print(f"  Hate Speech (1)  : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"  Not Hate Speech  : {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"  Avg Igbo ratio   : {df['igbo_ratio'].mean():.3f}")
    print(f"  Code-mixed >5%%  : {(df['igbo_ratio']>0.05).sum()}")
    print(f"\n  Sources:")
    for src, count in df["source"].value_counts().items():
        print(f"    {src:<40} {count:>5} ({count/len(df)*100:.1f}%)")

    report = {
        "generated_at"   : datetime.now().isoformat(),
        "total_samples"  : int(len(df)),
        "hate_samples"   : int(df["label"].sum()),
        "safe_samples"   : int((df["label"]==0).sum()),
        "avg_igbo_ratio" : float(df["igbo_ratio"].mean()),
        "sources_used"   : sources_used,
    }
    with open("data/dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  Report saved: data/dataset_report.json")
    return report


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE 06C: DATA COMBINER")
    print("  Merging HuggingFace + YouTube/Reddit + Igbo samples")
    print("=" * 60)

    print("\nLoading all data sources...")
    sources_used = []

    df_hf   = load_huggingface_data()
    df_sc   = load_scraped_data()
    df_igbo = load_igbo_handcrafted()
    df_m1   = load_project_dataset()

    if len(df_hf)   > 0: sources_used.append("huggingface")
    if len(df_sc)   > 0: sources_used.append("youtube_reddit")
    if len(df_igbo) > 0: sources_used.append("igbo_handcrafted")
    if len(df_m1)   > 0: sources_used.append("module01")

    if not sources_used:
        print("\n✗ No data found! Run these first:")
        print("    python huggingface-datasets.py")
        print("    python free-scraper.py --mock")
        exit(1)

    print("\nMerging and cleaning...")
    merged = merge_all([df_hf, df_sc, df_igbo, df_m1])

    if len(merged) == 0:
        print("✗ Merge produced empty dataset. Check source files.")
        exit(1)

    print(f"  Raw merged: {len(merged)} unique posts")

    print("\nBalancing classes...")
    balanced = balance_dataset(merged, max_per_class=3000)

    balanced.to_csv("data/full_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"\n✓ Full dataset saved: data/full_dataset.csv")

    print("\nCreating train/val/test splits...")
    split_and_save(balanced)

    generate_report(balanced, sources_used)

    print("\n" + "=" * 60)
    print("  ✓ DONE! Now run: python pipeline.py --skip-deep")
    print("=" * 60)