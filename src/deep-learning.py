"""
=============================================================================
MODULE 3: DEEP LEARNING + TRANSFORMER MODELS
Project: Automatic Hate Speech Detection in English-Igbo Code-Mixed Data
=============================================================================
This module covers:
  1. LSTM/BiLSTM with word embeddings (lightweight deep learning)
  2. Fine-tuning AfriBERTa (best for Igbo/African languages)
  3. Fine-tuning XLM-RoBERTa (strong multilingual baseline)
  4. Fine-tuning mBERT (multilingual BERT)
  5. Evaluation and model comparison
=============================================================================
INSTALLATION REQUIREMENTS:
  pip install transformers torch datasets scikit-learn pandas numpy
  pip install sentencepiece protobuf accelerate
  
RECOMMENDED MODEL: AfriBERTa (castorini/afriberta_large)
  - Trained on 11 African languages including Igbo
  - Best suited for English-Igbo code-mixed text
  - Outperforms mBERT on low-resource African NLP tasks
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: DEVICE SETUP
# =============================================================================

def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (M1/M2 GPU)")
    else:
        device = torch.device('cpu')
        print("Using CPU (no GPU detected)")
    return device


DEVICE = get_device()


# =============================================================================
# SECTION 2: PYTORCH DATASET CLASS
# =============================================================================

class HateSpeechDataset(Dataset):
    """
    PyTorch Dataset for hate speech detection.
    Works with both tokenizer-based (transformers) and manual encoding.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids'      : encoding['input_ids'].squeeze(),
            'attention_mask' : encoding['attention_mask'].squeeze(),
            'label'          : torch.tensor(label, dtype=torch.long)
        }


# =============================================================================
# SECTION 3: BiLSTM MODEL (Lightweight Deep Learning)
# =============================================================================

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for text classification.
    Lighter than transformers; good baseline for limited compute.
    Architecture: Embedding → BiLSTM → Attention → FC → Output
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=2, dropout=0.3, pad_idx=0):
        super(BiLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.dropout(self.embedding(input_ids))     # (B, T, E)
        lstm_out, _ = self.lstm(embedded)                       # (B, T, 2H)
        lstm_out = self.layer_norm(lstm_out)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
        attended = (lstm_out * attn_weights).sum(dim=1)                 # (B, 2H)

        output = self.dropout(attended)
        logits = self.fc(output)                                         # (B, C)
        return logits


class SimpleVocab:
    """Simple vocabulary builder for BiLSTM."""
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.token2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2token = {0: '<PAD>', 1: '<UNK>'}

    def build(self, texts):
        from collections import Counter
        import re
        all_tokens = []
        for text in texts:
            tokens = re.findall(
                r"[a-zA-ZọỌụỤịỊṅṄńǹáàéèíìóòúù']+", text.lower()
            )
            all_tokens.extend(tokens)
        counts = Counter(all_tokens)
        for token, freq in counts.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        print(f"Vocabulary size: {len(self.token2idx)}")

    def encode(self, text, max_len=128):
        import re
        tokens = re.findall(
            r"[a-zA-ZọỌụỤịỊṅṄńǹáàéèíìóòúù']+", text.lower()
        )
        ids = [self.token2idx.get(t, 1) for t in tokens[:max_len]]
        # Pad or truncate
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.token2idx)


class BiLSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.data = [
            (torch.tensor(vocab.encode(t, max_len), dtype=torch.long),
             torch.tensor(l, dtype=torch.long))
            for t, l in zip(texts, labels)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, label = self.data[idx]
        return {'input_ids': input_ids, 'label': label}


def train_bilstm(X_train, y_train, X_val, y_val, epochs=10, batch_size=16, lr=1e-3):
    """Train the BiLSTM model."""
    print("\nTraining BiLSTM Classifier...")

    # Build vocabulary
    vocab = SimpleVocab(min_freq=1)
    vocab.build(X_train)

    # Datasets
    train_dataset = BiLSTMDataset(X_train, y_train, vocab)
    val_dataset   = BiLSTMDataset(X_val, y_val, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    ).to(DEVICE)

    # Class weights for imbalance
    class_counts = np.bincount(y_train)
    weights = torch.tensor(
        [1.0 / c for c in class_counts], dtype=torch.float
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_f1 = 0
    best_state  = None
    history     = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            labels    = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds, val_labels, val_loss_total = [], [], 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                labels    = batch['label'].to(DEVICE)
                logits = model(input_ids)
                val_loss_total += criterion(logits, labels).item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch['label'].numpy())

        val_f1   = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        avg_val_loss = val_loss_total / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = model.state_dict().copy()

        scheduler.step()
        print(f"  Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {val_f1:.4f}"
              + (" ← Best" if val_f1 == best_val_f1 else ""))

    # Load best weights
    if best_state:
        model.load_state_dict(best_state)

    print(f"\nBest Validation F1: {best_val_f1:.4f}")
    return model, vocab, history


# =============================================================================
# SECTION 4: TRANSFORMER MODEL FINE-TUNING
# =============================================================================

# Recommended model options:
TRANSFORMER_MODELS = {
    'afriberta'   : 'castorini/afriberta_large',    # Best for Igbo/African languages
    'xlm_roberta' : 'xlm-roberta-base',              # Strong multilingual baseline
    'mbert'       : 'bert-base-multilingual-cased',  # Classic multilingual BERT
    'afro_xlmr'   : 'Davlan/afro-xlmr-base',        # Afro-centric XLM-R
}


class TransformerClassifier(nn.Module):
    """
    Fine-tuneable transformer for hate speech classification.
    Adds a classification head on top of any HuggingFace encoder.
    """

    def __init__(self, model_name, num_labels=2, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        from transformers import AutoModel
        self.encoder  = AutoModel.from_pretrained(model_name)
        hidden_size   = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return loss, logits


def get_tokenizer(model_name):
    """Load tokenizer for a given model."""
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {model_name}")
    return AutoTokenizer.from_pretrained(model_name)


def train_transformer(
    X_train, y_train, X_val, y_val,
    model_name='castorini/afriberta_large',
    epochs=5, batch_size=16, lr=2e-5, max_length=128,
    output_dir="models/transformer"
):
    """
    Fine-tune a transformer model on the hate speech dataset.
    Uses AdamW with linear warmup schedule (standard for BERT fine-tuning).
    """
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    print(f"\nFine-tuning Transformer: {model_name}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Datasets
    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, max_length)
    val_dataset   = HateSpeechDataset(X_val, y_val, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = TransformerClassifier(model_name, num_labels=2).to(DEVICE)

    # Optimizer: different LR for encoder vs classifier head
    optimizer_params = [
        {'params': model.encoder.parameters(),    'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr * 10}
    ]
    optimizer = AdamW(optimizer_params, weight_decay=0.01)

    # Linear warmup scheduler
    total_steps  = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_f1 = 0
    best_state  = None
    history     = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids   = batch['input_ids'].to(DEVICE)
            attn_mask   = batch['attention_mask'].to(DEVICE)
            labels      = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            loss, _ = model(input_ids, attn_mask, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds, val_true, val_loss_total = [], [], 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attn_mask = batch['attention_mask'].to(DEVICE)
                labels    = batch['label'].to(DEVICE)
                loss, logits = model(input_ids, attn_mask, labels)
                val_loss_total += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch['label'].numpy())

        val_f1       = f1_score(val_true, val_preds, average='weighted', zero_division=0)
        avg_val_loss = val_loss_total / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {val_f1:.4f}"
              + (" ← Best" if val_f1 == best_val_f1 else ""))

    # Load best checkpoint
    if best_state:
        model.load_state_dict(best_state)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to: {output_dir}/")
    print(f"Best Validation F1: {best_val_f1:.4f}")

    return model, tokenizer, history


# =============================================================================
# SECTION 5: EVALUATION ON TEST SET
# =============================================================================

def evaluate_transformer(model, tokenizer, X_test, y_test,
                          batch_size=16, max_length=128):
    """Evaluate transformer model on test set."""
    test_dataset = HateSpeechDataset(X_test, y_test, tokenizer, max_length)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attn_mask = batch['attention_mask'].to(DEVICE)
            _, logits = model(input_ids, attn_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(batch['label'].numpy())

    print("\n" + "=" * 60)
    print("  TRANSFORMER TEST SET EVALUATION")
    print("=" * 60)
    print(f"  Accuracy  : {accuracy_score(all_true, all_preds):.4f}")
    print(f"  F1 (w-avg): {f1_score(all_true, all_preds, average='weighted'):.4f}")
    print(f"  F1 (hate) : {f1_score(all_true, all_preds, pos_label=1, zero_division=0):.4f}")
    print("\n" + classification_report(
        all_true, all_preds,
        target_names=['Not Hate Speech', 'Hate Speech'],
        digits=4
    ))

    return all_preds, all_true


# =============================================================================
# SECTION 6: INFERENCE
# =============================================================================

def predict_transformer(text, model, tokenizer, max_length=128):
    """Predict hate speech for a single text using transformer."""
    model.eval()
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids   = encoding['input_ids'].to(DEVICE)
    attn_mask   = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        _, logits = model(input_ids, attn_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred  = int(np.argmax(probs))

    return {
        'prediction' : pred,
        'label'      : 'HATE SPEECH' if pred == 1 else 'NOT HATE SPEECH',
        'confidence' : round(float(probs[pred]), 4),
        'prob_hate'  : round(float(probs[1]), 4),
        'prob_safe'  : round(float(probs[0]), 4),
    }


# =============================================================================
# SECTION 7: MODEL COMPARISON TABLE
# =============================================================================

def print_model_comparison(results_list):
    """
    Print a comparison table of all models.
    results_list: list of dicts with keys: model, accuracy, f1_weighted, f1_hate
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values('F1 Weighted', ascending=False)

    print("\n" + "=" * 70)
    print("  FULL MODEL COMPARISON (Classical + Deep Learning + Transformers)")
    print("=" * 70)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\n★ Best Model: {best['Model']} with F1={best['F1 Weighted']:.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("MODULE 3: Deep Learning + Transformer Models")
    print("=" * 60)

    # Load data
    try:
        train_df = pd.read_csv("data/train.csv")
        val_df   = pd.read_csv("data/val.csv")
        test_df  = pd.read_csv("data/test.csv")
        X_train = train_df['text'].values
        y_train = train_df['label'].values
        X_val   = val_df['text'].values
        y_val   = val_df['label'].values
        X_test  = test_df['text'].values
        y_test  = test_df['label'].values
    except FileNotFoundError:
        print("Run Module 1 first to generate data files.")
        exit(1)

    # --- Option A: Train BiLSTM (no GPU required) ---
    print("\nOption A: BiLSTM (lightweight, runs on CPU)")
    bilstm_model, vocab, history = train_bilstm(
        X_train, y_train, X_val, y_val, epochs=10
    )

    # --- Option B: Fine-tune Transformer (GPU recommended) ---
    print("\nOption B: AfriBERTa Transformer (GPU recommended)")
    print("NOTE: Uncomment the block below and run with GPU for best results.")
    print("      AfriBERTa is pre-trained on Igbo and 10 other African languages.")
    print()
    print("  # model, tokenizer, history = train_transformer(")
    print("  #     X_train, y_train, X_val, y_val,")
    print("  #     model_name='castorini/afriberta_large',")
    print("  #     epochs=5, batch_size=16, lr=2e-5")
    print("  # )")
    print()
    print("  # Alternative (stronger multilingual baseline):")
    print("  # model_name='xlm-roberta-base'")

    print("\nModule 3 complete.")