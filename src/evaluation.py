"""
=============================================================================
MODULE 4: COMPREHENSIVE EVALUATION, VISUALIZATION & ERROR ANALYSIS
Project: Automatic Hate Speech Detection in English-Igbo Code-Mixed Data
=============================================================================
This module covers:
  1. Full evaluation metrics suite
  2. Confusion matrix visualization
  3. ROC and Precision-Recall curves
  4. Error analysis (false positives/negatives)
  5. Feature importance for classical models
  6. Bias analysis (fairness across groups)
  7. Learning curves
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score
)
from sklearn.model_selection import learning_curve, StratifiedKFold

plt.style.use('seaborn-v0_8')
COLORS = ['#2E86AB', '#E84855', '#3BB273', '#FFC857', '#6A4C93']

os.makedirs("outputs/figures", exist_ok=True)


# =============================================================================
# SECTION 1: METRICS COMPUTATION
# =============================================================================

def compute_all_metrics(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Compute comprehensive set of evaluation metrics.
    y_prob: probability of positive class (optional, for AUC)
    """
    metrics = {
        'model'     : model_name,
        'accuracy'  : round(accuracy_score(y_true, y_pred), 4),
        'precision_macro' : round(float(np.mean([
            sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c) /
            max(sum(1 for p in y_pred if p == c), 1)
            for c in [0, 1]
        ])), 4),
        'recall_macro' : round(float(np.mean([
            sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c) /
            max(sum(1 for t in y_true if t == c), 1)
            for c in [0, 1]
        ])), 4),
        'f1_weighted' : round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'f1_macro'    : round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'f1_hate'     : round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        'f1_safe'     : round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
    }

    if y_prob is not None:
        try:
            metrics['roc_auc'] = round(roc_auc_score(y_true, y_prob), 4)
            metrics['avg_precision'] = round(average_precision_score(y_true, y_prob), 4)
        except Exception:
            metrics['roc_auc'] = None
            metrics['avg_precision'] = None

    return metrics


def print_full_report(y_true, y_pred, model_name="Model"):
    """Print a full classification report."""
    print(f"\n{'='*65}")
    print(f"  EVALUATION REPORT: {model_name}")
    print(f"{'='*65}")
    print(classification_report(
        y_true, y_pred,
        target_names=['Not Hate Speech (0)', 'Hate Speech (1)'],
        digits=4
    ))


# =============================================================================
# SECTION 2: CONFUSION MATRIX PLOT
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """Plot and optionally save a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ['Confusion Matrix (Counts)', 'Confusion Matrix (Normalized)'],
        ['d', '.2%']
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=['Not Hate', 'Hate Speech'],
            yticklabels=['Not Hate', 'Hate Speech'],
            ax=ax, linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8}
        )
        ax.set_title(f"{model_name}\n{title}", fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Actual Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix: {save_path}")
    plt.close()


# =============================================================================
# SECTION 3: ROC CURVE
# =============================================================================

def plot_roc_curves(results_list, save_path=None):
    """
    Plot ROC curves for multiple models on the same axes.
    results_list: list of dicts with keys: name, y_true, y_prob
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    ax = axes[0]
    for i, result in enumerate(results_list):
        y_true = result['y_true']
        y_prob = result.get('y_prob')
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)],
                lw=2, label=f"{result['name']} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
    ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves – Model Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Precision-Recall Curve
    ax = axes[1]
    for i, result in enumerate(results_list):
        y_true = result['y_true']
        y_prob = result.get('y_prob')
        if y_prob is None:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, color=COLORS[i % len(COLORS)],
                lw=2, label=f"{result['name']} (AP={ap:.3f})")

    baseline = sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1,
               label=f'Baseline ({baseline:.2f})')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC/PR curves: {save_path}")
    plt.close()


# =============================================================================
# SECTION 4: MODEL COMPARISON BAR CHART
# =============================================================================

def plot_model_comparison(summary_df, save_path=None):
    """
    Bar chart comparing all models across metrics.
    summary_df: DataFrame with columns: Model, Accuracy, F1 (Weighted), F1 (Hate), ...
    """
    metrics = ['Accuracy', 'F1 (Weighted)', 'F1 (Hate)']
    available = [m for m in metrics if m in summary_df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(summary_df))
    width = 0.25

    for i, metric in enumerate(available):
        bars = ax.bar(
            x + i * width - width,
            summary_df[metric],
            width=width,
            label=metric,
            color=COLORS[i],
            alpha=0.87,
            edgecolor='white'
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Model'], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title('Model Performance Comparison\nEnglish-Igbo Code-Mixed Hate Speech Detection',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved model comparison: {save_path}")
    plt.close()


# =============================================================================
# SECTION 5: ERROR ANALYSIS
# =============================================================================

def analyze_errors(X_test, y_true, y_pred, n=10):
    """
    Analyze false positives and false negatives.
    Helps identify patterns in model errors.
    """
    results = []
    for i, (text, true, pred) in enumerate(zip(X_test, y_true, y_pred)):
        if true != pred:
            error_type = 'False Positive' if pred == 1 else 'False Negative'
            results.append({
                'text'       : text,
                'true_label' : 'Hate' if true == 1 else 'Safe',
                'pred_label' : 'Hate' if pred == 1 else 'Safe',
                'error_type' : error_type
            })

    error_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("  ERROR ANALYSIS")
    print("=" * 65)
    total_errors = len(error_df)
    print(f"Total errors: {total_errors} / {len(y_true)} "
          f"({total_errors/len(y_true)*100:.1f}%)")

    if len(error_df) == 0:
        print("No errors! Perfect prediction.")
        return error_df

    fp = error_df[error_df['error_type'] == 'False Positive']
    fn = error_df[error_df['error_type'] == 'False Negative']

    print(f"\nFalse Positives (predicted hate, was safe): {len(fp)}")
    for _, row in fp.head(min(n, len(fp))).iterrows():
        print(f"  → {row['text'][:80]}")

    print(f"\nFalse Negatives (missed hate speech): {len(fn)}")
    for _, row in fn.head(min(n, len(fn))).iterrows():
        print(f"  → {row['text'][:80]}")

    return error_df


# =============================================================================
# SECTION 6: FEATURE IMPORTANCE (for TF-IDF + classical models)
# =============================================================================

def plot_feature_importance(model, vectorizer, top_n=20, save_path=None):
    """
    Plot most important features for logistic regression / linear SVM.
    Shows which words/n-grams are most associated with hate speech.
    """
    try:
        # Get feature names
        if hasattr(vectorizer, 'word_vec'):
            feature_names = (
                list(vectorizer.word_vec.get_feature_names_out()) +
                list(vectorizer.char_vec.get_feature_names_out())
            )
        else:
            feature_names = list(vectorizer.get_feature_names_out())

        # Get coefficients
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        else:
            print("Model does not support feature importance visualization.")
            return

        # Top features for each class
        top_hate_idx = np.argsort(coef)[-top_n:][::-1]
        top_safe_idx = np.argsort(coef)[:top_n]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, indices, title, color in [
            (axes[0], top_hate_idx, 'Top Features → HATE SPEECH',    '#E84855'),
            (axes[1], top_safe_idx, 'Top Features → NOT HATE SPEECH', '#2E86AB')
        ]:
            feats  = [feature_names[i][:25] for i in indices]
            scores = [abs(coef[i]) for i in indices]

            bars = ax.barh(feats[::-1], scores[::-1], color=color, alpha=0.8)
            ax.set_title(title, fontsize=12, fontweight='bold', color=color)
            ax.set_xlabel('|Coefficient Weight|', fontsize=10)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle('Feature Importance: Logistic Regression\nEnglish-Igbo Hate Speech Detection',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved feature importance: {save_path}")
        plt.close()

    except Exception as e:
        print(f"Feature importance plot error: {e}")


# =============================================================================
# SECTION 7: LEARNING CURVES
# =============================================================================

def plot_learning_curves(model, X, y, save_path=None):
    """
    Plot learning curves to diagnose overfitting/underfitting.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer as TV

    pipe = Pipeline([
        ('tfidf', TV(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ('clf', model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=cv, scoring='f1_weighted',
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1),
                    alpha=0.15, color=COLORS[0])
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1),
                    alpha=0.15, color=COLORS[1])

    ax.plot(train_sizes, train_scores.mean(1), 'o-',
            color=COLORS[0], lw=2, label='Training Score')
    ax.plot(train_sizes, val_scores.mean(1), 's-',
            color=COLORS[1], lw=2, label='Validation Score (CV)')

    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('F1 Score (Weighted)', fontsize=11)
    ax.set_title('Learning Curves\nEnglish-Igbo Hate Speech Detection',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved learning curves: {save_path}")
    plt.close()


# =============================================================================
# SECTION 8: TRAINING HISTORY PLOT (Deep Learning)
# =============================================================================

def plot_training_history(history, model_name="Model", save_path=None):
    """Plot training/validation loss and F1 over epochs."""
    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', color=COLORS[0],
                 lw=2, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 's-', color=COLORS[1],
                 lw=2, label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} – Loss Curves', fontweight='bold')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # F1
    axes[1].plot(epochs, history['val_f1'], 's-', color=COLORS[2],
                 lw=2, label='Val F1 (Weighted)')
    best_epoch = int(np.argmax(history['val_f1'])) + 1
    best_f1    = max(history['val_f1'])
    axes[1].axvline(x=best_epoch, color=COLORS[3], linestyle='--',
                    lw=1.5, label=f'Best Epoch={best_epoch} (F1={best_f1:.4f})')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1 Score')
    axes[1].set_title(f'{model_name} – F1 Score', fontweight='bold')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history: {save_path}")
    plt.close()


# =============================================================================
# SECTION 9: DATASET STATISTICS VISUALIZATION
# =============================================================================

def plot_dataset_stats(df, save_path=None):
    """Visualize dataset label distribution and token length stats."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Label distribution pie
    label_counts = df['label'].value_counts()
    axes[0].pie(
        label_counts,
        labels=['Not Hate Speech', 'Hate Speech'],
        colors=[COLORS[0], COLORS[1]],
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
    )
    axes[0].set_title('Label Distribution', fontweight='bold')

    # Token length histogram
    axes[1].hist(df['token_count'], bins=15, color=COLORS[2], alpha=0.8,
                 edgecolor='white')
    axes[1].set_xlabel('Token Count')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Token Length Distribution', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Igbo ratio by label
    for i, (label, group) in enumerate(df.groupby('label')):
        name = 'Hate Speech' if label == 1 else 'Not Hate Speech'
        axes[2].hist(group['igbo_ratio'], bins=10, alpha=0.6,
                     color=COLORS[i], label=name, edgecolor='white')
    axes[2].set_xlabel('Igbo Language Ratio')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Igbo Ratio by Label', fontweight='bold')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle('Dataset Statistics: English-Igbo Code-Mixed Posts',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved dataset stats: {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("MODULE 4: Evaluation, Visualization & Error Analysis")
    print("=" * 60)

    # Demonstrate with dummy data (replace with real model outputs)
    np.random.seed(42)
    n = 30
    y_true = np.array([0]*15 + [1]*15)
    y_pred = y_true.copy()
    # Inject some errors
    flip_idx = np.random.choice(n, size=4, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    y_prob = np.where(y_true == 1, np.random.uniform(0.6, 1.0, n),
                                    np.random.uniform(0.0, 0.4, n))

    # Print report
    print_full_report(y_true, y_pred, "Demo Model")

    # Metrics
    metrics = compute_all_metrics(y_true, y_pred, y_prob, "Demo Model")
    print("\nComputed Metrics:", json.dumps(metrics, indent=2))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, "Demo Model",
                          save_path="outputs/figures/confusion_matrix.png")

    # ROC/PR
    plot_roc_curves(
        [{'name': 'Demo Model', 'y_true': y_true, 'y_prob': y_prob}],
        save_path="outputs/figures/roc_pr_curves.png"
    )

    print("\nAll figures saved to outputs/figures/")
    print("Module 4 complete.")