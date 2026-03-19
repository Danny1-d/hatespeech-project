"""
=============================================================================
MODULE 2: FEATURE EXTRACTION + CLASSICAL ML MODELS
Project: Automatic Hate Speech Detection in English-Igbo Code-Mixed Data
=============================================================================
This module covers:
  1. TF-IDF feature extraction (unigrams + bigrams + character n-grams)
  2. Classical ML models: Naive Bayes, SVM, Logistic Regression, Random Forest
  3. Cross-validation and hyperparameter tuning
  4. Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
=============================================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: FEATURE EXTRACTORS
# =============================================================================

def build_tfidf_word(ngram_range=(1, 2), max_features=10000):
    """
    Word-level TF-IDF with unigrams and bigrams.
    Good at capturing word sequences like 'kill all' or 'dirty people'.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,         # Replace TF with 1 + log(TF) - reduces impact of frequent words
        strip_accents=None,        # IMPORTANT: Keep Igbo diacritics!
        analyzer='word',
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-ZọỌụỤịỊṅṄńǹáàéèíìóòúù']{2,}\b"
    )


def build_tfidf_char(ngram_range=(2, 5), max_features=8000):
    """
    Character-level TF-IDF.
    Excellent for handling morphological variations in Igbo
    and catching misspellings/slang in English.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents=None,
        analyzer='char_wb',  # char_wb pads word boundaries
        min_df=1
    )


def build_combined_tfidf():
    """
    Returns both word and char vectorizers for combined feature space.
    The combination generally outperforms either alone.
    """
    word_vec = build_tfidf_word(ngram_range=(1, 2))
    char_vec  = build_tfidf_char(ngram_range=(2, 4))
    return word_vec, char_vec


from scipy.sparse import hstack

class CombinedTFIDF:
    """Combines word-level and character-level TF-IDF features."""

    def __init__(self):
        self.word_vec = build_tfidf_word(ngram_range=(1, 2))
        self.char_vec = build_tfidf_char(ngram_range=(2, 4))

    def fit(self, texts):
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        return self

    def transform(self, texts):
        word_feat = self.word_vec.transform(texts)
        char_feat  = self.char_vec.transform(texts)
        return hstack([word_feat, char_feat])

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


# =============================================================================
# SECTION 2: CLASSICAL ML MODELS
# =============================================================================

def get_models():
    """
    Returns a dictionary of classical ML models to evaluate.
    Each entry is: model_name -> sklearn estimator
    """
    return {
        "Naive Bayes (Complement)": ComplementNB(alpha=0.5),

        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),

        "Linear SVM": LinearSVC(
            C=1.0,
            max_iter=2000,
            class_weight='balanced',
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
    }


# =============================================================================
# SECTION 3: TRAINING AND EVALUATION
# =============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Train a model and compute all evaluation metrics.
    Returns a results dictionary.
    """
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_hate   = f1_score(y_test, y_pred, pos_label=1, zero_division=0)  # F1 for hate class

    # Cross-validation F1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')

    results = {
        'model_name'    : model_name,
        'accuracy'      : round(acc, 4),
        'precision'     : round(precision, 4),
        'recall'        : round(recall, 4),
        'f1_weighted'   : round(f1, 4),
        'f1_hate_class' : round(f1_hate, 4),
        'cv_f1_mean'    : round(cv_f1.mean(), 4),
        'cv_f1_std'     : round(cv_f1.std(), 4),
        'y_pred'        : y_pred,
        'model'         : model
    }

    return results


def print_evaluation(results, y_test):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  MODEL: {results['model_name']}")
    print(f"{'='*60}")
    print(f"  Accuracy           : {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision (w-avg)  : {results['precision']:.4f}")
    print(f"  Recall    (w-avg)  : {results['recall']:.4f}")
    print(f"  F1-Score  (w-avg)  : {results['f1_weighted']:.4f}")
    print(f"  F1-Score  (hate)   : {results['f1_hate_class']:.4f}")
    print(f"  5-Fold CV F1       : {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, results['y_pred'],
        target_names=['Not Hate Speech', 'Hate Speech'],
        digits=4
    ))

    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, results['y_pred'])
    print(f"  {'':>20} Predicted: No  Predicted: Yes")
    print(f"  {'Actual: No':>20} {cm[0][0]:<16} {cm[0][1]}")
    print(f"  {'Actual: Yes':>20} {cm[1][0]:<16} {cm[1][1]}")


def run_all_classical_models(X_train, y_train, X_test, y_test, vectorizer_type='combined'):
    """
    Run all classical models with specified vectorizer.
    Returns a summary DataFrame of all results.
    """
    print(f"\nRunning Classical ML Models with {vectorizer_type.upper()} TF-IDF features...")
    print("=" * 60)

    # Feature extraction
    if vectorizer_type == 'combined':
        vectorizer = CombinedTFIDF()
    elif vectorizer_type == 'word':
        vectorizer = build_tfidf_word()
    else:
        vectorizer = build_tfidf_char()

    X_train_feat = vectorizer.fit_transform(X_train)
    X_test_feat  = vectorizer.transform(X_test)

    print(f"Feature shape - Train: {X_train_feat.shape}, Test: {X_test_feat.shape}")

    models = get_models()
    all_results = []

    for name, model in models.items():
        print(f"\nTraining: {name}...")
        results = evaluate_model(model, X_train_feat, y_train, X_test_feat, y_test, name)
        print_evaluation(results, y_test)
        all_results.append({
            'Model'          : results['model_name'],
            'Accuracy'       : results['accuracy'],
            'Precision'      : results['precision'],
            'Recall'         : results['recall'],
            'F1 (Weighted)'  : results['f1_weighted'],
            'F1 (Hate)'      : results['f1_hate_class'],
            'CV F1 Mean'     : results['cv_f1_mean'],
            'CV F1 Std'      : results['cv_f1_std'],
        })

    summary_df = pd.DataFrame(all_results).sort_values('F1 (Weighted)', ascending=False)

    print("\n" + "=" * 60)
    print("  CLASSICAL MODELS COMPARISON SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    return summary_df, vectorizer, models


# =============================================================================
# SECTION 4: HYPERPARAMETER TUNING (Best Model)
# =============================================================================

def tune_best_model(X_train_feat, y_train, X_test_feat, y_test):
    """
    Grid search hyperparameter tuning for Logistic Regression.
    (Typically best performing classical model for text classification.)
    """
    print("\nTuning Logistic Regression with GridSearchCV...")

    param_grid = {
        'C'            : [0.01, 0.1, 1.0, 5.0, 10.0],
        'solver'       : ['lbfgs', 'liblinear'],
        'class_weight' : ['balanced', None],
        'max_iter'     : [500, 1000],
    }

    lr = LogisticRegression(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        lr, param_grid, cv=cv,
        scoring='f1_weighted', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train_feat, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1     : {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_feat)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Test F1 (tuned): {f1:.4f}")

    return best_model


# =============================================================================
# SECTION 5: SAVE MODELS
# =============================================================================

def save_models(models_dict, vectorizer, output_dir="models/classical"):
    """Save trained models and vectorizer to disk."""
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved: {output_dir}/vectorizer.pkl")

    for name, model in models_dict.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        filepath = f"{output_dir}/{safe_name}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved: {filepath}")


def load_model(model_path, vectorizer_path):
    """Load a saved model and vectorizer."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# =============================================================================
# SECTION 6: INFERENCE / PREDICTION
# =============================================================================

def predict_single(text, model, vectorizer, preprocessor=None):
    """
    Predict hate speech for a single text input.
    Returns: label (0/1), confidence score
    """
    if preprocessor:
        text = preprocessor.preprocess(text)

    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    # Get probability if model supports it
    if hasattr(model, 'predict_proba'):
        confidence = model.predict_proba(features)[0][prediction]
    elif hasattr(model, 'decision_function'):
        raw_score = model.decision_function(features)[0]
        confidence = 1 / (1 + np.exp(-raw_score))  # Sigmoid
    else:
        confidence = None

    label = 'HATE SPEECH' if prediction == 1 else 'NOT HATE SPEECH'
    return prediction, label, confidence


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Load data from Module 1 output
    try:
        train_df = pd.read_csv("data/train.csv")
        test_df  = pd.read_csv("data/test.csv")
        X_train, y_train = train_df['text'].values, train_df['label'].values
        X_test,  y_test  = test_df['text'].values,  test_df['label'].values
    except FileNotFoundError:
        print("Data files not found. Run Module 1 first to generate them.")
        print("Using sample data instead...\n")

        # Fallback: run module 1 inline
        import sys
        sys.path.insert(0, '.')
        from importlib import import_module
        m1 = import_module('01_data_collection_preprocessing')
        df = m1.create_dataset()
        data = m1.save_processed_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = data

    # Run all classical models
    summary_df, vectorizer, trained_models = run_all_classical_models(
        X_train, y_train, X_test, y_test, vectorizer_type='combined'
    )

    # Test prediction
    test_texts = [
        "Chukwu gozie gị! I love my Igbo culture so much.",
        "Kill all those dirty people from that tribe! Gbuo ha!",
        "Nna, the market was full of life today. Obi ụtọ!",
        "These stupid people should leave our country. We hate them all!",
    ]

    print("\n" + "=" * 60)
    print("  SAMPLE PREDICTIONS (Best Model - Logistic Regression)")
    print("=" * 60)

    # Use LR as default best model
    lr_model = trained_models["Logistic Regression"]
    lr_model.fit(vectorizer.fit_transform(X_train), y_train)

    for text in test_texts:
        feat = vectorizer.transform([text])
        pred = lr_model.predict(feat)[0]
        label = "⚠️  HATE SPEECH" if pred == 1 else "✅ NOT HATE SPEECH"
        print(f"\nText  : {text[:70]}")
        print(f"Result: {label}")

    print("\nModule 2 complete.")