"""
=============================================================================
PIPELINE.PY  —  FULL END-TO-END RUNNER  (FIXED FOR WINDOWS)
Project: Automatic Hate Speech Detection in English-Igbo Code-Mixed Data
=============================================================================
FIX APPLIED:
  Python cannot use import_module() on files whose names start with numbers
  (e.g. 01_data_collection_preprocessing.py).
  This version uses importlib.util.spec_from_file_location() instead,
  which loads any .py file directly by its path — works on Windows & Mac.

Usage:
  python pipeline.py               # Run everything
  python pipeline.py --skip-deep   # Skip BiLSTM (faster, recommended first run)
=============================================================================
"""

import os
import sys
import json
import argparse
import warnings
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')

# =============================================================================
# FIX: Load modules by FILE PATH (not by module name)
# This works even when filenames start with numbers
# =============================================================================

# Folder where all your .py files live
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

def load_module(filename, alias):
    """
    Load any Python file by its actual path on disk.
    Works on Windows, Mac, Linux.
    Works even if filename starts with a number.
    """
    filepath = os.path.join(SRC_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n\nFile not found: {filepath}"
            f"\nMake sure all .py files are in the same folder as pipeline.py"
            f"\nExpected folder: {SRC_DIR}"
        )

    spec   = importlib.util.spec_from_file_location(alias, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# Lazy loaders — each module is only loaded when actually needed
def get_module1():
    return load_module('data-collection.py',  'data_preprocessing')

def get_module2():
    return load_module('feature-extraction.py', 'classical_ml')

def get_module3():
    return load_module('deep-learning.py',      'deep_learning')

def get_module4():
    return load_module('evaluation.py',        'evaluation')


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data
    'test_size'     : 0.2,
    'val_size'      : 0.1,
    'random_state'  : 42,

    # Classical ML
    'tfidf_type'    : 'combined',

    # BiLSTM
    'bilstm_epochs' : 15,
    'bilstm_batch'  : 8,
    'bilstm_lr'     : 1e-3,

    # Transformer (if enabled with --transformer flag)
    'transformer_model'  : 'castorini/afriberta_large',
    'transformer_epochs' : 5,
    'transformer_batch'  : 8,
    'transformer_lr'     : 2e-5,

    # Output directories
    'output_dir'  : 'outputs',
    'figures_dir' : 'outputs/figures',
    'models_dir'  : 'models',
    'data_dir'    : 'data',
}


# =============================================================================
# STEP 1: DATA
# =============================================================================

def step1_data(cfg):
    print("\n" + "█" * 60)
    print("  STEP 1: DATA COLLECTION & PREPROCESSING")
    print("█" * 60)

    m1 = get_module1()

    df = m1.create_dataset()
    m1.analyze_dataset(df)

    # Dataset visualization
    try:
        m4 = get_module4()
        os.makedirs(cfg['figures_dir'], exist_ok=True)
        m4.plot_dataset_stats(df, save_path=f"{cfg['figures_dir']}/01_dataset_stats.png")
    except Exception as e:
        print(f"  [Warning] Could not plot dataset stats: {e}")

    # Save train/val/test splits
    splits = m1.save_processed_data(df, output_dir=cfg['data_dir'])
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    print(f"\n✓ Step 1 complete.")
    return df, X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# STEP 2: CLASSICAL ML MODELS
# =============================================================================

def step2_classical(cfg, X_train, y_train, X_test, y_test):
    print("\n" + "█" * 60)
    print("  STEP 2: CLASSICAL ML MODELS")
    print("█" * 60)

    m2 = get_module2()
    m4 = get_module4()

    summary_df, vectorizer, trained_models = m2.run_all_classical_models(
        X_train, y_train, X_test, y_test,
        vectorizer_type=cfg['tfidf_type']
    )

    # Save model comparison chart
    try:
        m4.plot_model_comparison(
            summary_df,
            save_path=f"{cfg['figures_dir']}/02_classical_model_comparison.png"
        )
    except Exception as e:
        print(f"  [Warning] Could not plot comparison: {e}")

    # Evaluate best model (Logistic Regression)
    best_model = trained_models['Logistic Regression']
    feat       = vectorizer.fit_transform(X_train)
    best_model.fit(feat, y_train)

    X_test_feat = vectorizer.transform(X_test)
    y_pred      = best_model.predict(X_test_feat)
    y_prob      = None
    if hasattr(best_model, 'predict_proba'):
        y_prob = best_model.predict_proba(X_test_feat)[:, 1]

    # Confusion matrix
    try:
        m4.plot_confusion_matrix(
            y_test, y_pred, "Logistic Regression",
            save_path=f"{cfg['figures_dir']}/03_confusion_matrix_lr.png"
        )
    except Exception as e:
        print(f"  [Warning] Could not plot confusion matrix: {e}")

    # Error analysis
    m4.analyze_errors(X_test, y_test, y_pred)

    # Save models
    os.makedirs(f"{cfg['models_dir']}/classical", exist_ok=True)
    m2.save_models(trained_models, vectorizer,
                   output_dir=f"{cfg['models_dir']}/classical")

    metrics = m4.compute_all_metrics(
        y_test, y_pred, y_prob, "Logistic Regression (TF-IDF)"
    )

    print(f"\n✓ Step 2 complete. Best F1: {metrics['f1_weighted']:.4f}")
    return summary_df, vectorizer, trained_models, metrics


# =============================================================================
# STEP 3: BILSTM (optional — skip with --skip-deep)
# =============================================================================

def step3_deep(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "█" * 60)
    print("  STEP 3: BILSTM + ATTENTION")
    print("█" * 60)

    m3 = get_module3()
    m4 = get_module4()

    model, vocab, history = m3.train_bilstm(
        X_train, y_train, X_val, y_val,
        epochs=cfg['bilstm_epochs'],
        batch_size=cfg['bilstm_batch'],
        lr=cfg['bilstm_lr']
    )

    # Training history plot
    try:
        m4.plot_training_history(
            history, "BiLSTM",
            save_path=f"{cfg['figures_dir']}/06_bilstm_training.png"
        )
    except Exception as e:
        print(f"  [Warning] Could not plot training history: {e}")

    # Evaluate on test set
    import torch
    from torch.utils.data import DataLoader

    device       = m3.DEVICE
    test_dataset = m3.BiLSTMDataset(X_test, y_test, vocab)
    test_loader  = DataLoader(test_dataset, batch_size=16)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            inp    = batch['input_ids'].to(device)
            logits = model(inp)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    y_pred_bilstm = np.array(all_preds)

    m4.print_full_report(y_test, y_pred_bilstm, "BiLSTM (Attention)")

    try:
        m4.plot_confusion_matrix(
            y_test, y_pred_bilstm, "BiLSTM",
            save_path=f"{cfg['figures_dir']}/07_confusion_matrix_bilstm.png"
        )
    except Exception as e:
        print(f"  [Warning] Could not plot BiLSTM confusion matrix: {e}")

    metrics = m4.compute_all_metrics(y_test, y_pred_bilstm, None, "BiLSTM")

    # Save model
    import pickle
    os.makedirs(f"{cfg['models_dir']}/bilstm", exist_ok=True)
    with open(f"{cfg['models_dir']}/bilstm/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    torch.save(model.state_dict(), f"{cfg['models_dir']}/bilstm/model.pt")

    print(f"\n✓ Step 3 complete. BiLSTM F1: {metrics['f1_weighted']:.4f}")
    return model, vocab, metrics


# =============================================================================
# STEP 4: FINAL REPORT
# =============================================================================

def step4_report(cfg, all_metrics, df):
    print("\n" + "█" * 60)
    print("  STEP 4: FINAL REPORT")
    print("█" * 60)

    os.makedirs(cfg['output_dir'], exist_ok=True)

    # Comparison table
    comparison = [{
        'Model'       : m['model'],
        'Accuracy'    : m['accuracy'],
        'F1 Weighted' : m['f1_weighted'],
        'F1 (Hate)'   : m['f1_hate'],
        'F1 (Safe)'   : m['f1_safe'],
    } for m in all_metrics]

    comp_df = pd.DataFrame(comparison).sort_values('F1 Weighted', ascending=False)

    try:
        m4 = get_module4()
        m4.plot_model_comparison(
            comp_df.rename(columns={'F1 Weighted': 'F1 (Weighted)'}),
            save_path=f"{cfg['figures_dir']}/08_final_comparison.png"
        )
    except Exception as e:
        print(f"  [Warning] Could not plot final comparison: {e}")

    # Save JSON report
    report = {
        'project'    : 'Hate Speech Detection — English-Igbo Code-Mixed',
        'timestamp'  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset'    : {
            'total' : int(len(df)),
            'hate'  : int(df['label'].sum()),
            'safe'  : int((df['label'] == 0).sum()),
        },
        'models'     : all_metrics,
        'best_model' : comp_df.iloc[0]['Model'],
        'best_f1'    : float(comp_df.iloc[0]['F1 Weighted']),
    }

    with open(f"{cfg['output_dir']}/results_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(comp_df.to_string(index=False))
    print(f"\n★ Best Model : {report['best_model']}")
    print(f"★ Best F1    : {report['best_f1']:.4f}")
    print(f"\nReport saved : {cfg['output_dir']}/results_report.json")
    print(f"Figures saved: {cfg['figures_dir']}/")

    return report


# =============================================================================
# INFERENCE DEMO
# =============================================================================

def demo_inference(vectorizer, model, preprocessor):
    test_texts = [
        "Chukwu gozie gị! I love my Igbo culture so much.",
        "Kill those useless people! Gbuo ha niile!",
        "Nna, obi ụtọ nke ukwuu today at work!",
        "These dirty people should be removed from this country!",
    ]

    print("\n" + "=" * 60)
    print("  INFERENCE DEMO")
    print("=" * 60)

    m2 = get_module2()
    for text in test_texts:
        clean = preprocessor.preprocess(text)
        pred, label, conf = m2.predict_single(clean, model, vectorizer)
        icon = "⚠️ " if pred == 1 else "✅"
        conf_str = f"(conf: {conf:.2f})" if conf is not None else ""
        print(f"\n{icon} {label} {conf_str}")
        print(f"   Text: {text[:75]}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hate Speech Detection Pipeline — English-Igbo"
    )
    parser.add_argument(
        '--skip-deep', action='store_true',
        help='Skip BiLSTM training (faster — good for first run)'
    )
    parser.add_argument(
        '--transformer', action='store_true',
        help='Also run AfriBERTa transformer (requires GPU)'
    )
    args = parser.parse_args()

    print("\n" + "█" * 65)
    print("  HATE SPEECH DETECTION — ENGLISH-IGBO CODE-MIXED DATA")
    print("  Data → Features → Models → Evaluation")
    print("█" * 65)

    cfg = CONFIG

    # Make all output dirs
    for d in [cfg['output_dir'], cfg['figures_dir'], cfg['models_dir']]:
        os.makedirs(d, exist_ok=True)

    all_metrics = []

    # ── Step 1: Data ──────────────────────────────────────────────────────
    df, X_train, X_val, X_test, y_train, y_val, y_test = step1_data(cfg)

    # Combine train + val for classical ML
    X_train_full = np.concatenate([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # ── Step 2: Classical ML ──────────────────────────────────────────────
    summary_df, vectorizer, trained_models, metrics_lr = step2_classical(
        cfg, X_train_full, y_train_full, X_test, y_test
    )
    all_metrics.append(metrics_lr)

    # Add other classical model metrics
    try:
        m4   = get_module4()
        m2   = get_module2()
        feat = vectorizer.fit_transform(X_train_full)
        feat_test = vectorizer.transform(X_test)
        for name, model in trained_models.items():
            if name != 'Logistic Regression':
                try:
                    model.fit(feat, y_train_full)
                    y_p = model.predict(feat_test)
                    m   = m4.compute_all_metrics(y_test, y_p, None, name)
                    all_metrics.append(m)
                except Exception as e:
                    print(f"  Skipped {name}: {e}")
    except Exception as e:
        print(f"  [Warning] Could not evaluate all classical models: {e}")

    # ── Step 3: BiLSTM (optional) ─────────────────────────────────────────
    if not args.skip_deep:
        try:
            _, _, metrics_bilstm = step3_deep(
                cfg, X_train, y_train, X_val, y_val, X_test, y_test
            )
            all_metrics.append(metrics_bilstm)
        except Exception as e:
            print(f"  [Warning] BiLSTM failed: {e}")
            print("  Try running with --skip-deep flag")
    else:
        print("\n[Skipped BiLSTM — remove --skip-deep to include it]")

    # ── Step 4: Report ────────────────────────────────────────────────────
    step4_report(cfg, all_metrics, df)

    # ── Demo inference ────────────────────────────────────────────────────
    try:
        m1 = get_module1()
        preprocessor = m1.CodeMixPreprocessor()
        demo_inference(vectorizer, trained_models['Logistic Regression'], preprocessor)
    except Exception as e:
        print(f"  [Warning] Demo skipped: {e}")

    print("\n" + "█" * 65)
    print("  PIPELINE COMPLETE!")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 65)


if __name__ == "__main__":
    main()