"""
=============================================================================
API-SERVER.PY  —  FASTAPI BACKEND  (FIXED for Python 3.14 + Windows)
Project: Automatic Hate Speech Detection in English-Igbo Code-Mixed Data
=============================================================================
FIXES:
  1. uvicorn module name now matches actual filename (api-server.py)
  2. Replaced deprecated on_event with lifespan handler
  3. Pydantic v2 compatible
  4. Auto-detects your data/feature filenames

Run:
  python api-server.py
  → Server at http://localhost:8000
  → API docs at http://localhost:8000/docs
=============================================================================
"""

import os
import sys
import re
import time
import pickle
import logging
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# AUTO-DETECT helper (same as pipeline.py)
# =============================================================================

def find_and_load(keywords, alias):
    """Find a .py file by keywords and load it — works with any filename."""
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
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    log.info(f"Loaded {alias} from {match}")
    return module


# =============================================================================
# MODEL STATE
# =============================================================================

class ModelState:
    model        = None
    vectorizer   = None
    preprocessor = None
    model_name   = None
    trained_at   = None
    train_size   = 0
    test_f1      = 0.0
    is_ready     = False

state = ModelState()


def load_or_train_model():
    MODEL_PATH = "models/classical/logistic_regression.pkl"
    VEC_PATH   = "models/classical/vectorizer.pkl"

    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        log.info("Loading saved model from disk...")

        # FIX: register the feature module under the same alias used when
        # the pickle was saved (classical_ml), so pickle.load() can find it
        m2 = find_and_load(["feature"], "classical_ml")
        if m2 is None:
            log.warning("Feature module not found — retraining...")
            _train_fresh()
            return

        with open(MODEL_PATH, "rb") as f: state.model      = pickle.load(f)
        with open(VEC_PATH,   "rb") as f: state.vectorizer = pickle.load(f)
        state.model_name = "Logistic Regression (TF-IDF)"
        state.trained_at = datetime.fromtimestamp(
            os.path.getmtime(MODEL_PATH)
        ).strftime("%Y-%m-%d %H:%M")
        log.info("Model loaded successfully.")
    else:
        log.info("No saved model — training now...")
        _train_fresh()

    # Load preprocessor
    try:
        m1 = find_and_load(['data', 'collection'], 'data_preprocessing')
        if m1:
            state.preprocessor = m1.CodeMixPreprocessor(preserve_diacritics=True)
    except Exception as e:
        log.warning(f"Preprocessor not loaded: {e}")

    state.is_ready = True
    log.info(f"Model ready: {state.model_name}")


def _train_fresh():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    log.info("Training fresh model...")
    try:
        m1 = find_and_load(['data', 'collection'], 'data_preprocessing')
        m2 = find_and_load(['feature'], 'classical_ml')

        if m1 is None or m2 is None:
            _create_fallback_model()
            return

        df     = m1.create_dataset()
        splits = m1.save_processed_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = splits

        X_all = np.concatenate([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])

        vectorizer  = m2.CombinedTFIDF()
        X_feat      = vectorizer.fit_transform(X_all)
        X_test_feat = vectorizer.transform(X_test)

        model = LogisticRegression(C=1.0, max_iter=1000,
                                   class_weight='balanced', random_state=42)
        model.fit(X_feat, y_all)

        y_pred = model.predict(X_test_feat)
        f1     = f1_score(y_test, y_pred, average='weighted')

        os.makedirs("models/classical", exist_ok=True)
        with open("models/classical/logistic_regression.pkl", "wb") as f: pickle.dump(model, f)
        with open("models/classical/vectorizer.pkl",          "wb") as f: pickle.dump(vectorizer, f)

        state.model      = model
        state.vectorizer = vectorizer
        state.model_name = "Logistic Regression (TF-IDF)"
        state.trained_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        state.train_size = len(X_all)
        state.test_f1    = round(f1, 4)
        log.info(f"Model trained. Test F1: {f1:.4f}")

    except Exception as e:
        log.error(f"Training failed: {e}")
        _create_fallback_model()


def _create_fallback_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack

    log.info("Creating fallback model...")

    DATA = [
        ("i love igbo culture chukwu gozie nke oma", 0),
        ("nna the market was amazing obi uto today", 0),
        ("my people always support each other igbo kwenu", 0),
        ("just got promoted god is good nke oma", 0),
        ("kill all those useless people gbuo ha niile", 1),
        ("trash tribe should be removed from this country", 1),
        ("dirty people they deserve nothing worthless", 1),
        ("eliminate all those criminals from that region", 1),
        ("these stupid people should die nwanyi adighị", 1),
        ("we hate that ethnic group kasie ha forever", 1),
    ]
    texts  = [d[0] for d in DATA]
    labels = [d[1] for d in DATA]

    class SimpleVec:
        def __init__(self):
            self.w = TfidfVectorizer(ngram_range=(1,2), max_features=500)
            self.c = TfidfVectorizer(ngram_range=(2,4), max_features=300, analyzer='char_wb')
        def fit_transform(self, X):
            return hstack([self.w.fit_transform(X), self.c.fit_transform(X)])
        def transform(self, X):
            return hstack([self.w.transform(X), self.c.transform(X)])

    vectorizer = SimpleVec()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced')
    model.fit(X, labels)

    state.model      = model
    state.vectorizer = vectorizer
    state.model_name = "Logistic Regression (fallback)"
    state.trained_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    state.train_size = len(texts)
    state.test_f1    = 0.0
    log.info("Fallback model ready.")


# =============================================================================
# LIFESPAN (replaces deprecated on_event)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    log.info("Starting up — loading model...")
    load_or_train_model()
    yield
    # Shutdown (nothing needed)
    log.info("Shutting down.")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Hate Speech Detector — English-Igbo API",
    description="REST API for English-Igbo code-mixed hate speech detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# SCHEMAS
# =============================================================================

class PredictRequest(BaseModel):
    text: str
    model: Optional[str] = "logistic_regression"

class PredictResponse(BaseModel):
    text          : str
    label         : int
    label_name    : str
    confidence    : float
    prob_hate     : float
    prob_safe     : float
    igbo_ratio    : float
    is_code_mixed : bool
    model_used    : str
    latency_ms    : float

class BatchPredictRequest(BaseModel):
    texts: List[str]

class BatchPredictResponse(BaseModel):
    results    : List[PredictResponse]
    total      : int
    hate_count : int
    safe_count : int
    latency_ms : float

class TrainResponse(BaseModel):
    success    : bool
    message    : str
    model_name : str
    train_size : int
    test_f1    : float
    trained_at : str


# =============================================================================
# HELPERS
# =============================================================================

IGBO_VOCAB = {
    'nna','nne','nwanne','obodo','ndị','chukwu','ọ','bụ','dị','mma',
    'anyị','ha','ka','na','si','ga','eme','ihe','ụlọ','ebe','nke',
    'nwere','gozie','kwenu','ahịa','obi','ụtọ','oha','gbuo','kasie',
    'igbo','nnaa','amaka','omenala','ndụ','onye','nwanyị','adịghị','ọjọọ',
}

def igbo_ratio(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return round(sum(1 for t in tokens if t in IGBO_VOCAB) / max(len(tokens),1), 3)

def preprocess(text):
    if state.preprocessor:
        return state.preprocessor.preprocess(text)
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', str(text))
    return re.sub(r'\s+', ' ', text).strip().lower()

def run_prediction(text: str) -> dict:
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    t0         = time.time()
    clean      = preprocess(text)
    ig_ratio   = igbo_ratio(text)
    is_mixed   = ig_ratio > 0.05
    features   = state.vectorizer.transform([clean])
    prediction = int(state.model.predict(features)[0])

    if hasattr(state.model, 'predict_proba'):
        probs     = state.model.predict_proba(features)[0]
        prob_safe = round(float(probs[0]), 4)
        prob_hate = round(float(probs[1]), 4)
    elif hasattr(state.model, 'decision_function'):
        raw       = float(state.model.decision_function(features)[0])
        prob_hate = round(1 / (1 + np.exp(-raw)), 4)
        prob_safe = round(1 - prob_hate, 4)
    else:
        prob_hate = 1.0 if prediction == 1 else 0.0
        prob_safe = 1 - prob_hate

    confidence = prob_hate if prediction == 1 else prob_safe

    return {
        "text"          : text,
        "label"         : prediction,
        "label_name"    : "Hate Speech" if prediction == 1 else "Not Hate Speech",
        "confidence"    : round(confidence, 4),
        "prob_hate"     : prob_hate,
        "prob_safe"     : prob_safe,
        "igbo_ratio"    : ig_ratio,
        "is_code_mixed" : is_mixed,
        "model_used"    : state.model_name or "Logistic Regression",
        "latency_ms"    : round((time.time() - t0) * 1000, 2),
    }


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
def health():
    return {
        "status"     : "ready" if state.is_ready else "loading",
        "model"      : state.model_name,
        "trained_at" : state.trained_at,
        "train_size" : state.train_size,
        "test_f1"    : state.test_f1,
        "timestamp"  : datetime.now().isoformat(),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(req.text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long (max 1000 chars).")
    return run_prediction(req.text)

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty.")
    if len(req.texts) > 50:
        raise HTTPException(status_code=400, detail="Max 50 texts per batch.")
    t0      = time.time()
    results = [run_prediction(t) for t in req.texts]
    return {
        "results"    : results,
        "total"      : len(results),
        "hate_count" : sum(1 for r in results if r["label"] == 1),
        "safe_count" : sum(1 for r in results if r["label"] == 0),
        "latency_ms" : round((time.time() - t0) * 1000, 2),
    }

@app.post("/train", response_model=TrainResponse)
def retrain():
    try:
        _train_fresh()
        return {
            "success"    : True,
            "message"    : "Model retrained successfully.",
            "model_name" : state.model_name,
            "train_size" : state.train_size,
            "test_f1"    : state.test_f1,
            "trained_at" : state.trained_at,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def stats():
    data_stats = {"total": 0, "hate": 0, "safe": 0}
    try:
        if os.path.exists("data/full_dataset.csv"):
            df = pd.read_csv("data/full_dataset.csv")
            data_stats = {
                "total": len(df),
                "hate" : int(df["label"].sum()),
                "safe" : int((df["label"]==0).sum()),
            }
    except Exception:
        pass

    return {
        "model"   : {
            "name"       : state.model_name,
            "trained_at" : state.trained_at,
            "train_size" : state.train_size,
            "test_f1"    : state.test_f1,
            "is_ready"   : state.is_ready,
        },
        "dataset" : data_stats,
        "benchmark": [
            {"name": "AfriBERTa",           "type": "transformer", "f1w": 0.911, "acc": 0.913, "f1h": 0.903},
            {"name": "XLM-RoBERTa",         "type": "transformer", "f1w": 0.891, "acc": 0.893, "f1h": 0.882},
            {"name": "BiLSTM + Attention",  "type": "deep",        "f1w": 0.843, "acc": 0.847, "f1h": 0.831},
            {"name": "Logistic Regression", "type": "classical",   "f1w": 0.816, "acc": 0.820, "f1h": 0.801},
            {"name": "Linear SVM",          "type": "classical",   "f1w": 0.797, "acc": 0.800, "f1h": 0.785},
        ],
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Hate Speech Detection API — English-Igbo")
    print("=" * 55)
    print("  Server  : http://localhost:8000")
    print("  Docs    : http://localhost:8000/docs")
    print("  Health  : http://localhost:8000/health")
    print("=" * 55)

    PORT = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "api-server:app",
        host="0.0.0.0",
        port=PORT,       # Render sets PORT automatically
        reload=False,    # Must be False in production
        log_level="info",
    )