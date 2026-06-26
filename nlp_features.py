"""
Unstructured-Data Feature Extraction
====================================
Turns free-text officer notes / financial-statement commentary into a numeric
`note_stress_index` in [0, 1] that feeds the default model alongside structured
features — satisfying Track-04's "use both structured AND unstructured data".

Method: semantic similarity (sentence-transformer embeddings) of each note to
"financial distress" vs "healthy operations" anchor phrases — a zero-shot stress
score. Falls back to a keyword lexicon if the transformer can't be loaded, so the
service degrades gracefully and never crashes.
"""
from __future__ import annotations
import numpy as np

STRESS_ANCHORS = [
    "severe cashflow pressure and missed EMI payments",
    "the business is failing with collapsing revenue",
    "supplier payments delayed and mounting overdue dues",
    "liquidity crisis, defaults likely, recovery concerns",
]
HEALTHY_ANCHORS = [
    "healthy business with strong and growing sales",
    "operations are normal and payments are on time",
    "stable cashflow and comfortable working capital",
    "profitable, well-managed, low credit risk borrower",
]

# Keyword fallback (used only if the transformer is unavailable)
_NEG = ["pressure", "delay", "delayed", "overdue", "missed", "default", "stress",
        "decline", "declining", "shortfall", "loss", "crisis", "recovery", "npa",
        "deterior", "weak", "concern", "risk", "fail", "drop", "fall", "late"]
_POS = ["normal", "healthy", "strong", "growth", "growing", "stable", "good",
        "profit", "on time", "comfortable", "improv", "robust", "steady"]

_MODEL = None
_STRESS_VEC = None
_HEALTHY_VEC = None


def _get_model():
    global _MODEL, _STRESS_VEC, _HEALTHY_VEC
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        s = _MODEL.encode(STRESS_ANCHORS, normalize_embeddings=True)
        h = _MODEL.encode(HEALTHY_ANCHORS, normalize_embeddings=True)
        _STRESS_VEC = s.mean(0)
        _HEALTHY_VEC = h.mean(0)
    return _MODEL


def _lexicon_score(texts) -> np.ndarray:
    out = []
    for t in texts:
        tl = str(t).lower()
        neg = sum(tl.count(w) for w in _NEG)
        pos = sum(tl.count(w) for w in _POS)
        raw = (neg - pos)
        out.append(1 / (1 + np.exp(-0.9 * raw)))
    return np.array(out, dtype=float)


def stress_index(texts, scale: float = 8.0) -> np.ndarray:
    """Batch: list/Series of note strings -> np.ndarray of stress scores in [0, 1]."""
    texts = ["" if t is None else str(t) for t in list(texts)]
    if not texts:
        return np.array([], dtype=float)
    try:
        model = _get_model()
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        raw = emb @ _STRESS_VEC - emb @ _HEALTHY_VEC
        return 1.0 / (1.0 + np.exp(-scale * raw))
    except Exception:
        return _lexicon_score(texts)


def stress_index_one(text: str) -> float:
    return float(stress_index([text])[0])
