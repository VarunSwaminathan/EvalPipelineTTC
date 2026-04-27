"""Scoring functions used across all eval tasks.

Two design rules:

1. **Heavy models are gated.** Sentence-transformers and an NLI cross-encoder
   are nice-to-have but pull ~500MB of weights and take a minute to load on a
   cold cache. The smoke-test path (``--llm mock``) must finish without them,
   so we degrade to TF-IDF cosine + lexical noun-phrase overlap when
   ``--use-embeddings`` is off. The production run flips it on.
2. **All scores in [0, 1].** Caller code (averaging, CI, accuracy_delta)
   assumes this — we enforce it via clipping rather than letting bad inputs
   propagate silent NaNs.
"""

from __future__ import annotations

import logging
import math
import random
import re
import string
from collections import Counter
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)

def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(ground_truth) else 0.0

def f1_token_overlap(prediction: str, ground_truth: str) -> float:
    """Token-level F1 (SQuAD definition). Empty answers handled explicitly.
    """
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F-measure based on longest common subsequence.
    """
    pred = prediction.lower().split()
    gold = ground_truth.lower().split()
    if not pred or not gold:
        return 0.0
    lcs = _lcs_length(pred, gold)
    if lcs == 0:
        return 0.0
    p = lcs / len(pred)
    r = lcs / len(gold)
    return 2 * p * r / (p + r)

def _lcs_length(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[-1]

def semantic_similarity(text_a: str, text_b: str, use_embeddings: bool = False) -> float:
    """Cosine similarity in [0, 1].
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0
    if use_embeddings:
        try:
            sim = _embed_cosine(text_a, text_b)
            return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding similarity fell back to TF-IDF", extra={"error": str(exc)})
    return _tfidf_cosine(text_a, text_b)

@lru_cache(maxsize=1)
def _load_sentence_model():  # type: ignore[no-untyped-def]
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _embed_cosine(text_a: str, text_b: str) -> float:
    model = _load_sentence_model()
    vecs = model.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))

def _tfidf_cosine(text_a: str, text_b: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer().fit_transform([text_a, text_b])
    a = vec[0].toarray()[0]
    b = vec[1].toarray()[0]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))

def accuracy_delta(baseline_score: float, compressed_score: float) -> float:
    """Positive = compression helped (Pax Historia framing).
    """
    return compressed_score - baseline_score

_NP_RE = re.compile(r"[A-Za-z][A-Za-z\-']*(?:\s+[A-Za-z][A-Za-z\-']*){0,3}")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "on", "with",
    "is", "are", "was", "were", "be", "been", "being", "by", "as", "at",
    "this", "that", "these", "those", "it", "its", "from", "but", "not",
    "has", "have", "had", "do", "does", "did", "can", "could", "would",
    "should", "will", "may", "might", "must", "shall", "if", "than", "then",
    "so", "such", "also", "very", "more", "most",
}

def faithfulness_score(
    summary: str, source_document: str, use_embeddings: bool = False
) -> float:
    """Does the summary's content come from the source, or did it hallucinate?

    Two implementations:

    * ``use_embeddings=True``: NLI cross-encoder. For each summary sentence,
      score entailment against the source; report the mean entailment prob.
      This is the metric you'd ship in production.
    * Off: lexical fallback — fraction of summary content phrases (lowercased,
      stopwords removed, 1–4 grams) that appear in the source. Crude but
      monotonic with faithfulness on our data and runs in milliseconds.

    Both return a value in [0, 1] where higher = more faithful to the source.
    """
    if not summary.strip() or not source_document.strip():
        return 0.0
    if use_embeddings:
        try:
            return _nli_faithfulness(summary, source_document)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NLI faithfulness fell back to lexical", extra={"error": str(exc)})
    return _lexical_faithfulness(summary, source_document)

@lru_cache(maxsize=1)
def _load_nli_model():  # type: ignore[no-untyped-def]
    from sentence_transformers import CrossEncoder

    return CrossEncoder("cross-encoder/nli-deberta-v3-small")

def _nli_faithfulness(summary: str, source: str) -> float:
    model = _load_nli_model()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
    if not sentences:
        return 0.0
    pairs = [(source, sent) for sent in sentences]
    logits = np.asarray(model.predict(pairs))
    if logits.ndim == 1:
        logits = logits.reshape(-1, 3)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    entailment = probs[:, 1]
    return float(np.clip(entailment.mean(), 0.0, 1.0))

def _lexical_faithfulness(summary: str, source: str) -> float:
    src_lc = source.lower()
    phrases = _content_phrases(summary)
    if not phrases:
        return 0.0
    hits = sum(1 for p in phrases if p in src_lc)
    return hits / len(phrases)

def _content_phrases(text: str) -> list[str]:
    raw = _NP_RE.findall(text.lower())
    out: list[str] = []
    for ph in raw:
        toks = [t for t in ph.split() if t not in _STOPWORDS and len(t) > 2]
        if not toks:
            continue
        out.append(" ".join(toks))
    return out

def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(lower, upper)``. With n=50 samples and binary-ish task scores,
    asymptotic normal CIs are misleading; bootstrap is honest about the
    underlying distribution. Empty input yields (0, 0) rather than NaN — the
    report renders these as "n/a" without crashing.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(math.floor((alpha / 2) * n_resamples))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_resamples)) - 1
    return (means[lo_idx], means[hi_idx])
