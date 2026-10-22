"""Sparse TF-IDF and local dense embeddings behind a common score interface.

Each retriever returns one similarity score per passage. Semantic models load
lazily from local files. A shared cross-encoder separately ranks question–source
relevance; ranking scores never certify truth. TF-IDF and dense cosine values
have different scales.
"""

import os
import time
import threading
import models

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDER_MODEL = models.EMBEDDER_MODEL


class TfidfRetriever:
    """The original sparse retriever. Behavior-identical to the pre-refactor
    inline code (same vectorizer settings, same junk handling)."""

    def __init__(self, passages, match_texts, junk_passages=frozenset()):
        self.passages = passages
        self.match_texts = match_texts
        self.junk_passages = set(junk_passages)
        started = time.perf_counter()
        # sublinear_tf dampens repeated words; (1,2)-grams let matches like
        # "machine learning" survive as a unit.
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=1,
            tokenizer=str.split, token_pattern=None, lowercase=False,
        )
        self.matrix = self.vectorizer.fit_transform(self.match_texts)
        self.index_seconds = time.perf_counter() - started

    def scores(self, question: str, q_norm: str = None):
        """TF-IDF cosine of the (normalized) question against every passage.
        Title-junk passages are zeroed — they match every section-title
        question verbatim but never hold an answer."""
        if q_norm is None:
            q_norm = question or ""
        q_vec = self.vectorizer.transform([q_norm])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        if 0 < len(self.junk_passages) < len(self.passages):
            for j in self.junk_passages:
                scores[j] = 0.0
        return scores

    def top_k(self, question: str, k: int = 5, q_norm: str = None):
        scores = self.scores(question, q_norm)
        order = scores.argsort()[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]


_SHARED_EMBEDDERS: dict[str, object] = {}
_EMBEDDER_LOCK = threading.RLock()
_ENCODE_LOCK = threading.Lock()
_RERANKER = None
_RERANKER_LOCK = threading.RLock()
_RERANKER_FAILURE_AT = 0.0


def reranker_ready():
    return _RERANKER is not None


def semantic_rerank(question: str, passages):
    """Query–passage cross-attention logits, not probabilities or entailment scores."""
    global _RERANKER, _RERANKER_FAILURE_AT
    with _RERANKER_LOCK:
        if _RERANKER is None:
            if time.monotonic() - _RERANKER_FAILURE_AT < 30:
                return None
            try:
                from sentence_transformers import CrossEncoder
                _RERANKER = CrossEncoder(models.local_path(models.RERANKER_MODEL), device='cpu', local_files_only=True, max_length=384)
            except Exception:
                _RERANKER_FAILURE_AT = time.monotonic()
                return None
        if not passages:
            return []
        return _RERANKER.predict([(question, p) for p in passages], batch_size=16, show_progress_bar=False)


class SemanticRetriever:
    """Dense retriever over sentence-transformer embeddings.

    The embedding model downloads once (~90 MB) on first use and the passage
    matrix is cached on the instance — one per QAEngine, so a new dataset
    re-indexes automatically. Batch-encodes passages on CPU.
    """

    MODEL = EMBEDDER_MODEL

    def __init__(self, passages, model_name: str = EMBEDDER_MODEL):
        self.passages = passages
        self.model_name = model_name
        self._model = None
        self._matrix = None
        self.index_seconds = None
        self._lock = threading.RLock()

    def _ensure(self):
        with self._lock:
            self._ensure_locked()

    def _ensure_locked(self):
        """Load the model + encode the passages (once, lazily)."""
        if self._matrix is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed — run "
                "pip install -r requirements.txt to use semantic mode."
            ) from exc
        started = time.perf_counter()
        try:
            with _EMBEDDER_LOCK:
                if self.model_name not in _SHARED_EMBEDDERS:
                    _SHARED_EMBEDDERS[self.model_name] = SentenceTransformer(models.local_path(self.model_name), device="cpu", local_files_only=True)
                self._model = _SHARED_EMBEDDERS[self.model_name]
        except Exception as exc:
            raise RuntimeError(
                'Semantic retrieval is unavailable. Run python prepare_models.py to prepare local models, then retry.'
            ) from exc
        # Batch-encode; normalize so dot product == cosine similarity.
        with _ENCODE_LOCK:
            self._matrix = self._model.encode(
                self.passages, batch_size=32, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False)
        self.index_seconds = time.perf_counter() - started

    def index_time_ms(self):
        return None if self.index_seconds is None else round(self.index_seconds * 1000)

    def scores(self, question: str, q_norm: str = None):
        self._ensure()
        with _ENCODE_LOCK:
            q = self._model.encode([question or ''], convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)[0]
        return self._matrix @ q

    def top_k(self, question: str, k: int = 5, q_norm: str = None):
        scores = self.scores(question)
        order = scores.argsort()[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]
