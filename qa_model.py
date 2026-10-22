"""Offline local model loading and bounded CPU inference.

The default MiniLM SQuAD2 reader compares source spans with a null answer.
Scores are uncalibrated extraction scores, not probabilities of truth.
Failed local loads retry; model preparation is explicit. Generated summaries
are accepted only as exact source-aligned sentences.
"""

import os
import re
import time
import threading
import models

_MODEL_NAME = models.QA_MODEL
# Lightweight abstractive summarizer for the "explain the document" feature
# (~550 MB download, cached after the first run; CPU-friendly).
_SUMMARIZER_NAME = models.SUMMARIZER_MODEL
_SUMMARIZER_MAX_CHARS = 3600  # ≈ 900 tokens, inside the model's 1024 limit

# Below this the model is essentially guessing on a context that contains no
# real answer — treat as "no information available". Contexts are already
# strictly retrieval-filtered by qa_engine, so a low score here reliably means
# "this passage is topical but does not actually answer the question".
# ~0.05–0.1 is the useful band: legitimate spans on out-of-domain text often
# land around 0.08–0.3.
MIN_MODEL_SCORE = 0.05
# Chunks are searched in retrieval-rank order; a later chunk must beat the
# current best by this margin to win, so near-ties keep the better-ranked one.
RANK_TIE_EPSILON = 0.02

_pipeline = None
_load_failed = False
_load_error = None

_summarizer = None
_summarizer_failed = False
_model_lock = threading.RLock()
_inference_gate = threading.BoundedSemaphore(1)
_last_load_attempt = 0.0
_last_summary_attempt = 0.0
_load_fingerprint = None


def invoke(pipe, *args, **kwargs):
    """Bound queued CPU inference; callers already bound input/context counts."""
    if not _inference_gate.acquire(timeout=20):
        raise RuntimeError('The local model is busy. Please try again shortly.')
    try:
        return pipe(*args, **kwargs)
    finally:
        _inference_gate.release()


def reset_failed_loads():
    """Explicit recovery after preparing resources, without replacing loaded models."""
    global _load_failed, _summarizer_failed, _last_load_attempt, _last_summary_attempt
    with _model_lock:
        _load_failed = _summarizer_failed = False
        _last_load_attempt = _last_summary_attempt = 0.0


def get_summarizer():
    """Return the abstractive summarization pipeline (loaded once, lazily)."""
    global _summarizer, _summarizer_failed, _last_summary_attempt
    with _model_lock:
        if _summarizer is not None:
            return _summarizer
        if _summarizer_failed and time.monotonic() - _last_summary_attempt < 30:
            return None
        _last_summary_attempt = time.monotonic()
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            name = models.local_path(_SUMMARIZER_NAME)
            _summarizer = pipeline('summarization', model=AutoModelForSeq2SeqLM.from_pretrained(name, local_files_only=True),
                                   tokenizer=AutoTokenizer.from_pretrained(name, local_files_only=True), device=-1)
            _summarizer_failed = False
        except Exception:
            _summarizer_failed = True
            _summarizer = None
        return _summarizer


def warm_up() -> bool:
    """Force the one-time QA-model load at app startup. True when ready."""
    return get_pipeline() is not None


def warm_up_summarizer() -> bool:
    """Force the one-time summarizer load at app startup. True when ready."""
    return get_summarizer() is not None


def abstractive_summary(text: str, max_length: int = 190) -> str | None:
    """Paraphrase-style summary in the model's own words, or None."""
    pipe = get_summarizer()
    if pipe is None:
        return None
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) < 200:
        return None  # too little text for an abstractive pass to add value
    try:
        out = invoke(pipe,
            text[:_SUMMARIZER_MAX_CHARS],
            max_length=max_length,
            min_length=40,
            do_sample=False,
            truncation=True,
        )
    except Exception:
        return None
    if not out or not out[0].get("summary_text"):
        return None
    summary = out[0]["summary_text"].strip()
    # Never label an unverified free paraphrase as grounded. These exact source
    # sentences can still be selected/ranked by the generative proposal.
    from summary_engine import align_to_sources
    aligned = align_to_sources(summary, [text])
    if not aligned:
        return None
    summary = ' '.join(aligned)
    # BART-style models end without a period sometimes; normalize.
    return (summary if summary.endswith((".", "!", "?")) else summary + ".").replace(" .", ".")


def get_pipeline():
    """Return the shared QA pipeline, loading it on first use (once)."""
    global _pipeline, _load_failed, _load_error, _last_load_attempt, _load_fingerprint
    with _model_lock:
        if _pipeline is not None:
            return _pipeline
        path = models.local_path(_MODEL_NAME)
        fingerprint = (path, os.path.exists(os.path.join(path, 'model.safetensors')))
        if _load_failed and fingerprint == _load_fingerprint and time.monotonic() - _last_load_attempt < 30:
            return None
        _last_load_attempt = time.monotonic()
        _load_fingerprint = fingerprint
        try:
            import torch
            from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
            torch.set_num_threads(int(os.environ.get('QUILLQA_CPU_THREADS', '2')))
            _pipeline = pipeline('question-answering', model=AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=True),
                                 tokenizer=AutoTokenizer.from_pretrained(path, local_files_only=True), device=-1)
            _load_failed = False
            _load_error = None
        except Exception as exc:
            _load_failed = True
            _load_error = type(exc).__name__ + ': QA model unavailable; run python prepare_models.py or retry after preparing local resources.'
            _pipeline = None
        return _pipeline


def capabilities():
    from retrievers import _SHARED_EMBEDDERS, reranker_ready
    return {'qa': {'loaded': _pipeline is not None, 'model': _MODEL_NAME,
                   'supports_no_answer_training': 'squad2' in _MODEL_NAME.lower(), 'error': _load_error},
            'summarizer': {'loaded': _summarizer is not None, 'model': _SUMMARIZER_NAME},
            'embedder': {'loaded': bool(_SHARED_EMBEDDERS), 'model': models.EMBEDDER_MODEL},
            'reranker': {'loaded': reranker_ready(), 'model': models.RERANKER_MODEL},
            'verifier': {'loaded': __import__('verifier').ready(), 'model': models.VERIFIER_MODEL},
            'network_during_inference': False}


def answer_candidates(question: str, contexts) -> list:
    """Compare an extracted answer with an explicit SQuAD2 null answer per context."""
    pipe = get_pipeline()
    if pipe is None:
        return []
    candidates = []
    for index, context in enumerate(contexts):
        if not context.strip():
            continue
        try:
            predictions = invoke(pipe, question=question, context=context, handle_impossible_answer=True,
                                 top_k=6, max_answer_len=80, max_seq_len=384, doc_stride=128)
            if isinstance(predictions, dict):
                predictions = [predictions]
            null = max((float(x['score']) for x in predictions if not x.get('answer', '').strip()), default=0.0)
            selected = []
            for result in predictions:
                if not result.get('answer', '').strip() or float(result['score']) < MIN_MODEL_SCORE or float(result['score']) <= null:
                    continue
                result = dict(result)
                start, end = result.get('start'), result.get('end')
                if not isinstance(start, int) or not isinstance(end, int) or context[start:end] != result['answer']:
                    continue
                if any(start < old_end and end > old_start for old_start, old_end in selected):
                    continue
                if re.fullmatch(r'(?:he|she|it|they|his|her|its|their)', result['answer'], re.I):
                    continue
                result.update(context_index=index, null_score=null, score_type='uncalibrated_squad2_span_score')
                candidates.append(result)
                selected.append((start, end))
                if len(selected) >= 2:
                    break
        except RuntimeError:
            raise
        except Exception:
            continue
    return candidates



def load_error() -> str | None:
    """Reason the model is unavailable, or None if it loaded fine."""
    get_pipeline()
    return _load_error


def answer_from(question: str, contexts) -> dict | None:
    """Extract the best answer span across the retrieved contexts.

    Runs the QA model once per context chunk and keeps the span with the
    highest model confidence:
        {"answer", "score", "start", "end", "context_index"}
    Returns None when the model is unavailable, every run failed, or the best
    score is below MIN_MODEL_SCORE (context probably has no direct answer).
    """
    candidates = answer_candidates(question, contexts)
    return max(candidates, key=lambda x: (x['score'], -x['context_index']), default=None)
