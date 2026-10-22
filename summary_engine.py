"""Bounded local model proposals with exact source-aligned executive bullets.

Map/reduce inputs respect character and tokenizer budgets. Generated proposals
are filtered to source sentences; extractive selection supplies unsupported
proposals. A bounded thread-safe cache preserves complete coverage metadata.
Executive summaries use the first 24 passages and do not increment QA stats.
"""

import hashlib
import os
import time
import re
import copy
import threading

import preprocessing
import qa_model

SUMMARY_MODEL = os.environ.get(
    "QUILLQA_SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-6-6"
)

MIN_WORDS = 100          # below this, a model call is overkill
MAX_CHUNKS = 24          # cap on passages fed to the map step
GROUP_CHAR_LIMIT = 3200  # ≈ 800 tokens, safely inside the 1024-token window
MAP_MAX_LEN = 130
MAP_MIN_LEN = 40
REDUCE_MAX_LEN = 190
REDUCE_MIN_LEN = 60
BULLET_COUNT = 4

MAX_CACHE_ENTRIES = 50
# (model_name, corpus_hash) → response dict
_CACHE: dict = {}
_CACHE_LOCK = threading.RLock()


def cache_put(key, val):
    with _CACHE_LOCK:
        if key not in _CACHE and len(_CACHE) >= MAX_CACHE_ENTRIES:
            _CACHE.pop(next(iter(_CACHE)), None)
        _CACHE[key] = copy.deepcopy(val)


def cache_get(key):
    with _CACHE_LOCK:
        return copy.deepcopy(_CACHE.get(key))


def last_coverage(passages, chunks_used):
    return {'chunks_used': chunks_used, 'total_passages': len(passages),
            'omitted_passages': max(0, len(passages)-chunks_used),
            'selection': 'first passages', 'source_aligned': True}


def corpus_hash(passages) -> str:
    return hashlib.sha256("\n\n".join(passages).encode("utf-8")).hexdigest()[:16]


def _bulletize(text: str) -> list:
    """Split summary text into 3-4 complete-sentence bullets."""
    bullets = []
    for sent in preprocessing.sentence_split(text):
        sent = sent.strip()
        if len(sent.split()) >= 4:
            if not sent.endswith((".", "!", "?")):
                sent += "."
            bullets.append(sent)
        if len(bullets) >= BULLET_COUNT:
            break
    return bullets


def _call(pipe, text: str, max_length: int, min_length: int) -> str:
    out = qa_model.invoke(pipe, text, max_length=max_length, min_length=min_length,
                          do_sample=False, truncation=True)
    if not out or not out[0].get("summary_text", "").strip():
        raise RuntimeError("empty summary output")
    summary = out[0]["summary_text"].strip()
    # BART-style models leave stray spaces before periods; tidy them.
    summary = summary.replace(" .", ".")
    return summary if summary.endswith((".", "!", "?")) else summary + "."


def _groups(texts, limit=GROUP_CHAR_LIMIT):
    groups,current=[],''
    for text in texts:
        for part in preprocessing.bounded_windows(text, limit=limit, overlap=0):
            if current and len(current)+1+len(part)>limit:
                groups.append(current);current=''
            current=(current+' '+part).strip()
    if current:groups.append(current)
    return groups


def _model_groups(texts, pipe):
    groups = _groups(texts)
    tokenizer = getattr(pipe, 'tokenizer', None)
    if tokenizer is None:
        return groups
    maximum = min(1024, getattr(tokenizer, 'model_max_length', 1024)) - 32
    bounded = []
    for group in groups:
        encoded = tokenizer(group, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded['offset_mapping']
        start = 0
        for pos in range(0, len(offsets), maximum):
            end = offsets[min(pos + maximum, len(offsets)) - 1][1]
            bounded.append(group[start:end])
            start = end
        if start < len(group) and bounded:
            bounded[-1] += group[start:]
    return bounded


def align_to_sources(proposal, passages):
    """Only source-identical sentences can be certified without an entailment model.

    Word-set overlap and cosine cannot prove a new number/entity relationship.
    Unsupported free paraphrases are replaced with actual source sentences.
    """
    def key(text):return re.sub(r'\s+',' ',text).strip().casefold().rstrip('.!?')
    original={key(s):s for p in passages for s in preprocessing.sentence_split(p)}
    aligned=[]
    for sentence in preprocessing.sentence_split(proposal):
        source=original.get(key(sentence))
        if source and source not in aligned:aligned.append(source)
    return aligned


def _extractive_bullets(passages):
    """Stable source selection with passage diversity and bounded sentence work."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    candidates=[]
    for pid,passage in enumerate(passages):
        for index,sentence in enumerate(preprocessing.sentence_split(passage)):
            if len(sentence.split())>=4 and len(sentence)<=500:
                candidates.append((pid,index,sentence))
    if not candidates:return []
    if len(candidates)>500:
        candidates=[candidates[i] for i in np.linspace(0,len(candidates)-1,500,dtype=int)]
    try:
        matrix=TfidfVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False).fit_transform([preprocessing.normalize(x[2]) for x in candidates])
        scores=cosine_similarity(matrix,np.asarray(matrix.mean(axis=0)))[:,0]
    except ValueError:scores=np.ones(len(candidates))
    order=sorted(range(len(candidates)),key=lambda i:(float(scores[i])+.08/(1+candidates[i][1]),-i),reverse=True)
    chosen=[];owners=set();seen=set()
    for i in order:
        pid,index,sentence=candidates[i]
        if pid in owners or sentence.casefold() in seen:continue
        chosen.append(i);owners.add(pid);seen.add(sentence.casefold())
        if len(chosen)>=BULLET_COUNT:break
    for i in order:
        if len(chosen)>=BULLET_COUNT:break
        if candidates[i][2].casefold() not in seen:
            chosen.append(i);seen.add(candidates[i][2].casefold())
    return [candidates[i][2] for i in sorted(chosen)]


def summarize_passages(passages, pipe=None, model_name=SUMMARY_MODEL):
    """Map-reduce over passages. Returns (bullets, coverage) — raises on
    model failure; callers translate exceptions into friendly errors."""
    if model_name != SUMMARY_MODEL and pipe is None:
        raise RuntimeError('Configure QUILLQA_SUMMARIZER_MODEL before starting the application to select another model.')
    if pipe is None:
        pipe = qa_model.get_summarizer()
    if pipe is None:
        raise RuntimeError("summarization model is not available")

    selected = [p for p in passages if p and p.strip()][:MAX_CHUNKS]
    if not selected:
        raise RuntimeError('No source passages are available for summarization.')

    # MAP: group passages under the model's input limit, summarize each group
    groups = _model_groups(selected, pipe)

    partials = [
        _call(pipe, group, max_length=MAP_MAX_LEN, min_length=MAP_MIN_LEN)
        for group in groups
    ]

    # REDUCE: summarize the concatenated partial summaries
    # Hierarchical reduction preserves every partial; no hard slicing that
    # silently pretends all selected evidence reached the last model call.
    while len(_model_groups(partials, pipe)) > 1:
        groups = _model_groups(partials, pipe)
        reduced=[_call(pipe,g,max_length=MAP_MAX_LEN,min_length=MAP_MIN_LEN) for g in groups]
        if len(' '.join(reduced))>=len(' '.join(partials)):
            return _extractive_bullets(selected), len(selected)
        partials=reduced
    final = _call(pipe, ' '.join(partials), max_length=REDUCE_MAX_LEN, min_length=REDUCE_MIN_LEN)
    aligned = align_to_sources(final, selected)
    # Filling a partially verified proposal with arbitrary generated sentences
    # would reintroduce hallucination. Use one coherent source-only selection.
    bullets = _bulletize(' '.join(aligned)) if len(aligned)>=3 else _extractive_bullets(selected)
    if not bullets:
        # Degenerate model output (e.g. garbled input text) — no crash, but
        # nothing usable either.
        raise RuntimeError("the model produced no usable summary sentences")
    return bullets, len(selected)
