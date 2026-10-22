"""Deterministic corpus normalization, sentence segmentation and bounded windows.

Normalized text uses Porter stemming and a fixed stoplist while preserving
negation, signed numbers, decimals and single-character identifiers. Displayed
evidence retains original text. NLTK resource downloads require explicit opt-in.
"""

import re
import os

import nltk

# ------------------------------------------------------------------ NLTK bootstrap

def _have(resource: str) -> bool:
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        try:
            nltk.data.find(resource + '.zip')
            return True
        except LookupError:
            return False

def _try_download(name: str) -> bool:
    try:
        return bool(nltk.download(name, quiet=True))
    except Exception:
        return False


def ensure_nltk_data():
    """Download whatever NLTK corpora/tokenizers we need, once, quietly."""
    for resource, name in (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),  # nltk >= 3.9 uses punkt_tab
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ):
        if not _have(resource):
            _try_download(name)


# Resource installation is explicit; serving a request must never initiate downloads.
if os.environ.get('QUILLQA_DOWNLOAD_NLTK') == '1':
    ensure_nltk_data()

# Bundled fallback (subset of NLTK's English list) so the app degrades
# gracefully on a machine with no internet access at all.
_FALLBACK_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when
where which while who whom why will with you your yours yourself yourselves
""".split())

from nltk.stem import PorterStemmer

STOPWORDS = _FALLBACK_STOPWORDS - {'not', 'no', 'nor'}

_stemmer = PorterStemmer()

# Porter stemming is deterministic across online/offline installations.
_LEMMATIZER_OK = False


# ------------------------------------------------------------------ tokenization

def _word_tokenize(text: str):
    return re.findall(r'[+-]?\d+(?:[.,:]\d+)*|[^\W\d_]+', text, re.UNICODE)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_split(text: str) -> list:
    """Sentence segmentation with a regex fallback when punkt is unavailable."""
    text = text.strip()
    if not text:
        return []
    parts, start = [], 0
    for match in re.finditer(r'(?<=[.!?])\s+', text):
        prefix = text[start:match.start()]
        if re.search(r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|e\.g|i\.e)\.$', prefix, re.I):
            continue
        parts.append(prefix.strip())
        start = match.end()
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def normalize(text: str) -> str:
    """Lowercase → split hyphens → tokenize → drop stopwords → lemmatize.

    Hyphens/slashes become spaces so "TF-IDF" matches the words "tf" and "idf"
    on both the question side and the corpus side.
    """
    text = re.sub(r'[−–](?=\d)', '-', (text or '').lower())
    text = re.sub(r'(?<=[^\W\d_])[-–—_/]|[-–—_/](?=[^\W\d_])', ' ', text)
    tokens = _word_tokenize(text)
    kept = []
    for index, token in enumerate(tokens):
        identifier = index > 0 and tokens[index - 1] in {'class','channel','mode','type','grade'}
        if token not in STOPWORDS or identifier:
            kept.append(token.replace(',', '') if re.fullmatch(r'[+-]?\d[\d,]*(?:\.\d+)?', token) else token)
    # Porter stems collapse derivational variants the WordNet lemmatizer
    # misses ("educational" -> "educ" matches "education" -> "educ").
    # Match text only — the displayed answer is never stemmed.
    kept = [_stemmer.stem(t) if t.isalpha() else t for t in kept]
    return " ".join(kept)


# ------------------------------------------------------------------ chunking

PARAGRAPH_MAX_CHARS = 900   # paragraphs longer than this get sentence-windowed
SENTENCES_PER_WINDOW = 3    # sliding window size
WINDOW_STRIDE = 2           # overlap of 1 sentence between neighbouring windows
MIN_PASSAGE_CHARS = 40      # throw away fragments shorter than this
MAX_CORPUS_CHARS = 2_000_000
MAX_PASSAGES = 5000


def bounded_windows(text: str, limit: int = PARAGRAPH_MAX_CHARS, overlap: int = 120) -> list:
    """Hard character bound, including long words and delimiter-free input."""
    if len(text) <= limit:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            space = text.rfind(' ', start + limit // 2, end)
            if space > start:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        next_start = max(start + 1, end - overlap)
        space = text.find(' ', next_start, end)
        start = space + 1 if space >= 0 else next_start
    return chunks


def build_passages(raw_text: str) -> list:
    """Split a raw corpus into retrieval-sized passages.

    Strategy: paragraphs (blank-line separated) are the natural unit. PDFs and
    slide decks rarely have blank lines, so any paragraph that is too long is
    re-cut into sliding windows of 3 sentences (stride 2 ⇒ 1-sentence overlap),
    which keeps answers findable even when they straddle a window boundary.
    """
    if len(raw_text or '') > MAX_CORPUS_CHARS:
        raise ValueError(f'Document text exceeds the {MAX_CORPUS_CHARS:,}-character processing limit.')
    text = (raw_text or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        # Soft line breaks (one \n, e.g. PDF line wrapping) become spaces;
        # hard blank lines already split the block above.
        block = re.sub(r"\s*\n\s*", " ", block).strip()
        if block:
            paragraphs.append(block)
    if not paragraphs:
        paragraphs = [re.sub(r"\s+", " ", text)]

    # Merge bare section headers ("EDUCATION", "ACHIEVEMENTS" — resumes and
    # reports are full of them) into the passage that follows. The header
    # carries exactly the word a question like "educational details" uses, so
    # dropping it orphans its content; a trailing header has nothing to
    # anchor and is dropped as before.
    merged = []
    for para in paragraphs:
        if merged and len(merged[-1]) < MIN_PASSAGE_CHARS:
            sep = " " if merged[-1].endswith((".", "!", "?", ":")) else ". "
            merged[-1] = merged[-1] + sep + para
        else:
            merged.append(para)
    if merged and len(merged[-1]) < MIN_PASSAGE_CHARS and len(merged) > 1:
        last = merged.pop()
        sep = " " if merged[-1].endswith((".", "!", "?", ":")) else ". "
        merged[-1] = merged[-1] + sep + last

    passages, seen = [], set()
    for para in merged:
        if len(para) <= PARAGRAPH_MAX_CHARS:
            candidates = [para]
        else:
            sentences = sentence_split(para)
            candidates = [
                " ".join(sentences[i : i + SENTENCES_PER_WINDOW])
                for i in range(0, len(sentences), WINDOW_STRIDE)
            ]
        candidates = [part for candidate in candidates for part in bounded_windows(candidate)]
        for cand in candidates:
            cand = re.sub(r"\s+", " ", cand).strip()
            if (len(cand) >= MIN_PASSAGE_CHARS or not passages) and cand not in seen and cand:
                seen.add(cand)
                passages.append(cand)
                if len(passages) > MAX_PASSAGES:
                    raise ValueError(f'Document exceeds the {MAX_PASSAGES}-passage processing limit.')
    return passages
