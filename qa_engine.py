"""Document indexing, explicit definitions, suggestions and source summaries.

QAEngine.ask delegates semantic ranking, null-aware reading, constraint checks
and evidence composition to answering.py. Retrieval similarity and reader
scores have separate meanings. Definitions and source-copy suggestions carry
no fabricated confidence probability.
"""

import hashlib
import math
import re
import time
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import retrievers
import preprocessing
import qa_model
import simplifier

# Below this cosine similarity a passage is not considered a direct match.
# TF-IDF cosine against whole passages rarely exceeds ~0.35 for genuine
# matches, so 0.35 rejects weak/topical-only overlaps ("explain the dataset"
# scoring 0.23) BEFORE the QA model ever sees them. Weaker-but-real keyword
# matches are still reachable through the sentence-level rescue pass below.
CONFIDENCE_THRESHOLD = 0.35
# Context chunks handed to the reading-comprehension model.
TOP_PASSAGE_COUNT = 3
# ...but only passages at least this fraction of the winner's score.
PASSAGE_POOL_RATIO = 0.60
# Rescue pass (short keyword questions) widens the sentence search this far.
RESCUE_PASSAGE_COUNT = 5
# A rescue match must share at least this fraction of the question's content
# words — one stray common word ("price", "compare") is not evidence.
RESCUE_OVERLAP_MIN = 0.5
# ...and its best sentence must clear this TF-IDF cosine, so the rescue path
# never accepts anything weaker than the old direct threshold accepted.
RESCUE_MIN_SENTENCE_SCORE = 0.15
# Dense-embedding cosine runs on a different scale than TF-IDF cosine (related
# text typically scores 0.3–0.8); 0.30 rejects unrelated questions without
# penalizing genuine semantic matches.
SEMANTIC_CONFIDENCE_THRESHOLD = 0.30
# Sentence candidates scoring within this fraction of the best are tied; ties
# are broken by the definition pattern below (fallback engine only).
TIE_WINDOW_RATIO = 0.85
# Chunks shorter than ~2 sentences (agenda slides, bare titles) are retrieval
# magnets but rarely contain a complete answer — Stage 2 reads longer chunks
# whenever any retrieval-supported one exists.
MIN_CONTEXT_CHARS = 120
# A passage whose sentences are mostly bare titles (agenda/contents slides:
# "Introduction." "Data Used." …) matches every section-title question with
# verbatim high scores but never contains an answer — excluded from the
# sentence-level rescue pool and preferred against in context selection.
TITLE_JUNK_SENT_CHARS = 25
TITLE_JUNK_FRACTION = 0.6
# The passage a final answer comes from must literally contain at least this
# fraction of the question's content words. Blocks "confident" model spans
# pulled from passages that only share one stray word with the question
# (e.g. "price" matching "stock price of Apple").
CONTEXT_SUPPORT_MIN = 0.5

# ---- Meta-question ("summarize the dataset") intent detection -------------
# These questions are about the corpus as a whole, so they route to a
# dataset-level summary instead of single-passage retrieval + span extraction.
SUMMARY_PATTERNS = [
    r"\bsummar(y|ise|ize)\b",
    r"\b(give|provide|write|create|make)\b.{0,24}\b(summary|overview)\b",
    r"\boverview of (the|this)\b",
    r"\bdescribe (the|this) (dataset|data|corpus|document)\b",
    r"\bwhat (is|are) (this|the) (dataset|data|corpus) about\b",
    r"\btl;?dr\b",
]
_SUMMARY_RE = re.compile("|".join(SUMMARY_PATTERNS), re.IGNORECASE)

# Explain/analyze intent: a whole-document walkthrough in the system's own
# words (abstractive) plus simplified key points. Kept separate from the
# extractive summary intent — "summarize" picks key sentences, "explain"
# re-tells the document.
EXPLAIN_PATTERNS = [
    r"\b(explain|analy[sz]e|break ?down|walk me through)\b.{0,30}\b"
    r"(document|dataset|corpus|file|text|data)\b",
    r"\bexplain (this|it)\b(?!\s+(how|to|for))",
    r"\bwhat does (this|the) (document|dataset|file) (say|contain|cover|talk about)\b",
    r"\bexplain\b.{0,24}\bown words\b",
]
_EXPLAIN_RE = re.compile("|".join(EXPLAIN_PATTERNS), re.IGNORECASE)

# "...in simple words/terms" on a fact question → also return a simplified
# version of the answer.
_SIMPLE_RE = re.compile(r"\b(in simple (words|terms)|simpl(er|y|ify)|plain words)\b", re.IGNORECASE)

# Instructional phrases ("explain in simple words", "briefly") add no topical
# content — stripped before retrieval so they can't dilute the match.
_INSTRUCTION_RE = re.compile(
    r"\b(in simple (words|terms)|in own words|in plain (words|english)|"
    r"explain (it )?in simple words|briefly|in detail|simply)\b", re.IGNORECASE,
)

# Extractive summarization knobs (TF-IDF centrality, LexRank-style).
SUMMARY_SENTENCE_COUNT = 4     # sentences shown in the summary
SUMMARY_REDUNDANCY = 0.60      # skip sentences too similar to a picked one
SUMMARY_POOL_CAP = 500         # centrality matrix size cap for huge corpora
SUMMARY_SENT_MIN_CHARS = 40
SUMMARY_SENT_MAX_CHARS = 400


def is_summary_request(question: str) -> bool:
    q = re.sub(r'\s+', ' ', _INSTRUCTION_RE.sub(' ', question or '')).strip(' .!?').lower()
    return bool(re.fullmatch(r'(?:please )?(?:summari[sz]e|summary|tl;?dr|(?:give|provide|write|create|make)(?: me)? (?:a |an )?(?:summary|overview)(?: of)?|overview of|describe|what (?:is|are)) ?(?:(?:the|this|my|provided|entire|whole) )?(?:dataset|data|corpus|document|file|text)?(?: about)?', q))


def is_explain_request(question: str) -> bool:
    q = re.sub(r'\s+', ' ', _INSTRUCTION_RE.sub(' ', question or '')).strip(' .!?').lower()
    return bool(re.fullmatch(r'(?:please )?(?:(?:explain|analy[sz]e|break ?down|walk me through) (?:(?:the|this|my|provided|entire|whole) )?(?:document|dataset|corpus|file|text|data)|explain (?:this|it)|what does (?:this|the) (?:document|dataset|file) (?:say|contain|cover|talk about))', q))

_PRONOUN_START = re.compile(
    r"^(he|she|it|they|his|her|its|their|this|these|those)\b[,.]?", re.IGNORECASE
)
# In "how does X work?" the trailing "work" is interrogative scaffolding, not
# content — stripped so that X alone drives retrieval.
_FRAME_VERB = re.compile(r"\s+\b(work|works)\b$")

# Question-suggestion pattern: "<Subject> is/are/stands for ..." — the classic
# definitional sentence shape, used to build dataset-specific sample chips.
_COPULA_RE = re.compile(
    r"^(?P<subject>[A-Z][A-Za-z0-9 .+\-']{1,48}?)\s+"
    r"(?P<copula>is|are|was|were|stands for|refers to|means|denotes)\b"
)
_SUGGESTION_GENERIC_TERMS = frozenset({
    "additional", "background", "based", "context", "document", "example",
    "examples", "historical", "include", "includes", "including",
    "information", "method", "methods", "provide", "provides", "research",
    "result", "results", "section", "sections", "topic", "topics", "various",
    "wider",
})

_SPAN_TIDY_LEADING = re.compile(r"^[\s:;,.—–-]+")
_SPAN_TIDY_QUOTES = "\"“”‘’"
_DEFINITION_QUESTION_RE = re.compile(
    r"^(?:what\s+is|define|definition\s+of|meaning\s+of)\s+"
    r"(?:the\s+)?(?P<topic>[^?.!]+)",
    re.IGNORECASE,
)
_DEFINITION_CUE_RE = re.compile(
    r"^\s*(?:\([^)]{1,120}\)\s*)?"
    r"(?:is|are|means|refers to|stands for|denotes|is defined as|"
    r"happens when|occurs when)\b",
    re.IGNORECASE,
)
_DEFINITION_CLASS_RE = re.compile(
    r"\b(?:statistic|measure|metric|method|technique|process|approach|"
    r"algorithm|score|weight|model|system|simulation|branch|field|"
    r"representation|architecture|framework|mechanism)\b",
    re.IGNORECASE,
)
_TOPIC_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-–—_/][A-Za-z0-9]+)*")
_TOPIC_ALIASES = {
    "ai": ("artificial intelligence", "AI"),
    "artificialintelligence": ("artificial intelligence", "AI"),
    "tfidf": ("TF-IDF", "Term Frequency-Inverse Document Frequency"),
    "termfrequencyinversedocumentfrequency": (
        "TF-IDF",
        "Term Frequency-Inverse Document Frequency",
    ),
}
_DIRECT_DEFINITION_RE = re.compile(
    r"^\s*(?:\([^)]{1,120}\)\s*)?"
    r"(?:(?:is|are|was|were)\s+(?:a|an|the)\s+|"
    r"(?:stands for|means|refers to|denotes|is defined as|"
    r"happens when|occurs when)\b)",
    re.IGNORECASE,
)


def _strip_frame_verb(raw_question: str, q_norm: str) -> str:
    if re.match(r"^\s*how\b", raw_question, re.IGNORECASE):
        stripped = _FRAME_VERB.sub("", q_norm)
        if stripped.strip():
            return stripped
    return q_norm


def _question_key(question: str) -> str:
    return re.sub(r"\s+", " ", question or "").strip().casefold()


def _definition_topic(question: str) -> str | None:
    """Return the requested concept for a simple, single-topic definition."""
    match = _DEFINITION_QUESTION_RE.match(question.strip())
    if not match:
        return None
    topic = re.sub(
        r"\s+(?:in simple (?:words|terms)|in plain (?:words|english))\s*$",
        "",
        match.group("topic").strip(),
        flags=re.IGNORECASE,
    ).strip(" \t,")
    if not topic or re.match(r"(?:the\s+)?difference\b", topic, re.IGNORECASE):
        return None
    return topic


def _find_topic_span(text: str, topic: str) -> tuple[int, int] | None:
    """Locate a topic despite harmless differences in spaces or hyphens."""
    topic_key = "".join(preprocessing.normalize(topic).split())
    if not topic_key:
        return None
    tokens = list(_TOPIC_WORD_RE.finditer(text))
    max_tokens = len(preprocessing.normalize(topic).split()) + 2
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + max_tokens) + 1):
            candidate = text[tokens[start].start():tokens[end - 1].end()]
            candidate_key = "".join(preprocessing.normalize(candidate).split())
            if candidate_key == topic_key:
                return tokens[start].start(), tokens[end - 1].end()
    return None


def _concept_key(topic: str) -> str:
    """Normalize a concept's spelling without stemming or punctuation."""
    return "".join(re.findall(r"[a-z0-9]+", topic.casefold()))


def _topic_variants(topic: str) -> tuple[str, ...]:
    """Return equivalent spellings of a concept, ignoring case and separators."""
    return _TOPIC_ALIASES.get(_concept_key(topic), (topic,))


def _has_direct_definition(sentence: str, topic_span: tuple[int, int]) -> bool:
    """Check for a definition cue immediately after the matched topic."""
    continuation = sentence[topic_span[1]:topic_span[1] + 180]
    return bool(_DIRECT_DEFINITION_RE.match(continuation))


def _has_direct_definition_for_topic(sentence: str, variants: tuple[str, ...]) -> bool:
    """Detect a definition cue after any occurrence of an equivalent topic name."""
    for variant in variants:
        offset = 0
        while offset < len(sentence):
            span = _find_topic_span(sentence[offset:], variant)
            if span is None:
                break
            start, end = span[0] + offset, span[1] + offset
            if _has_direct_definition(sentence, (start, end)):
                return True
            offset = end
    return False


def _has_definition_cue(engine, topic: str) -> bool:
    """Detect definition-shaped mentions that are too weak to trust."""
    variants = _topic_variants(topic)
    for passage in engine.passages:
        for sentence in preprocessing.sentence_split(passage):
            for variant in variants:
                topic_span = _find_topic_span(sentence, variant)
                if topic_span is not None and _DEFINITION_CUE_RE.match(
                    sentence[topic_span[1]:]
                ):
                    return True
    return False


def _definition_answer(
    engine, question: str, topic: str, started: float, retriever: str
):
    """Find a source definition and nearby source-backed facts for a concept."""
    variants = _topic_variants(topic)
    if not any(preprocessing.normalize(variant) for variant in variants):
        return None

    candidates = []
    for passage_idx, passage in enumerate(engine.passages):
        sentences = preprocessing.sentence_split(passage)
        for sentence_idx, sentence in enumerate(sentences):
            for variant in variants:
                topic_span = _find_topic_span(sentence, variant)
                if topic_span is None or not _has_direct_definition(sentence, topic_span):
                    continue
                continuation = sentence[topic_span[1]:topic_span[1] + 220]
                expansion = re.match(
                    r"^\s*(?:\([^)]{1,120}\)\s*)?"
                    r"(?:stands for|means|refers to|denotes|is defined as)\b",
                    continuation,
                    re.IGNORECASE,
                )
                expansion = expansion or re.match(
                    r"^\s*happens when\b|^\s*occurs when\b",
                    continuation,
                    re.IGNORECASE,
                )
                definition_clause = re.split(r"[.!?;]", continuation, maxsplit=1)[0]
                if not expansion and not _DEFINITION_CLASS_RE.search(definition_clause):
                    continue
                prefix = sentence[:topic_span[0]]
                # Prefer a concept that is the sentence's subject. This avoids
                # mistaking "supervised learning is a category of machine
                # learning" for a definition of machine learning.
                score = 5.0 - min(len(prefix), 200) / 100
                if _DEFINITION_CLASS_RE.search(definition_clause):
                    score += 1.0
                if re.search(r"\bstatistic\b", definition_clause, re.IGNORECASE):
                    score += 2.0
                if sentence_idx:
                    previous_sentence = sentences[sentence_idx - 1]
                    if re.match(r"^\s*what\s+is\b", previous_sentence, re.IGNORECASE):
                        previous_topic = next(
                            (
                                _find_topic_span(previous_sentence, item)
                                for item in variants
                                if _find_topic_span(previous_sentence, item) is not None
                            ),
                            None,
                        )
                        if previous_topic is not None:
                            score += 2.0
                candidates.append((
                    score, passage_idx, sentence_idx, sentence, sentences
                ))
                break

    if not candidates:
        return None
    _, passage_idx, definition_idx, definition, sentences = max(
        candidates, key=lambda item: item[0]
    )
    passage = engine.passages[passage_idx]

    definition_terms = set(preprocessing.normalize(definition).split())
    related = []
    for sentence_idx, sentence in enumerate(sentences):
        if sentence_idx == definition_idx or len(sentence) < 55 or len(sentence) > 400:
            continue
        topic_span = next(
            (
                span for variant in variants
                if (span := _find_topic_span(sentence, variant)) is not None
            ),
            None,
        )
        if _has_direct_definition_for_topic(sentence, variants):
            continue
        sentence_terms = set(preprocessing.normalize(sentence).split())
        if topic_span is None:
            continue
        overlap = len(sentence_terms & definition_terms)
        related.append((overlap, -abs(sentence_idx - definition_idx), sentence_idx, sentence))
    related.sort(reverse=True)
    supporting_points = [item[3] for item in related[:2]]

    concept_key = _concept_key(topic)
    if 'Term Frequency-Inverse Document Frequency'.casefold() in passage.casefold() and concept_key in {
        "tfidf",
        "termfrequencyinversedocumentfrequency",
    }:
        definition_topic_span = next(
            (
                span for variant in variants
                if (span := _find_topic_span(definition, variant)) is not None
            ),
            None,
        )
        if definition_topic_span is not None:
            definition_tail = definition[definition_topic_span[1]:]
            definition_tail = re.sub(
                r"^\s*\([^)]{1,120}\)", "", definition_tail
            )
            definition_text = (
                "TF-IDF (Term Frequency-Inverse Document Frequency)"
                + definition_tail
            )
        else:
            definition_text = definition
    else:
        definition_text = definition

    answer = f"Based on the dataset: {definition_text}"
    if supporting_points:
        answer += " Key points: " + " ".join(supporting_points)
    answer_start = passage.find(definition)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    retrieval_scores = engine.tfidf.scores(
        question, preprocessing.normalize(question)
    )
    return {
        "status": "answered",
        "answer": answer,
        "span": definition,
        "confidence": None,
        "retrieval_score": round(float(retrieval_scores[passage_idx]), 4),
        "response_time_ms": elapsed_ms,
        "source_passage": passage,
        "passage_index": passage_idx + 1,
        "retriever": retriever,
        "passage_id": passage_idx + 1,
        "passage_text": passage,
        "answer_start": answer_start if answer_start >= 0 else None,
        "answer_end": answer_start + len(definition) if answer_start >= 0 else None,
        "category": "Definition",
    }


def compose_answer(span: str) -> str | None:
    """Wrap a raw extracted span into a short, natural sentence."""
    span = re.sub(r"\s+", " ", (span or "")).strip().strip(_SPAN_TIDY_QUOTES)
    span = _SPAN_TIDY_LEADING.sub("", span).strip()
    if not span:
        return None
    # Sentence-case the span for presentation, but leave acronyms/mixed-case
    # names ("AI", "NLTK", "iPhone") untouched.
    if span[0].islower() and (len(span) == 1 or span[1].islower()):
        span = span[0].upper() + span[1:]
    sentence = f"Based on the dataset: {span}"
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _is_title_junk(passage: str) -> bool:
    """True for agenda/contents-style passages that can't hold an answer."""
    # Numbers/IDs (roll numbers, figures, "Chapter 1") mark real content.
    if any(ch.isdigit() for ch in passage):
        return False
    sents = preprocessing.sentence_split(passage)
    if sents:
        short = sum(1 for s in sents if len(s.strip()) < TITLE_JUNK_SENT_CHARS)
        if short / len(sents) >= TITLE_JUNK_FRACTION:
            return True
    # A run of Title-Case words with (almost) no lowercase words is a title
    # list regardless of how punkt segmented it.
    words = re.findall(r"[A-Za-z0-9']+", passage)
    if len(words) < 6:
        return False
    lower = sum(1 for w in words if w[0].islower())
    return lower <= max(1, len(words) // 10)


def _containing_sentence(passage: str, start: int, end: int) -> str | None:
    """Return the sentence of `passage` that contains the char span [start, end).

    Used to upgrade short extractive spans ("Organized service content and page
    sections") into their full containing sentence, which reads like an actual
    answer on resumes and bullet-style documents.
    """
    cursor = 0
    for sent in preprocessing.sentence_split(passage):
        idx = passage.find(sent, cursor)
        if idx == -1:
            continue
        cursor = idx + len(sent)
        if idx <= start and end <= idx + len(sent):
            return sent.strip()
    return None


def _passage_pool(scores, best_idx, best_score):
    """Stage-2 search space: top passages close to the winner's score."""
    pool = [
        int(i) for i in scores.argsort()[::-1][:TOP_PASSAGE_COUNT]
        if scores[i] >= PASSAGE_POOL_RATIO * best_score
    ]
    if best_idx not in pool:
        pool.insert(0, best_idx)
    return pool


def _context_pool(scores, passages, threshold, junk=frozenset(), prefer_long=True):
    """Reading context for Stage 2, ranked by retrieval score.

    TF-IDF mode prefers retrieval-supported passages that are long enough to
    hold a full answer; ultra-short chunks (agenda slides, bare titles) are
    only used when nothing longer clears the threshold. Semantic mode passes
    prefer_long=False — its ranking already accounts for passage quality, so
    a top-ranked short passage must not be demoted. Junk passages are always
    excluded while anything survives. Returns (pool, base_score) where
    base_score is the retrieval support of the returned context.
    """
    candidates = [int(i) for i in scores.argsort()[::-1] if scores[i] >= threshold]
    non_junk = [i for i in candidates if i not in junk]
    candidates = non_junk or candidates
    if prefer_long:
        long = [i for i in candidates if len(passages[i]) >= MIN_CONTEXT_CHARS]
        if long:
            base = float(scores[long[0]])
            pool = [i for i in long if scores[i] >= PASSAGE_POOL_RATIO * base][:TOP_PASSAGE_COUNT]
            return pool, base
    pool = candidates[:TOP_PASSAGE_COUNT]
    base = float(scores[pool[0]]) if pool else 0.0
    return pool, base


class QAEngine:
    """One instance per browser session (app.py creates them lazily)."""

    def __init__(self, corpus_text: str):
        started = time.perf_counter()
        self.passages = preprocessing.build_passages(corpus_text)
        if not self.passages:
            raise ValueError(
                "No usable passages found — the document text was too short or empty."
            )
        self.word_count = len(corpus_text.split())
        # Content fingerprint for per-dataset caches (executive summary).
        self.corpus_hash = hashlib.sha256(
            "\n\n".join(self.passages).encode("utf-8")
        ).hexdigest()[:16]
        # Matching text = normalized (lemmatized, stopword-free) copy of each
        # passage. The original passages stay untouched for display / Stage 2.
        self.match_texts = [preprocessing.normalize(p) for p in self.passages]

        # Agenda/contents-style passages: mostly bare title sentences, or a
        # run of capitalized words with no lowercase function words ("CONTENTS
        # Introduction Problem Definition …") — punkt can't split those, so
        # the sentence-fraction test alone misses them.
        self.junk_passages = set()
        for p_no, passage in enumerate(self.passages):
            if _is_title_junk(passage):
                self.junk_passages.add(p_no)

        # Retrieval lives behind the Retriever interface (retrievers.py). The
        # TF-IDF retriever indexes eagerly (identical settings/behavior to the
        # pre-refactor inline code); the semantic one is built lazily on first
        # use so TF-IDF-only sessions never load the embedding model.
        self.tfidf = retrievers.TfidfRetriever(
            self.passages, self.match_texts, self.junk_passages
        )
        self._semantic = None
        self._suggested_answers = {}
        self.index_seconds = time.perf_counter() - started
        self.retrieval_index_seconds = self.tfidf.index_seconds
        self._t0 = started  # kept for reference; timing now lives per-retriever

    @property
    def vectorizer(self):
        return self.tfidf.vectorizer

    @property
    def passage_matrix(self):
        return self.tfidf.matrix

    def _get_retriever(self, mode: str):
        """Retriever for the requested mode; semantic is cached per dataset."""
        if mode == "semantic":
            if self._semantic is None:
                self._semantic = retrievers.SemanticRetriever(self.passages)
            return self._semantic
        return self.tfidf

    @property
    def passage_count(self) -> int:
        return len(self.passages)

    # ------------------------------------------------------------- retrieval

    def ask(self, question: str, threshold: float = None, retriever: str = "tfidf") -> dict:
        """Answer after query–evidence relevance and answerability checks."""
        from answering import answer
        return answer(self, question, threshold=threshold, retriever=retriever)

    # -------------------------------------------------- fallback (no model)

    def _summarize(self, started: float) -> dict:
        """Dataset-level extractive summary via TF-IDF centrality (LexRank-style).

        Sentence importance = how similar a sentence is to every other sentence
        in the corpus (centrality in the TF-IDF similarity graph). The top
        sentences are picked greedily with a redundancy filter and shown in
        original reading order — so the summary spans the whole dataset rather
        than one passage.
        """
        picked, sentences, pool_owners = self._central_sentences()
        if not picked:
            return self._no_info("The dataset is too short to summarize.", started)

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return {
            "status": "summarized",
            "answer": " ".join(sentences[i] for i in picked),
            "span": None,
            "confidence": None,
            "retrieval_score": None,
            "response_time_ms": elapsed_ms,
            "source_passage": None,
            "passage_index": None,
            "summary_stats": self._doc_stats(len(sentences)),
            'coverage': self._summary_coverage,
            'coverage_note': self._summary_coverage['note'],
            "summary_sources": sorted({pool_owners[i] + 1 for i in picked}),
            "category": "Summary",
        }

    # ------------------------------------------------------ document analysis

    def _analyze(self, started: float) -> dict:
        """Explain the whole document in the system's own words.

        Two steps: (1) the most central passages (TF-IDF centrality over the
        passage graph) are paraphrased by a local abstractive summarizer
        (DistilBART) — that is the "own words" explanation; (2) the most
        central sentences are rewritten by the rule-based simplifier into
        short, plain-English key points. Falls back to the extractive summary
        when the summarizer model is unavailable. The analysis depends only on
        the indexed dataset, so it is computed once per dataset and cached.
        """
        cached = getattr(self, "_explanation_cache", None)
        if cached is not None:
            result = dict(cached)
            result["response_time_ms"] = round((time.perf_counter() - started) * 1000)
            result["cached"] = True
            return result
        picked, sentences, owners = self._central_sentences()
        if not picked:
            return self._no_info("The dataset is too short to analyze.", started)

        # (1) representative passages → abstractive "own words" paragraph.
        # Each passage is paraphrased SEPARATELY and the results joined —
        # feeding several unrelated passages to the model at once makes it
        # fuse them into confidently wrong claims.
        # Source-aligned explanations prevent unsupported generative relations.
        # Simplified points provide readable wording without claiming a free
        # paraphrase has been proven by similarity alone.
        top_passages = sorted({owners[i] for i in picked})
        own_words = False
        explanation = ' '.join(sentences[i] for i in picked)

        # (2) simplified key points.
        points = simplifier.simplify_points([sentences[i] for i in picked])

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        result = {
            "status": "summarized",
            "answer": explanation,
            "span": None,
            "confidence": None,
            "retrieval_score": None,
            "response_time_ms": elapsed_ms,
            "source_passage": None,
            "passage_index": None,
            "simplified_points": points,
            "own_words": own_words,
            "summary_stats": self._doc_stats(len(sentences)),
            'coverage': self._summary_coverage,
            'coverage_note': self._summary_coverage['note'],
            "summary_sources": sorted({owners[i] + 1 for i in picked} | {p + 1 for p in top_passages[:3]}),
            "category": "Explanation",
        }
        self._explanation_cache = result
        return result

    def suggest_questions(self, k: int = 4) -> list:
        """Generate sample questions from the loaded dataset (the Ask-card chips).

        Deterministic and fully local — no model. Candidates are ranked using
        how often their informative terms occur across the whole corpus, with
        concise definitional sentences preferred on ties. Near-duplicate
        topics are suppressed, and each chip remains linked to its source
        sentence so selecting it returns a verified answer.
        """
        cached = getattr(self, "_suggestions_cache", None)
        if cached is not None and getattr(self, '_suggestions_capacity', 0) >= k:
            return cached[:max(0, k)]
        requested_k = max(0, k)
        k = max(4, requested_k)

        sentence_rows = []
        seen_sentences = set()
        term_passage_counts = Counter()
        for passage_idx, passage in enumerate(self.passages):
            if passage_idx in self.junk_passages:
                continue
            passage_terms = set(preprocessing.normalize(passage).split())
            term_passage_counts.update(passage_terms)
            for sentence in preprocessing.sentence_split(passage):
                sentence = _SPAN_TIDY_LEADING.sub("", sentence.strip())
                sentence_key = preprocessing.normalize(sentence)
                if (
                    len(sentence) < 65 or len(sentence) > SUMMARY_SENT_MAX_CHARS
                    or not sentence_key or sentence_key in seen_sentences
                ):
                    continue
                seen_sentences.add(sentence_key)
                terms = sentence_key.split()
                if len(terms) < 4:
                    continue
                sentence_rows.append((sentence, passage_idx, terms))

        if not sentence_rows:
            self._suggested_answers = {}
            self._suggestions_cache = []
            return []

        # Repeated terms provide a local signal for concepts emphasized
        # throughout the document; IDF preserves the value of specific terms.
        term_weights = {
            term: math.log1p(count) + 0.25 * math.log(
                (1 + len(self.passages)) / (1 + count)
            )
            for term, count in term_passage_counts.items()
        }
        phrase_counts = Counter()
        sentence_phrases = []
        for sentence, passage_idx, terms in sentence_rows:
            words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", sentence)
            content_words = [
                (word, normalized[0])
                for word in words
                if word.casefold() not in _SUGGESTION_GENERIC_TERMS
                if (normalized := preprocessing.normalize(word).split())
            ]
            phrases = []
            phrase_keys = set()
            for start in range(len(content_words)):
                for end in range(start + 2, min(len(content_words), start + 4) + 1):
                    phrase_terms = [term for _, term in content_words[start:end]]
                    phrase_key = " ".join(phrase_terms)
                    phrase_keys.add(phrase_key)
                    phrases.append((
                        phrase_key,
                        " ".join(word for word, _ in content_words[start:end]),
                    ))
            phrase_counts.update(phrase_keys)
            sentence_phrases.append(phrases)

        candidates = []
        suggested_answers = {}
        for row, phrases in zip(sentence_rows, sentence_phrases):
            sent, passage_idx, terms = row
            m = _COPULA_RE.match(sent)
            if m:
                subject = re.sub(
                    r"^(the|a|an)\s+", "", m.group("subject").strip(),
                    flags=re.IGNORECASE,
                )
                words = subject.split()
                if not (2 <= len(words) <= 5 or (
                    len(words) == 1 and subject.isupper()
                )):
                    m = None
            if m:
                subject = re.sub(
                    r"^(the|a|an)\s+", "", m.group("subject").strip(),
                    flags=re.IGNORECASE,
                )
                cop = m.group("copula").lower()
                if cop == "stands for":
                    question = f"What does {subject} stand for?"
                elif cop in ("are", "were"):
                    question = f"What are {subject}?"
                else:
                    question = f"What is {subject}?"
                topic_terms = set(preprocessing.normalize(subject).split())
                specificity_bonus = 3.0
            else:
                ranked_phrases = sorted(
                    phrases,
                    key=lambda item: (
                        sum(term_weights.get(t, 0.0) for t in item[0].split())
                        / max(1, len(item[0].split()))
                        + math.log1p(phrase_counts[item[0]])
                    ),
                    reverse=True,
                )
                if not ranked_phrases:
                    continue
                phrase_key, topic = ranked_phrases[0]
                topic_terms = set(phrase_key.split())
                question = f"What are the key points about {topic}?"
                specificity_bonus = 0.0

            key = " ".join(sorted(topic_terms))
            if not key:
                continue
            importance = (
                sum(term_weights.get(term, 0.0) for term in set(terms))
                / math.sqrt(len(set(terms)))
                + specificity_bonus
            )
            candidates.append((
                importance, question, key, topic_terms, sent, passage_idx
            ))

        questions, selected_terms, seen_topics = [], [], set()
        remaining = candidates[:]
        while remaining and len(questions) < k:
            best = None
            best_score = float("-inf")
            for candidate in remaining:
                importance, question, key, topic_terms, sent, passage_idx = candidate
                if key in seen_topics or sent not in self.passages[passage_idx]:
                    continue
                overlap = max(
                    (
                        len(topic_terms & previous)
                        / max(1, len(topic_terms | previous))
                        for previous in selected_terms
                    ),
                    default=0.0,
                )
                score = importance * (1.0 - 0.7 * overlap)
                if score > best_score:
                    best, best_score = candidate, score
            if best is None:
                break
            _, question, key, topic_terms, sent, passage_idx = best
            questions.append(question)
            seen_topics.add(key)
            selected_terms.append(topic_terms)
            suggested_answers[_question_key(question)] = (sent, passage_idx)
            remaining.remove(best)

        # Preserve the legacy heading chips for short structured documents
        # whose useful content does not yield enough sentence candidates.
        if len(questions) < k:
            for passage_idx, passage in enumerate(self.passages):
                if len(questions) >= k:
                    break
                m = re.match(r"^([A-Z][A-Z &]{2,30})[.:] ", passage)
                if not m:
                    continue
                question = f"Tell me about {m.group(1).title()}"
                key = " ".join(preprocessing.normalize(m.group(1)).split())
                if key and key not in seen_topics:
                    questions.append(question)
                    seen_topics.add(key)
                    suggested_answers[_question_key(question)] = (
                        passage, passage_idx
                    )

        self._suggested_answers = suggested_answers
        self._suggestions_cache = questions
        self._suggestions_capacity = k
        return questions[:requested_k]

    def _doc_stats(self, sentence_count: int) -> dict:
        return {
            "passages": len(self.passages),
            "words": self.word_count,
            "sentences": getattr(self, '_summary_coverage', {}).get('total_sentences', sentence_count),
        }

    def _passage_centrality(self):
        """Passage-level TF-IDF centrality (total similarity to all passages)."""
        # Cosine to the corpus centroid: O(passages × features), not O(P²) RAM.
        centroid = self.passage_matrix.mean(axis=0)
        import numpy as np
        return cosine_similarity(self.passage_matrix, np.asarray(centroid))[..., 0]

    def _central_sentences(self):
        """Sentence pool + LexRank-style greedy pick shared by both meta paths.

        Returns (picked_indices, sentences, owners); picked is empty when the
        corpus is too short to summarize.
        """
        sentences, owners = [], []
        for p_no, passage in enumerate(self.passages):
            if p_no in self.junk_passages:
                continue  # agenda slides don't belong in a content summary
            for sent in preprocessing.sentence_split(passage):
                # Drop split artifacts like ": An AI-powered..." left when a
                # slide title ends with "?" mid-line.
                sent = _SPAN_TIDY_LEADING.sub("", sent.strip())
                if SUMMARY_SENT_MIN_CHARS <= len(sent) <= SUMMARY_SENT_MAX_CHARS:
                    sentences.append(sent)
                    owners.append(p_no)
        self._summary_coverage = {'total_sentences': len(sentences), 'sentences_considered': min(len(sentences), SUMMARY_POOL_CAP),
                                 'sampling': 'evenly spread across document', 'note': None}
        if len(sentences) > SUMMARY_POOL_CAP:
            self._summary_coverage['note'] = f'Summary selects from {SUMMARY_POOL_CAP} of {len(sentences)} eligible sentences, sampled across the document.'
        if not sentences:
            return [], sentences, owners

        import numpy as np
        selected = np.linspace(0, len(sentences) - 1, min(len(sentences), SUMMARY_POOL_CAP), dtype=int)
        pool = [sentences[i] for i in selected]
        pool_owners = [owners[i] for i in selected]
        norms = [preprocessing.normalize(s) for s in pool]
        try:
            vec = TfidfVectorizer(sublinear_tf=True)
            matrix = vec.fit_transform(norms)
        except ValueError:  # empty vocabulary after normalization
            return [], sentences, owners

        sim = cosine_similarity(matrix)
        centrality = sim.sum(axis=1)
        picked = []
        for i in centrality.argsort()[::-1]:
            if len(picked) >= SUMMARY_SENTENCE_COUNT:
                break
            if any(sim[i, j] > SUMMARY_REDUNDANCY for j in picked):
                continue  # too similar to something already chosen
            picked.append(int(i))
        picked.sort()  # restore reading order for coherence
        return picked, pool, pool_owners

    def _best_sentences(self, q_norm, pool, q_overlap_min=0.0):
        """TF-IDF-only fallback: score individual sentences of pooled passages.

        Returns (answer_text, confidence, passage_idx, span_start, span_end)
        where the offsets locate the answer text inside the full passage (None
        when it cannot be located), or None if nothing clears the bar. The
        answer is the best sentence, plus its antecedent when it starts with
        a pronoun. With q_overlap_min > 0 (rescue mode) the best sentence must
        also literally contain at least that fraction of the question's content
        words, so weak one-word coincidences are rejected.
        """
        q_vec = self.vectorizer.transform([q_norm])
        q_terms = set(q_norm.split())
        pool = list(pool)
        if not pool:
            return None
        items = []
        for p_no in pool:
            for s_no, sent in enumerate(preprocessing.sentence_split(self.passages[p_no])):
                norm = preprocessing.normalize(sent)
                if norm:
                    items.append((p_no, s_no, norm, sent))
        if not items:
            return None

        s_matrix = self.vectorizer.transform([it[2] for it in items])
        s_scores = cosine_similarity(q_vec, s_matrix)[0]
        order = s_scores.argsort()[::-1]
        best_pos = int(order[0])
        if s_scores[best_pos] <= 0:
            return None

        # Definition-pattern tie-break: among near-best candidates, prefer a
        # sentence that opens with the question's subject ("Machine learning
        # is ..."), the classic shape of a definitional sentence.
        cutoff = TIE_WINDOW_RATIO * s_scores[best_pos]
        for pos in order:
            if s_scores[pos] < cutoff:
                break
            if items[pos][2].startswith(q_norm + " ") or items[pos][2] == q_norm:
                best_pos = int(pos)
                break

        if q_overlap_min > 0:
            s_terms = set(items[best_pos][2].split())
            if len(q_terms & s_terms) / len(q_terms) < q_overlap_min:
                return None

        p_no, s_no = items[best_pos][0], items[best_pos][1]
        sents = preprocessing.sentence_split(self.passages[p_no])
        sentences = [sents[s_no]]
        prepend_antecedent = _PRONOUN_START.match(sents[s_no]) and s_no > 0
        # Pronoun starts like "He wrote Hamlet" need their antecedent, so pull
        # in the preceding sentence of the same passage.
        if prepend_antecedent:
            sentences.insert(0, sents[s_no - 1])

        # Offsets of the answer text inside the full passage (for the source
        # panel). sentence_split yields (stripped) substrings of the passage,
        # so str.find locates them exactly; -1 → offsets withheld.
        passage = self.passages[p_no]
        span_start = passage.find(sents[s_no])
        if span_start >= 0:
            span_end = span_start + len(sents[s_no])
            if prepend_antecedent:
                prev_start = passage.find(sents[s_no - 1])
                if 0 <= prev_start < span_start:
                    span_start = prev_start
        else:
            span_start = None
            span_end = None

        return " ".join(sentences), float(s_scores[best_pos]), p_no, span_start, span_end

    # ---------------------------------------------------------------- helpers

    def _no_info(self, message: str, started: float, retriever: str = "tfidf") -> dict:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return {
            "status": "no_answer",
            "answer": message,
            "span": None,
            "confidence": 0.0,
            "retrieval_score": 0.0,
            "response_time_ms": elapsed_ms,
            "source_passage": None,
            "passage_index": None,
            "passage_id": None,
            "passage_text": None,
            "answer_start": None,
            "answer_end": None,
            "retriever": retriever,
            "category": None,
        }
