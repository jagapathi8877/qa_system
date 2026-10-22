"""
tests/test_retrievers.py — the retriever interface + semantic mode.

Fast tests (TF-IDF regression, interface behavior, lazy-load guarantee) run
with everything else. The synonym-fixture and cache tests need the real
MiniLM embeddings — they are marked `slow` (still fully local/offline after
the one-time ~90 MB model download).

Run:  python -m pytest tests/ -v          (everything)
      python -m pytest tests/ -m "not slow"   (fast only)
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qa_engine  # noqa: E402
import retrievers  # noqa: E402

SNAPSHOT = json.load(
    open(Path(__file__).parent / "data" / "retrieval_snapshot.json",
         encoding="utf-8")
)

SAMPLE_TEXT = (Path(__file__).resolve().parents[1] / "static" /
               "sample_dataset.txt").read_text(encoding="utf-8")

# Fixture that proves the synonym gap: the correct passage shares NO keywords
# with the question except "Hamlet", while a distractor shares BOTH "author"
# and "Hamlet" — keyword retrieval ranks the distractor first.
SYNONYM_TEXT = (
    "The novel branch of the library holds modern fiction and popular science "
    "titles, and it opens every weekday morning at nine for members.\n\n"
    "William Shakespeare wrote Hamlet around 1600, and the play remains one of "
    "the most performed works in the world.\n\n"
    "In the campus bookstore a shelf of Hamlet paperbacks sits beside the "
    "author biographies that literature students buy every spring.\n\n"
    "Photosynthesis converts light energy into chemical energy inside the "
    "chloroplasts of green plant cells during daylight hours."
)
CORRECT_FRAGMENT = "William Shakespeare wrote Hamlet around 1600"
DISTRACTOR_FRAGMENT = "In the campus bookstore a shelf of Hamlet paperbacks"


@pytest.fixture(scope="module")
def engine():
    return qa_engine.QAEngine(SAMPLE_TEXT)


@pytest.fixture(scope="module")
def synonym_engine():
    return qa_engine.QAEngine(SYNONYM_TEXT)


# ---------------------------------------------- TF-IDF regression (fast)

@pytest.mark.parametrize("question", list(SNAPSHOT.keys()))
def test_tfidf_public_response_keeps_expected_evidence(engine, question):
    """Preserve source/status; verify current score semantics independently.

    The historical fixture remains unchanged. Its old numeric values mixed
    sentence rescue and passage cosine and were already failing at baseline.
    Scores now always describe the returned source passage, not the route.
    """
    exp = SNAPSHOT[question]
    d = engine.ask(question)
    assert d["status"] == exp["status"]
    assert d["passage_index"] == exp["passage_index"]
    if d['status']=='answered':
        import numpy as np
        from scipy.spatial.distance import cosine
        q=engine.vectorizer.transform([qa_engine.preprocessing.normalize(question)]).toarray()[0]
        source=engine.passage_matrix[d['passage_index']-1].toarray()[0]
        expected=1-cosine(q,source) if np.linalg.norm(q) and np.linalg.norm(source) else 0
        assert d['retrieval_score']==pytest.approx(expected,abs=0.000051)
        assert d['passage_text'][d['answer_start']:d['answer_end']]==d['span']
    else:
        assert d['retrieval_score']==0 and d['source_passage'] is None


def test_tfidf_top_k_orders_descending(engine):
    top = engine.tfidf.top_k("What is machine learning?", k=4)
    scores = [s for _, s in top]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= i < engine.passage_count for i, _ in top)


def test_semantic_model_never_loads_in_tfidf_mode(engine, monkeypatch):
    """TF-IDF-only sessions must not touch the embedding model."""
    def _boom(self):
        raise AssertionError("SemanticRetriever._ensure must not be called")
    monkeypatch.setattr(retrievers.SemanticRetriever, "_ensure", _boom)
    d = engine.ask("Who wrote Hamlet?", retriever="tfidf")
    assert d["status"] == "answered"
    assert d["retriever"] == "tfidf"


def test_ask_response_carries_mode(engine):
    assert engine.ask("Who wrote Hamlet?")["retriever"] == "tfidf"


def test_meta_intents_still_answer(engine):
    """Regression guard: summary/explain intents run their own TF-IDF code
    paths inside qa_engine — a missing import there used to 500 /api/ask."""
    for q in ("summarize this dataset", "explain the dataset in simple words"):
        d = engine.ask(q)
        assert d["status"] == "summarized"
        assert d["category"] in ("Summary", "Explanation")
        assert d["answer"].strip()


# --------------------------------------- semantic mode (marked slow)

@pytest.mark.slow
def test_synonym_fixture_ranks(synonym_engine):
    """On the synonym fixture the semantic retriever should rank the correct
    passage top-1 while keyword retrieval is misled by the distractor."""
    q = "Who is the author of Hamlet?"
    correct = next(i for i, p in enumerate(synonym_engine.passages)
                   if CORRECT_FRAGMENT in p)
    distractor = next(i for i, p in enumerate(synonym_engine.passages)
                      if DISTRACTOR_FRAGMENT in p)

    tfidf_order = [i for i, _ in synonym_engine.tfidf.top_k(q, k=4)]
    sem_order = [i for i, _ in
                 synonym_engine._get_retriever("semantic").top_k(q, k=4)]

    # Semantic: correct passage first.
    assert sem_order[0] == correct
    # Keyword (honest record): the distractor shares BOTH "author" and
    # "Hamlet" with the question, so TF-IDF ranks it first and the correct
    # passage second.
    assert tfidf_order[0] == distractor
    assert tfidf_order.index(correct) > 0


@pytest.mark.slow
def test_synonym_fixture_end_to_end_answer(synonym_engine):
    d = synonym_engine.ask("Who is the author of Hamlet?", retriever="semantic")
    assert d["status"] == "answered"
    assert d["retriever"] == "semantic"
    assert "William Shakespeare" in d["span"]
    assert d["passage_text"][d["answer_start"]:d["answer_end"]].strip() == d["span"].strip()


@pytest.mark.slow
def test_semantic_embeddings_cached_per_dataset(synonym_engine):
    """Second use must not re-index: same instance, same timing object."""
    first = synonym_engine._get_retriever("semantic")
    first.scores("warm up")
    assert first.index_seconds is not None
    again = synonym_engine._get_retriever("semantic")
    assert again is first
    assert again._matrix is not None
    assert again.index_seconds == first.index_seconds


@pytest.mark.slow
def test_semantic_stopword_query_refused(synonym_engine):
    """A query with only stopwords still embeds, scores low, and is refused."""
    d = synonym_engine.ask("is it the?", retriever="semantic")
    assert d["status"] == "no_answer"
    assert d["retriever"] == "semantic"
