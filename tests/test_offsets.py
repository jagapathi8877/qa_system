"""
tests/test_offsets.py — correctness of the answer-span character offsets.

Run with:  python -m pytest tests/ -v   (after `pip install -r requirements-dev.txt`)

The QA model reads the stored passage directly, so the offsets returned by the
Hugging Face pipeline are already relative to the full stored passage. These
tests pin the core invariant — passage_text[answer_start:answer_end] equals the
answer span — across every code path that can produce an answer:
direct retrieval, windowed (multi-window) paragraphs, the title-echo retry
(which re-asks against a shortened context and must shift offsets back), and
the TF-IDF fallback that runs when the model is unavailable.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qa_engine  # noqa: E402
import qa_model  # noqa: E402


def norm(s):
    """Whitespace-normalized comparison (the pipeline strips/collapses spaces)."""
    return " ".join(s.split())


# --------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def engine():
    qa_engine_wordlist = importlib.reload(qa_engine)
    text = (
        "William Shakespeare was an English playwright and poet born in 1564 in "
        "Stratford upon Avon, widely regarded as the greatest writer in the English "
        "language. He wrote around thirty seven plays, including tragedies such as "
        "Hamlet, Macbeth, and King Lear.\n\n"
        "Machine learning is a branch of artificial intelligence in which a system "
        "learns patterns from data instead of following hand-written rules. Common "
        "applications include spam filtering, speech recognition, and medical "
        "diagnosis.\n\n"
        "Photosynthesis is the process by which green plants convert light energy "
        "into chemical energy inside organelles called chloroplasts."
    )
    return qa_engine_wordlist.QAEngine(text)


def _assert_span_invariant(result):
    """The acceptance invariant: passage_text[start:end] equals the span."""
    assert result["status"] == "answered"
    start, end = result["answer_start"], result["answer_end"]
    assert start is not None and end is not None, "offsets missing on answered result"
    assert isinstance(start, int) and isinstance(end, int)
    passage = result["passage_text"]
    assert result["passage_id"] == result["passage_index"]
    assert result["passage_text"] == result["source_passage"]
    assert 0 <= start < end <= len(passage)
    assert norm(passage[start:end]) == norm(result["span"]), (
        f"slice {passage[start:end]!r} != span {result['span']!r}"
    )


# ------------------------------------------------------------------ tests

def test_hamlet_offsets(engine):
    """The screenshot scenario: 'Who wrote Hamlet?' → 'William Shakespeare'."""
    result = engine.ask("Who wrote Hamlet?")
    _assert_span_invariant(result)
    assert norm(result["span"]) == "William Shakespeare"
    assert "William Shakespeare" in result["passage_text"]


def test_offsets_within_bounds_and_consistent(engine):
    for question in ("When was Shakespeare born?", "What is machine learning?",
                     "What is photosynthesis?"):
        result = engine.ask(question)
        _assert_span_invariant(result)


def test_no_answer_has_no_source(engine):
    result = engine.ask("What is the capital of France?")
    assert result["status"] == "no_answer"
    assert result["passage_id"] is None
    assert result["passage_text"] is None
    assert result["answer_start"] is None
    assert result["answer_end"] is None


def test_long_paragraph_multi_window(engine):
    """A paragraph > PARAGRAPH_MAX_CHARS is re-cut into sliding windows; the
    stored window IS the passage the model reads, so offsets must index the
    window's text — proven here with an answer that lives in the last window."""
    filler = " ".join(
        f"Sentence number {i} provides additional filler detail about the topic "
        "without adding any new factual content at all."
        for i in range(30)
    )
    long_paragraph = (
        "Quantum tunneling is a phenomenon where particles pass through barriers. "
        + filler +
        " The probability of tunneling decreases exponentially with barrier width, "
        "which is the key limiting factor for practical quantum devices today."
    )
    assert len(long_paragraph) > qa_engine.preprocessing.PARAGRAPH_MAX_CHARS
    passages = qa_engine.preprocessing.build_passages(long_paragraph)
    assert len(passages) > 1, "corpus fixture must produce multiple windows"

    windowed_engine = qa_engine.QAEngine(long_paragraph)
    assert windowed_engine.passage_count == len(passages)

    result = windowed_engine.ask("What decreases exponentially with barrier width?")
    _assert_span_invariant(result)
    # offsets are relative to the stored WINDOW, not the raw paragraph:
    # the answer window must be a later window and sized like a window.
    assert result["passage_index"] > 1
    assert len(result["passage_text"]) <= qa_engine.preprocessing.PARAGRAPH_MAX_CHARS + 200
    assert result["passage_text"] in passages


def test_echo_retry_offsets_shifted():
    """Title-echo retry re-asks on context minus its first sentence; the
    returned offsets must be shifted back to the FULL context/passage."""
    passage = (
        "Project Nightingale. Project Nightingale is the internal codename for the "
        "search platform that the platform engineering team built during the last "
        "two years and it now serves the whole organisation across three departments."
    )
    engine = qa_engine.QAEngine(passage)
    result = engine.ask("What is the project codename?")
    _assert_span_invariant(result)


def test_fallback_path_offsets(engine, monkeypatch):
    """Without the QA model, the TF-IDF fallback locates the answer sentence
    inside the passage and still satisfies the span invariant."""
    monkeypatch.setattr(qa_model, "get_pipeline", lambda: None)
    result = engine.ask("Who wrote Hamlet?")
    _assert_span_invariant(result)


def test_passage_text_is_raw_text_not_html(engine):
    """The API returns the passage verbatim; escaping is the frontend's job
    (it renders text nodes only), so the payload must be untouched."""
    text = (
        "Malware reflections. Some documentation contains raw markup like "
        "<script>alert(1)</script> inside its body text and the system must pass "
        "it through unchanged for the UI to escape it safely."
    )
    eng = qa_engine.QAEngine(text)
    result = eng.ask("What contains raw markup?")
    if result["status"] == "answered" and result.get("passage_text"):
        assert "<script>alert(1)</script>" in result["passage_text"]
        if result["answer_start"] is not None:
            _assert_span_invariant(result)
