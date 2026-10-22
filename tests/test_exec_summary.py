"""
tests/test_exec_summary.py — the "Generate Executive Summary" feature.

Fast tests mock the summarization pipeline; one integration test
(`pytest -m slow`) runs the real local model on the sample dataset.

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module  # noqa: E402
import summary_engine  # noqa: E402


class FakePipe:
    """Stands in for the HF summarization pipeline.

    Echoes a deterministic, sentence-bearing summary of whatever it gets and
    counts its calls, so tests can assert map-reduce behavior.
    """

    def __init__(self, fail=False, empty=False):
        self.calls = []
        self.fail = fail
        self.empty = empty

    def __call__(self, text, **kwargs):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("model exploded")
        if self.empty:
            return [{"summary_text": ""}]
        first = (text.split(".")[0].strip() or "empty chunk")[:90]
        return [{"summary_text": (
            f"The document explains {first}. Key ideas are summarized clearly. "
            "The main findings remain consistent throughout. A final observation "
            "closes the text."
        )}]


@pytest.fixture()
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client


def _load_sample(client):
    resp = client.post("/api/sample")
    assert resp.status_code == 200
    return resp.get_json()


def _clear_cache():
    summary_engine._CACHE.clear()


def _patch_pipe(monkeypatch, pipe):
    monkeypatch.setattr(summary_engine.qa_model, "get_summarizer", lambda: pipe)


# ------------------------------------------------------------ unit tests

def test_bulletize_caps_at_four_complete_sentences():
    text = ("The system learns from labelled examples. It builds a model for "
            "predictions. Results are cached between runs. A final check "
            "validates everything. An extra sentence exists here.")
    bullets = summary_engine._bulletize(text)
    assert len(bullets) == summary_engine.BULLET_COUNT == 4
    assert all(b.endswith(".") and len(b.split()) >= 4 for b in bullets)


def test_map_reduce_groups_long_corpus(monkeypatch):
    pipe = FakePipe()
    passages = [
        f"Passage {i} discusses topic number {i} in some detail. " * 6
        for i in range(30)
    ]
    bullets, chunks_used = summary_engine.summarize_passages(
        passages, pipe=pipe
    )
    # chunk cap: only MAX_CHUNKS passages summarized
    assert chunks_used == summary_engine.MAX_CHUNKS
    # map step ran once per group (multiple groups for 24 long passages)
    assert 1 < len(pipe.calls) <= summary_engine.MAX_CHUNKS
    # every map input stayed inside the model's char budget
    assert all(len(c) <= summary_engine.GROUP_CHAR_LIMIT for c in pipe.calls)
    # reduce step ran on the concatenated partials
    assert "The document explains" in pipe.calls[-1]
    assert 1 <= len(bullets) <= summary_engine.BULLET_COUNT


def test_short_passages_single_group(monkeypatch):
    pipe = FakePipe()
    passages = ["Short passage about cats." for _ in range(3)]
    _, chunks_used = summary_engine.summarize_passages(passages, pipe=pipe)
    assert chunks_used == 3
    assert len(pipe.calls) == 2  # one map group + one reduce


def test_model_failure_raises(monkeypatch):
    with pytest.raises(RuntimeError):
        summary_engine.summarize_passages(["Some text here."], pipe=FakePipe(fail=True))


def test_empty_model_output_raises(monkeypatch):
    with pytest.raises(RuntimeError):
        summary_engine.summarize_passages(["Some text here."], pipe=FakePipe(empty=True))


# --------------------------------------------------------- endpoint tests

def test_endpoint_requires_dataset(client):
    resp = client.post("/api/summarize")
    assert resp.status_code == 409


def test_endpoint_too_short(client, monkeypatch):
    _clear_cache()
    client.post("/api/paste", json={
        "text": "Just a few short sentences here. The dataset stays small on "
                "purpose. It has fewer than one hundred words in total."
    })
    _patch_pipe(monkeypatch, FakePipe())  # must never be called
    resp = client.post("/api/summarize")
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["status"] == "too_short"
    assert "too short" in body["message"]


def test_endpoint_ok_then_cached_stats_unchanged(client, monkeypatch):
    _clear_cache()
    _load_sample(client)
    stats_before = client.get("/api/stats").get_json()

    pipe = FakePipe()
    _patch_pipe(monkeypatch, pipe)
    first = client.post("/api/summarize").get_json()
    assert first["status"] == "ok"
    assert first["cached"] is False
    assert 3 <= len(first["bullets"]) <= 4
    assert all(b.endswith((".", "!", "?")) for b in first["bullets"])
    calls_after_first = len(pipe.calls)

    second = client.post("/api/summarize").get_json()
    assert second["cached"] is True
    assert second["bullets"] == first["bullets"]
    assert second["response_time_ms"] <= first["response_time_ms"]
    assert len(pipe.calls) == calls_after_first  # model not re-invoked on cache hit

    # summaries do NOT count toward dashboard counters
    stats_after = client.get("/api/stats").get_json()
    assert stats_after["total_queries"] == stats_before["total_queries"] == 0


def test_endpoint_model_unavailable(client, monkeypatch):
    _clear_cache()
    _load_sample(client)
    _patch_pipe(monkeypatch, None)
    resp = client.post("/api/summarize")
    assert resp.status_code == 503
    assert "Summary failed" in resp.get_json()["message"]


def test_endpoint_large_document_capped(client, monkeypatch):
    _clear_cache()
    _load_sample(client)
    # simulate a 50-page document: many passages re-loaded into the engine
    big = " ".join(
        f"Section {i} explains another aspect of the system in complete sentences "
        "with plenty of connecting words to fill the group window realistically."
        for i in range(300)
    )
    client.post("/api/paste", json={"text": big})
    engine = app_module._engine()
    assert engine.passage_count > summary_engine.MAX_CHUNKS

    _patch_pipe(monkeypatch, FakePipe())
    resp = client.post("/api/summarize")
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["coverage"]["chunks_used"] == summary_engine.MAX_CHUNKS
    assert body["coverage_note"] and "first" in body["coverage_note"]


# ------------------------------------------------------ integration (slow)

@pytest.mark.slow
def test_integration_real_model(client):
    _clear_cache()
    _load_sample(client)
    resp = client.post("/api/summarize")
    body = resp.get_json()
    assert resp.status_code == 200, body
    assert body["status"] == "ok"
    assert 3 <= len(body["bullets"]) <= 4
    assert all(len(b.split()) >= 4 for b in body["bullets"])
    assert body["response_time_ms"] < 120_000  # CPU budget for 3 map + 1 reduce
