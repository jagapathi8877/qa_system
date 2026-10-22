"""
tests/test_bug_fixes.py — regression tests for bug fixes.
"""

import io
import json
import pytest

import app as app_module
import ingestion
import preprocessing
from stats import classify_query


def test_json_ingestion_booleans_and_numbers():
    raw_dict = {
        "years": [1999, 2020, 2024],
        "active": True,
        "score": 98.5,
        "title": "NLP Project"
    }
    raw = json.dumps(raw_dict).encode("utf-8")
    extracted = ingestion.extract_text("test.json", io.BytesIO(raw))
    assert "years: 1999, 2020, 2024" in extracted
    assert "active: True" in extracted
    assert "score: 98.5" in extracted
    assert "title: NLP Project" in extracted


def test_csv_ingestion_none_omitted():
    csv_data = b"Name,Age,Role\nAlice,30,Engineer\nBob,25\n"
    extracted = ingestion.extract_text("test.csv", io.BytesIO(csv_data))
    assert "Alice" in extracted
    assert "None" not in extracted


def test_preprocessing_trailing_passage_retained():
    text = (
        "This is an introductory paragraph that definitely exceeds forty characters in total length.\n\n"
        "Contact: info@domain.com"
    )
    passages = preprocessing.build_passages(text)
    assert len(passages) >= 1
    assert "info@domain.com" in passages[0]


def test_preprocessing_short_single_passage_retained():
    text = "Short text under forty chars."
    passages = preprocessing.build_passages(text)
    assert len(passages) == 1
    assert passages[0] == text


def test_stats_comparison_query_priority():
    q = "What is the difference between supervised and unsupervised learning?"
    assert classify_query(q) == "Comparison"

    q2 = "How to compare two machine learning models?"
    assert classify_query(q2) == "Comparison"

    q3 = "What is deep learning?"
    assert classify_query(q3) == "Definition"


def test_app_paste_invalid_type():
    client = app_module.app.test_client()
    resp = client.post("/api/paste", json={"text": 12345})
    assert resp.status_code == 400
    assert "Paste some text first" in resp.get_json()["message"]


def test_app_ask_invalid_type():
    client = app_module.app.test_client()
    client.post("/api/sample")
    before = client.get('/api/stats').get_json()['total_queries']
    resp = client.post("/api/ask", json={"question": 12345, "retriever": "tfidf"})
    assert resp.status_code == 400
    assert resp.get_json()["status"] == "error"
    assert client.get('/api/stats').get_json()['total_queries'] == before
