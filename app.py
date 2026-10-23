"""
app.py — QNext, a session-scoped document QA system.

Run with:  python app.py   →  http://127.0.0.1:5000

Flask routes
------------
GET  /                     → the single-page UI
POST /api/upload           → file upload  {file}
POST /api/paste            → pasted text  {text}
GET  /api/dataset          → current dataset state (or 404-style flag)
DELETE /api/dataset        → reset session (new dataset can be uploaded)
POST /api/ask              → answer a question {question, threshold?}
GET  /api/stats            → dashboard metrics for this session

Every browser gets its own corpus + stats via a signed session cookie, so two
people using the app at the same time never see each other's data.
"""

import os
import time
import uuid
import math
import secrets
import copy
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).with_name('.env'), override=False)

from flask import Flask, jsonify, render_template, request, session, g
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

import ingestion
import qa_model
import summary_engine
from qa_engine import QAEngine
from stats import SessionStats, AIStats, classify_query
from ai_explainer import AIExplainer, disabled_result
from gemini_gateway import classify_error, log_event
import logging

ai_explainer = AIExplainer()

app = Flask(__name__)
_ai_logger = logging.getLogger('qnext.ai')
_ai_logger.setLevel(logging.INFO)
if not _ai_logger.hasHandlers():
    _ai_logger.addHandler(logging.StreamHandler())
_configured_secret = os.environ.get('QUILLQA_SECRET')
if os.environ.get('QUILLQA_ENV') == 'production' and (not _configured_secret or len(_configured_secret) < 32):
    raise RuntimeError('Production requires a private QUILLQA_SECRET of at least 32 characters.')
app.secret_key = _configured_secret or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB uploads
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')

import threading
from collections import OrderedDict

MAX_SESSIONS = 50
SESSION_TTL_SECONDS = 3600
_STATE: OrderedDict[str, dict] = OrderedDict()
_STATE_LOCK = threading.Lock()


# ------------------------------------------------------------------ session

def _state(create: bool = False) -> dict:
    """Empty readers do not allocate capacity; active datasets are never evicted."""
    sid = session.get('sid') or getattr(g, 'qa_sid', None) or uuid.uuid4().hex
    g.qa_sid = sid
    with _STATE_LOCK:
        st = getattr(g, 'qa_state', None) or _STATE.get(sid)
        if st is None:
            st = {'engine': None, 'stats': SessionStats(), 'source': None,
                  'lock': threading.RLock(), 'last_seen': time.monotonic(), 'history': [],
                  'ai_stats': AIStats(), 'dataset_revision': uuid.uuid4().hex}
        if create and sid not in _STATE:
            now = time.monotonic()
            for key, value in list(_STATE.items()):
                if now - value['last_seen'] > SESSION_TTL_SECONDS and value['lock'].acquire(blocking=False):
                    try:
                        _STATE.pop(key, None)
                    finally:
                        value['lock'].release()
            if len(_STATE) >= MAX_SESSIONS:
                raise RuntimeError('Dataset capacity reached. Reset an unused session or try again later.')
            _STATE[sid] = st
            session['sid'] = sid
        if sid in _STATE:
            st['last_seen'] = time.monotonic()
            _STATE.move_to_end(sid)
        g.qa_state = st
        return st


def _engine() -> QAEngine | None:
    return _state()["engine"]


def _error(message: str, status: int = 400):
    return jsonify({"status": "error", "message": message}), status


@app.before_request
def _serialize_session():
    if request.path.startswith('/api/') and request.path != '/api/health':
        # g belongs to an application context, which can span more than one
        # request in test clients/embedded use. Never reuse another request's sid.
        g.qa_sid = session.get('sid') or uuid.uuid4().hex
        g.qa_state = None
        state = _state()
        if not state['lock'].acquire(timeout=30):
            return _error('This session is busy. Please try again shortly.', 503)
        g.qa_lock = state['lock']


@app.teardown_request
def _release_session(_exception):
    lock = getattr(g, 'qa_lock', None)
    if lock is not None:
        lock.release()
        g.qa_lock = None


@app.errorhandler(BadRequest)
def _bad_request(exc):
    return _error(exc.description, 400)


@app.errorhandler(RequestEntityTooLarge)
def _request_too_large(_exc):
    return _error('The request exceeds the 25 MiB limit.', 413)


def _payload():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise BadRequest('Send a valid JSON object.')
    return payload


# ------------------------------------------------------------------- routes

@app.route("/")
def index():
    return render_template("index.html", ai_default_enabled=ai_explainer.config.enabled)


def _ingest(raw_text: str, source_label: str):
    """Shared by /api/upload and /api/paste."""
    started = time.perf_counter()
    try:
        engine = QAEngine(raw_text)
    except ValueError as exc:
        return _error(str(exc), 422)
    try:
        state = _state(create=True)
    except RuntimeError as exc:
        return _error(str(exc), 503)
    state["engine"] = engine
    state["source"] = source_label
    state['dataset_revision'] = uuid.uuid4().hex
    return jsonify({
        "status": "ok",
        "message": f"Dataset loaded — {engine.passage_count} passages indexed",
        "passages": engine.passage_count,
        "index_time_ms": round(engine.index_seconds * 1000),
        "source": source_label,
        "chars": len(raw_text),
        'corpus_id': engine.corpus_hash,
        'processing_time_ms': round((time.perf_counter() - started) * 1000),
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return _error("No file part in the request.")
    f = request.files["file"]
    if not f.filename:
        return _error("No file selected.")
    try:
        raw_text = ingestion.extract_text(f.filename, f.stream)
    except ingestion.UnsupportedFormatError as exc:
        return _error(str(exc), 415)
    except ingestion.ExtractionError as exc:
        return _error(str(exc), 422)
    except Exception:
        app.logger.exception('Document extraction failed')
        return _error('Could not extract text from this file. Check its format and encoding.', 415)
    if not raw_text.strip():
        return _error("No extractable text found in this file.", 422)
    return _ingest(raw_text, f.filename)


@app.route("/api/paste", methods=["POST"])
def paste():
    payload = _payload()
    raw_text = payload.get("text", "")
    text = raw_text.strip() if isinstance(raw_text, str) else ""
    if not text:
        return _error("Paste some text first (at least a couple of sentences).")
    if len(text) < 60:
        return _error("That text is too short to index — paste at least a paragraph.")
    return _ingest(text, "Pasted text")


@app.route("/api/dataset", methods=["GET", "DELETE"])
def dataset():
    state = _state()
    if request.method == "DELETE":
        state["engine"] = None
        state["source"] = None
        state["stats"] = SessionStats()
        state['ai_stats'] = AIStats()
        state['history'] = []
        with _STATE_LOCK:
            _STATE.pop(session.get('sid'), None)
        return jsonify({"status": "ok", "message": "Session reset."})
    engine = state["engine"]
    if engine is None:
        return jsonify({"status": "empty", "loaded": False})
    return jsonify({
        "status": "ok",
        "loaded": True,
        "source": state["source"],
        "passages": engine.passage_count,
        "index_time_ms": round(engine.index_seconds * 1000),
        'corpus_id': engine.corpus_hash,
    })


@app.route("/api/sample", methods=["POST"])
def sample():
    """One-click demo corpus served from static/sample_dataset.txt."""
    path = os.path.join(app.static_folder, "sample_dataset.txt")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return _error("sample_dataset.txt is missing from the static folder.", 500)
    return _ingest(text, "Sample dataset")


@app.route("/api/suggest")
def suggest():
    """Sample questions generated from the loaded dataset (the Ask-card chips)."""
    engine = _engine()
    if engine is None:
        return jsonify({"status": "empty", "questions": []})
    return jsonify({"status": "ok", "questions": engine.suggest_questions()})


@app.route("/api/summarize", methods=["POST"])
def summarize():
    """Executive summary of the loaded dataset.

    Deliberately separate from /api/ask: summaries are cached per dataset,
    are NOT recorded in the dashboard counters, and never touch retrieval or
    the QA model.
    """
    engine = _engine()
    if engine is None:
        return _error("Load a dataset before requesting a summary.", 409)

    started = time.perf_counter()
    if engine.word_count < summary_engine.MIN_WORDS:
        return jsonify({
            "status": "too_short",
            "message": (
                "This dataset is too short for an executive summary — it has "
                f"only {engine.word_count} words. Ask a question instead."
            ),
        })

    # Per-dataset cache: a second click on the same content returns instantly.
    cache_key = (summary_engine.SUMMARY_MODEL, engine.corpus_hash)
    cached = summary_engine.cache_get(cache_key)
    if cached is not None:
        return jsonify({
            "status": "ok",
            **cached,
            "model": summary_engine.SUMMARY_MODEL,
            "cached": True,
            "response_time_ms": round((time.perf_counter() - started) * 1000),
        })

    try:
        bullets, chunks_used = summary_engine.summarize_passages(engine.passages)
    except RuntimeError as exc:
        # model unavailable (offline first run / bad load) or degenerate output
        return _error(
            "Summary failed: local model unavailable. Run prepare_models.py --models summarizer "
            "to prepare the local model, then try again.", 503
        )
    except Exception as exc:  # unexpected — never crash the UI
        app.logger.exception('Executive summary failed')
        return _error('The executive summary could not be completed.', 500)

    coverage_note = None
    if engine.passage_count > summary_engine.MAX_CHUNKS:
        coverage_note = (
            f"Summary covers the first {chunks_used} of {engine.passage_count} "
            "sections of the document."
        )
    details = summary_engine.last_coverage(engine.passages, chunks_used)
    summary_engine.cache_put(cache_key, {'bullets': bullets, 'coverage': details,
                                       'coverage_note': coverage_note,
                                       'grounding': 'source-aligned sentences; no unverified free paraphrases'})
    return jsonify({
        "status": "ok",
        "bullets": bullets,
        "model": summary_engine.SUMMARY_MODEL,
        "cached": False,
        "coverage": details,
        "coverage_note": coverage_note,
        'grounding': 'source-aligned sentences; no unverified free paraphrases',
        "response_time_ms": round((time.perf_counter() - started) * 1000),
    })


@app.route("/api/ask", methods=["POST"])
def ask():
    payload = _payload()
    raw_question = payload.get("question")
    question = raw_question.strip() if isinstance(raw_question, str) else ""
    if not question:
        return _error('Please type a nonempty question.')
    if len(question) > 1000:
        return _error('Questions must contain at most 1000 characters.')
    # Omitted mode is retrieval-only for legacy API consumers. The UI sends
    # its default-ON preference explicitly; strings such as "false" are invalid.
    ai_mode = payload.get('ai_mode', False)
    defer_ai = payload.get('defer_ai', False)
    if type(ai_mode) is not bool or type(defer_ai) is not bool:
        return _error('ai_mode and defer_ai must be JSON booleans.')
    retriever = payload.get("retriever", "tfidf")
    if retriever not in ("tfidf", "semantic"):
        return _error('Retriever must be tfidf or semantic.')
    # No explicit threshold → each retriever mode uses its own default.
    threshold = payload.get("threshold")
    if threshold is not None:
        try:
            if isinstance(threshold, bool):
                raise ValueError()
            threshold = float(threshold)
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 0.9:
                raise ValueError()
        except (TypeError, ValueError):
            return _error('Threshold must be a finite number between 0 and 0.9.')

    engine = _engine()
    if engine is None and (not ai_mode or not ai_explainer.config.enabled):
        return _error('Turn on AI explanations or load a dataset before asking questions.'
                      if ai_explainer.config.enabled else 'AI is disabled. Load a dataset before asking questions.', 409)

    try:
        # General questions also need durable, session-scoped history for the
        # deferred explanation route. Do not create an empty retrieval index.
        state = _state(create=True)
    except RuntimeError as exc:
        return _error(str(exc), 503)
    if engine is None:
        result = {'status': 'no_answer', 'reason': 'no_dataset', 'answer': None,
                  'sources': [], 'response_time_ms': 0, 'retriever': None,
                  'confidence': None, 'retrieval_score': None}
    else:
        try:
            result = engine.ask(question, threshold=threshold, retriever=retriever)
        except RuntimeError as exc:
            # e.g. semantic model can't be loaded (offline first run)
            return _error(str(exc), 503)
        except Exception:
            app.logger.exception('QA request failed')
            return _error('The answer could not be completed. Please try again.', 500)
    result["category"] = result.get("category") or classify_query(question)
    if engine is not None:
        state["stats"].record(
            category=result["category"],
            answered=result["status"] in ("answered", "summarized"),
            response_ms=result["response_time_ms"],
        )
    result['corpus_id'] = engine.corpus_hash if engine is not None else None
    result['source'] = state['source']
    retrieval = copy.deepcopy(result)
    request_id = uuid.uuid4().hex
    if ai_mode and defer_ai and ai_explainer.config.enabled:
        ai = {'enabled': True, 'status': 'pending', 'basis': 'unavailable', 'answer': None}
    else:
        ai = _explain(question, retrieval, engine) if ai_mode else disabled_result()
        state['ai_stats'].record(ai)
    result.update(retrieval=retrieval, ai=ai, request_id=request_id)
    state['history'].append({'question': question, 'corpus_id': result['corpus_id'],
                             'source': state['source'], 'request_id': request_id,
                             'dataset_revision': state['dataset_revision'],
                             'result': copy.deepcopy(result)})
    state['history'] = state['history'][-1000:]
    return jsonify(result)


def _explain(question, retrieval, engine):
    started = time.perf_counter()
    try:
        result = ai_explainer.explain(question, retrieval, engine)
    except Exception as exc:
        # Deliberate isolation boundary. No provider exception text or traceback
        # may leak secrets/prompts; retrieval has already succeeded independently.
        failure = classify_error(exc)
        category = failure.category
        log_event('ai_explanation', category=category, outcome='failure', exception_type=type(exc).__name__,
                  http_status=failure.http_status, reason=failure.reason,
                  latency_ms=round((time.perf_counter() - started) * 1000))
        result = {'enabled': True, 'status': 'unavailable', 'basis': 'unavailable', 'answer': None,
                'error_category': category, 'message': 'AI is unavailable. The retrieved result is still available.',
                'latency_ms': round((time.perf_counter() - started) * 1000), 'attempts': 0,
                'failover_count': 0, 'failures': [category]}
    if engine is None and result['status'] == 'unavailable':
        result['message'] = 'AI could not answer right now. Try again, or load a dataset to use document retrieval.'
    return result


@app.route('/api/explain', methods=['POST'])
def explain():
    """Complete an explicitly AI-enabled ask, without rerunning retrieval.

    History and corpus identity are checked under the existing session lock.
    A replacement/reset/stale ID cannot send another document's evidence.
    Duplicate completion requests return the saved result without regeneration.
    """
    payload = _payload()
    if not isinstance(payload.get('request_id'), str) or type(payload.get('ai_mode')) is not bool:
        return _error('Send a request_id and a boolean ai_mode.')
    state = _state()
    engine = state['engine']
    record = next((r for r in reversed(state['history']) if r['request_id'] == payload['request_id']), None)
    corpus_id = engine.corpus_hash if engine is not None else None
    if (record is None or record['corpus_id'] != corpus_id
            or record['dataset_revision'] != state['dataset_revision']):
        return _error('This question no longer belongs to the active document. Ask it again.', 409)
    result = record['result']
    if result['ai']['status'] != 'pending':
        return jsonify({'ai': result['ai'], 'request_id': record['request_id']})
    ai = _explain(record['question'], result['retrieval'], engine) if payload['ai_mode'] else disabled_result()
    result['ai'] = ai
    state['ai_stats'].record(ai)
    return jsonify({'ai': ai, 'request_id': record['request_id']})


@app.route('/api/ai/status')
def ai_status():
    return jsonify(ai_explainer.config.public_status())


@app.route('/api/ai/stats')
def ai_stats():
    return jsonify(_state()['ai_stats'].snapshot())


@app.route("/api/stats")
def stats():
    snap = _state()["stats"].snapshot()
    engine = _engine()
    snap["dataset_loaded"] = engine is not None
    snap["passages"] = engine.passage_count if engine else 0
    return jsonify(snap)


@app.route('/api/history')
def history():
    return jsonify({'status': 'ok', 'records': _state()['history']})


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'models': qa_model.capabilities(),
                    'state_storage': 'single-process, ephemeral',
                    'session_capacity': MAX_SESSIONS,
                    'session_idle_ttl_seconds': SESSION_TTL_SECONDS})


if __name__ == "__main__":
    print("Loading prepared local reading-comprehension model (no network access)…")
    if qa_model.warm_up():
        print("Reading-comprehension model ready.")
    else:
        print(
            "WARNING: QA model unavailable — using conservative relation-based "
            f"extraction. ({qa_model.load_error()})"
        )
    print("Loading prepared local summarizer…")
    if qa_model.warm_up_summarizer():
        print("Document-explanation model ready.")
    else:
        print("WARNING: summarizer unavailable — explanations fall back to "
              "extractive key sentences.")
    print("QNext running at http://127.0.0.1:5000  (Ctrl+C to stop)")
    # use_reloader=False: the reloader would re-import the app and reload the
    # models on every code change — worse than a manual restart here.
    app.run(debug=os.environ.get('QUILLQA_DEBUG') == '1', use_reloader=False, port=5000)
