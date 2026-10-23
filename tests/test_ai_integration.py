import copy
from dataclasses import replace

import pytest
import app as app_module
from ai_config import AIConfig
from ai_explainer import AIExplainer
from stats import AIStats


TEXT = ('Luminescence is a process for emitting light without heating a material. '
        'The laboratory uses luminescence to study mineral samples and energy changes.')


@pytest.fixture
def client(monkeypatch):
    app_module._STATE.clear()
    monkeypatch.setitem(app_module.app.config, 'TESTING', True)
    with app_module.app.test_client() as client:
        yield client
    app_module._STATE.clear()


class Explainer:
    config = AIConfig()
    def __init__(self): self.calls = []
    def explain(self, question, retrieval, engine):
        self.calls.append((question, copy.deepcopy(retrieval), engine.corpus_hash if engine else None))
        return {'enabled': True, 'status': 'success', 'answer': 'A separate generated explanation.',
                'basis': 'document_evidence' if engine else 'general_knowledge', 'latency_ms': 1, 'attempts': 1,
                'credential_slot': 1, 'model': 'mock', 'failures': []}


def load(client):
    assert client.post('/api/paste', json={'text': TEXT}).status_code == 200


@pytest.mark.parametrize('payload', [{}, {'ai_mode': False}, {'ai_mode': False, 'defer_ai': True}])
def test_disabled_mode_never_calls_provider(client, monkeypatch, payload):
    explainer = Explainer(); monkeypatch.setattr(app_module, 'ai_explainer', explainer)
    load(client)
    result = client.post('/api/ask', json={'question': 'What is luminescence?', **payload}).get_json()
    assert result['ai']['status'] == 'disabled' and not explainer.calls
    client.post('/api/explain', json={'request_id': result['request_id'], 'ai_mode': True})
    assert not explainer.calls  # Cannot upgrade an OFF question through deferred route.
    assert client.get('/api/ai/stats').get_json()['counts']['disabled_requests'] == 1


@pytest.mark.parametrize('mode', ['false', 1, None, [], {}])
def test_mode_requires_boolean_before_stats(client, mode):
    load(client)
    response = client.post('/api/ask', json={'question': 'What is luminescence?', 'ai_mode': mode})
    assert response.status_code == 400
    assert client.get('/api/stats').get_json()['total_queries'] == 0


def test_combined_response_preserves_every_original_field_and_history(client, monkeypatch):
    explainer = Explainer(); monkeypatch.setattr(app_module, 'ai_explainer', explainer)
    load(client)
    result = client.post('/api/ask', json={'question': 'What is luminescence?', 'ai_mode': True}).get_json()
    assert result['ai']['status'] == 'success'
    for name, value in result['retrieval'].items(): assert result[name] == value
    assert explainer.calls[0][1] == result['retrieval']
    assert client.get('/api/history').get_json()['records'][-1]['result'] == result
    assert client.get('/api/stats').get_json()['total_queries'] == 1


def test_deferred_generation_runs_once_and_cancellation_does_not_call(client, monkeypatch):
    explainer = Explainer(); monkeypatch.setattr(app_module, 'ai_explainer', explainer)
    load(client)
    result = client.post('/api/ask', json={'question': 'What is luminescence?', 'ai_mode': True, 'defer_ai': True}).get_json()
    assert result['ai']['status'] == 'pending' and not explainer.calls
    payload = {'request_id': result['request_id'], 'ai_mode': True}
    for _ in range(2): assert client.post('/api/explain', json=payload).get_json()['ai']['status'] == 'success'
    assert len(explainer.calls) == 1
    assert client.get('/api/ai/stats').get_json()['counts']['requests'] == 1
    result = client.post('/api/ask', json={'question': 'What is luminescence?', 'ai_mode': True, 'defer_ai': True}).get_json()
    client.post('/api/explain', json={'request_id': result['request_id'], 'ai_mode': False})
    assert len(explainer.calls) == 1


def test_replacement_and_other_session_cannot_complete_old_question(client, monkeypatch):
    explainer = Explainer(); monkeypatch.setattr(app_module, 'ai_explainer', explainer)
    load(client)
    result = client.post('/api/ask', json={'question': 'What is luminescence?', 'ai_mode': True, 'defer_ai': True}).get_json()
    payload = {'request_id': result['request_id'], 'ai_mode': True}
    other = app_module.app.test_client()
    assert other.post('/api/explain', json=payload).status_code == 409
    client.post('/api/paste', json={'text': 'A different document describes coastal measurements and ocean observations in a field station.'})
    assert client.post('/api/explain', json=payload).status_code == 409
    assert not explainer.calls


def test_same_content_replacement_also_invalidates_pending_explanation(client, monkeypatch):
    explainer = Explainer(); monkeypatch.setattr(app_module, 'ai_explainer', explainer)
    load(client)
    result = client.post('/api/ask', json={'question': 'What is luminescence?', 'ai_mode': True, 'defer_ai': True}).get_json()
    load(client)
    assert client.post('/api/explain', json={'request_id': result['request_id'], 'ai_mode': True}).status_code == 409
    assert not explainer.calls


@pytest.mark.parametrize('deferred', [True, False])
def test_without_document_ai_answers_and_persists_in_own_session(client, monkeypatch, deferred):
    explainer = Explainer(); monkeypatch.setattr(app_module, 'ai_explainer', explainer)
    response = client.post('/api/ask', json={'question': 'What is light?', 'ai_mode': True, 'defer_ai': deferred})
    assert response.status_code == 200
    result = response.get_json()
    assert result['reason'] == 'no_dataset' and result['sources'] == [] and result['corpus_id'] is None
    assert result['answer'] is None and result['retriever'] is None
    if deferred:
        assert result['ai']['status'] == 'pending' and not explainer.calls
    payload = {'request_id': result['request_id'], 'ai_mode': True}
    other = app_module.app.test_client()
    assert other.post('/api/explain', json=payload).status_code == 409
    for _ in range(2):
        ai = client.post('/api/explain', json=payload).get_json()['ai']
        assert ai['status'] == 'success' and ai['basis'] == 'general_knowledge'
    assert len(explainer.calls) == 1 and explainer.calls[0][2] is None
    assert not client.get('/api/dataset').get_json()['loaded']
    assert client.get('/api/stats').get_json()['total_queries'] == 0  # No retrieval was performed.
    assert client.get('/api/ai/stats').get_json()['counts']['requests'] == 1
    assert client.get('/api/history').get_json()['records'][-1]['result']['ai'] == ai
    assert not other.get('/api/history').get_json()['records']


@pytest.mark.parametrize('mode', [False, None])
def test_without_document_ai_off_requires_dataset_and_never_calls_provider(client, monkeypatch, mode):
    explainer = Explainer(); monkeypatch.setattr(app_module, 'ai_explainer', explainer)
    payload = {'question': 'What is light?'}
    if mode is not None: payload['ai_mode'] = mode
    assert client.post('/api/ask', json=payload).status_code == 409
    assert not explainer.calls


@pytest.mark.parametrize('change', ['upload', 'reset'])
def test_general_question_cannot_be_completed_after_context_changes(client, monkeypatch, change):
    explainer = Explainer(); monkeypatch.setattr(app_module, 'ai_explainer', explainer)
    result = client.post('/api/ask', json={'question': 'What is light?', 'ai_mode': True, 'defer_ai': True}).get_json()
    if change == 'upload': load(client)
    else: client.delete('/api/dataset')
    assert client.post('/api/explain', json={'request_id': result['request_id'], 'ai_mode': True}).status_code == 409
    assert not explainer.calls


def test_general_question_can_be_cancelled_without_provider_call(client, monkeypatch):
    explainer = Explainer(); monkeypatch.setattr(app_module, 'ai_explainer', explainer)
    result = client.post('/api/ask', json={'question': 'What is light?', 'ai_mode': True, 'defer_ai': True}).get_json()
    payload = {'request_id': result['request_id'], 'ai_mode': False}
    assert client.post('/api/explain', json=payload).get_json()['ai']['status'] == 'disabled'
    assert not explainer.calls


@pytest.mark.parametrize('enabled', [True, False])
def test_general_question_without_credentials_or_with_master_switch_off(client, monkeypatch, enabled):
    explainer = AIExplainer(AIConfig(enabled=enabled))
    monkeypatch.setattr(app_module, 'ai_explainer', explainer)
    try:
        response = client.post('/api/ask', json={'question': 'What is light?', 'ai_mode': True})
        if not enabled:
            assert response.status_code == 409
        else:
            assert response.status_code == 200
            ai = response.get_json()['ai']
            assert ai['status'] == 'unavailable' and ai['attempts'] == 0
            assert 'retrieved result is still available' not in ai['message']
    finally:
        explainer.gateway.close()


def test_ai_failure_cannot_erase_retrieval_or_leak_exception(client, monkeypatch, caplog):
    class Failing(Explainer):
        def explain(self, *args): raise RuntimeError('SECRET-AND-STUDENT-TEXT')
    monkeypatch.setattr(app_module, 'ai_explainer', Failing())
    load(client)
    response = client.post('/api/ask', json={'question': 'What is luminescence?', 'ai_mode': True})
    result = response.get_json()
    assert response.status_code == 200 and result['status'] == 'answered'
    assert result['ai']['status'] == 'unavailable' and result['sources']
    assert 'SECRET-AND-STUDENT-TEXT' not in str(result) + caplog.text
    assert client.get('/api/dataset').get_json()['loaded']


def test_no_credentials_graceful_and_backend_master_switch(client, monkeypatch):
    load(client)
    for config, expected in [(AIConfig(), 'unavailable'), (AIConfig(enabled=False), 'disabled')]:
        monkeypatch.setattr(app_module, 'ai_explainer', AIExplainer(config))
        result = client.post('/api/ask', json={'question': 'What is luminescence?', 'ai_mode': True}).get_json()
        assert result['status'] == 'answered' and result['ai']['status'] == expected
        app_module.ai_explainer.gateway.close()


def test_ai_metrics_latency_and_failures_are_separate():
    stats = AIStats()
    for i in range(20):
        stats.record({'enabled': True, 'status': 'success', 'latency_ms': i + 1,
                      'attempts': 2, 'failover_count': 1, 'failures': ['rate_limit'],
                      'evidence_state': 'no_sufficient_answer'})
    snap = stats.snapshot()
    assert snap['p50_ms'] == 10 and snap['p95_ms'] == 19
    assert snap['counts']['provider_attempts'] == 40
    assert snap['failures']['rate_limit'] == 20


def test_health_and_ai_status_have_no_document_or_credentials(client, monkeypatch):
    monkeypatch.setattr(app_module, 'ai_explainer', AIExplainer(AIConfig()))
    assert client.get('/api/ai/status').get_json()['status'] == 'not_configured'
    assert client.get('/api/ai/stats').get_json()['latency_sample_size'] == 0
    app_module.ai_explainer.gateway.close()


@pytest.mark.parametrize('enabled', [True, False])
def test_rendered_switch_uses_configured_default_without_helper_text(client, monkeypatch, enabled):
    from html.parser import HTMLParser
    class Inputs(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag == 'input' and attributes.get('id') == 'aiMode': self.switch = attributes
    explainer = Explainer(); explainer.config = AIConfig(enabled=enabled)
    monkeypatch.setattr(app_module, 'ai_explainer', explainer)
    html = client.get('/').get_data(as_text=True)
    parser = Inputs(); parser.feed(html)
    assert ('checked' in parser.switch) is enabled
    assert parser.switch['role'] == 'switch' and parser.switch['aria-label'] == 'AI explanations'
    assert 'aria-describedby' not in parser.switch
    assert 'When on, your question and selected evidence are sent to Google Gemini.' not in html
    assert 'aiConfigStatus' not in html


def test_unexpected_explanation_failure_logs_classification_without_content(client, monkeypatch, caplog):
    from google.genai.errors import ClientError
    class Failing(Explainer):
        def explain(self, *args):
            raise ClientError(403, {'error': {'message': 'SECRET AUTH HEADER AND PROMPT'}})
    monkeypatch.setattr(app_module, 'ai_explainer', Failing())
    load(client)
    result = client.post('/api/ask', json={'question': 'What is luminescence?', 'ai_mode': True}).get_json()
    assert result['ai']['error_category'] == 'permission_or_billing'
    assert 'permission_or_billing' in caplog.text and 'SECRET AUTH' not in caplog.text
    assert result['status'] == 'answered'
