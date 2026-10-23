"""Deterministic provider and failure-policy checks; never calls Google."""
from dataclasses import replace
from types import SimpleNamespace as NS
import threading
import time

import pytest

from ai_config import AIConfig, Credential, MODEL_REGISTRY
from gemini_gateway import GeminiGateway, GoogleTransport, ProviderFailure, classify_error, normalize_response


MODELS = tuple(MODEL_REGISTRY)


def credentials(projects=('project-one', 'project-two', 'project-three')):
    return tuple(Credential(i, f'test-only-placeholder-key-{i}', project, f'account-{i}', MODELS)
                 for i, project in enumerate(projects, 1))


class Error(Exception):
    def __init__(self, code, details=(), headers=None):
        super().__init__('DO NOT LOG: provider may include a secret or a document')
        self.code = code
        self.response_json = {'error': {'details': list(details)}}
        self.response = NS(headers=headers or {})


class Transport:
    def __init__(self, outcomes=()):
        self.outcomes, self.calls = list(outcomes), []

    def generate(self, credential, model, system, prompt, config, timeout):
        self.calls.append((credential.slot, model, config, timeout))
        value = self.outcomes.pop(0) if self.outcomes else 'A complete explanation.'
        if isinstance(value, Exception):
            raise value
        return value


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


@pytest.fixture
def gateway():
    active = []
    def make(outcomes=(), projects=('project-one', 'project-two', 'project-three'), **kwargs):
        transport, clock = Transport(outcomes), Clock()
        instance = GeminiGateway(AIConfig(credentials=credentials(projects), **kwargs), transport,
                                 clock=clock, sleep=clock.sleep, jitter=lambda: 0)
        active.append(instance)
        return instance, transport, clock
    yield make
    for instance in active: instance.close()


def generate(gateway): return gateway.generate('system', 'synthetic question', 800)


def quota(model=None, daily=False, account=False):
    dimensions = {'model': model} if model else {}
    if account: dimensions['account'] = 'account-one'
    return {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [
        {'quotaMetric': 'generate_requests_per_day' if daily else 'generate_requests_per_minute',
         'quotaDimensions': dimensions}]}


@pytest.mark.parametrize('count', [0, 1, 3])
def test_credential_counts(gateway, count):
    instance, transport, _ = gateway(projects=('project-one', 'project-two', 'project-three')[:count])
    result = generate(instance)
    assert result['status'] == ('success' if count else 'unavailable')
    assert len(transport.calls) == bool(count)


def test_invalid_key_keeps_preferred_model(gateway):
    instance, transport, _ = gateway([Error(401)])
    assert generate(instance)['status'] == 'success'
    assert [(s, m) for s, m, *_ in transport.calls] == [(1, MODELS[0]), (2, MODELS[0])]
    generate(instance)
    assert transport.calls[-1][0] == 2


def test_two_model_failures_reach_lightweight_fallback(gateway):
    instance, transport, _ = gateway([Error(503), Error(503)])
    result = generate(instance)
    assert result['status'] == 'success' and result['model'] == MODELS[2]
    assert [call[1] for call in transport.calls] == list(MODELS)
    assert result['attempts'] == 3


def test_all_three_unhealthy_credentials_stop_promptly(gateway):
    instance, transport, _ = gateway([Error(401)] * 3)
    result = generate(instance)
    assert result['status'] == 'unavailable' and result['attempts'] == 3
    assert [call[0] for call in transport.calls] == [1, 2, 3]


def test_timeout_recovers_through_different_model_and_leases_original_slot():
    release = threading.Event()
    class StalledPrimary(Transport):
        def generate(self, credential, model, *args):
            self.calls.append((credential.slot, model))
            if model == MODELS[0]:
                release.wait(1)
                return 'Late primary response must be discarded.'
            return 'A complete secondary explanation.'
    transport = StalledPrimary()
    instance = GeminiGateway(AIConfig(credentials=credentials(), attempt_timeout=.03, deadline=.4), transport)
    try:
        result = generate(instance)
        assert result['status'] == 'success' and result['model'] == MODELS[1]
        assert result['answer'] == 'A complete secondary explanation.'
        assert result['failures'] == ['timeout'] and result['attempts'] == 2
        assert transport.calls == [(1, MODELS[0]), (2, MODELS[1])]
        assert 1 in instance._busy  # The original transport is still leased.
    finally:
        release.set(); instance.close()


def test_single_slot_waits_for_timed_out_transport_before_fallback():
    class SlowPrimary(Transport):
        def generate(self, credential, model, *args):
            self.calls.append((credential.slot, model))
            if model == MODELS[0]:
                time.sleep(.07)
                return 'Late primary.'
            return 'A complete fallback.'
    transport = SlowPrimary()
    instance = GeminiGateway(AIConfig(credentials=credentials(('project-one',)), attempt_timeout=.03, deadline=.4), transport)
    try:
        result = generate(instance)
        assert result['status'] == 'success' and result['model'] == MODELS[1]
        assert transport.calls == [(1, MODELS[0]), (1, MODELS[1])]
    finally: instance.close()


def test_attempt_logs_have_status_and_context_without_secrets(gateway, caplog):
    import logging,json
    caplog.set_level(logging.INFO, logger='qnext.ai')
    instance, _, _ = gateway([Error(503)])
    result = generate(instance)
    records = [json.loads(r.message) for r in caplog.records if r.name == 'qnext.ai']
    failed = next(r for r in records if r['event'] == 'gemini_attempt' and r['outcome'] == 'failure')
    assert failed['model'] == MODELS[0] and failed['slot'] == 1
    assert failed['http_status'] == 503 and failed['category'] == 'server_error'
    assert failed['diagnostic_id'] == result['diagnostic_id'] and failed['latency_ms'] >= 0
    assert 'test-only-placeholder' not in caplog.text
    assert 'DO NOT LOG' not in caplog.text and 'synthetic question' not in caplog.text


@pytest.mark.parametrize('message,reason', [
    ('Thinking level minimal is not supported. SECRET VALUE', 'generation_parameter_rejected'),
    ('Manually set deadline 8s is too short. Minimum allowed deadline is 10s.', 'deadline_below_provider_minimum')])
def test_unsupported_parameters_are_terminal_and_safely_classified(gateway, message, reason):
    error = Error(400); error.message = message
    instance, transport, _ = gateway([error])
    result = generate(instance)
    assert result['error_category'] == 'unsupported_parameters' and len(transport.calls) == 1
    assert classify_error(error).reason == reason
    assert message not in str(result)


def test_structured_invalid_key_400_classified_as_credential(gateway):
    instance, transport, _ = gateway([Error(400, [{'reason': 'API_KEY_INVALID'}])])
    assert generate(instance)['status'] == 'success'
    assert transport.calls[-1][0] == 2


def test_real_sdk_error_details_and_retry_info():
    from google.genai.errors import ClientError
    error = ClientError(429, {'error': {'details': [quota(daily=True),
                        {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '45s'}]}})
    failure = classify_error(error)
    assert failure.category == 'quota' and failure.scope == 'project' and failure.retry_after == 45
    error = ClientError(400, {'error': {'details': [{'reason': 'API_KEY_INVALID'}]}})
    assert classify_error(error).category == 'authentication'


def test_known_and_unknown_projects_cannot_claim_independent_quota(gateway):
    instance, transport, _ = gateway([Error(429, [quota(daily=True)])], projects=('known-one', '', 'known-three'))
    assert generate(instance)['status'] == 'success'
    assert transport.calls[1][0] == 3
    instance, transport, _ = gateway([Error(429, [quota(daily=True)])], projects=('', 'known-two', 'known-three'))
    assert generate(instance)['status'] == 'unavailable'
    assert len(transport.calls) == 1


@pytest.mark.parametrize('projects,expected', [
    (('project-one', 'project-one', 'project-three'), 3),
    (('project-one', 'project-two', 'project-three'), 2),
])
def test_daily_project_quota_skips_shared_keys(gateway, projects, expected):
    instance, transport, _ = gateway([Error(429, [quota(daily=True)])], projects=projects)
    assert generate(instance)['status'] == 'success'
    assert transport.calls[1][:2] == (expected, MODELS[0])


def test_all_slots_same_project_stop_after_one_daily_quota(gateway):
    instance, transport, _ = gateway([Error(429, [quota(daily=True)])], projects=('shared-one',) * 3)
    assert generate(instance)['error_category'] == 'quota'
    assert len(transport.calls) == 1


def test_unknown_projects_are_not_assumed_independent(gateway):
    instance, transport, _ = gateway([Error(429)], projects=('', '', ''))
    result = generate(instance)
    assert result['attempts'] <= 4
    # Every same-group retry waits, rather than immediately rotating keys.
    assert transport.calls[1][0] == 1


def test_model_rate_limit_allows_same_project_other_model(gateway):
    instance, transport, _ = gateway([Error(429, [quota(model=MODELS[0])])], projects=('shared-one',) * 3)
    assert generate(instance)['status'] == 'success'
    assert transport.calls[1][:2] == (1, MODELS[1])


@pytest.mark.parametrize('code,category', [(400, 'request_error'), (422, 'request_error')])
def test_permanent_bad_request_never_rotates(gateway, code, category):
    instance, transport, _ = gateway([Error(code)])
    assert generate(instance)['error_category'] == category
    assert len(transport.calls) == 1


@pytest.mark.parametrize('code,model', [(404, MODELS[1]), (503, MODELS[1])])
def test_unavailable_model_uses_verified_fallback(gateway, code, model):
    instance, transport, _ = gateway([Error(code)])
    assert generate(instance)['status'] == 'success'
    assert transport.calls[1][1] == model


@pytest.mark.parametrize('code', [402, 403])
def test_billing_and_permission_skip_shared_project(gateway, code):
    instance, transport, _ = gateway([Error(code)], projects=('shared-one', 'shared-one', 'independent-three'))
    assert generate(instance)['status'] == 'success'
    assert transport.calls[1][0] == 3


def test_safety_refusal_is_terminal(gateway):
    instance, transport, _ = gateway([ProviderFailure('safety', 'terminal')])
    assert generate(instance)['error_category'] == 'safety'
    assert len(transport.calls) == 1


@pytest.mark.parametrize('output', ['', None, {'text': 'bad'}, ProviderFailure('incomplete_output')])
def test_invalid_outputs_are_not_success(gateway, output):
    instance, transport, _ = gateway([output] * 6, max_attempts=4)
    assert generate(instance)['status'] == 'unavailable'
    assert len(transport.calls) <= 4


def test_truncation_is_reported(gateway):
    instance, _, _ = gateway([ProviderFailure('incomplete_output')], max_attempts=1)
    result = generate(instance)
    assert result['truncated'] and result['answer'] is None


def test_global_attempt_budget(gateway):
    instance, transport, _ = gateway([Error(401)] * 10, max_attempts=2)
    assert generate(instance)['attempts'] == len(transport.calls) == 2


def test_retry_after_is_respected_and_circuit_recovers(gateway):
    instance, transport, clock = gateway([Error(429, headers={'Retry-After': '25'})], projects=('project-one',))
    assert generate(instance)['status'] == 'unavailable'
    assert generate(instance)['error_category'] == 'circuit_open'
    assert len(transport.calls) == 1
    clock.sleep(26)
    assert generate(instance)['status'] == 'success'
    assert len(transport.calls) == 2


def test_transient_backoff_and_jitter_are_bounded(gateway):
    instance, transport, clock = gateway([Error(429)], projects=('project-one',), models=(MODELS[0],))
    assert generate(instance)['status'] == 'success'
    assert clock.now == pytest.approx(100.4)
    assert len(transport.calls) == 2


def test_request_deadline_with_real_slow_transport():
    class Slow(Transport):
        def generate(self, *args):
            time.sleep(.25)
            return 'Late response.'
    instance = GeminiGateway(AIConfig(credentials=credentials(), deadline=.04, attempt_timeout=.04), Slow())
    started = time.monotonic()
    result = generate(instance)
    assert time.monotonic() - started < .20
    assert result['error_category'] == 'timeout' and result['attempts'] == 1
    instance.close()


def test_concurrent_recovery_probe_is_exclusive():
    entered, release = threading.Event(), threading.Event()
    class Blocking(Transport):
        def generate(self, *args):
            entered.set(); release.wait(2)
            return 'A complete response.'
    instance = GeminiGateway(AIConfig(credentials=credentials(('project-one',))), Blocking())
    result = []
    worker = threading.Thread(target=lambda: result.append(generate(instance)))
    worker.start(); assert entered.wait(1)
    assert generate(instance)['error_category'] == 'circuit_open'
    release.set(); worker.join(); instance.close()
    assert result[0]['status'] == 'success'


def test_known_secrets_in_output_are_rejected(gateway):
    instance, transport, _ = gateway(['test-only-placeholder-key-1'])
    result = generate(instance)
    assert result['error_category'] == 'sensitive_output'
    assert 'test-only-placeholder' not in str(result)


def response(text='Complete response.', finish='STOP', block=None):
    return NS(prompt_feedback=NS(block_reason=block), candidates=[NS(
        finish_reason=finish, content=NS(parts=[NS(text=text, thought=False)]))])


@pytest.mark.parametrize('finish,category', [('MAX_TOKENS', 'incomplete_output'), ('SAFETY', 'safety'),
                                           ('RECITATION', 'safety'), (None, 'invalid_output')])
def test_response_finish_reasons(finish, category):
    with pytest.raises(ProviderFailure, match=category): normalize_response(response(finish=finish))


def test_response_normalization_does_not_expose_thinking():
    item = response()
    item.candidates[0].content.parts.insert(0, NS(text='private thought', thought=True))
    assert normalize_response(item) == 'Complete response.'
    with pytest.raises(ProviderFailure): normalize_response(response(text=''))
    with pytest.raises(ProviderFailure): normalize_response(response(block='SAFETY'))


def test_official_sdk_transport_contract_without_network(monkeypatch):
    from google import genai
    calls = []
    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.models = NS(generate_content=self.generate)
        def generate(self, **kwargs):
            calls.append(kwargs)
            return response()
        def close(self): pass
    monkeypatch.setattr(genai, 'Client', Client)
    transport = GoogleTransport()
    for _ in range(2):
        assert transport.generate(credentials()[0], MODELS[0], 'system', 'question',
                                  {'max_output_tokens': 500, 'thinking_config': {'thinking_level': 'low'}}, 1.)
    assert len(calls) == 3  # One client, two generations.
    assert calls[0]['http_options'].retry_options.attempts == 1
    assert calls[1]['config'].http_options.timeout == 10000
    assert calls[1]['config'].automatic_function_calling.disable


def test_config_safe_validation():
    env = {f'GEMINI_API_KEY_{i}': 'A' * 30 + str(i) for i in range(1, 4)}
    env.update({f'GEMINI_PROJECT_ID_{i}': f'project-{i}' for i in range(1, 4)})
    config = AIConfig.from_env(env)
    assert len(config.credentials) == 3 and not config.errors
    assert 'A' * 30 not in repr(config)
    assert 'A' * 30 not in str(config.public_status())
    assert not AIConfig.from_env({}).credentials


def test_authorization_key_format_is_accepted_without_exposure():
    key = 'AQ.' + 'synthetic_test_only_' * 3
    config = AIConfig.from_env({'GEMINI_API_KEY_1': key, 'GEMINI_PROJECT_ID_1': 'project-one'})
    assert not config.errors and len(config.credentials) == 1
    assert config.credentials[0].key == key
    assert key not in repr(config) + str(config.public_status())


@pytest.mark.parametrize('key', ['test.only-key-with-a-period', 'AQ.bad key with spaces', 'AQ.' + 'x' * 20 + '\nmalformed'])
def test_invalid_key_punctuation_and_controls_still_rejected(key):
    assert AIConfig.from_env({'GEMINI_API_KEY_1': key}).errors


@pytest.mark.parametrize('name,value', [('GEMINI_DEADLINE', 'NaN'), ('GEMINI_MAX_ATTEMPTS', '2.5'),
    ('GEMINI_MODELS', 'unverified-model'), ('GEMINI_THINKING_LEVEL', 'minimal'),
    ('QNEXT_AI_ENABLED', 'yes'), ('GEMINI_API_KEY_1', 'your-key-here'),
    ('GEMINI_PROJECT_ID_1', 'INVALID'), ('GEMINI_ACCOUNT_LABEL_1', 'private@example.org')])
def test_invalid_config_never_crashes_retrieval(name, value):
    env = {'GEMINI_API_KEY_1': 'A' * 30, name: value}
    assert AIConfig.from_env(env).errors
