"""Bounded Gemini generation with independent credential and model selection.

Only normalized, fixed error categories leave this module. Provider exceptions
can contain credentials/prompts and must never be logged or returned verbatim.
Health state and client reuse are single-process and protected by a lock.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import random
import math
import threading
import time
import json
import logging
import hashlib
import uuid

from ai_config import AIConfig, Credential, MODEL_REGISTRY

logger = logging.getLogger('qnext.ai')


def log_event(event: str, **fields):
    """Only fixed diagnostic fields; never exception strings or request data."""
    allowed = {'diagnostic_id', 'model', 'slot', 'project_group', 'attempt', 'category',
               'scope', 'http_status', 'latency_ms', 'outcome', 'exception_type', 'reason'}
    record = {'event': event, **{key: value for key, value in fields.items() if key in allowed}}
    method = logger.warning if record.get('outcome') == 'failure' else logger.info
    method(json.dumps(record, sort_keys=True))


class ProviderFailure(Exception):
    def __init__(self, category: str, scope: str = 'pair', retry_after: float = 0,
                 http_status: int | None = None, reason: str | None = None):
        super().__init__(category)
        self.category, self.scope, self.retry_after = category, scope, retry_after
        self.http_status, self.reason = http_status, reason


def _seconds(value) -> float:
    try:
        seconds = float(str(value).rstrip('s'))
        return max(0., min(seconds, 86400.)) if math.isfinite(seconds) else 0.
    except (TypeError, ValueError):
        try:
            return max(0., (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0.


def classify_error(exc: Exception) -> ProviderFailure:
    if isinstance(exc, ProviderFailure):
        return exc
    if isinstance(exc, (TimeoutError, FutureTimeout)):
        return ProviderFailure('timeout', 'model', reason='caller_deadline')
    # HTTPX transport failures never carry an HTTP response code.
    import httpx
    if isinstance(exc, httpx.TimeoutException):
        return ProviderFailure('timeout', 'model', reason='transport_timeout')
    if isinstance(exc, httpx.TransportError):
        return ProviderFailure('network')
    code = getattr(exc, 'code', None)
    def failure(category, scope='pair', retry_after=0, reason=None):
        return ProviderFailure(category, scope, retry_after,
                               code if isinstance(code, int) else None, reason)
    body = getattr(exc, 'response_json', None) or getattr(exc, 'details', {}) or {}
    if isinstance(body, list):
        body = {'details': body}
    if not isinstance(body, dict):
        body = {}
    error = body.get('error', body)
    details = error.get('details', []) if isinstance(error, dict) else []
    retry_after, reasons, violations = 0., [], []
    for detail in details if isinstance(details, list) else []:
        if not isinstance(detail, dict):
            continue
        retry_after = max(retry_after, _seconds(detail.get('retryDelay', 0)))
        reasons.append(str(detail.get('reason', '')).upper())
        violations.extend(detail.get('violations', []) or [])
    response = getattr(exc, 'response', None)
    if response is not None:
        retry_after = max(retry_after, _seconds(response.headers.get('Retry-After', 0)))
    if code == 429:
        daily, model_only, account = False, bool(violations), False
        for violation in violations:
            if not isinstance(violation, dict):
                model_only = False
                continue
            metric = (str(violation.get('quotaMetric', '')) + str(violation.get('quotaId', ''))).lower()
            dimensions = violation.get('quotaDimensions', {}) or {}
            daily |= 'day' in metric or str(violation.get('quotaValue', '')) == '0'
            model_only &= bool(dimensions.get('model'))
            account |= any('account' in str(k).lower() for k in dimensions)
        scope = 'account' if account else 'pair' if model_only and not daily else 'project'
        return failure('quota' if daily else 'rate_limit', scope, retry_after)
    if code == 401 or any(reason in ('API_KEY_INVALID', 'API_KEY_EXPIRED') for reason in reasons):
        return failure('authentication', 'slot')
    if code in (402, 403):
        account_restricted = any(reason in ('ACCOUNT_DISABLED', 'ACCOUNT_RESTRICTED', 'ACCOUNT_SUSPENDED', 'CONSUMER_SUSPENDED') for reason in reasons)
        return failure('permission_or_billing', 'account' if account_restricted else 'project')
    if code == 404:
        return failure('model_unavailable', 'model')
    if code in (408, 504):
        return failure('timeout', 'model', reason='provider_deadline')
    if isinstance(code, int) and 500 <= code < 600:
        return failure('server_error', 'model')
    if code in (400, 422):
        # Inspect messages only to choose fixed reason codes. Raw provider text
        # can contain secrets and must never leave this classification boundary.
        message = str(getattr(exc, 'message', '') or '').casefold()
        if 'api key' in message and any(term in message for term in ('not valid', 'invalid', 'expired')):
            return failure('authentication', 'slot')
        if 'deadline' in message and ('too short' in message or 'minimum' in message):
            return failure('unsupported_parameters', 'terminal', reason='deadline_below_provider_minimum')
        if any(term in message for term in ('not supported', 'unsupported', 'unknown field', 'unrecognized field', 'invalid parameter')):
            return failure('unsupported_parameters', 'terminal', reason='generation_parameter_rejected')
        return failure('request_error', 'terminal')
    return ProviderFailure('internal_error', 'terminal')


def _enum(value) -> str:
    return str(getattr(value, 'value', value) or '').upper()


def normalize_response(response) -> str:
    feedback = getattr(response, 'prompt_feedback', None)
    if feedback and _enum(getattr(feedback, 'block_reason', None)) not in ('', 'BLOCK_REASON_UNSPECIFIED'):
        raise ProviderFailure('safety', 'terminal')
    candidates = getattr(response, 'candidates', None)
    if not candidates or len(candidates) != 1:
        raise ProviderFailure('invalid_output')
    candidate = candidates[0]
    finish = _enum(getattr(candidate, 'finish_reason', None))
    if finish in ('SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'IMAGE_SAFETY'):
        raise ProviderFailure('safety', 'terminal')
    if finish == 'MAX_TOKENS':
        raise ProviderFailure('incomplete_output')
    if finish != 'STOP':
        raise ProviderFailure('invalid_output')
    parts = getattr(getattr(candidate, 'content', None), 'parts', None) or []
    text = ''.join(p.text for p in parts if isinstance(getattr(p, 'text', None), str)
                   and not getattr(p, 'thought', False)).strip()
    if not text or len(text) > 24000 or any(ord(c) < 32 and c not in '\n\r\t' for c in text):
        raise ProviderFailure('invalid_output')
    return text


class GoogleTransport:
    """Reuse clients; SDK retries are disabled so the gateway owns the budget."""
    def __init__(self):
        self._clients, self._lock = {}, threading.Lock()

    def generate(self, credential, model, system, prompt, config, timeout):
        from google import genai
        from google.genai import types
        with self._lock:
            if credential.slot not in self._clients:
                self._clients[credential.slot] = genai.Client(
                    api_key=credential.key, vertexai=False,
                    http_options=types.HttpOptions(api_version='v1beta', timeout=10000,
                                                  retry_options=types.HttpRetryOptions(attempts=1)))
            client = self._clients[credential.slot]
        response = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system, **config,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                # Google rejects generateContent deadlines below its minimum.
                # The gateway's independent bounded future still enforces the
                # configured shorter caller deadline, without duplicate calls.
                http_options=types.HttpOptions(timeout=max(int(MODEL_REGISTRY[model].minimum_server_timeout * 1000), int(timeout * 1000)),
                                              retry_options=types.HttpRetryOptions(attempts=1))))
        return normalize_response(response)

    def close(self):
        for client in self._clients.values():
            client.close()


@dataclass
class Health:
    until: float = 0.
    failures: int = 0
    category: str | None = None


class GeminiGateway:
    def __init__(self, config: AIConfig, transport=None, clock=time.monotonic,
                 sleep=time.sleep, jitter=random.random):
        self.config, self.transport = config, transport or GoogleTransport()
        self.clock, self.sleep, self.jitter = clock, sleep, jitter
        self._lock = threading.RLock()
        self._health, self._busy, self._success = {}, set(), {}
        self._workers = ThreadPoolExecutor(max_workers=3, thread_name_prefix='qnext-gemini')

    def close(self):
        self._workers.shutdown(wait=True)
        if hasattr(self.transport, 'close'):
            self.transport.close()

    @staticmethod
    def _scopes(credential, model):
        return [('slot', credential.slot), ('project', credential.quota_group),
                ('account', credential.account or credential.quota_group), ('model', model),
                ('pair', credential.quota_group, model)]

    def _select(self, excluded, waiting_slots=frozenset()):
        with self._lock:
            now, next_ready = self.clock(), None
            for model in self.config.models:
                for credential in self.config.credentials:
                    pair = (credential.slot, model)
                    if pair in excluded or (credential.models and model not in credential.models):
                        continue
                    if credential.slot in self._busy:
                        # Only wait for this request's timed-out worker, never
                        # queue behind another user's generation. The total
                        # deadline bounds this short wait for a single-slot app.
                        if credential.slot in waiting_slots:
                            soon = now + .05
                            next_ready = soon if next_ready is None else min(next_ready, soon)
                        continue
                    until = max((self._health.get(scope, Health()).until for scope in self._scopes(credential, model)), default=0)
                    if until > now:
                        next_ready = until if next_ready is None else min(next_ready, until)
                        continue
                    self._busy.add(credential.slot)  # one controlled recovery probe per slot
                    return (credential, model), next_ready
            return None, next_ready

    def _release(self, slot):
        with self._lock:
            self._busy.discard(slot)

    def _log_late_completion(self, future, diagnostic_id, model, slot, attempt, started):
        fields = dict(diagnostic_id=diagnostic_id, model=model, slot=slot, attempt=attempt,
                      latency_ms=round((self.clock() - started) * 1000))
        try:
            future.result()
            log_event('gemini_late_completion', outcome='discarded', http_status=200, **fields)
        except Exception as exc:
            failure = classify_error(exc)
            log_event('gemini_late_completion', outcome='failure', category=failure.category,
                      http_status=failure.http_status, reason=failure.reason, **fields)

    def _fail(self, credential, model, failure, attempt):
        cfg = self.config
        if failure.scope == 'terminal':
            return
        scope = next(s for s in self._scopes(credential, model) if s[0] == failure.scope)
        delay = cfg.backoff * 2 ** min(attempt - 1, 4) * (1 + self.jitter() * .25)
        if failure.category in ('authentication', 'permission_or_billing', 'model_unavailable'):
            delay = cfg.auth_cooldown
        elif failure.category == 'quota':
            delay = max(60., cfg.cooldown)
        elif failure.category in ('server_error', 'timeout', 'network', 'invalid_output', 'incomplete_output'):
            delay = cfg.cooldown
        with self._lock:
            health = self._health.setdefault(scope, Health())
            health.failures += 1
            if health.failures >= 2:
                delay = max(delay, cfg.cooldown)
            health.until = max(health.until, self.clock() + max(delay, failure.retry_after))
            health.category = failure.category
            if failure.scope == 'account' and not credential.account:
                for other in self.config.credentials:
                    shared = self._health.setdefault(('account', other.account or other.quota_group), Health())
                    shared.until = max(shared.until, health.until)
                    shared.category = failure.category
            if failure.scope == 'project' and failure.category in ('quota', 'rate_limit'):
                # Unknown ownership cannot prove independence from an exhausted
                # known project. An unknown failing project's scope could be any
                # configured project, so conservatively cool all candidates.
                for other in self.config.credentials:
                    if not credential.project or not other.project:
                        shared = self._health.setdefault(('project', other.quota_group), Health())
                        shared.until = max(shared.until, health.until)
                        shared.category = failure.category

    def generate(self, system: str, prompt: str, output_tokens: int) -> dict:
        started = self.clock()
        cfg, events, attempts = self.config, [], 0
        base = {'enabled': True, 'status': 'unavailable', 'basis': 'unavailable',
                'answer': None, 'model': None, 'credential_slot': None, 'truncated': False,
                'diagnostic_id': uuid.uuid4().hex}

        def finish(category):
            messages = {
                'timeout': 'The AI service took too long to respond. Your retrieved result is still available.',
                'deadline': 'The AI service took too long to respond. Your retrieved result is still available.',
                'server_error': 'The AI service is temporarily busy. Your retrieved result is still available.',
                'rate_limit': 'AI is temporarily at capacity. Your retrieved result is still available.',
                'quota': 'AI is temporarily at capacity. Your retrieved result is still available.',
                'unsupported_parameters': 'AI configuration needs attention. Your retrieved result is still available.',
            }
            log_event('gemini_request', diagnostic_id=base['diagnostic_id'], model=base['model'],
                      slot=base['credential_slot'], category=category, outcome='failure',
                      latency_ms=round((self.clock() - started) * 1000), attempt=attempts)
            return dict(base, error_category=category,
                        message='AI could not provide an explanation for this question.' if category == 'safety' else
                                messages.get(category, 'AI is temporarily unavailable. The retrieved result is still available.'),
                        latency_ms=round((self.clock() - started) * 1000), attempts=attempts,
                        failover_count=max(0, attempts - 1), failures=events,
                        truncated='incomplete_output' in events)

        if not cfg.enabled:
            return dict(finish('disabled'), enabled=False, status='disabled', basis='disabled')
        if cfg.errors:
            return finish('configuration_error')
        if not cfg.credentials:
            return dict(finish('not_configured'), message=cfg.public_status()['message'])
        excluded, waiting_slots, last_category = set(), set(), 'circuit_open'
        while attempts < cfg.max_attempts:
            remaining = cfg.deadline - (self.clock() - started)
            if remaining <= .01:
                return finish('timeout' if last_category == 'timeout' else 'deadline')
            selection, ready = self._select(excluded, waiting_slots)
            if selection is None:
                if ready is not None and ready - self.clock() + .01 < remaining:
                    self.sleep(max(.001, ready - self.clock()))
                    continue
                return finish(last_category)
            credential, model = selection
            attempts += 1
            attempt_started = self.clock()
            base.update(model=model, credential_slot=credential.slot)
            spec = MODEL_REGISTRY[model]
            thinking = spec.thinking if cfg.thinking == 'auto' else cfg.thinking
            generation = {'temperature': cfg.temperature, 'max_output_tokens': min(output_tokens, cfg.max_output_tokens),
                          'thinking_config': {'thinking_level': thinking}}
            future = self._workers.submit(self.transport.generate, credential, model, system, prompt,
                                          generation, min(cfg.attempt_timeout, remaining))
            # Release only when the real call ends, even after the caller times out.
            future.add_done_callback(lambda _f, slot=credential.slot: self._release(slot))
            try:
                text = future.result(timeout=min(cfg.attempt_timeout, remaining))
                if not isinstance(text, str) or not text.strip() or len(text) > 24000:
                    raise ProviderFailure('invalid_output')
                if any(c.key in text for c in cfg.credentials):
                    raise ProviderFailure('sensitive_output', 'terminal')
                if self.clock() - started > cfg.deadline:
                    return finish('deadline')
                with self._lock:
                    self._success[credential.slot] = time.time()
                    for scope in self._scopes(credential, model):
                        # An older concurrent success must not erase a newer
                        # failure/cooldown observed by another request.
                        if self._health.get(scope, Health()).until <= self.clock():
                            self._health.pop(scope, None)
                log_event('gemini_attempt', diagnostic_id=base['diagnostic_id'], model=model,
                          slot=credential.slot, attempt=attempts, outcome='success', http_status=200,
                          project_group=hashlib.sha256(credential.quota_group.encode()).hexdigest()[:12],
                          latency_ms=round((self.clock() - attempt_started) * 1000))
                return dict(base, status='success', answer=text.strip(), error_category=None,
                            latency_ms=round((self.clock() - started) * 1000), attempts=attempts,
                            failover_count=max(0, attempts - 1), failures=events)
            except Exception as exc:
                # Catch only at the provider boundary; expose a classified failure,
                # never exception text, traceback, headers, or request content.
                failure = classify_error(exc)
                last_category = failure.category
                events.append(failure.category)
                log_event('gemini_attempt', diagnostic_id=base['diagnostic_id'], model=model,
                          slot=credential.slot, attempt=attempts, outcome='failure', category=failure.category,
                          scope=failure.scope, reason=failure.reason, http_status=failure.http_status,
                          project_group=hashlib.sha256(credential.quota_group.encode()).hexdigest()[:12],
                          latency_ms=round((self.clock() - attempt_started) * 1000),
                          exception_type=type(exc).__name__)
                self._fail(credential, model, failure, attempts)
                if failure.scope == 'terminal':
                    return finish(failure.category)
                if failure.category in ('authentication', 'permission_or_billing', 'quota', 'model_unavailable'):
                    excluded.add((credential.slot, model))
                if failure.category == 'timeout':
                    # Never repeat an uncertain generation on the same model.
                    # Keep its slot leased until the real call ends, then allow
                    # a different configured model within the shared budget.
                    excluded.update((item.slot, model) for item in cfg.credentials)
                    if not future.done():
                        waiting_slots.add(credential.slot)
                        future.add_done_callback(lambda done, request_id=base['diagnostic_id'],
                                                 selected_model=model, slot=credential.slot, number=attempts,
                                                 began=attempt_started:
                                                 self._log_late_completion(done, request_id, selected_model, slot, number, began))
        return finish(last_category)
