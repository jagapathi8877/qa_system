"""Server-only Gemini configuration. No network calls or credential discovery."""
from dataclasses import dataclass, field
import math
import os
import re
from typing import Mapping


@dataclass(frozen=True)
class ModelSpec:
    name: str
    thinking: str
    levels: tuple[str, ...]
    minimum_server_timeout: float = 10.0


# Verified against Google's official model pages and thinking guide, 2026-10-23.
MODEL_REGISTRY = {
    'gemini-3.8-flash': ModelSpec('gemini-3.8-flash', 'low', ('low', 'medium', 'high')),
    'gemini-3.6-flash': ModelSpec('gemini-3.6-flash', 'minimal', ('minimal', 'low', 'medium', 'high')),
    'gemini-3.5-flash-lite': ModelSpec('gemini-3.5-flash-lite', 'minimal', ('minimal', 'low', 'medium', 'high')),
}


@dataclass(frozen=True)
class Credential:
    slot: int
    key: str = field(repr=False)
    project: str = ''
    account: str = ''  # Opaque operator label, never an email address.
    models: tuple[str, ...] = ()

    @property
    def quota_group(self) -> str:
        # Unknown ownership must never be assumed independent.
        return self.project or 'unverified-project'


@dataclass(frozen=True)
class AIConfig:
    credentials: tuple[Credential, ...] = ()
    models: tuple[str, ...] = tuple(MODEL_REGISTRY)
    enabled: bool = True
    attempt_timeout: float = 8.0
    deadline: float = 20.0
    max_attempts: int = 4
    cooldown: float = 30.0
    auth_cooldown: float = 300.0
    backoff: float = 0.4
    max_output_tokens: int = 1800
    context_chars: int = 10000
    max_passages: int = 6
    temperature: float = 0.2
    thinking: str = 'auto'
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> 'AIConfig':
        env = os.environ if env is None else env
        errors, warnings, credentials = [], [], []

        def number(name, default, low, high, integer=False):
            try:
                value = float(env.get(name, str(default)))
                if not math.isfinite(value) or not low <= value <= high or (integer and not value.is_integer()):
                    raise ValueError()
                return int(value) if integer else value
            except (ValueError, TypeError):
                errors.append(name + ' is outside its supported range.')
                return default

        models = tuple(x.strip() for x in env.get('GEMINI_MODELS', ','.join(MODEL_REGISTRY)).split(',') if x.strip())
        if not models or len(set(models)) != len(models) or any(x not in MODEL_REGISTRY for x in models):
            errors.append('GEMINI_MODELS must contain unique, verified model IDs from ai_config.MODEL_REGISTRY.')
        enabled = env.get('QNEXT_AI_ENABLED', 'true').lower()
        if enabled not in ('true', 'false'):
            errors.append('QNEXT_AI_ENABLED must be true or false.')
        thinking = env.get('GEMINI_THINKING_LEVEL', 'auto').lower()
        if thinking != 'auto' and any(thinking not in MODEL_REGISTRY[x].levels for x in models if x in MODEL_REGISTRY):
            errors.append('GEMINI_THINKING_LEVEL is unsupported by a configured model.')
        keys = set()
        for slot in range(1, 4):
            key = env.get(f'GEMINI_API_KEY_{slot}', '').strip()
            project = env.get(f'GEMINI_PROJECT_ID_{slot}', '').strip()
            account = env.get(f'GEMINI_ACCOUNT_LABEL_{slot}', '').strip()
            if not key:
                continue
            # Google AI Studio authorization keys use the AQ. prefix. Keep
            # whitespace/control characters forbidden without rejecting that
            # documented key family; provider validation establishes validity.
            if len(key) < 20 or not re.fullmatch(r'(?:AQ\.)?[A-Za-z0-9_-]+', key) or key.lower().startswith(('your', 'replace', 'placeholder')):
                errors.append(f'Credential slot {slot} has a malformed or placeholder key.')
                continue
            if key in keys:
                warnings.append(f'Credential slot {slot} duplicates another key and is ignored.')
                continue
            keys.add(key)
            if project and not re.fullmatch(r'[a-z][a-z0-9-]{4,61}[a-z0-9]', project):
                errors.append(f'Credential slot {slot} has an invalid project ID.')
                continue
            if account and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', account):
                errors.append(f'Credential slot {slot} needs an opaque account label, not personal information.')
                continue
            supported = tuple(x.strip() for x in env.get(f'GEMINI_MODELS_{slot}', ','.join(models)).split(',') if x.strip())
            if not supported or any(x not in models for x in supported):
                errors.append(f'Credential slot {slot} has invalid model preferences.')
                continue
            if not project:
                warnings.append(f'Credential slot {slot} has unverified project metadata; independence is not assumed.')
            credentials.append(Credential(slot, key, project, account, supported))
        projects = [c.quota_group for c in credentials]
        if len(set(projects)) != len(projects):
            warnings.append('Some credentials share a project quota group.')
        if credentials and (not all(c.account for c in credentials) or len({c.account for c in credentials}) != len(credentials)):
            warnings.append('Independent accounts have not been declared for every slot.')
        values = dict(
            attempt_timeout=number('GEMINI_ATTEMPT_TIMEOUT', 8., .1, 30),
            deadline=number('GEMINI_DEADLINE', 20., .1, 60),
            max_attempts=number('GEMINI_MAX_ATTEMPTS', 4, 1, 6, True),
            cooldown=number('GEMINI_COOLDOWN', 30., .1, 3600),
            auth_cooldown=number('GEMINI_AUTH_COOLDOWN', 300., 1, 86400),
            backoff=number('GEMINI_BACKOFF', .4, .01, 5),
            max_output_tokens=number('GEMINI_MAX_OUTPUT_TOKENS', 1800, 128, 4096, True),
            context_chars=number('GEMINI_CONTEXT_CHARS', 10000, 512, 24000, True),
            max_passages=number('GEMINI_MAX_PASSAGES', 6, 1, 12, True),
            temperature=number('GEMINI_TEMPERATURE', .2, 0, 1),
        )
        return cls(credentials=tuple(credentials), models=models, enabled=enabled == 'true',
                   thinking=thinking, errors=tuple(errors), warnings=tuple(warnings), **values)

    def public_status(self) -> dict:
        return {'available': self.enabled and bool(self.credentials) and not self.errors,
                'enabled': self.enabled, 'configured_slots': len(self.credentials),
                'status': 'configuration_error' if self.errors else 'configured' if self.credentials else 'not_configured',
                'message': 'AI configuration needs attention. Retrieval remains available.' if self.errors else
                           'AI explanations are configured.' if self.credentials else
                           'AI is not configured. Add server-side Gemini credentials; retrieval remains available.'}
