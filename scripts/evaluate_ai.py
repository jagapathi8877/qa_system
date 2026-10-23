"""Repeatable synthetic explanation evaluation. Offline by default.

Offline reference outputs test contracts, NOT Gemini quality or provider speed.
--live sends only these synthetic cases to Google and may incur charges.
Results retain outputs for human review; substring checks do not prove truth.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env', override=False)
from ai_config import AIConfig
from ai_explainer import AIExplainer
from gemini_gateway import GeminiGateway, normalize_response
from qa_engine import QAEngine


def fixture(case):
    passages = list(case['passages'])
    if case.get('repeat'):
        # Distinct paragraphs exercise deterministic context limits rather than
        # preprocessing's intentional duplicate-window suppression.
        passages = [f'Section {i}. ' + passages[0] for i in range(case['repeat'])]
    engine = QAEngine('\n\n'.join(passages))
    result = {'status': case['status'], 'reason': case.get('reason'),
              'answer': 'No sufficient answer.' if case.get('reason') else passages[0],
              'sources': [] if case['id'] == 'absent' else [
                  {'passage_id': i + 1, 'passage_text': p} for i, p in enumerate(engine.passages)]}
    return engine, result


class ReferenceTransport:
    def __init__(self, case): self.case = case
    def generate(self, *_args):
        finish = 'MAX_TOKENS' if self.case.get('output_kind') == 'truncated' else 'STOP'
        return normalize_response(NS(prompt_feedback=None, candidates=[NS(
            finish_reason=finish, content=NS(parts=[NS(text=self.case['reference'], thought=False)]))]))


def run(live=False, models=None, limit=None):
    config = AIConfig.from_env()
    if live and not config.public_status()['available']:
        return {'mode': 'live', 'status': 'skipped', 'reason': 'No usable server-side credentials or invalid configuration.'}
    from ai_config import Credential
    if not live:
        config = AIConfig(credentials=(Credential(1, 'synthetic-only-not-a-real-key', 'synthetic-project'),), max_attempts=1)
    cases = json.loads((ROOT / 'tests/data/ai_evaluation.json').read_text())
    selected_models = models or [config.models[0]]
    rows = []
    for model in selected_models:
        for case in cases[:limit]:
            if live and case.get('output_kind'):
                continue  # Empty/truncated faults are injected offline only.
            engine, result = fixture(case)
            gateway = GeminiGateway(replace(config, models=(model,)), None if live else ReferenceTransport(case))
            started = time.perf_counter()
            ai = AIExplainer(gateway.config, gateway).explain(case['question'], result, engine)
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            gateway.close()
            rows.append({'id': case['id'], 'category': case['category'], 'question': case['question'],
                         'model': model, 'ai': ai, 'elapsed_ms': elapsed,
                         'required_fact_screen': {fact: fact.casefold() in (ai.get('answer') or '').casefold() for fact in case['required']}})
    timings = sorted(r['elapsed_ms'] for r in rows)
    import math
    return {'mode': 'live' if live else 'offline_reference_contracts', 'status': 'completed',
            'python': platform.python_version(), 'platform': platform.platform(), 'machine': platform.machine(),
            'sample_size': len(rows), 'includes_context_construction': True,
            'p50_ms': timings[(len(timings) - 1) // 2] if timings else None,
            'p95_ms': timings[max(0, math.ceil(len(timings) * .95) - 1)] if timings else None,
            'provider_latency_measured': live, 'results': rows,
            'quality_claim': 'Human review required. Offline reference output and substring screens do not measure Gemini correctness.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--all-models', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1: parser.error('--limit must be positive')
    data = run(args.live, list(AIConfig.from_env().models) if args.all_models else None, args.limit)
    serialized = json.dumps(data, indent=2, ensure_ascii=False)
    if args.output: args.output.write_text(serialized, encoding='utf-8')
    print(json.dumps({k: v for k, v in data.items() if k != 'results'}, indent=2))
