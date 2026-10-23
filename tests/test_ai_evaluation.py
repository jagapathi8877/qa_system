import json
from pathlib import Path
from dataclasses import replace
import pytest
from scripts.evaluate_ai import fixture, ReferenceTransport
from ai_config import AIConfig, Credential
from ai_explainer import AIExplainer
from gemini_gateway import GeminiGateway

CASES = json.loads((Path(__file__).parent / 'data/ai_evaluation.json').read_text())


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_synthetic_explanation_contract(case):
    engine, retrieval = fixture(case)
    gateway = GeminiGateway(AIConfig(credentials=(Credential(1, 'synthetic-not-a-real-key', 'project-one'),), max_attempts=1), ReferenceTransport(case))
    result = AIExplainer(gateway.config, gateway).explain(case['question'], retrieval, engine)
    gateway.close()
    assert result['status'] == ('unavailable' if case.get('output_kind') else 'success')
    if case['id'] == 'absent': assert result['basis'] == 'general_knowledge'
    if case['id'] == 'contradiction': assert result['basis'] == 'conflicting_evidence'
    if case['id'] == 'ambiguous': assert result['basis'] == 'incomplete_evidence'
    if case['id'] == 'long': assert result['context_truncated']
