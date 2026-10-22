"""Backend API, cache, state and safety-policy regressions; no frontend edits."""
import io,json,math
import pytest
import app as app_module
import qa_model,retrievers,summary_engine
from qa_engine import QAEngine

@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setitem(app_module.app.config,'TESTING',True)
    app_module._STATE.clear()
    with app_module.app.test_client() as c:yield c

@pytest.mark.parametrize('payload',[['bad'],'bad',7,True,None])
@pytest.mark.parametrize('path',['/api/paste','/api/ask'])
def test_top_level_schema_errors_are_json(client,path,payload):
    r=client.post(path,data=json.dumps(payload),content_type='application/json')
    assert r.status_code==400 and r.is_json

@pytest.mark.parametrize('value',['NaN','Infinity',-1,1,True,[],{}])
def test_invalid_threshold_rejected_before_counters(client,value):
    before=client.get('/api/stats').get_json()['total_queries']
    r=client.post('/api/ask',json={'question':'What happened?','threshold':value})
    assert r.status_code==400
    assert client.get('/api/stats').get_json()['total_queries']==before

def test_capacity_readers_cannot_evict_active_dataset(client,monkeypatch):
    monkeypatch.setattr(app_module,'MAX_SESSIONS',1)
    assert client.post('/api/paste',json={'text':'An ocean observatory records pressure measurements and retains its daily scientific logs.'}).status_code==200
    for _ in range(55):app_module.app.test_client().get('/api/stats')
    assert client.get('/api/dataset').get_json()['loaded']
    other=app_module.app.test_client()
    assert other.post('/api/paste',json={'text':'A different observatory records scientific measurements in a separate coastal region.'}).status_code==503
    assert client.get('/api/dataset').get_json()['loaded']

def test_secret_is_not_public_development_key():
    assert app_module.app.secret_key!='quillqa-dev-key-change-me'
    assert app_module.app.config['SESSION_COOKIE_HTTPONLY']

def test_cache_preserves_coverage_and_is_copied():
    summary_engine._CACHE.clear()
    value={'bullets':['Source fact.'],'coverage':{'chunks_used':24,'total_passages':25},'coverage_note':'first 24 of 25'}
    summary_engine.cache_put(('test','corpus'),value)
    first=summary_engine.cache_get(('test','corpus'));first['coverage_note']=None
    assert summary_engine.cache_get(('test','corpus'))==value

def test_summary_group_budget_includes_unbroken_passages():
    groups=summary_engine._groups(['Term '*1800,'More evidence. '*400])
    assert all(len(g)<=summary_engine.GROUP_CHAR_LIMIT for g in groups)
    assert len(groups)>1

def test_shared_words_do_not_certify_a_summary_relation():
    source=['The project began in Pune. The team installed sensors along the western coast.']
    assert summary_engine.align_to_sources('The team installed sensors along the western coast of Pune.',source)==[]
    assert summary_engine.align_to_sources('The project began in Pune.',source)==['The project began in Pune.']

def test_meaning_and_numeric_simplification_preserved():
    import simplifier
    assert simplifier.simplify('Utilize individual sensors at -10 degrees and 37.50 units.')=='Use individual sensors at -10 degrees and 37.50 units.'

def test_scoped_meta_words_do_not_trigger_whole_corpus_summary():
    import qa_engine
    assert not qa_engine.is_summary_request('What is the executive summary budget?')
    assert not qa_engine.is_explain_request('Explain the dataset retention period.')
    assert qa_engine.is_summary_request('summarize this dataset')
    assert qa_engine.is_explain_request('explain the dataset in simple words')

def test_summary_total_count_and_balanced_sampling():
    engine=QAEngine('\n\n'.join(f'Station {i} records scientifically validated ocean observations in coastal sector {i}.' for i in range(510)))
    result=engine.ask('summarize this dataset')
    assert result['summary_stats']['sentences']==510
    assert result['coverage']['sentences_considered']==500
    assert '500 of 510' in result['coverage_note']

def test_suggestions_allow_larger_later_request():
    text=(__import__('pathlib').Path(__file__).parents[1]/'static/sample_dataset.txt').read_text()
    engine=QAEngine(text)
    assert len(engine.suggest_questions(1))==1
    assert len(engine.suggest_questions(4))==4

def test_explicit_definition_obeys_threshold_and_simple_words():
    engine=QAEngine('Luminescence is a process for emitting light from a material without heating it to a high temperature.')
    assert engine.ask('What is luminescence?',threshold=.9)['status']=='no_answer'
    result=engine.ask('What is luminescence in simple words?')
    assert result['status']=='answered' and result['answer_simple']
    assert result['confidence'] is None and not result['confidence_calibrated']

def test_health_does_not_load_models(client,monkeypatch):
    monkeypatch.setattr(qa_model,'get_pipeline',lambda:pytest.fail('Health must not load models'))
    result=client.get('/api/health')
    assert result.status_code==200 and result.get_json()['models']['network_during_inference'] is False

@pytest.mark.parametrize('question',['What is the field station salary?','What is the field station phone number?','How much does it cost?'])
def test_model_free_fallback_abstains_on_unproven_relation(monkeypatch,question):
    monkeypatch.setattr(qa_model,'get_pipeline',lambda:None)
    monkeypatch.setattr(retrievers,'semantic_rerank',lambda *args:None)
    monkeypatch.setattr(retrievers.SemanticRetriever,'_ensure',lambda *args:(_ for _ in ()).throw(RuntimeError('offline')))
    engine=QAEngine('The field station director is Maya Rao. The field station budget is 12000 rupees and its equipment costs 35 rupees.')
    assert engine.ask(question)['status']=='no_answer'
