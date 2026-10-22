"""Actual local-model regressions: relevance is not mere shared vocabulary."""
import pytest
from qa_engine import QAEngine

pytestmark=pytest.mark.slow

@pytest.mark.parametrize('mode',['tfidf','semantic'])
def test_relation_paraphrase_beats_keyword_distractor(mode):
    engine=QAEngine(
        'Cass Neri wrote the novel Amber Coast in 2018. The novel was published by a small independent press.\n\n'
        'The Amber Coast author biographies are stored on a shelf beside several popular novels in the campus bookstore.'
    )
    result=engine.ask('Who is the author of Amber Coast?',retriever=mode)
    assert result['status']=='answered' and 'Cass Neri' in result['answer']
    assert result['passage_index']==1

@pytest.mark.parametrize('question',['What is the Amber Coast author salary?','What is the bookstore telephone number?'])
def test_same_subject_is_not_evidence_for_an_absent_property(question):
    engine=QAEngine('Cass Neri wrote Amber Coast in 2018. The bookstore sells Amber Coast for 280 rupees and keeps author biographies beside the novel.')
    result=engine.ask(question)
    assert result['status']=='no_answer'
    assert result['source_passage'] is None

def test_conflicting_measurements_need_resolution():
    engine=QAEngine('The Argo prototype has a mass of 17 kilograms. This laboratory record has no date.\n\nThe Argo prototype has a mass of 19 kilograms. This second laboratory record has no date.')
    result=engine.ask('What is the mass of the Argo prototype?')
    assert result['status']=='no_answer' and result['reason']=='conflicting_evidence'
    assert len(result['sources'])>=2

def test_steps_are_complete_and_spans_remain_exact():
    engine=QAEngine('To service the Umber pump, first isolate the power supply. Next open the drain valve. Finally replace the seal and record the inspection result.')
    result=engine.ask('How do I service the Umber pump?')
    assert result['status']=='answered'
    for phrase in ['isolate','open','replace','record']:assert phrase in result['answer']
    assert result['passage_text'][result['answer_start']:result['answer_end']]==result['span']

def test_multifact_answer_has_separate_verifiable_sources():
    engine=QAEngine('The Topaz project director is Imani Cole. The team researches coastal erosion using a network of field instruments.\n\nThe Topaz annual budget is 47000 rupees. The budget is approved by the laboratory finance board every year.')
    result=engine.ask('Who directs Topaz and how much is its annual budget?')
    assert result['status']=='answered'
    assert 'Imani Cole' in result['answer'] and '47000' in result['answer']
    assert len(result['sources'])>=2
    for source in result['sources']:
        assert source['passage_text'][source['answer_start']:source['answer_end']]==source['span']

def test_query_polarity_is_respected():
    engine=QAEngine('The Umber pump is not submersible. The Umber pump is designed for dry indoor laboratories and must remain above the liquid level during operation.')
    result=engine.ask('Is the Umber pump submersible?')
    assert result['status']=='answered' and 'not submersible' in result['answer']


@pytest.mark.parametrize('question,expected',[
    ('Do Vesta sensors measure humidity?','do not measure humidity'),
    ('What does Vesta filtering use?','ceramic membrane'),
    ('What is the price of the Vesta kit?','145 euros'),
])
def test_property_and_polarity_survive_same_paragraph_distractors(question,expected):
    engine=QAEngine('Vesta sensors measure pressure. Vesta sensors do not measure humidity. Vesta sensors cost 145 euros.\n\n'
                    'Vesta filtering is a process for removing salt from samples. Vesta filtering uses a ceramic membrane.\n\n'
                    'The Vesta kit has a mass of 2 kilograms. The Vesta kit costs 145 euros. The Vesta kit runs for 8 hours.')
    result=engine.ask(question)
    assert result['status']=='answered' and expected in result['answer']
