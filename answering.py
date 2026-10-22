"""Evidence-first answering: semantic relevance, null-aware reading, constraints.

Reranker logits are used for ordering, never as truth probabilities. Every
answer keeps an exact source span. Unresolved alternatives fail closed.
"""
import re,time
import numpy as np
import preprocessing,qa_model,retrievers,simplifier
import verifier
from decimal import Decimal

MAX_CANDIDATES = 12
MAX_CONTEXTS = 4
_QUESTION_WORDS = {'Who','What','Where','When','Why','How','Which','Compare','Define','Tell','Explain','The','In','On','Give','Please','Is','Are','Do','Does','Did','Can','Could'}
_YEAR = re.compile(r'\b(?:18|19|20|21)\d{2}\b')
_NUMBER = re.compile(r'[+-]?\d+(?:[.,:]\d+)*|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|thirty|hundred|thousand)\b',re.I)
_DURATION = re.compile(r'\b(?:second|minute|hour|day|week|month|year)s?\b',re.I)
_NEGATIVE = re.compile(r"\b(?:not|never|cannot|no)\b|n['’]t\b",re.I)
_RELATIONS = [
    {'direct','director','manage','manager','head','lead','charge','run','curate','curator','responsible'},
    {'write','wrote','written','author'}, {'begin','began','start','open','establish','found'},
    {'last','life','lifetime','endurance','duration'}, {'cost','price','fee','charge'},
    {'retain','retention','keep','store'}, {'locate','location','city','base','where'},
    {'weigh','weight','heavy','mass'}, {'use','utilize','employ'},
]
_RELATION_STEMS = [{preprocessing.normalize(x) for x in group} for group in _RELATIONS]


def _entities(question):
    if question == question.upper():return []
    ignored={x.casefold() for x in _QUESTION_WORDS}|{'dr','mr','ms','mrs','i'}
    return [x for x in re.findall(r'\b[A-Z][\w-]*\b',question) if x.casefold() not in ignored]


def _tokens(text):
    return set(preprocessing.normalize(text).split())


def _expanded(tokens):
    out=set(tokens)
    for group in _RELATION_STEMS:
        if group & tokens:out.update(group)
    return out


def _kind(q):
    q=q.casefold().strip()
    if re.match(r'^how\b',q) and re.search(r'\b(?:work|works|function|functions|operate|operates)\??$',q):return 'mechanism'
    if re.match(r'^(?:who|whom)\b',q):return 'who'
    if re.match(r'^(?:where|in (?:which|what) (?:city|place|location))\b',q):return 'where'
    if re.match(r'^(?:when|on (?:what|which) date|in (?:what|which) year|what (?:time|year|date)|on which day)\b',q):return 'when'
    if re.match(r'^how (?:long|many (?:hours|days|months|years|seconds|minutes))\b',q):return 'duration'
    if re.match(r'^how (?:many|much|heavy)\b',q) or re.search(r'\b(maximum|minimum|temperature|weight|mass|capacity|price|budget)\b',q):return 'quantity'
    if re.match(r'^(?:is|are|do|does|did|can|could|has|have)\b',q):return 'boolean'
    if re.match(r'^(?:how (?:to|do|does|can|should)|what (?:are the )?steps)\b',q):return 'procedure'
    return 'fact'


def _sentence(passage,start,end):
    cursor=0
    for sentence in preprocessing.sentence_split(passage):
        at=passage.find(sentence,cursor);cursor=at+len(sentence)
        if at<=start<end<=cursor:return sentence,at,cursor
    return passage,0,len(passage)


def _clause(passage,start,end):
    sentence,at,finish=_sentence(passage,start,end)
    cursor=at
    for part in sentence.split(';'):
        position=passage.find(part,cursor);cursor=position+len(part)+1
        if position<=start<end<=position+len(part):return part.strip(),position,position+len(part)
    return sentence,at,finish


def _read_candidates(engine, question, ids):
    contexts=[engine.passages[i] for i in ids]
    locations=[(i,0) for i in ids]
    extra=0
    for pid in ids:
        passage=engine.passages[pid];cursor=0;previous=None
        for sentence in preprocessing.sentence_split(passage):
            at=passage.find(sentence,cursor);end=at+len(sentence);cursor=end
            start=previous if previous is not None and re.match(r'^(?:he|she|it|they|his|her|its|their)\b',sentence,re.I) else at
            previous=at
            window=passage[start:end]
            if len(window)<25 or window==passage or extra>=8:continue
            contexts.append(window);locations.append((pid,start));extra+=1
    raw=qa_model.answer_candidates(question,contexts)
    out=[]
    for item in raw:
        pid,shift=locations[item['context_index']]
        out.append(dict(item,pid=pid,start=item['start']+shift,end=item['end']+shift))
    return out


def _qualifier_conflict(question, sentence, passage):
    for category,identifier in re.findall(r'\b(class|channel|mode|type|grade)\s+([\w-]+)',question,re.I):
        present=re.findall(r'\b'+category+r'\s+([\w-]+)',sentence,re.I)
        if present and identifier.casefold() not in {x.casefold() for x in present}:return True
    # Contrastive modifiers are discovered in this passage, not a domain list.
    # Only reject a clause that explicitly uses a competing modifier for the
    # same noun; a bare pronoun/subject remains eligible for contextual reading.
    modifiers={}
    for s in preprocessing.sentence_split(passage):
        words=re.findall(r'[^\W\d_]+',s,re.UNICODE)
        for first,head in zip(words,words[1:]):
            if first.casefold() in preprocessing.STOPWORDS or _NEGATIVE.fullmatch(first):continue
            key=preprocessing.normalize(head);value=preprocessing.normalize(first)
            if key and value:modifiers.setdefault(key,set()).add(value)
    q=_tokens(question)
    for head,values in modifiers.items():
        wanted=values&q
        if len(values)<2 or head not in q or not wanted:continue
        words=re.findall(r'[^\W\d_]+',sentence)
        for first,last in zip(words,words[1:]):
            if preprocessing.normalize(last)==head and preprocessing.normalize(first) in values-wanted:return True
    return False


def _canonical(span):
    return re.sub(r'\b(?:dr|mr|mrs|ms|the|a|an)\b','',preprocessing.normalize(span)).strip()


def _wrong_dimension(question, span):
    """Explicit answer type constraints override an optimistic model score."""
    money=r'\b(?:rupees?|dollars?|euros?|pounds?|USD|INR|EUR|free)\b|[$€£₹]'
    if re.search(r'\b(?:price|cost|salary|budget|funding|fee)\b|^how much (?:is|does .* cost)',question,re.I):
        if re.search(r'\b(?:seconds?|minutes?|hours?|days?|years?|grams?|litres?|liters?|Celsius)\b',span,re.I):return True
    if re.search(r'\b(?:capacity|volume|reservoir)\b',question,re.I) and re.search(r'\b(?:seconds?|minutes?|hours?|days?|years?|grams?|rupees?)\b',span,re.I):return True
    if re.match(r'^what time\b',question,re.I) and not re.search(r'\d{1,2}:\d{2}|\b(?:am|pm|noon|midnight)\b',span,re.I):return True
    if re.match(r'^how long\b',question,re.I) and re.search(money,span,re.I):return True
    if re.search(r'\b(?:battery life|duration|delivery time)\b',question,re.I) and not _DURATION.search(span):return True
    return False


def _explicit_bound(question, candidate, passage):
    requested=re.search(r'\b(minimum|maximum|lowest|highest)\b',question,re.I)
    if not requested:return candidate
    kinds=[(r'temperatur|celsius|fahrenheit|kelvin',r'(?:degrees?\s+)?(?:Celsius|Fahrenheit|kelvin|°\s*[CF])'),
           (r'pressure',r'(?:kilo)?pascals?'),(r'voltage',r'volts?'),(r'power|watt',r'watts?'),
           (r'capacity|volume',r'(?:milli)?lit(?:re|er)s?'),(r'mass|weight',r'(?:kilo)?grams?')]
    for name,unit in kinds:
        if not re.search(name,question,re.I):continue
        for match in re.finditer(r'\b(?:between|from)\s+([+-]?\d+(?:\.\d+)?)\s+(?:and|to)\s+([+-]?\d+(?:\.\d+)?)\s+('+unit+r')\b',passage,re.I):
            if not (candidate['start']<match.end() and candidate['end']>match.start()):continue
            minimum=requested[1].casefold() in ('minimum','lowest')
            left,right=Decimal(match[1]),Decimal(match[2]);group=1 if ((left<=right)==minimum) else 2
            start,end=match.start(group),match.end(group)
            if group==2:end=match.end(3)
            return dict(candidate,answer=passage[start:end],start=start,end=end,score=None,
                        meaning_check={'method':'explicit_numeric_bound','accepted':True,'requested':requested[1],
                                       'bounds':[match[1],match[2]],'unit':match[3]})
    return candidate


def _source(engine,pid,start,end):
    passage=engine.passages[pid]
    return {'passage_id':pid+1,'passage_index':pid+1,'passage_text':passage,'source_passage':passage,
            'span':passage[start:end],'answer_start':start,'answer_end':end}


def _refuse(engine,question,started,mode,message,reason,sources=None):
    result=engine._no_info(message,started,mode)
    result.update(reason=reason,confidence_type='not_a_probability',confidence_calibrated=False)
    if sources:result['sources']=sources
    return result


def _render(engine,question,started,mode,source,score,retrieval_score,route,semantic_score=None,rerank_score=None):
    from qa_engine import compose_answer
    passage=source['passage_text'];sentence,at,end=_sentence(passage,source['answer_start'],source['answer_end'])
    if len(source['span'].split())<=10 and len(sentence)<=500:
        display=sentence
    else:display=source['span']
    result={'status':'answered','answer':compose_answer(display),'confidence':score,
            'retrieval_score':round(float(retrieval_score),4),'retriever':mode,
            'response_time_ms':round((time.perf_counter()-started)*1000),'category':None,**source,
            'sources':[dict(source)],'answer_method':route,'confidence_type':'uncalibrated_model_span_score' if score is not None else 'source_evidence_no_probability',
            'confidence_calibrated':False,'semantic_score':semantic_score,'reranker_logit':rerank_score}
    if re.search(r'\b(?:simple words|simple terms|plain english|plain words)\b',question,re.I):
        result['answer_simple']=simplifier.simplify(display)
    return result


def _candidate_pool(engine,question,mode,threshold):
    norm=preprocessing.normalize(question)
    keyword=engine.tfidf.scores(question,norm)
    semantic=None
    if mode=='semantic':
        semantic=engine._get_retriever('semantic').scores(question)
        floor=.30 if threshold is None else threshold
        ids=[int(i) for i in np.argsort(semantic)[::-1] if semantic[i]>=floor][:MAX_CANDIDATES]
        if threshold is None and ids:
            # Check lexically plausible alternatives too. A weaker embedding
            # match must not hide a contradictory value or a second battery.
            ids=list(dict.fromkeys(ids+[int(i) for i in np.argsort(keyword)[::-1] if keyword[i]>=.05]))[:MAX_CANDIDATES]
    else:
        # Keep top topical candidates even when whole-passage cosine is diluted;
        # the cross-encoder/reader, not one shared token, decides answer support.
        floor=0.01 if threshold is None else threshold
        ids=[int(i) for i in np.argsort(keyword)[::-1] if keyword[i]>=floor][:MAX_CANDIDATES]
        if threshold is None and (not ids or float(keyword.max())<.12):
            try:
                semantic=engine._get_retriever('semantic').scores(question)
                ids=list(dict.fromkeys(ids+[int(i) for i in np.argsort(semantic)[::-1] if semantic[i]>=.30]))[:MAX_CANDIDATES]
            except RuntimeError:
                pass
    # A source must carry named subjects, not just a relation shared by another topic.
    entities=_entities(question)
    named=[i for i in ids if all(re.search(r'\b'+re.escape(entity)+r'\b', engine.passages[i],re.I) for entity in entities)]
    if named:ids=named
    elif entities:ids=[]
    nonjunk=[i for i in ids if i not in engine.junk_passages]
    ids=nonjunk or ids
    if not ids:return [],keyword,semantic,{}
    reranked=retrievers.semantic_rerank(question,[engine.passages[i] for i in ids])
    logits={}
    if reranked is not None:
        logits={i:float(s) for i,s in zip(ids,reranked)}
        ids=sorted(ids,key=lambda i:logits[i],reverse=True)
        ids=[i for i in ids if logits[i]>=0.0]
    return ids[:MAX_CONTEXTS],keyword,semantic,logits


def _fallback(engine,question,ids):
    """No reader: require explicit relation/type evidence, otherwise abstain."""
    kind=_kind(question);q=_tokens(question);expanded=_expanded(q)
    entities={preprocessing.normalize(x) for x in _entities(question)}
    relation=q-entities-{'what','who','where','when','how','much','mani','long','doe','did','pleas'}
    candidates=[]
    for pid in ids:
        passage=engine.passages[pid]
        for sentence in preprocessing.sentence_split(passage):
            terms=_tokens(sentence)
            if kind=='who' and not re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',sentence):continue
            if kind=='when' and not (_YEAR.search(sentence) or re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|\d{1,2}:\d{2}',sentence,re.I)):continue
            if kind=='where' and not re.search(r'\b(?:in|at|based|located|from)\b',sentence,re.I):continue
            if kind=='duration' and not _DURATION.search(sentence):continue
            if kind=='quantity' and not _NUMBER.search(sentence):continue
            if kind=='procedure' and not re.search(r'\b(?:first|next|finally|step)\b',sentence,re.I):continue
            # A specific relation (salary, founder, retention, etc.) must match,
            # independently of entity overlap. A phrase about price is not a date.
            distinctive=relation-{'maximum','minimum','project','initi','system','unit'}
            if any(not (_expanded({term}) & terms) for term in distinctive):continue
            if kind=='fact' and len(expanded&terms)/max(1,len(q))<.65:continue
            start=passage.find(sentence);end=start+len(sentence)
            if kind=='procedure':start,end=0,len(passage)
            if re.match(r'^(?:he|she|it|they|his|her|its|their)\b',sentence,re.I) and start>0:
                preceding=preprocessing.sentence_split(passage[:start])
                if preceding:start=passage.rfind(preceding[-1],0,start)
            candidates.append({'answer':passage[start:end],'start':start,'end':end,'pid':pid,'score':None,
                               'rank':len(expanded&terms)/max(1,len(q))})
    return sorted(candidates,key=lambda x:x['rank'],reverse=True)


def _split(question):
    """Small compositional grammar; unresolved structures are never invented."""
    parts=re.split(r'\s+and\s+(?=(?:who|what|when|where|how|which|why|in which|in what|on what)\b)',question,flags=re.I)
    if len(parts)>1:
        entities=_entities(parts[0]);subject=' '.join(entities)
        if subject:
            parts=[parts[0]]+[re.sub(r'\b(?:its|their|his|her)\b',subject+"'s",re.sub(r'\b(?:it|they|he|she)\b',subject,p,flags=re.I),flags=re.I) for p in parts[1:]]
        return [p.strip(' ?.')+'?' for p in parts], 'multipart'
    match=re.match(r'^(?:compare|contrast)\s+(?:the\s+)?(.+?)\s+of\s+(.+?)\s+(?:and|versus|vs\.?)\s+(.+?)[?.!]*$',question,re.I)
    if match:
        aspect,left,right=match.groups()
        return [f'What is the {aspect} of {left}?',f'What is the {aspect} of {right.rstrip("?.!")}?'],'comparison'
    match=re.match(r'^(?:compare|contrast|what is the difference between)\s+(.+?)\s+(?:and|versus|vs\.?)\s+(.+?)[?.!]*$',question,re.I)
    if match:
        left,right=match.groups();right=right.rstrip('?.!');lw,rw=left.split(),right.split()
        if len(rw)>1:
            shared=' '.join(rw[1:]);prefix=' '.join(lw[:-1]);suffix=re.sub(r'\b(prices|times|costs)\b',lambda m:m.group()[:-1],shared)
            return [f'What is {left} {suffix}?',f'What is {prefix+" " if prefix else ""}{rw[0]} {suffix}?'],'comparison'
        return [f'What is {left}?',f'What is {right}?'],'comparison'
    return [question],None


def _location_chain(question):
    match=re.match(r'^(?:where|in (?:which|what) city)\s+(?:is|was)\s+(.+?)\s+(?:that\s+)?(operating|running|managing|operates|runs|manages|built|owns|maintains|that built)\s+(.+?)(?:\s+(?:based|located))?\??$',question,re.I)
    if not match:return None
    head,verb,obj=match.groups();obj=re.sub(r'\s+(?:based|located)\??$','',obj,flags=re.I).rstrip('?.!')
    verb={'operating':'operates','running':'runs','managing':'manages','that built':'built'}.get(verb.casefold(),verb)
    return f'Who {verb} {obj}?'


def answer(engine,question,threshold=None,retriever='tfidf',_depth=0):
    from qa_engine import (_definition_topic,_definition_answer,_has_definition_cue,
                           is_summary_request,is_explain_request,_question_key,_SIMPLE_RE)
    started=time.perf_counter();question=(question or '').strip();mode=retriever if retriever in ('tfidf','semantic') else 'tfidf'
    if not question:return engine._no_info('Please type a question first.',started,mode)
    if len(question)>1000:return _refuse(engine,question,started,mode,'Please shorten the question to 1000 characters.','input_limit')
    if is_summary_request(question):return engine._summarize(started)
    if is_explain_request(question):return engine._analyze(started)
    if re.search(r'\b(?:it|its|they|their|his|her)\b',question,re.I) and not _entities(question):
        return _refuse(engine,question,started,mode,'Please specify the person, item or topic you mean; this question has an unresolved reference.','clarification_required')
    if not _depth:
        parts,composition=_split(question)
        if composition:
            if len(parts)>3:return _refuse(engine,question,started,mode,'Please ask at most three related questions at a time.','question_scope_limit')
            answers=[answer(engine,p,threshold,mode,1) for p in parts]
            if any(x['status']!='answered' for x in answers):
                return _refuse(engine,question,started,mode,'The dataset does not support every part of this question. '+ ' '.join(x['answer'] for x in answers if x['status']!='answered'),'incomplete_evidence')
            result=dict(answers[0]);sources=[s for a in answers for s in a.get('sources',[a])]
            result.update(answer='Based on the dataset: '+' '.join(a['answer'].removeprefix('Based on the dataset: ').strip() for a in answers),
                          sources=sources,subquestions=parts,category='Comparison' if composition=='comparison' else None,
                          answer_method='composed_source_evidence',response_time_ms=round((time.perf_counter()-started)*1000))
            scores=[a['confidence'] for a in answers]
            result['confidence']=min(scores) if all(x is not None for x in scores) else None
            return result
    chain=_location_chain(question) if _depth<2 else None
    if chain:
        owner=answer(engine,chain,threshold,mode,_depth+1)
        if owner['status']=='answered':
            location=answer(engine,f"Where is {owner['span']} located?",threshold,mode,_depth+1)
            if location['status']=='answered':
                location['sources']=owner.get('sources',[])+location.get('sources',[])
                location['subquestions']=[chain,f"Where is {owner['span']} located?"]
                location['answer_method']='linked_source_evidence'
                location['response_time_ms']=round((time.perf_counter()-started)*1000)
                return location
    suggested=engine._suggested_answers.get(_question_key(question))
    if suggested is not None and threshold is None:
        text,pid=suggested;start=engine.passages[pid].find(text)
        if start>=0:
            result=_render(engine,question,started,mode,_source(engine,pid,start,start+len(text)),None,0.0,'suggested_source_evidence')
            result['retrieval_score']=None
            return result
    topic=_definition_topic(question)
    if topic:
        definition=_definition_answer(engine,question,topic,started,mode)
        if definition is not None:
            if mode=='semantic':
                semantic_scores=engine._get_retriever('semantic').scores(question)
                definition['retrieval_score']=round(float(semantic_scores[definition['passage_index']-1]),4)
            if threshold is not None and definition['retrieval_score']<threshold:
                return _refuse(engine,question,started,mode,'The definition does not meet the requested retrieval threshold.','below_threshold')
            definition.update(confidence=None,confidence_type='source_definition_no_probability',confidence_calibrated=False,
                              answer_method='explicit_source_definition',sources=[{k:definition[k] for k in ['passage_id','passage_index','passage_text','source_passage','span','answer_start','answer_end']}])
            if _SIMPLE_RE.search(question):definition['answer_simple']=simplifier.simplify(definition['answer'].removeprefix('Based on the dataset: '))
            return definition
        if _has_definition_cue(engine,topic):return _refuse(engine,question,started,mode,f'No clear definition of {topic} was found in the dataset.','no_explicit_definition')
    # No keyword-only instruction tokens should distort candidate retrieval.
    from qa_engine import _INSTRUCTION_RE
    clean=re.sub(r'\s+',' ',_INSTRUCTION_RE.sub(' ',question)).strip()
    ids,keyword,semantic,logits=_candidate_pool(engine,clean,mode,threshold)
    if not ids:return _refuse(engine,question,started,mode,'The information in the provided dataset is not enough to answer this question.','no_relevant_evidence')
    reader=qa_model.get_pipeline()
    if reader is not None:
        candidates=_read_candidates(engine,clean,ids)
    else:candidates=_fallback(engine,clean,ids)
    if _kind(clean)=='boolean' and reader is not None:
        # An explicit full proposition can certify polarity even when an
        # extractive reader offers only a fragment or prefers its null span.
        candidates+=_fallback(engine,clean,ids)
    kind=_kind(clean);years=set(_YEAR.findall(clean));entities=_entities(clean)
    validated=[];verification_unavailable=False
    for c in candidates:
        passage=engine.passages[c['pid']]
        c=_explicit_bound(clean,c,passage)
        sentence,_,_=_clause(passage,c['start'],c['end'])
        if _wrong_dimension(clean,c['answer']):continue
        action=re.match(r'^(?:what|which)\b.*?\b(?:does|do|did)\s+(.+)',clean,re.I)
        clause=verifier._clause(action[1]) if action else None
        if clause:
            predicate=_tokens(clause[1]);known=next((group for group in _RELATION_STEMS if group&predicate),None)
            if known and not known&_tokens(sentence):continue
        if _qualifier_conflict(clean,sentence,passage):continue
        if years and not years.issubset(set(_YEAR.findall(sentence))):continue
        if kind=='who' and _NUMBER.fullmatch(c['answer'].strip()):continue
        if kind=='duration' and not _DURATION.search(sentence):continue
        if kind=='quantity' and not _NUMBER.search(c['answer']):continue
        if kind=='when' and not (_YEAR.search(sentence) or _NUMBER.search(c['answer']) or re.search(r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)s?\b',c['answer'],re.I)):continue
        if kind=='where' and reader is not None:
            prefix=passage[max(0,c['start']-35):c['start']]
            if not re.search(r'\b(?:in|at|from|near|outside|inside|within|to)\s+(?:(?:the|city|town|of)\s+)*$',prefix,re.I):continue
        if kind=='procedure':
            if not re.search(r'\b(?:first|next|finally|step)\b',passage,re.I):continue
            c=dict(c,answer=passage,start=0,end=len(passage))
        if kind=='boolean':
            relation=_tokens(clean)-_tokens('is are was were do does did can could has have')-{preprocessing.normalize(x) for x in entities}
            if not relation.issubset(_expanded(_tokens(sentence))):continue
            _,start,end=_clause(passage,c['start'],c['end'])
            c=dict(c,answer=passage[start:end],start=start,end=end)
        c['sentence']=sentence
        if kind=='who':
            appointment=re.search(r'\b(?:names?|appoints?|appointed|named)\s+'+re.escape(c['answer'])+r'\s+as\s+([^.;]+)',sentence,re.I)
            role=_tokens(clean)-_tokens('who is was the current latest present now')-{preprocessing.normalize(x) for x in entities}
            if appointment and role and role.issubset(_tokens(appointment[1])):
                c['meaning_check']={'method':'explicit_role_appointment','accepted':True}
        if reader is not None and kind not in ('procedure','mechanism') and not c.get('meaning_check'):
            claim=verifier.hypothesis(clean,c['answer'])
            if claim:
                proof=verifier.literal_support(passage,claim,entities)
                if proof is None:
                    proof=verifier.check(sentence,claim)
                    # Context can resolve an explicit pronoun or a relative
                    # subject, but may not certify an unrelated weak span.
                    if not proof.get('accepted') and sentence!=passage and float(c['score'] or 0)>=.5:
                        full=verifier.check(passage,claim)
                        if full.get('accepted'):proof=full
                if not proof.get('accepted'):
                    if not proof.get('available',True):verification_unavailable=True
                    continue
                c['meaning_check']=proof
        validated.append(c)
    # Relevance belongs to the clause containing this answer, not merely its
    # paragraph. A paragraph may discuss unrelated properties of one entity.
    if validated:
        evidence_scores=retrievers.semantic_rerank(clean,[c['sentence'] for c in validated])
        if evidence_scores is not None:
            for c,s in zip(validated,evidence_scores):c['evidence_rank']=float(s)
    def rank(c):return c.get('evidence_rank',logits.get(c['pid'],c.get('rank',0)))
    candidates=sorted(validated,key=lambda c:(rank(c),float(c['score'] or 0)),reverse=True)
    if not candidates:
        # A false positive premise can be corrected with explicit negative evidence.
        if re.match(r'^why\b',clean,re.I) and not _NEGATIVE.search(clean):
            qterms=_tokens(clean)-{'whi'}
            for pid in ids:
                for sentence in preprocessing.sentence_split(engine.passages[pid]):
                    if _NEGATIVE.search(sentence) and len(qterms&_tokens(sentence))/max(1,len(qterms))>=.6:
                        start=engine.passages[pid].find(sentence)
                        return _render(engine,question,started,mode,_source(engine,pid,start,start+len(sentence)),None,keyword[pid],'explicit_premise_correction',rerank_score=logits.get(pid))
        if verification_unavailable:
            return _refuse(engine,question,started,mode,'The local meaning-verification model is unavailable. Prepare the verifier model before retrying this question.','meaning_verifier_unavailable')
        return _refuse(engine,question,started,mode,'The dataset mentions this topic, but it does not provide enough evidence to answer the question directly.','reader_abstained' if reader is not None else 'unverified_without_reader')
    # Explicit current/latest requests can use dated evidence; historic years
    # were already enforced against the actual answer sentence above.
    if re.search(r'\b(?:current|latest|now|present)\b',clean,re.I):
        dated=[c for c in candidates if _YEAR.search(c['sentence'])]
        if dated:
            candidates=sorted(dated,key=lambda c:max(map(int,_YEAR.findall(c['sentence']))),reverse=True)
    best=candidates[0]
    alternatives=[]
    if kind not in ('procedure','mechanism') and not years and not re.search(r'\b(?:current|latest|now|present)\b',clean,re.I):
        for c in candidates[1:]:
            if _canonical(c['answer'])==_canonical(best['answer']):continue
            if c['pid']==best['pid'] and c['start']<best['end'] and c['end']>best['start']:continue
            if c['pid']==best['pid'] and _clause(engine.passages[c['pid']],c['start'],c['end'])[1:]==_clause(engine.passages[best['pid']],best['start'],best['end'])[1:]:continue
            # Clause relevance can be low when its subject is supplied by an
            # earlier sentence. It must not hide an already supported competing
            # answer from that paragraph (or a similarly relevant paragraph).
            gap=logits.get(best['pid'],best.get('rank',0))-logits.get(c['pid'],c.get('rank',0))
            if gap<=2.0:alternatives.append(c)
    if alternatives:
        sources=[_source(engine,c['pid'],c['start'],c['end']) for c in [best]+alternatives]
        reason='conflicting_evidence' if entities else 'clarification_required'
        message=('The dataset contains conflicting answers' if entities else 'More than one item in the dataset fits this question; please specify which one you mean')
        message+=': '+'; '.join(dict.fromkeys(c['answer'] for c in [best]+alternatives))+'.'
        return _refuse(engine,question,started,mode,message,reason,sources)
    pid=best['pid'];score=semantic[pid] if mode=='semantic' else keyword[pid]
    result = _render(engine,question,started,mode,_source(engine,pid,best['start'],best['end']),best['score'],score,
                   'semantic_rerank_squad2' if reader is not None else 'verified_relation_fallback',
                   semantic_score=float(semantic[pid]) if semantic is not None else None,rerank_score=logits.get(pid))
    result['meaning_check']=best.get('meaning_check',{'method':'explicit_procedure' if kind=='procedure' else 'reader_answerability_and_semantic_relevance',
                                                     'score_is_probability_of_truth':False})
    return result
