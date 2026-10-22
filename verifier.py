"""Meaning check for supported question forms using natural-language inference.

The model estimates textual entailment, not real-world truth. Exact numbers,
dates and source spans are checked separately. Uncalibrated scores are exposed
as diagnostics, never a probability that the final answer is true.
"""
import re,threading,time
import numpy as np
import models

_MODEL=None
_LOCK=threading.RLock()
_FAILED_AT=0.0
_VERBS=r'(?:use|cost|weigh|hold|contain|measure|retain|last|open|close|begin|start|direct|manage|head|operate|build|write|manufacture|employ|keep|charge|store|produce|ship|install|draw|detect|include|lead|run|curate|collect|record|release|convert|support)'
_PAST={'begin':'began','build':'built','write':'wrote','hold':'held','keep':'kept','lead':'led','run':'ran','draw':'drew'}


def _verb(verb,aux):
    if aux=='did':return _PAST.get(verb,verb+'d' if verb.endswith('e') else verb+'ed')
    if aux=='do':return verb
    if verb=='have':return 'has'
    return verb+'es' if verb.endswith(('s','sh','ch','x','o')) else verb+'s'


def _clause(tail):
    match=re.match(r'(.+?)\s+('+_VERBS+r')\b\s*(.*)',tail,re.I)
    return match.groups() if match else None


def hypothesis(question,span):
    """Return an explicit answer claim, or None when inversion is ambiguous."""
    q=question.strip(' \t?.!');a=span.strip(' \t?.!')
    q=re.sub(r'\b(?:current|latest|present)\s+','',q,flags=re.I)
    if not a:return None
    match=re.match(r'^(?:what|which)\s+(is|are|was|were)\s+(.+)$',q,re.I)
    if match:return f'{match[2]} {match[1]} {a}.'
    match=re.match(r'^who\s+(.+)$',q,re.I)
    if match:
        if re.match(r'(?:did|does|do|can)\b',match[1],re.I):return None
        return f'{a} {match[1]}.'
    match=re.match(r'^(?:how much|how heavy)\s+(is|are|was|were)\s+(.+)$',q,re.I)
    if match:
        verb='weighs' if q.lower().startswith('how heavy') else ('costs' if not re.search(r'\b(?:budget|salary|funding|revenue|mass|weight|capacity|volume)\b',match[2],re.I) else match[1])
        return f'{match[2]} {verb} {a}.'
    match=re.match(r'^how long\s+(is|are|was|were)\s+(.+?)\s+(retained|stored|kept|held|maintained|saved)$',q,re.I)
    if match:return f'{match[2]} {match[1]} {match[3]} for {a}.'
    match=re.match(r'^on (?:which|what) day\s+(is|are|was|were)\s+(.+?)\s+(open|closed)$',q,re.I)
    if match:return f'{match[2]} {match[1]} {match[3]} on {a}.'
    match=re.match(r'^(?:what|which)\s+(.*?)\s*(does|did|do)\s+(.+)$',q,re.I)
    if match:
        clause=_clause(match[3])
        if clause:
            subject,verb,rest=clause
            return f'{subject} {_verb(verb.lower(),match[2].lower())} {a} {rest}.'.replace(' .','.')
    match=re.match(r'^how (many|much|long)\s*(.*?)\s+(does|did|do)\s+(.+)$',q,re.I)
    if match:
        kind,obj,aux,tail=match.groups();clause=_clause(tail)
        if clause:
            subject,verb,rest=clause;verb=_verb(verb.lower(),aux.lower())
            if kind.lower()=='long':return f'{subject} {verb} {rest} for {a}.'.replace('  ',' ')
            value=a if not obj or obj.casefold() in a.casefold() else f'{a} {obj}'
            return f'{subject} {verb} {value} {rest}.'.replace(' .','.')
    match=re.match(r'^(?:when|on (?:what|which) date|in (?:what|which) year)\s+(did|does|do)\s+(.+)$',q,re.I)
    if match:
        clause=_clause(match[2])
        if clause:
            subject,verb,rest=clause;prep='in' if re.fullmatch(r'\d{4}',a) else 'on'
            return f'{subject} {_verb(verb.lower(),match[1].lower())} {rest} {prep} {a}.'.replace('  ',' ')
    match=re.match(r'^when\s+(is|are|was|were)\s+(.+)$',q,re.I)
    if match:
        tail=match[2];parts=tail.rsplit(' ',1)
        if len(parts)==2:return f'{parts[0]} {match[1]} {parts[1]} in {a}.'
    match=re.match(r'^(?:where|in (?:which|what) (?:city|place))\s+(did|does|do)\s+(.+)$',q,re.I)
    if match:
        clause=_clause(match[2])
        if clause:
            subject,verb,rest=clause
            return f'{subject} {_verb(verb.lower(),match[1].lower())} {rest} in {a}.'.replace('  ',' ')
    match=re.match(r'^(?:where|in (?:which|what) (?:city|place))\s+(is|are|was|were)\s+(.+)$',q,re.I)
    if match:
        tail=re.sub(r'\s+(?:located|based)$','',match[2],flags=re.I)
        return f'{tail} {match[1]} located in {a}.'
    match=re.match(r'^(is|are|was|were)\s+(.+)\s+(\w+)$',q,re.I)
    if match:
        neg=' not' if re.search(r'\b(?:not|no|never)\b',a,re.I) else ''
        return f'{match[2]} {match[1]}{neg} {match[3]}.'
    match=re.match(r'^(does|do|did)\s+(.+)$',q,re.I)
    if match:
        clause=_clause(match[2])
        if clause:
            subject,verb,rest=clause
            if re.search(r'\b(?:not|no|never)\b',a,re.I):return f'{subject} {match[1]} not {verb} {rest}.'
            return f'{subject} {_verb(verb.lower(),match[1].lower())} {rest}.'
    return None


def ready():return _MODEL is not None


def literal_support(premise, claim, named_subjects=()):
    """A contiguous source clause is stronger evidence than a model score.

    The only allowed omission is an explicit leading subject that already
    scopes a following clause in this same passage (e.g. Atlas / class B).
    Do not treat an unordered set of matching words as a witness.
    """
    import preprocessing
    wanted=preprocessing.normalize(claim)
    for sentence in preprocessing.sentence_split(premise):
        source=preprocessing.normalize(sentence)
        if wanted and (' '+wanted+' ') in (' '+source+' '):
            return {'available':True,'accepted':True,'method':'contiguous_source_clause','claim':claim}
    for subject in named_subjects:
        prefix=preprocessing.normalize(subject)
        if wanted.startswith(prefix+' '):
            remainder=wanted[len(prefix)+1:]
            if len(remainder.split())<4:continue
            for sentence in preprocessing.sentence_split(premise):
                source=preprocessing.normalize(sentence)
                if (' '+remainder+' ') in (' '+source+' '):
                    before=premise[:premise.find(sentence)]
                    if re.search(r'\b'+re.escape(subject)+r'\b',before,re.I):
                        return {'available':True,'accepted':True,'method':'contiguous_clause_with_explicit_subject_scope','claim':claim}
    return None


def check(premise,claim):
    """Return an NLI result; unavailable verification never reports entailment."""
    global _MODEL,_FAILED_AT
    with _LOCK:
        if _MODEL is None:
            if time.monotonic()-_FAILED_AT<30:return {'available':False}
            try:
                from sentence_transformers import CrossEncoder
                _MODEL=CrossEncoder(models.local_path(models.VERIFIER_MODEL),device='cpu',max_length=512,local_files_only=True)
            except Exception:
                _FAILED_AT=time.monotonic();return {'available':False}
        import torch
        logits=np.asarray(_MODEL.predict([(premise,claim)],activation_fct=torch.nn.Identity(),show_progress_bar=False))[0]
        scores=np.exp(logits-logits.max());scores=scores/scores.sum()
        labels=['contradiction','entailment','neutral']
        label=labels[int(scores.argmax())]
        return {'available':True,'claim':claim,'label':label,'entailment_score':float(scores[1]),
                'accepted':label=='entailment' and float(scores[1])>=.65,
                'score_type':'uncalibrated_textual_entailment','model':models.VERIFIER_MODEL}
