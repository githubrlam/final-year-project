"""Development-only financial-row retrieval / input-check factorial experiment.

No benchmark-specific selection rules. References enter only saved-output analysis.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re

METHODS=('baseline','row_retrieval','input_binding','combined')
DEV_SHA='84cd505aa1dbb4d5f348b33a69367af22ba6508273278800f126f17e9864d334'
COMMON=dict(top_k=4,evidence_tokens=2000,context_tokens=6144,output_tokens=1024,
            thinking=False,max_model_calls=4,max_retrieval_rounds=2,seed=42,
            prompt_revision='input_ablation',retrieval_candidates=100,rrf_constant=60)
# General accounting vocabulary; no source values, IDs or expected answers.
FIELDS={
 'ppe':(r'property\s*,?\s*plant\s*(?:,?\s*and|&)\s*equipment|\bPP&E\b','property plant and equipment net'),
 'revenue':(r'(?:total|net)\s+(?:revenues?|sales)\b|^\s*(?:revenues?|sales)\b','total net revenues sales'),
 'net_income':(r'\bnet\s+(?:income|earnings|profit)\b','net income net earnings'),
 'cost_of_sales':(r'cost\s+of\s+(?:sales|revenues?|goods\s+sold)\b','cost of sales cost of revenues cost of goods sold'),
 'inventory':(r'\binventor(?:y|ies)\b','inventories inventory'),
 'current_assets':(r'\btotal\s+current\s+assets\b','total current assets'),
 'current_liabilities':(r'\btotal\s+current\s+liabilities\b','total current liabilities'),
 'receivables':(r'\b(?:accounts?|trade)\s+(?:and\s+other\s+)?receivables?\b','accounts trade receivables net'),
}


def requested_fields(question):
    names=[]
    rules=[(r'\bPPNE\b|\bPP[&E]+\b|property.?plant|fixed.asset', ['ppe']),
           (r'(?:net\s+)?profit\s+margin|net\s+margin',['revenue','net_income']),
           (r'fixed.asset\s+turnover',['revenue','ppe']),
           (r'inventory\s+turnover',['cost_of_sales','inventory']),
           (r'current\s+assets',['current_assets']),
           (r'current\s+liabilities',['current_liabilities']),
           (r'\bAR\b|receivables?',['receivables']),
           (r'revenue\s+growth|growth\s+(?:in|of)\s+(?:net\s+)?revenues?',['revenue'])]
    for pattern,fields in rules:
        if re.search(pattern,question,re.I):names.extend(fields)
    return [dict(name=n,pattern=FIELDS[n][0],query=FIELDS[n][1]) for n in dict.fromkeys(names)]


def row_hit(field,text):
    return any(re.search(field['pattern'],line,re.I) and re.search(r'\d',line) for line in text.splitlines())


class RowRetriever:
    """Supplement a scoped hybrid pool with lexical row anchors, then cover fields."""
    def __init__(self,base,question,study):
        self.base,self.study=base,study
        self.fields=requested_fields(question)
        self.events=[]
        self.indexes={}

    def retrieve(self,query,doc_name=None,previous=None):
        if not isinstance(doc_name,str) or not doc_name.strip():raise ValueError('Document scope is required.')
        if any(c['doc_name']!=doc_name for c in previous or []):raise ValueError('Previous candidates outside scope.')
        from development_repair import financial_query
        from rank_bm25 import BM25Okapi
        settings=self.study.CONFIG
        original_reranker=settings.get('reranker')
        expanded=financial_query(query)
        if self.fields:expanded+='\nRequired financial rows: '+'; '.join(f['query'] for f in self.fields)
        # Rerank exactly once, after supplementing the hybrid pool.
        settings['reranker']=None
        try:ranked=self.base.retrieve(expanded,doc_name=doc_name,previous=previous)
        finally:settings['reranker']=original_reranker
        if any(c['doc_name']!=doc_name for c in ranked):raise ValueError('Base retrieval escaped scope.')
        eligible=[c for c in self.base.chunks if c['doc_name']==doc_name]
        anchors=[]
        if self.fields and eligible:
            if doc_name not in self.indexes:
                self.indexes[doc_name]=getattr(self.base,'document_indexes',{}).get(doc_name) or BM25Okapi([self.study.tokens(c['text']) for c in eligible])
            index=self.indexes[doc_name]
            for field in self.fields:
                scores=index.get_scores(self.study.tokens(field['query']))
                hits=sorted((i for i,c in enumerate(eligible) if row_hit(field,c['text'])),
                            key=lambda i:(-float(scores[i]),eligible[i]['id']))[:2]
                anchors.extend({**eligible[i],'retrieval_score':0.0} for i in hits)
        pool=list({c['id']:c for c in [*anchors,*ranked]}.values())[:settings['retrieval_candidates']]
        reranked=self.study.rerank_candidates(expanded,pool,original_reranker,generator=settings['generator']) if original_reranker else pool
        remaining=list(reranked);ordered=[];uncovered={f['name'] for f in self.fields}
        while uncovered and remaining:
            hits=[{f['name'] for f in self.fields if f['name'] in uncovered and row_hit(f,c['text'])} for c in remaining]
            best=max(range(len(remaining)),key=lambda i:len(hits[i]))
            if not hits[best]:break
            uncovered-=hits[best];ordered.append(remaining.pop(best))
        ordered.extend(remaining)
        self.events.append({'query':query,'expanded_query':expanded,'doc_name':doc_name,
            'requested_fields':[f['name'] for f in self.fields],'lexical_field_searches':len(self.fields),
            'anchor_ids':list(dict.fromkeys(c['id'] for c in anchors)),'candidate_count':len(pool),
            'reranked_ids':[c['id'] for c in reranked],'selection_order':[c['id'] for c in ordered]})
        return ordered


BINDING='''
Before answering, identify each required financial input's exact row label, fiscal year,
unit and source chunk. Use consolidated net income for net profit margin, not operating
income or a mixture of continuing-operation and attributable-income definitions.
Check numerator, denominator, opening/closing years, signs and scale before calculation.
Do not substitute segment revenue for assets. Thousands to millions requires division
by 1,000; a percentage and a percentage-point change are different quantities.
If a needed input is absent, abstain or request that missing row in the permitted retry.
For a calculator request, put compact input records in the existing units field:
row label | fiscal year | value | source unit | exact source chunk ID; then output unit.
Use only the evidenced input values; put formula constants only in the expression.
This is a concise source record, not a request for hidden reasoning. Do not add JSON keys.
Return only the requested final value/units (or complete requested list) in answer.
'''


@contextmanager
def ablation_scope(study,method):
    if method not in METHODS:raise ValueError('Unknown development method.')
    config=dict(study.CONFIG);old=study.PROMPTS.get('input_ablation')
    prompts=dict(study.PROMPTS['v5'])
    if method in ('input_binding','combined'):
        from development_repair import FINANCE
        prompts['SYSTEM']+='\n'+FINANCE
        prompts['ASSESS']+='\n'+BINDING.split('For a calculator request,')[0]
        prompts['ANSWER']+='\n'+BINDING
    study.CONFIG.update(COMMON,input_ablation_method=method)
    study.PROMPTS['input_ablation']=prompts
    try:yield
    finally:
        study.CONFIG.clear();study.CONFIG.update(config)
        if old is None:study.PROMPTS.pop('input_ablation',None)
        else:study.PROMPTS['input_ablation']=old


def validate_development(path):
    path=Path(path)
    if path.name!='dev.jsonl' or hashlib.sha256(path.read_bytes()).hexdigest()!=DEV_SHA:
        raise ValueError('Only the pinned ten-question development split is allowed.')
    questions=[json.loads(line) for line in path.read_text().splitlines() if line]
    if len(questions)!=10 or len({q['financebench_id'] for q in questions})!=10:raise ValueError('Invalid development set.')
    return questions

