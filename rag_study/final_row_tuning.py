"""One final development retrieval revision; no gold-directed retrieval or new prompts."""
from contextlib import contextmanager
import json
from pathlib import Path
import re

from input_ablation import COMMON, FIELDS, requested_fields, validate_development, ablation_scope

METHODS=('baseline','row_verified')
ROW_PATTERNS={**{name:spec[0] for name,spec in FIELDS.items()},
    'ppe':r'(?:property\s*,?\s*(?:plant\s*,?\s*(?:and|&)?\s*)?(?:and\s+)?equipment|PP&E)',
    'revenue':r'(?:(?:total|consolidated|net)\s+)*(?:revenues?|sales)',
    'net_income':r'net\s+(?:income|earnings|profit)(?:\s+including\s+noncontrolling\s+interests|\s+attributable\s+to\s+.+?\s+shareholders)?',
    'current_assets':r'(?:total\s+)?current\s+assets',
    'current_liabilities':r'(?:total\s+)?current\s+liabilities',
}
NUMBER=r'\(?[+−-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?'


def row_cells(field,line):
    # Require a row-label boundary followed exclusively by financial-value cells.
    # Prose, units embedded in prose, and ambiguous single-space layouts are not boosted.
    pattern=rf'^\s*(?:{ROW_PATTERNS[field]})(?:,?\s+net)?(?:\s*\([a-z0-9]\))?(?:\s*\|\s*|\s{{2,}})(.+?)\s*$'
    match=re.match(pattern,line,re.I)
    if not match:return []
    suffix=match[1]
    if not re.fullmatch(r'[\s|$€£()\d,.+−\-–—]+',suffix):return []
    return re.findall(NUMBER,suffix)


def row_matches(field,line):
    return bool(row_cells(field,line))


def requested_years(question):
    text=re.sub(r'\bFY\s?(\d{2})(?!\d)',r'FY20\1',question,flags=re.I)
    years={int(y) for y in re.findall(r'(?<!\d)((?:19|20)\d{2})(?!\d)',text)}
    for a,b in re.findall(r'((?:19|20)\d{2})\s*(?:-|–|—|to)\s*(?:FY\s*)?((?:19|20)\d{2})',text,re.I):
        if 0<=int(b)-int(a)<=10:years.update(range(int(a),int(b)+1))
    return years


def field_years(question,field):
    years=requested_years(question)
    turnover=bool(re.search(r'(?:fixed.asset|inventory)\s+turnover',question,re.I))
    if turnover and years:
        if field in ('ppe','inventory'):years=years|{max(years)-1}
        elif field in ('revenue','cost_of_sales'):years={max(years)}
    return years


def header_years(text):
    years=set()
    for line in text.splitlines():
        found={int(y) for y in re.findall(r'(?<!\d)((?:19|20)\d{2})(?!\d)',line)}
        if not found:continue
        remaining=re.sub(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|as|of|fiscal|year|years|ended|ending|unaudited|in|millions|million|thousands|thousand|billions|billion|USD|dollars|except|per|share|amounts)\b','',line,flags=re.I)
        # Dates/year-only table headers are allowed; narrative mentions are not.
        if re.fullmatch(r'[\s\d,./|()\-–—:]*',remaining):years.update(found)
    return years


def verified_field(field,text,question):
    needed=field_years(question,field)
    if not needed or not needed<=header_years(text):return False
    return any(len(row_cells(field,line))>=len(needed) for line in text.splitlines())


class VerifiedRowRetriever:
    """Heuristic table-row/year priority, not semantic or source-unit validation."""
    def __init__(self,base,question,study):
        self.base,self.question,self.study=base,question,study
        self.fields=requested_fields(question);self.events=[];self.indexes={};self.qualified={}

    def retrieve(self,query,doc_name=None,previous=None):
        from development_repair import financial_query
        from rank_bm25 import BM25Okapi
        if not isinstance(doc_name,str) or not doc_name.strip():raise ValueError('Document scope required.')
        if any(c['doc_name']!=doc_name for c in previous or []):raise ValueError('Previous candidates outside scope.')
        config=self.study.CONFIG;reranker=config.get('reranker')
        expanded=financial_query(query)
        if self.fields:expanded+='\nRequired statement rows: '+'; '.join(f['query'] for f in self.fields)
        config['reranker']=None
        try:ranked=self.base.retrieve(expanded,doc_name=doc_name,previous=previous)
        finally:config['reranker']=reranker
        if any(c['doc_name']!=doc_name for c in ranked):raise ValueError('Base retrieval escaped scope.')
        eligible=[c for c in self.base.chunks if c['doc_name']==doc_name]
        anchors=[];qualified={}
        if eligible and self.fields:
            if doc_name not in self.indexes:
                self.indexes[doc_name]=getattr(self.base,'document_indexes',{}).get(doc_name) or BM25Okapi([self.study.tokens(c['text']) for c in eligible])
            for f in self.fields:
                name=f['name'];key=(doc_name,name)
                if key not in self.qualified:
                    self.qualified[key]={c['id'] for c in eligible if verified_field(name,c['text'],self.question)}
                qualified[name]=self.qualified[key]
                scores=self.indexes[doc_name].get_scores(self.study.tokens(f['query']))
                hits=sorted((i for i,c in enumerate(eligible) if c['id'] in qualified[name]),key=lambda i:(-float(scores[i]),eligible[i]['id']))[:2]
                anchors.extend({**eligible[i],'retrieval_score':0.0} for i in hits)
        pool=list({c['id']:c for c in [*anchors,*ranked]}.values())[:config['retrieval_candidates']]
        ordered=self.study.rerank_candidates(expanded,pool,reranker,generator=config['generator']) if reranker else list(pool)
        original_order=[c['id'] for c in ordered]
        remaining=list(ordered);chosen=[];missing={f['name'] for f in self.fields}
        while missing and remaining:
            coverage=[{name for name in missing if c['id'] in qualified.get(name,set())} for c in remaining]
            best=max(range(len(remaining)),key=lambda i:len(coverage[i]))
            if not coverage[best]:break
            missing-=coverage[best];chosen.append(remaining.pop(best))
        chosen.extend(remaining)
        self.events.append(dict(query=query,expanded_query=expanded,doc_name=doc_name,
            field_years={f['name']:sorted(field_years(self.question,f['name'])) for f in self.fields},
            lexical_field_searches=len(self.fields),anchor_ids=list(dict.fromkeys(c['id'] for c in anchors)),
            qualified_ids={k:sorted(v) for k,v in qualified.items()},candidate_count=len(pool),
            reranked_ids=original_order,selection_order=[c['id'] for c in chosen],
            limitation='Row/year-header co-occurrence is heuristic; no complete column binding or unit validation.'))
        return chosen


@contextmanager
def method_scope(study,method):
    if method not in METHODS:raise ValueError('Unknown final development method.')
    with ablation_scope(study,'baseline'):
        study.CONFIG['final_row_method']=method
        yield


def summarise(study,folder,questions):
    import pandas as pd
    from judge_pilot_run import paired_interval
    from numeric_verification import check_numeric
    rows=study.read_jsonl(folder/'predictions.jsonl')
    expected={(m,v,q['financebench_id']) for m in METHODS for v in ('B0','B1','B2') for q in questions}
    actual=[(r['configuration'],r['variant'],r['financebench_id']) for r in rows]
    if len(actual)!=60 or len(set(actual))!=60 or set(actual)!=expected:raise ValueError('Expected 60 unique development outputs.')
    qs={q['financebench_id']:q for q in questions}
    numeric=[]
    for r in rows:
        q=qs[r['financebench_id']]
        check={'status':'not_checked','reason':'Abstention or invalid output.'} if r['invalid'] or r['abstain'] else check_numeric(r['answer'],q['answer'],q['question'])
        numeric.append({k:r[k] for k in ('configuration','variant','financebench_id')}|check)
    pd.DataFrame(numeric).to_csv(folder/'numeric_consistency.csv',index=False)
    frame=pd.read_csv(folder/'automatic_scores.csv',dtype={'financebench_id':str})
    runtime=pd.DataFrame(rows)[['configuration','variant','financebench_id','doc_name','model_calls','retrieval_rounds']]
    frame=frame.merge(runtime,on=['configuration','variant','financebench_id'],validate='one_to_one')
    summary=frame.groupby(['configuration','variant']).agg(n=('answer_correct','size'),correct=('answer_correct','sum'),
        invalid=('invalid','sum'),abstentions=('abstain','sum'),page_recall=('evidence_page_recall','mean'),
        median_seconds=('seconds','median'),input_tokens=('input_tokens','mean'),output_tokens=('output_tokens','mean'),
        model_calls=('model_calls','mean'),retrieval_rounds=('retrieval_rounds','mean')).reset_index()
    summary.to_csv(folder/'comparison.csv',index=False)
    intervals=[];cases=[]
    for policy in ('B0','B1','B2'):
        b=frame[(frame.configuration=='baseline')&(frame.variant==policy)]
        m=frame[(frame.configuration=='row_verified')&(frame.variant==policy)]
        pair=b.merge(m,on=['doc_name','financebench_id'],suffixes=('_baseline','_method'),validate='one_to_one')
        delta=[dict(doc_name=r.doc_name,difference=int(r.answer_correct_method)-int(r.answer_correct_baseline)) for r in pair.itertuples()]
        intervals.append(dict(policy=policy,**paired_interval(delta)))
        cases.extend(dict(policy=policy,question=r.financebench_id,baseline=int(r.answer_correct_baseline),method_match=int(r.answer_correct_method)) for r in pair.itertuples())
    pd.DataFrame(intervals).to_csv(folder/'paired_intervals.csv',index=False)
    pd.DataFrame(cases).to_csv(folder/'paired_cases.csv',index=False)
    print(summary.to_string(index=False),flush=True)
    (folder/'complete.json').write_text(json.dumps(dict(development_outputs=60,evaluation_outputs=0,methods=METHODS,complete=True,
        decision='No automatic promotion or evaluation; inspect numerical contradictions and paired regressions first.'),indent=2))


def main():
    import study
    from pilot_shared import run_matrix
    from evaluation import score_answer
    from transformers import AutoTokenizer
    questions=validate_development(study.DATA/'dev.jsonl')
    folder=study.OUT/'final_row_tuning_v1'
    if folder.exists():raise ValueError('Existing run; preserve all records.')
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B',revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',cache_dir=str(study.ROOT/'hf_cache/hub'),local_files_only=True)
    base=study.Retriever();original=study.run_question
    def run(question,doc_name,policy,retriever,token):
        adapter=VerifiedRowRetriever(retriever,question,study) if study.CONFIG['final_row_method']=='row_verified' else retriever
        row=original(question,doc_name,policy,adapter,token)
        row['row_retrieval_diagnostics']=getattr(adapter,'events',[])
        return row
    study.run_question=run
    try:run_matrix(study,questions,[(m,p) for m in METHODS for p in ('B0','B1','B2')],folder,lambda method:base,tokenizer,score_answer,scope=method_scope)
    finally:study.run_question=original
    summarise(study,folder,questions)

if __name__=='__main__':main()
