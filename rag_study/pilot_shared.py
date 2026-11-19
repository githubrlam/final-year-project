"""One shared development screen, predeclared selection, then frozen evaluation."""
from contextlib import contextmanager
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

METHODS = ('gen_v5', 'finance_v5', 'input_check_v5')
COMMON = dict(top_k=4, evidence_tokens=2000, context_tokens=6144, output_tokens=1024,
              thinking=False, max_model_calls=4, max_retrieval_rounds=2, seed=42,
              prompt_revision='final')
CHECK = '''
Before declaring evidence sufficient, check every requested input/entity:
- A list question requires evidence for the entire requested list, not one example.
- For each financial input, identify its exact row label, fiscal year and unit.
- Net profit margin uses consolidated net income/earnings, never operating income.
- Average annual margins require revenue and net earnings for every requested year.
- Fixed asset turnover requires annual revenue and opening/closing net PP&E.
- Inventory turnover requires annual cost of sales and opening/closing inventory.
If a required input is absent, name that row and fiscal year in missing_evidence
and target it in the retry query. Do not request evidence unrelated to missing inputs.
Never invent numbers or treat a page citation alone as proof that a row is present.
'''


def targeted_query(query):
    from development_repair import financial_query
    additions=[]
    if re.search(r'profit margin|net margin',query,re.I):
        additions.append('consolidated statements of earnings net income net earnings total revenue net sales')
    if re.search(r'fixed.asset turnover',query,re.I):
        additions.append('consolidated statements revenue balance sheets property plant equipment net opening closing')
    if re.search(r'inventory turnover',query,re.I):
        additions.append('consolidated statements cost of sales cost of revenues balance sheets inventories opening closing')
    if re.search(r'acquir|acquisition',query,re.I):
        additions.append('business combinations acquisitions acquired companies transactions')
    return financial_query(query)+('\nRequired source rows: '+'; '.join(additions) if additions else '')


class SearchAdapter:
    def __init__(self,base,method):self.base,self.method=base,method
    def retrieve(self,query,doc_name=None,previous=None):
        from development_repair import financial_query
        expand=targeted_query if self.method=='input_check_v5' else financial_query
        return self.base.retrieve(expand(query),doc_name=doc_name,previous=previous)


@contextmanager
def method_scope(study,method):
    from development_repair import FINANCE
    if method not in METHODS:raise ValueError('Unknown final method.')
    before=dict(study.CONFIG)
    prompts=dict(study.PROMPTS['v5'])
    if method!='gen_v5':
        prompts['SYSTEM']+='\n'+FINANCE
        prompts['ASSESS']+='\nUse financial synonyms; a computed result need not be quoted verbatim.'
    if method=='input_check_v5':
        prompts['ASSESS']+='\n'+CHECK
        prompts['ANSWER']+='\n'+CHECK
    old=study.PROMPTS.get('final')
    study.PROMPTS['final']=prompts
    study.CONFIG.update(COMMON,final_method=method)
    try:yield
    finally:
        study.CONFIG.clear();study.CONFIG.update(before)
        if old is None:study.PROMPTS.pop('final',None)
        else:study.PROMPTS['final']=old


def save_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run_matrix(study,questions,matrix,folder,retriever_for,tokenizer,score_answer,scope=method_scope):
    if folder.exists():raise ValueError('Existing run: preserve records; no silent rerun.')
    ids={q['financebench_id'] for q in questions}
    if len(ids)!=len(questions):raise ValueError('Duplicate question IDs.')
    folder.mkdir(parents=True)
    rows=[];scores=[]
    for method,policy in matrix:
        with scope(study,method):
            config_bytes=json.dumps(study.CONFIG,sort_keys=True).encode()
            config_path=folder/(method+'_'+policy+'_config.json');config_path.write_bytes(config_bytes)
            retriever=retriever_for(method)
            for q in questions:
                row=dict(study.run_question(q['question'],q['doc_name'],policy,retriever,tokenizer))
                if row['model_calls']>4 or row['retrieval_rounds']>2:raise ValueError('Execution limit exceeded.')
                _,chosen,tokens=study.pack_context(row['final_context'],tokenizer)
                if len(chosen)!=len(row['final_context']) or tokens>2000 or len(chosen)>4:
                    raise ValueError('Evidence limit exceeded.')
                row.update(configuration=method,variant=policy,financebench_id=q['financebench_id'],
                           question=q['question'],doc_name=q['doc_name'],evidence_tokens=tokens,
                           config_sha256=hashlib.sha256(config_bytes).hexdigest())
                rows.append(row)
                tmp=folder/'predictions.tmp';tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                tmp.replace(folder/'predictions.jsonl')
                # Reference labels are used only after the generated record has been saved.
                gold={(e.get('doc_name',e.get('evidence_doc_name')),int(e['evidence_page_num'])) for e in q['evidence']}
                actual={(c['doc_name'],int(c['page0'])) for c in row['final_context']}
                scores.append({k:row[k] for k in ['configuration','variant','financebench_id','invalid','abstain','seconds','input_tokens','output_tokens']}
                              |score_answer(row,q)|{'evidence_page_recall':len(gold&actual)/len(gold)})
                save_csv(folder/'automatic_scores.csv',scores)
                print(f'Saved: {len(rows)}/{len(questions)*len(matrix)}; {method}; {policy}; {q["financebench_id"]}; invalid={row["invalid"]}',flush=True)
    (folder/'manifest.json').write_text(json.dumps({'complete':True,'saved_outputs':len(rows),
        'question_ids':sorted(ids),'matrix':matrix,'evaluator':'reference_match_v1'},indent=2))
    return scores


def select_method(scores):
    counts=Counter(r['configuration'] for r in scores)
    if len(scores)!=30 or len(counts)!=3 or set(counts.values())!={10}:
        raise ValueError('Selection requires ten development outputs for each of three methods.')
    summary=[]
    for name in sorted(counts):
        part=[r for r in scores if r['configuration']==name]
        summary.append({'method':name,'correct':sum(int(r['answer_correct']) for r in part),
            'invalid':sum(bool(r['invalid']) for r in part),
            'tokens':sum(int(r['input_tokens'])+int(r['output_tokens']) for r in part)})
    ranked=sorted(summary,key=lambda r:(-r['correct'],r['invalid'],r['tokens'],r['method']))
    winner=ranked[0]
    return {'method':winner['method'],'gate_passed':winner['correct']>=5 and winner['invalid']==0,
            'ranking':ranked,'rule':'correct descending, invalid ascending, tokens ascending, name ascending',
            'evaluation_status':'previously inspected benchmark subset; exploratory evaluation'}

