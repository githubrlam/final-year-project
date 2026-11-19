"""Frozen selected-method evaluation; never tunes retrieval or changes grading."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from final_row_tuning import VerifiedRowRetriever,method_scope

MODEL_DIGEST='d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04'
TOKENIZER_REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_protocol(path,config,prompts,source_files,data_files,question_ids,documents,development_sha):
    path=Path(path)
    if path.exists():raise ValueError('Frozen protocol already exists; do not overwrite.')
    record=dict(method='row_verified',config=config,prompts=prompts,
        source_sha256={k:sha(p) for k,p in source_files.items()},data_sha256={k:sha(p) for k,p in data_files.items()},
        evaluation_question_ids=sorted(question_ids),evaluation_documents=sorted(documents),
        development_predictions_sha256=development_sha,model_digest=MODEL_DIGEST,tokenizer_revision=TOKENIZER_REVISION,
        expected_evaluation_outputs=156,scorer='reference_match_v1',
        evaluation_status='Exploratory; this subset has previously been inspected. No further tuning.',
        failure_policy='Retain all failures and raw outputs; no silent regeneration; invalids and abstentions remain in denominator.')
    with path.open('x') as f:json.dump(record,f,indent=2)
    return record


def verify_protocol(protocol,source_files,data_files):
    for key,files in [('source_sha256',source_files),('data_sha256',data_files)]:
        if set(files)!=set(protocol[key]) or any(sha(p)!=protocol[key][name] for name,p in files.items()):
            raise ValueError('Frozen source or data changed: '+key)


def validate_questions(questions,protocol):
    ids=[q['financebench_id'] for q in questions]
    if len(ids)!=52 or len(set(ids))!=52 or sorted(ids)!=protocol['evaluation_question_ids']:
        raise ValueError('Use exactly the frozen 52 evaluation questions.')
    if len({q['doc_name'] for q in questions})!=18 or sorted({q['doc_name'] for q in questions})!=protocol['evaluation_documents']:
        raise ValueError('Evaluation documents differ from the frozen 18-report set.')


def generate(study,questions,folder,base,tokenizer,run_matrix,score_answer,protocol):
    original=study.run_question
    def run(question,doc_name,policy,retriever,token):
        adapter=VerifiedRowRetriever(retriever,question,study)
        row=original(question,doc_name,policy,adapter,token)
        row['row_retrieval_diagnostics']=adapter.events
        return row
    @contextmanager
    def checked_scope(s,method):
        with method_scope(s,method):
            if s.CONFIG!=protocol['config'] or s.PROMPTS[s.CONFIG['prompt_revision']]!=protocol['prompts']:
                raise ValueError('Runtime settings/prompts differ from the frozen selected method.')
            yield
    study.run_question=run
    try:run_matrix(study,questions,[('row_verified',p) for p in ('B0','B1','B2')],folder,lambda method:base,tokenizer,score_answer,scope=checked_scope)
    finally:study.run_question=original


def main():
    import study,evaluation
    from pilot_shared import run_matrix
    from numeric_verification import check_numeric
    from transformers import AutoTokenizer
    import pandas as pd
    protocol_path=study.ROOT/'frozen_row_protocol.json';protocol=json.loads(protocol_path.read_text())
    sources={name:study.ROOT/name for name in protocol['source_sha256']}
    data={name:study.ROOT/name for name in protocol['data_sha256']}
    verify_protocol(protocol,sources,data)
    questions=study.read_jsonl(study.DATA/'test.jsonl');validate_questions(questions,protocol)
    study.CONFIG.clear();study.CONFIG.update(protocol['config'])
    out=study.OUT/'row_eval_v1'
    if out.exists():raise ValueError('Existing evaluation; preserve records, do not rerun silently.')
    out.mkdir()
    (out/'frozen_protocol.json').write_bytes(protocol_path.read_bytes())
    runtime=study.runtime_record()
    model=next(m for m in runtime['model_manifest']['models'] if m['name']==study.CONFIG['generator'])
    if model['digest']!=protocol['model_digest']:raise ValueError('Frozen model digest changed.')
    (out/'runtime_before_generation.json').write_text(json.dumps(runtime,indent=2))
    tokenizer=AutoTokenizer.from_pretrained(study.CONFIG['generator_tokenizer'],revision=protocol['tokenizer_revision'],cache_dir=str(study.ROOT/'hf_cache/hub'),local_files_only=True)
    base=study.Retriever()
    generate(study,questions,out/'evaluation',base,tokenizer,run_matrix,evaluation.score_answer,protocol)
    rows=study.read_jsonl(out/'evaluation/predictions.jsonl')
    keys=[(r['financebench_id'],r['variant'],r['configuration']) for r in rows]
    expected={(q['financebench_id'],p,'row_verified') for q in questions for p in ('B0','B1','B2')}
    if len(keys)!=156 or len(set(keys))!=156 or set(keys)!=expected:raise ValueError('Incomplete 156-output matrix.')
    saved=study.OUT
    try:
        study.OUT=out/'evaluation'
        (study.OUT/'test_predictions.jsonl').write_bytes((study.OUT/'predictions.jsonl').read_bytes())
        print(evaluation.metrics(study,'test').to_string(index=False),flush=True)
    finally:study.OUT=saved
    gold={q['financebench_id']:q for q in questions};numeric=[]
    for r in rows:
        q=gold[r['financebench_id']]
        check={'status':'not_checked','reason':'Abstention or invalid output.'} if r['invalid'] or r['abstain'] else check_numeric(r['answer'],q['answer'],q['question'])
        numeric.append({k:r[k] for k in ('variant','financebench_id')}|check)
    pd.DataFrame(numeric).to_csv(out/'evaluation/numeric_consistency.csv',index=False)
    verify_protocol(protocol,sources,data)
    if sha(protocol_path)!=sha(out/'frozen_protocol.json'):raise ValueError('Freeze record changed.')
    (out/'complete.json').write_text(json.dumps(dict(evaluation_outputs=156,development_outputs=0,method='row_verified',
        frozen_protocol_sha256=sha(protocol_path),complete=True,evaluation_status=protocol['evaluation_status']),indent=2))
    print('Frozen evaluation complete: 156 outputs; no further tuning.',flush=True)

if __name__=='__main__':main()
