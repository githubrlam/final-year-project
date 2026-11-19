"""Controlled B0/B1/B2 runs. Automatically evaluate development before freezing or testing."""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import phase4
import study

POLICIES = ('B0', 'B1', 'B2')


def now():
    return datetime.now(timezone.utc).isoformat()


def directory():
    return study.OUT / 'controlled_runs/phase5'


def schedule(questions):
    for index, question in enumerate(questions):
        offset = index % len(POLICIES)
        for variant in POLICIES[offset:] + POLICIES[:offset]:
            yield question, variant


def validate_pairs(rows, questions, complete=False):
    expected = {(q['financebench_id'], v) for q in questions for v in POLICIES}
    keys = [(r['financebench_id'], r['variant']) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError('Unexpected or duplicate question-policy records.')
    if complete and set(keys) != expected:
        raise ValueError(f'Incomplete run: {len(keys)}/{len(expected)} question-policy pairs.')


def validate_records(rows, signature):
    required = {'answer', 'citations', 'abstain', 'invalid', 'seconds', 'model_calls',
                'retrieval_rounds', 'input_tokens', 'output_tokens', 'final_context',
                'calculations', 'trace', 'evidence_tokens', 'session_index', 'created_utc'}
    for row in rows:
        if not required <= row.keys() or any(row.get(key) != signature['source_sha256'].get(name)
            for key, name in [('code_sha256', 'study.py'), ('config_sha256', 'config.json')]):
            raise ValueError('Prediction record is incomplete or came from different source/settings.')


def run_signature():
    signature, snapshot = phase4.run_signature()
    for name in ['phase5.py', 'evaluation.py', 'data/test.jsonl', 'data/source/provenance.json']:
        signature['source_sha256'][name] = study.digest(study.ROOT / name)
    return signature, snapshot


def readiness():
    """Read local review evidence only; never infer that the student reviewed it."""
    import pandas as pd
    dev = study.read_jsonl(study.DATA / 'dev.jsonl')
    predictions = study.OUT / 'dev_predictions.jsonl'
    rows = study.read_jsonl(predictions) if predictions.exists() else []
    checks = {}
    try:
        validate_pairs(rows, dev, complete=True)
        checks['thirty_development_outputs'] = True
    except ValueError:
        checks['thirty_development_outputs'] = False
    manifest_path = directory() / 'dev_manifest.json'
    recorded = json.loads(manifest_path.read_text()).get('signature', {}) if manifest_path.exists() else {}
    sources = recorded.get('source_sha256', {})
    checks['development_sources_current'] = bool(sources) and all(
        (study.ROOT / name).is_file() and study.digest(study.ROOT / name) == value
        for name, value in sources.items())
    result_path = directory() / 'dev_evaluation_manifest.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    checks['automatic_evaluation_current'] = bool(result) and all(
        (study.ROOT / name).is_file() and study.digest(study.ROOT / name) == value
        for name, value in result.get('inputs_sha256', {}).items()) and bool(result.get('inputs_sha256'))
    audit = study.OUT / 'data_audit/phase2_line_aware_v2/development_document_audit.csv'
    checks['document_audit_reviewed'] = bool(audit.exists() and len(pd.read_csv(audit)) == 13
        and pd.read_csv(audit).review_status.eq('reviewed').all())
    return {'development_outputs': len(rows), 'checks': checks, 'ready_to_freeze': all(checks.values()),
            'pending': [name for name, passed in checks.items() if not passed],
            'frozen': (study.OUT / 'frozen.json').exists()}


def freeze():
    state = readiness()
    if not state['ready_to_freeze']:
        raise ValueError('Development checks are incomplete: ' + ', '.join(state['pending']))
    if (study.OUT / 'frozen.json').exists():
        raise ValueError('Already frozen; do not overwrite the experiment protocol.')
    signature, _ = run_signature()
    manifest = json.loads((directory() / 'dev_manifest.json').read_text())
    if signature != manifest['signature']:
        raise ValueError('Development source/settings/runtime changed; rerun and evaluate before freezing.')
    # The evaluator and matching rules are frozen with the generator protocol.
    study.metrics('dev')
    record = study._freeze_record()
    extra = [study.ROOT / 'phase4.py', study.ROOT / 'phase5.py', study.ROOT / 'evaluation.py',
             study.DATA / 'source/provenance.json', study.OUT / 'dev_predictions.jsonl',
             study.OUT / 'dev_automatic_scores.csv', directory() / 'dev_manifest.json',
             directory() / 'dev_evaluation_manifest.json',
             study.OUT / 'data_audit/phase2_line_aware_v2/development_document_audit.csv']
    record['files'].update({str(p.relative_to(study.ROOT)): study.digest(p) for p in extra})
    record['controlled_run_signature'] = signature
    (study.OUT / 'frozen.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def run_batch(split='dev'):
    if split not in ('dev', 'test'):
        raise ValueError('Split must be dev or test.')
    frozen_path = study.OUT / 'frozen.json'
    if split == 'test' and not frozen_path.exists():
        raise ValueError('Protocol is not frozen. Complete development review first.')
    if split == 'dev' and frozen_path.exists():
        raise ValueError('Development is frozen; do not tune or rerun it within this protocol.')
    if split == 'test':
        study.check_frozen()
    signature, snapshot = run_signature()
    if split == 'test' and json.loads(frozen_path.read_text()).get('controlled_run_signature') != signature:
        raise ValueError('Frozen source/settings/runtime changed.')
    questions = study.read_jsonl(study.DATA / f'{split}.jsonl')
    if len(questions) != (10 if split == 'dev' else 52):
        raise ValueError('Unexpected split size.')
    directory().mkdir(parents=True, exist_ok=True)
    manifest_path = directory() / f'{split}_manifest.json'
    path = study.OUT / f'{split}_predictions.jsonl'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None and manifest['signature'] != signature:
        raise ValueError('Controlled-run source/settings/runtime changed; preserve and version the old run first.')
    if path.exists() and manifest is None:
        raise ValueError('Predictions have no run manifest; refusing to mix revisions.')
    rows = study.read_jsonl(path) if path.exists() else []
    validate_pairs(rows, questions)
    validate_records(rows, signature)
    done = {(r['financebench_id'], r['variant']) for r in rows}
    if len(done) == len(questions) * len(POLICIES):
        print(f'Reusing {len(rows)} matching {split} outputs; no generator calls.', flush=True)
        return rows

    import requests
    from transformers import AutoTokenizer
    manifest = manifest or {'signature': signature, 'sessions': [], 'split': split,
                            'order': 'Rotate B0/B1/B2 by question index; sequential execution.'}
    session = {'started_utc': now()}
    manifest['sessions'].append(session)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    retriever = study.Retriever()
    index = json.loads((study.DATA / 'index_signature.json').read_text())
    if retriever.collection.count() != index['count'] or index['embedding'] != study.CONFIG['embedding']:
        raise ValueError('Index signature/count does not match the corpus.')
    try:
        started = time.perf_counter()
        warm = requests.post(phase4.API + '/chat', json={
            'model': study.CONFIG['generator'], 'messages': [{'role': 'user', 'content': 'Reply Ready.'}],
            'stream': False, 'think': False, 'keep_alive': '10m',
            'options': {'temperature': 0, 'seed': study.CONFIG['seed'],
                        'num_ctx': study.CONFIG['context_tokens'], 'num_predict': 8}}, timeout=600)
        warm.raise_for_status()
        if warm.json().get('done') is not True:
            raise ValueError('Warm-up did not complete.')
        retriever.model.encode(['Represent this sentence for searching relevant passages: financial revenue'],
                               normalize_embeddings=True)
        session.update({'warmup_seconds': time.perf_counter() - started,
                        'warmup_included_in_question_latency': False})
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        for question, variant in schedule(questions):
            key = (question['financebench_id'], variant)
            if key in done:
                continue
            print(f'{split} {len(rows)+1}/{len(questions)*3}: {key}', flush=True)
            # No answer, justification, or annotated evidence is passed to generation.
            row = study.run_question(question['question'], question['doc_name'], variant, retriever, tokenizer)
            _, context, count = study.pack_context(row['final_context'], tokenizer)
            if len(context) != len(row['final_context']) or count > study.CONFIG['evidence_tokens']:
                raise ValueError('Recorded evidence violates the common context budget.')
            if row['model_calls'] > study.CONFIG['max_model_calls'] or row['retrieval_rounds'] > (1 if variant == 'B0' else 2):
                raise ValueError('Recorded controller exceeded its execution limits.')
            row.update({'financebench_id': key[0], 'doc_name': question['doc_name'], 'question': question['question'],
                        'evidence_tokens': count, 'created_utc': now(), 'session_index': len(manifest['sessions']) - 1,
                        'config_sha256': signature['source_sha256']['config.json'],
                        'code_sha256': signature['source_sha256']['study.py']})
            rows.append(row)
            # Replace atomically after each completed question-policy pair; retain failures.
            temporary = path.with_suffix('.tmp')
            study.write_jsonl(temporary, rows)
            temporary.replace(path)
            done.add(key)
            print(f"Saved: {row['seconds']:.2f}s, calls={row['model_calls']}, rounds={row['retrieval_rounds']}, invalid={row['invalid']}", flush=True)
        validate_pairs(rows, questions, complete=True)
        session['complete'] = True
    finally:
        session['finished_utc'] = now()
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        requests.post(phase4.API + '/generate', json={'model': study.CONFIG['generator'], 'keep_alive': 0}, timeout=30).raise_for_status()
    return rows


def prepare_evaluation(split='dev'):
    """Score against benchmark references after generation; no manual rating queue."""
    summary = study.metrics(split)
    names = ['study.py', 'evaluation.py', f'data/{split}.jsonl',
             f'outputs/{split}_predictions.jsonl', f'outputs/{split}_automatic_scores.csv']
    record = {'evaluation_mode': 'automatic_reference_match_v1', 'manual_assessment_required': False,
              'inputs_sha256': {name: study.digest(study.ROOT / name) for name in names},
              'primary_outcome': 'automated_reference_answer_accuracy',
              'citation_support_evaluated': False, 'created_utc': now()}
    directory().mkdir(parents=True, exist_ok=True)
    (directory() / f'{split}_evaluation_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return study.OUT / f'{split}_automatic_scores.csv'


if __name__ == '__main__':
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['dev', 'score', 'review', 'status', 'freeze', 'test'])
    args = parser.parse_args()
    if args.stage in ('dev', 'test'):
        run_batch(args.stage)
        print(prepare_evaluation(args.stage))
    elif args.stage in ('score', 'review'):
        print(prepare_evaluation())
    elif args.stage == 'freeze':
        freeze()
        print('Frozen. The controlled test command is now available.')
    else:
        print(json.dumps(readiness(), indent=2))
