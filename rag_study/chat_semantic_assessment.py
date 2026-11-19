"""Prepare and aggregate recorded chat judgements; never generate a correctness grade."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'rag_study/outputs/analysis/chat_semantic_assessment_20261113'
DIMENSIONS = ('answer_correctness', 'citation_support', 'citation_completeness')


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload(row, question):
    context = {c['id']: c for c in row.get('final_context', [])}
    cited = row.get('citations', [])
    return {'question': question['question'], 'reference_answer': question['answer'],
            'reference_justification': question.get('justification'),
            'reference_evidence': question['evidence'],
            'candidate_answer': row.get('answer', ''),
            'candidate_invalid': bool(row.get('invalid')),
            'candidate_abstain': bool(row.get('abstain')),
            'candidate_citations': cited,
            'missing_citation_ids': sorted(set(cited) - set(context)),
            'candidate_cited_passages': [context[c] for c in cited if c in context],
            'candidate_calculations': row.get('calculations', [])}


def terminal_rating(case):
    p = case['payload']
    if not (p.get('candidate_invalid') or p.get('candidate_abstain')):
        return None
    state = 'invalid output' if p.get('candidate_invalid') else 'valid abstention'
    return {'blind_id': case['blind_id'], 'answer_correctness': 'fail',
            'citation_support': 'not_assessable', 'citation_completeness': 'not_assessable',
            'reason_correctness': f'No answer-correctness credit for {state} on this answerable benchmark.',
            'reason_support': 'No valid substantive answer to assess.',
            'reason_completeness': 'No valid substantive answer to assess.',
            'assessed_by': 'predeclared terminal-state rule'}


def validate(cases, ratings):
    expected = {c['blind_id'] for c in cases}
    ids = [r['blind_id'] for r in ratings]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('Missing, duplicate or unknown rating IDs.')
    for r in ratings:
        for d in DIMENSIONS:
            if r[d] not in ('pass', 'fail', 'not_assessable') or not r['reason_' + d.removeprefix('answer_').replace('citation_', '')].strip():
                raise ValueError('Each dimension requires a valid rating and nonempty reason.')
        case = next(c for c in cases if c['blind_id'] == r['blind_id'])
        terminal = terminal_rating(case) if 'payload' in case else None
        if terminal and any(r[d] != terminal[d] for d in DIMENSIONS):
            raise ValueError('Terminal states must retain their predeclared ratings.')


def interval_counts(values, not_applicable=0):
    counts = Counter(values)
    n = len(values)
    unresolved = counts['not_assessable'] - not_applicable
    if not 0 <= not_applicable <= counts['not_assessable']:
        raise ValueError('Nonapplicable count must be part of the unassessable ratings.')
    result = {'n': n, 'pass': counts['pass'], 'fail': counts['fail'], 'unresolved': unresolved,
            'lower_bound': counts['pass'] / n,
            'upper_bound': (counts['pass'] + unresolved) / n,
            'accuracy': None if unresolved else counts['pass'] / n}
    if not_applicable:
        result['not_applicable'] = not_applicable
    return result


def validate_mapping(cases, mapping):
    lookup = {r['blind_id']: r for r in mapping}
    if len(lookup) != len(mapping) or set(lookup) != {c['blind_id'] for c in cases}:
        raise ValueError('Missing, duplicate or unknown mapping IDs.')
    for case in cases:
        sha = hashlib.sha256(json.dumps(case['payload'], sort_keys=True).encode()).hexdigest()
        if sha != lookup[case['blind_id']]['payload_sha256']:
            raise ValueError('Mapping payload identity mismatch.')


def verify_final_assessment(directory):
    path = Path(directory) / 'final_assessment_manifest.json'
    if path.exists():
        for name, sha in json.loads(path.read_text())['files'].items():
            if digest(Path(directory) / name) != sha:
                raise ValueError('Final assessment record changed: ' + name)


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / 'manifest.json').exists():
        raise ValueError('Existing assessment bundle; do not silently reprepare.')
    sources = {
        'development': ROOT / 'output/kaggle/final_row_tuning_v1/results/extracted/rag_study/outputs/final_row_tuning_v1/predictions.jsonl',
        'evaluation': ROOT / 'output/kaggle/row_eval_v1/results/extracted/rag_study/outputs/row_eval_v1/evaluation/test_predictions.jsonl'}
    manifest = {'judge_model': 'gpt-6.1-sol', 'interface': 'Codex chat session',
                'assessment_date': '2026-11-13', 'temperature': 'not exposed',
                'model_snapshot': 'not exposed', 'human_agreement': 'not measured',
                'prior_project_context': True, 'sources': {}, 'counts': {}}
    mapping = []
    for split, source in sources.items():
        qp = ROOT / ('rag_study/data/dev.jsonl' if split == 'development' else 'rag_study/data/test.jsonl')
        questions = {q['financebench_id']: q for q in read(qp)}
        rows = read(source)
        if split == 'development':
            rows = [r for r in rows if r.get('configuration') == 'row_verified']
        expected = {(q, v) for q in questions for v in ('B0', 'B1', 'B2')}
        if {(r['financebench_id'], r['variant']) for r in rows} != expected or len(rows) != len(expected):
            raise ValueError('Incomplete or duplicate source matrix.')
        random.Random(42).shuffle(rows)
        cases = []
        for i, r in enumerate(rows, 1):
            blind_id = ('D' if split == 'development' else 'E') + f'{i:03d}'
            p = payload(r, questions[r['financebench_id']])
            cases.append({'blind_id': blind_id, 'payload': p})
            mapping.append({'blind_id': blind_id, 'split': split, 'question_id': r['financebench_id'],
                            'doc_name': r['doc_name'], 'variant': r['variant'],
                            'payload_sha256': hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()})
        case_file = DEST / f'{split}_cases.jsonl'
        write(case_file, cases)
        write(DEST / f'{split}_terminal_ratings.jsonl', [t for c in cases if (t := terminal_rating(c))])
        manifest['counts'][split] = len(cases)
        manifest['sources'][str(source.relative_to(ROOT))] = digest(source)
        manifest['sources'][str(qp.relative_to(ROOT))] = digest(qp)
        manifest['sources'][str(case_file.relative_to(ROOT))] = digest(case_file)
    write(DEST / 'protected_mapping.jsonl', mapping)
    manifest['sources'][str((DEST / 'protected_mapping.jsonl').relative_to(ROOT))] = digest(DEST / 'protected_mapping.jsonl')
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['counts']))


def summarise():
    verify_final_assessment(DEST)
    manifest = json.loads((DEST / 'manifest.json').read_text())
    freeze = json.loads((DEST / 'rubric_freeze.json').read_text())
    if digest(DEST / 'Rubric.md') != freeze['rubric_sha256'] or digest(DEST / 'development_ratings.jsonl') != freeze['development_ratings_sha256']:
        raise ValueError('Frozen rubric or development judgements changed.')
    for path, sha in manifest['sources'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('Source identity changed: ' + path)
    map_rows = read(DEST / 'protected_mapping.jsonl')
    validate_mapping(read(DEST / 'development_cases.jsonl') + read(DEST / 'evaluation_cases.jsonl'), map_rows)
    mapping = {r['blind_id']: r for r in map_rows}
    summary = []; combined = []
    for split in ('development', 'evaluation'):
        cases = read(DEST / f'{split}_cases.jsonl')
        ratings = read(DEST / f'{split}_ratings.jsonl')
        validate(cases, ratings)
        for r in ratings:
            combined.append({**mapping[r['blind_id']], **r,
                             'supported_correctness': int(all(r[d] == 'pass' for d in DIMENSIONS))})
        for v in ('B0', 'B1', 'B2'):
            selected = [r for r in ratings if mapping[r['blind_id']]['variant'] == v]
            for d in DIMENSIONS:
                na = sum(r['assessed_by'] == 'predeclared terminal-state rule' for r in selected) if d != 'answer_correctness' else 0
                summary.append({'split': split, 'variant': v, 'dimension': d,
                                **interval_counts([r[d] for r in selected], na)})
    write(DEST / 'assessment_records.jsonl', combined)
    with (DEST / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) + ['not_applicable']); writer.writeheader(); writer.writerows(summary)
    (DEST / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps([r for r in summary if r['dimension'] == 'answer_correctness'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'summarise'])
    action = parser.parse_args().action
    prepare() if action == 'prepare' else summarise()
