"""Build and inspect a separate repaired corpus before explicitly activating it."""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import study

REVISION = study.DATA / 'revisions/line_aware_v2'
AUDIT = study.OUT / 'data_audit/chunking_repair_20261021'
HISTORY = study.OUT / 'history/before_chunking_repair_20261021'


def clipped_literals(page_text, chunks):
    """Count boundaries inside numeric literals, not financial correctness."""
    literals = list(re.finditer(r'[-+]?\(?\$?\d[\d,]*(?:\.\d+)?%?\)?', page_text))
    clipped = 0
    for chunk in chunks:
        if chunk['kind'] != 'text':
            continue
        start = page_text.find(chunk['text'])
        if start < 0:
            raise ValueError('Chunk is not an exact source-text slice.')
        end = start + len(chunk['text'])
        clipped += sum(m.start() < start < m.end() or m.start() < end < m.end() for m in literals)
    return clipped


def prepare():
    from collections import Counter
    import pandas as pd
    AUDIT.mkdir(parents=True, exist_ok=True)
    study.parse_corpus(REVISION)
    current = study.read_jsonl(REVISION / 'chunks.jsonl')
    old = study.read_jsonl(HISTORY / 'rag_study/data/chunks.jsonl')
    pages = [page for path in sorted((study.DATA / 'parsed').glob('*.jsonl')) for page in study.read_jsonl(path)]
    lookup = {(p['doc_name'], p['page0']): p for p in pages}
    dev = study.read_jsonl(study.DATA / 'dev.jsonl')
    keys = {(q['doc_name'], e['evidence_page_num']) for q in dev for e in q['evidence']}
    warnings = study.read_jsonl(REVISION / 'parsing_failures.jsonl')
    grouped, previous = {}, {}
    for chunks, destination in [(current, grouped), (old, previous)]:
        for chunk in chunks:
            destination.setdefault((chunk['doc_name'], chunk['page0']), []).append(chunk)
    nonempty = {key for key, page in lookup.items() if page['text'].strip()}
    text_pages = {key for key, chunks in grouped.items() if any(c['kind'] == 'text' for c in chunks)}
    comparisons = []
    for key in sorted(keys):
        p, before, after = lookup[key], previous.get(key, []), grouped.get(key, [])
        comparisons.append({'doc_name': key[0], 'pdf_page': key[1]+1,
            'old_text_chunks': sum(c['kind']=='text' for c in before),
            'new_text_chunks': sum(c['kind']=='text' for c in after),
            'old_table_chunks': sum(c['kind']=='table' for c in before),
            'new_table_chunks': sum(c['kind']=='table' for c in after),
            'old_clipped_numeric_boundaries': clipped_literals(p['text'], before),
            'new_clipped_numeric_boundaries': clipped_literals(p['text'], after)})
    frame = pd.DataFrame(comparisons)
    frame.to_csv(AUDIT / 'development_comparison.csv', index=False)
    events = Counter(w.get('event', 'extraction_error') for w in warnings)
    checks = {'all_nonempty_text_pages_retained': text_pages == nonempty,
        'all_chunks_within_350_tokens': all(c['embedding_tokens'] <= study.CONFIG['chunk_tokens'] for c in current),
        'development_numeric_boundaries_intact': int(frame.new_clipped_numeric_boundaries.sum()) == 0,
        'no_text_chunking_failures': not events['text_chunking_failure'],
        'no_over_budget_rejections': not events['chunk_rejected_over_budget']}
    summary = {'date': '21 October 2026', 'revision': 'line_aware_v2', 'checks': checks,
        'candidate_acceptable': all(checks.values()), 'source_pages': len(pages),
        'nonempty_text_pages': len(nonempty), 'retained_text_pages': len(text_pages),
        'old_chunks': len(old), 'new_chunks': len(current),
        'new_kinds': dict(Counter(c['kind'] for c in current)), 'events': dict(events),
        'development_pages': len(keys), 'development_detected_tables': sum(len(lookup[k]['tables']) for k in keys),
        'development_old_table_chunks': int(frame.old_table_chunks.sum()),
        'development_new_table_chunks': int(frame.new_table_chunks.sum()),
        'development_old_clipped_boundaries': int(frame.old_clipped_numeric_boundaries.sum()),
        'development_new_clipped_boundaries': int(frame.new_clipped_numeric_boundaries.sum()),
        'raw_pages_unchanged': True, 'correctness_scored': False, 'held_out_generation': False,
        'candidate_chunks_sha256': study.digest(REVISION/'chunks.jsonl'),
        'parser_sha256': study.parser_signature()}
    (AUDIT / 'repair_verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    for name in ['study.py','chunking.py','table_quality.py','repair_chunking.py']:
        destination = AUDIT / 'source_snapshot' / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(study.ROOT/name, destination)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary['candidate_acceptable']:
        raise ValueError('Candidate acceptance failed; inspect warnings before indexing or activation.')
    return summary


def index():
    record = json.loads((AUDIT/'repair_verification.json').read_text())
    if not record['candidate_acceptable'] or record['candidate_chunks_sha256'] != study.digest(REVISION/'chunks.jsonl'):
        raise ValueError('Rebuild/verify the candidate before indexing.')
    count = study.build_index(REVISION)
    record.update({'candidate_index_count': count, 'candidate_index_built': True})
    (AUDIT/'repair_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    return count


def activate():
    """Explicit local activation preserves the old index and generated results."""
    record = json.loads((AUDIT/'repair_verification.json').read_text())
    if not record.get('candidate_index_built') or (study.OUT/'frozen.json').exists():
        raise ValueError('Require a verified candidate index and an unfrozen study.')
    study.check_chunking_revision(REVISION)
    signature = json.loads((REVISION/'index_signature.json').read_text())
    if (signature['count'] != record['new_chunks'] or signature['parser_sha256'] != study.parser_signature()
            or signature['chunks_sha256'] != study.digest(REVISION/'chunks.jsonl')
            or signature['embedding'] != study.CONFIG['embedding']
            or record['candidate_chunks_sha256'] != study.digest(REVISION/'chunks.jsonl')):
        raise ValueError('Candidate index revision/count changed.')
    archived = HISTORY/'rag_study/data/index'
    if archived.exists():
        raise ValueError('Old index already archived; do not repeat activation.')
    (study.DATA/'index').rename(archived)
    # The active signature stays absent until every replacement has succeeded.
    (study.DATA/'index_signature.json').unlink(missing_ok=True)
    (REVISION/'index').rename(study.DATA/'index')
    for name in ['chunks.jsonl','parsing_failures.jsonl','chunking_manifest.json','index_signature.json']:
        shutil.copy2(REVISION/name,study.DATA/name)
    study.check_index_revision()
    record.update({'active_revision': True, 'old_index_preserved': str(archived.relative_to(study.ROOT)),
                   'new_development_generation_pending': True})
    (AUDIT/'repair_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Activated repaired chunks/index; earlier development outputs remain historical and must be rerun.',flush=True)


if __name__ == '__main__':
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','index','activate'])
    stage=parser.parse_args().stage
    {'prepare':prepare,'index':index,'activate':activate}[stage]()
