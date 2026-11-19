"""Guided B0 development run; reuse the shared experiment code unchanged."""
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import study

OUT = study.OUT / 'generation_audit/phase4'
API = 'http://127.0.0.1:11434/api'


def run_signature():
    """Resolve cached files and record the selected runtime, not unrelated tags."""
    import requests
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    for cache in [study.ROOT / 'hf_cache/hub', Path.home() / '.cache/huggingface/hub']:
        try:
            snapshot = Path(snapshot_download(study.CONFIG['generator_tokenizer'],
                                              cache_dir=cache, local_files_only=True))
            break
        except LocalEntryNotFoundError:
            continue
    else:
        raise FileNotFoundError('Restore the cached Phase 0 Qwen tokenizer first.')
    version = requests.get(API + '/version', timeout=10)
    version.raise_for_status()
    tags = requests.get(API + '/tags', timeout=10)
    tags.raise_for_status()
    model = next(m for m in tags.json()['models'] if m['name'] == study.CONFIG['generator'])
    assert model['details']['quantization_level'] == 'Q4_K_M'
    phase0 = json.loads((study.OUT / 'phase0_qwen35_environment.json').read_text())
    assert model['digest'] == phase0['ollama']['model']['digest'], 'Model changed since Phase 0.'
    sources = ['study.py', 'contracts.py', 'reranking.py', 'chunking.py', 'table_quality.py', 'phase4.py', 'config.json', 'requirements.lock.txt', 'data/dev.jsonl',
               'data/chunks.jsonl', 'data/manifest.json', 'data/index_signature.json']
    return {
        'config': study.CONFIG,
        'source_sha256': {name: study.digest(study.ROOT / name) for name in sources},
        'runtime': {'platform': platform.platform(), 'python': platform.python_version(),
                    'ollama_version': version.json()['version'], 'model': model},
        'tokenizer_revision': snapshot.name,
        'tokenizer_sha256': {str(p.relative_to(snapshot)): study.digest(p)
                             for p in snapshot.rglob('*') if p.is_file()},
    }, snapshot


def run_baseline(output_dir=OUT):
    """Run/reuse ten B0 outputs, refusing to mix source or runtime revisions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature, snapshot = run_signature()
    manifest_path = output_dir / 'run_manifest.json'
    path = output_dir / 'b0_predictions.jsonl'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None and manifest['signature'] != signature:
        raise ValueError('Phase 4 source/settings/runtime changed; archive this run before rerunning.')
    if path.exists() and manifest is None:
        raise ValueError('Predictions have no manifest; do not silently reuse them.')
    questions = study.read_jsonl(study.DATA / 'dev.jsonl')
    assert len(questions) == 10
    rows = study.read_jsonl(path) if path.exists() else []
    keys = [(row['financebench_id'], row['variant']) for row in rows]
    expected = {(q['financebench_id'], 'B0') for q in questions}
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise ValueError('Unexpected or duplicate question/policy records in this run.')
    if set(keys) == expected:
        print('Reusing all ten recorded B0 outputs; no generator calls.', flush=True)
        return rows

    import psutil
    import requests
    from transformers import AutoTokenizer

    manifest = manifest or {'signature': signature, 'sessions': []}
    session = {'started_utc': datetime.now(timezone.utc).isoformat(),
               'available_ram_before_bytes': psutil.virtual_memory().available}
    manifest['sessions'].append(session)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    retriever = study.Retriever()
    index = json.loads((study.DATA / 'index_signature.json').read_text())
    assert index['embedding'] == study.CONFIG['embedding']
    assert retriever.collection.count() == index['count'] == len(retriever.chunks)
    try:
        start = time.perf_counter()
        warm = requests.post(API + '/chat', json={
            'model': study.CONFIG['generator'], 'messages': [{'role': 'user', 'content': 'Reply Ready.'}],
            'stream': False, 'think': False, 'keep_alive': '10m',
            'options': {'temperature': 0, 'seed': study.CONFIG['seed'],
                        'num_ctx': study.CONFIG['context_tokens'], 'num_predict': 8}}, timeout=600)
        warm.raise_for_status()
        assert warm.json().get('done') is True
        retriever.model.encode(['Represent this sentence for searching relevant passages: financial revenue'],
                               normalize_embeddings=True)
        session['warmup_seconds'] = time.perf_counter() - start
        session['warmup_included_in_question_latency'] = False
        session['python_rss_bytes_after_warmup'] = psutil.Process().memory_info().rss
        loaded = requests.get(API + '/ps', timeout=30)
        loaded.raise_for_status()
        session['ollama_loaded_models'] = loaded.json()
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        done = set(keys)
        for question in questions:
            qid = question['financebench_id']
            if (qid, 'B0') in done:
                continue
            print(f'Generating B0: {qid}', flush=True)
            row = study.run_question(question['question'], question['doc_name'], 'B0', retriever, tokenizer)
            evidence, context, count = study.pack_context(row['final_context'], tokenizer)
            assert len(context) == len(row['final_context']) <= study.CONFIG['top_k']
            assert count == len(tokenizer.encode(evidence, add_special_tokens=False)) <= study.CONFIG['evidence_tokens']
            assert row['retrieval_rounds'] <= 1 and row['model_calls'] <= study.CONFIG['max_model_calls']
            row.update({'financebench_id': qid, 'doc_name': question['doc_name'], 'question': question['question'],
                        'evidence_tokens': count, 'session_index': len(manifest['sessions']) - 1,
                        'created_utc': datetime.now(timezone.utc).isoformat()})
            rows.append(row)
            study.write_jsonl(path, rows)
            print(f"Saved {len(rows)}/10: {row['seconds']:.2f}s; calls={row['model_calls']}; "
                  f"abstain={row['abstain']}; invalid={row['invalid']}", flush=True)
        session['complete'] = True
    finally:
        session['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        # Release the local generation model on the 8 GB Mac after this batch.
        release = requests.post(API + '/generate', json={'model': study.CONFIG['generator'], 'keep_alive': 0}, timeout=30)
        release.raise_for_status()
    return rows


if __name__ == '__main__':
    # The only network connection required is to the local Ollama server.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    run_baseline()
