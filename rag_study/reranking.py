"""Pinned cross-encoder ranking in a short-lived CPU process on the 8 GB Mac."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import requests


def rerank_candidates(query, candidates, settings, generator=None):
    if not candidates:
        return []
    started = time.perf_counter()
    if generator is not None:
        response = requests.post('http://127.0.0.1:11434/api/generate',
                                 json={'model':generator,'keep_alive':0},timeout=60)
        response.raise_for_status()
    payload = {'settings':settings,'pairs':[[query,c['text']] for c in candidates]}
    env = {**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false',
           'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    try:
        completed = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],
            input=json.dumps(payload),capture_output=True,text=True,check=True,timeout=300,env=env)
    except subprocess.CalledProcessError as exc:
        raise ValueError('Reranker worker failed: ' + (exc.stderr or str(exc))[-2000:]) from exc
    scores = json.loads(completed.stdout)
    if not isinstance(scores,list) or len(scores)!=len(candidates) or any(
        type(s) not in (int,float) or not math.isfinite(s) for s in scores):
        raise ValueError('Reranker must return one finite numerical score per candidate.')
    ranked = [{**c,'reranker_score':score} for c,score in zip(candidates,scores)]
    ranked.sort(key=lambda c:(-c['reranker_score'],c['id']))
    print(json.dumps({'event':'rerank_lifecycle','candidates':len(ranked),
          'generator_unloaded':generator is not None,'worker_exited':True,
          'seconds':time.perf_counter()-started}),file=sys.stderr,flush=True)
    return ranked


def worker():
    import torch
    from sentence_transformers import CrossEncoder
    torch.set_num_threads(2)
    payload = json.load(sys.stdin)
    settings = payload['settings']
    model = CrossEncoder(settings['model'],revision=settings['revision'],device='cpu',
                         max_length=settings['max_length'],local_files_only=True)
    scores = model.predict(payload['pairs'],batch_size=settings['batch_size'],
                           show_progress_bar=False,convert_to_numpy=True)
    print(json.dumps([float(s) for s in scores]))


if __name__=='__main__':
    if sys.argv[1:]!=['--worker']:
        raise SystemExit('This module is a retrieval worker, not an experiment runner.')
    worker()
