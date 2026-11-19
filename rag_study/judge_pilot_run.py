"""Thirty dev outputs per explicitly selected model; never run evaluation or train."""
import argparse
import json
from pathlib import Path
import sys


def validate_matrix(rows, questions):
    expected={(q['financebench_id'],v) for q in questions for v in ('B0','B1','B2')}
    actual=[(r['financebench_id'],r['variant']) for r in rows]
    if len(actual)!=30 or len(set(actual))!=30 or set(actual)!=expected:
        raise ValueError('Expected exactly ten development questions across all three policies.')


def paired_interval(rows):
    import numpy as np
    groups={}
    for row in rows:groups.setdefault(row['doc_name'],[]).append(float(row['difference']))
    if not groups:raise ValueError('No paired cases.')
    keys=sorted(groups);rng=np.random.default_rng(42)
    samples=[]
    for _ in range(10000):
        values=[v for i in rng.integers(0,len(keys),len(keys)) for v in groups[keys[i]]]
        samples.append(float(np.mean(values)))
    return {'difference':float(np.mean([r['difference'] for r in rows])),
            'lower':float(np.quantile(samples,.025)),'upper':float(np.quantile(samples,.975)),
            'reports':len(keys)}

