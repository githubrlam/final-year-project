"""Deterministic reference matching. Never a semantic or citation-support judge."""
import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation

VERSION = 'reference_match_v1'
NUMBER = r'(?<![\w.,])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?![\w,]|\.\d)'
SCALES = {'thousand': Decimal(1000), 'million': Decimal(1000000),
          'billion': Decimal(1000000000), 'trillion': Decimal(1000000000000)}


def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def precision(question, reference):
    match = re.search(r'round\w*.*?\b(one|two|three|four|zero|[0-9])\s+decimal places?', question.lower())
    if match:
        return int(match[1]) if match[1].isdigit() else ['zero', 'one', 'two', 'three', 'four'].index(match[1])
    if re.search(r'nearest (whole|integer)', question.lower()):
        return 0
    return max(0, -reference.normalize().as_tuple().exponent)


def scalar(text, question, default_scale=None, default_kind=None):
    numeric_text = text.replace('−', '-')
    numeric_text = re.sub(r'\b(?:USD|US)\b|[$€£]', '', numeric_text, flags=re.I)
    numeric_text = re.sub(r'([+-])\s+(?=[\d.])', r'\1', numeric_text)
    matches = list(re.finditer(NUMBER, numeric_text))
    if len(matches) != 1:
        return None
    match = matches[0]
    try:
        number = Decimal(match[0].replace(',', ''))
    except InvalidOperation:
        return None
    if numeric_text[:match.start()].rstrip().endswith('(') and numeric_text[match.end():].lstrip().startswith(')'):
        number = -abs(number)
    lower = text.lower()
    scale = next((value for name, value in SCALES.items() if re.search(r'\b'+name+r's?\b', lower)), default_scale)
    if scale is None:
        scale = next((value for name, value in SCALES.items() if re.search(r'\b'+name+r's?\b', question.lower())), Decimal(1))
    if re.search(r'\beur\b|€|\bgbp\b|£|\bcny\b|\bjpy\b|\ba\$|\bhk\$', lower):
        kind = 'other_currency'
    elif '%' in lower or re.search(r'\bpercent(?:age)?s?\b', lower):
        kind = 'percent'
    elif '$' in lower or re.search(r'\busd\b|\bdollars?\b', lower):
        kind = 'usd'
    elif default_kind:
        kind = default_kind
    elif re.search(r'\bpercent(?:age)?s?\b|%', question.lower()):
        kind = 'percent'
    elif re.search(r'\busd\b|\bdollars?\b|\bus\$', question.lower()):
        kind = 'usd'
    else:
        kind = 'number'
    return number, scale, kind


def score_answer(prediction, question):
    answer, reference = prediction.get('answer', ''), question['answer']
    tokens_a, tokens_r = normalize(answer).split(), normalize(reference).split()
    overlap = sum((Counter(tokens_a) & Counter(tokens_r)).values())
    f1 = 2 * overlap / (len(tokens_a) + len(tokens_r)) if tokens_a or tokens_r else 1.0
    exact = int(normalize(answer) == normalize(reference))
    result = {'answer_correct': 0, 'normalized_exact_match': exact, 'token_f1': f1,
              'match_method': 'normalized_exact_match', 'match_detail': ''}
    if prediction.get('invalid') or prediction.get('abstain'):
        return {**result, 'match_method': 'invalid' if prediction.get('invalid') else 'abstention'}
    # Only a scalar-only reference activates numeric matching; do not cherry-pick narrative numbers.
    ref_literal = re.sub(r'\b(USD|US|dollars?|thousands?|millions?|billions?|trillions?|percent(?:age)?s?)\b', '', reference.replace('−', '-'), flags=re.I)
    ref_literal = re.sub(r'[\s$%,()+\-\d.]', '', ref_literal)
    ref = scalar(reference, question['question']) if not ref_literal else None
    if ref is not None:
        got = scalar(answer, question['question'], ref[1], ref[2])
        digits = precision(question['question'], ref[0])
        tolerance = Decimal('0.5') * Decimal(10) ** (-digits) * ref[1]
        passed = not re.search(r'\b(not|cannot|unknown|insufficient|unable)\b', answer, re.I) and got is not None and got[2] == ref[2] and abs(got[0] * got[1] - ref[0] * ref[1]) <= tolerance
        return {**result, 'answer_correct': int(passed), 'match_method': 'numeric_equivalence',
                'match_detail': f'precision={digits}; tolerance={tolerance} in base units; exactly one predicted number required'}
    polarity_ref = re.match(r'^\s*(yes|no)\b', reference, re.I)
    if polarity_ref:
        polarity = re.match(r'^\s*(yes|no)\b', answer, re.I)
        return {**result, 'answer_correct': int(bool(polarity) and polarity[1].lower() == polarity_ref[1].lower()),
                'match_method': 'leading_yes_no', 'match_detail': 'Polarity only; additional claims are not assessed.'}
    return {**result, 'answer_correct': exact}


def metrics(study, split):
    import numpy as np
    import pandas as pd
    from phase5 import validate_pairs
    predictions = study.read_jsonl(study.OUT / f'{split}_predictions.jsonl')
    gold = {q['financebench_id']: q for q in study.read_jsonl(study.DATA / f'{split}.jsonl')}
    validate_pairs(predictions, list(gold.values()), complete=True)
    rows = []
    for p in predictions:
        q = gold[p['financebench_id']]
        expected = {(e.get('doc_name', e.get('evidence_doc_name')), int(e['evidence_page_num'])) for e in q['evidence']}
        actual = {(c['doc_name'], c['page0']) for c in p['final_context']}
        score = score_answer(p, q)
        rows.append({**{k:p[k] for k in ['financebench_id','variant','doc_name','seconds','model_calls','retrieval_rounds',
            'input_tokens','output_tokens','abstain','invalid']}, **score,
            'reference_answer':q['answer'], 'generated_answer':p.get('answer',''),
            'evidence_page_recall':len(expected & actual)/len(expected), 'evaluator_version':VERSION})
    frame = pd.DataFrame(rows)
    paired = frame.pivot(index=['doc_name','financebench_id'],columns='variant',values='answer_correct')
    if paired.isna().any().any() or len(paired) != len(gold):
        raise ValueError('Paired comparisons require all policies for every selected question.')
    summaries=[]
    for variant,part in frame.groupby('variant'):
        summaries.append({'variant':variant,'n':len(part),'answer_correct':part.answer_correct.mean(),
            'normalized_exact_match':part.normalized_exact_match.mean(),'token_f1':part.token_f1.mean(),
            'median_seconds':part.seconds.median(),'p95_seconds':part.seconds.quantile(.95),
            'mean_calls':part.model_calls.mean(),'mean_input_tokens':part.input_tokens.mean(),
            'mean_output_tokens':part.output_tokens.mean(),'evidence_page_recall':part.evidence_page_recall.mean(),
            'abstention_rate':(part.abstain & ~part.invalid).mean(),'invalid_rate':part.invalid.mean(),
            'mean_retrieval_rounds':part.retrieval_rounds.mean()})
    reports=paired.index.get_level_values('doc_name').unique().tolist()
    rng=np.random.default_rng(study.CONFIG['seed']);intervals=[]
    for comparator in ['B0','B1']:
        difference=paired.B2-paired[comparator]
        samples=[pd.concat([difference.xs(report,level='doc_name') for report in rng.choice(reports,size=len(reports),replace=True)]).mean()
                 for _ in range(study.CONFIG['bootstrap_replicates'])]
        low,high=np.quantile(samples,[.025,.975]);intervals.append({'comparison':'B2-'+comparator,
            'outcome':'automated_reference_answer_accuracy','difference':difference.mean(),'lower_95':low,'upper_95':high})
    summary=pd.DataFrame(summaries)
    frame.to_csv(study.OUT/f'{split}_automatic_scores.csv',index=False)
    frame.to_csv(study.OUT/f'{split}_question_metrics.csv',index=False)
    summary.to_csv(study.OUT/f'{split}_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(study.OUT/f'{split}_paired_intervals.csv',index=False)
    return summary
