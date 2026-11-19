"""Opt-in development adapters; frozen study defaults and scoring stay unchanged.

Method references: LlamaIndex sentence-window/parent retrieval and FinQA's
whitelisted numerical execution. Original implementation, not copied source.
"""
from collections import defaultdict
from contextlib import contextmanager
import math
import re
from typing import Literal
from pydantic import TypeAdapter, model_validator
from contracts import Answer, Abstain, Calculation

FINANCE = '''
Interpret financial terms using standard accounting definitions:
- PPNE / PP&E means property, plant and equipment (normally net), not revenue or performance obligations.
- Net profit can be labelled net income or net earnings; use the appropriate consolidated/shareholder measure.
- Unless specified otherwise, do not substitute gross/operating profit or only continuing operations for total net profit.
- FY20 means fiscal year 2020 and FY21 means fiscal year 2021. A fiscal year need not end in December.
- AR means accounts receivable; COGS means cost of goods sold (also cost of sales/revenues).
- A receivables balance labelled net or less an allowance already incorporates that allowance; do not subtract it again.
- Gains from disposals or deconsolidation can be non-recurring items; reason from the evidenced event, not a requirement for the phrase 'non-standard'.
- Derive a requested metric from the evidenced inputs even when its name/result is not quoted verbatim.
- Net profit margin is 100*net profit/revenue. An average of annual margins is the mean of those percentages.
- Fixed-asset turnover is revenue/average net PP&E; inventory turnover is COGS/average inventory, unless the question specifies another convention.
- Use the calculator for arithmetic; percentage_change([current,previous]) directly computes growth in percent.
Never invent an input or assume a missing row exists. Distinguish a different label from genuinely missing evidence.
Keep final answers concise. If abstaining, identify the missing fact in at most 20 words.
For a numeric answer, put only the requested final value and units in the answer field;
do not repeat input numbers or years there. Put source IDs in citations separately.
'''


def financial_query(query):
    query = re.sub(r'\bFY\s?(\d{2})(?!\d)', r'FY20\1', query, flags=re.I)
    aliases = [(r'\bPPNE\b|\bPP&E\b', 'property plant and equipment net'),
               (r'\b(?:net\s+)?AR\b', 'accounts receivable net'),
               (r'\bCOGS\b', 'cost of goods sold cost of sales'),
               (r'net profit', 'net income net earnings'),
               (r'fixed.asset turnover', 'revenue property plant and equipment net'),
               (r'inventory turnover', 'cost of sales inventories balance sheets')]
    additions = [words for pattern, words in aliases if re.search(pattern, query, re.I)]
    return query + ('\nFinancial search terms: ' + '; '.join(additions) if additions else '')


def calculate_compound(request, original):
    operation = request['operation']
    if operation not in ('percentage_ratio', 'divide_by_mean', 'mean_percentage_ratio'):
        if original is None:
            raise ValueError('Unsupported operation.')
        return original(request)
    values = request['operands']
    if not 2 <= len(values) <= 20 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Use two to twenty finite numeric operands.')
    if operation == 'percentage_ratio' and len(values) == 2:
        result = 100*values[0]/values[1]
    elif operation == 'divide_by_mean' and len(values) == 3:
        result = values[0]/((values[1]+values[2])/2)
    elif operation == 'mean_percentage_ratio' and len(values) % 2 == 0:
        ratios = [100*values[i]/values[i+1] for i in range(0, len(values), 2)]
        result = sum(ratios)/len(ratios)
    else:
        raise ValueError('Wrong operand count for compound arithmetic.')
    if not math.isfinite(result):
        raise ValueError('Non-finite calculation.')
    return result


class FinancialCalculation(Calculation):
    operation: Literal['add', 'subtract', 'multiply', 'divide', 'mean', 'percentage_change',
                       'percentage_ratio', 'divide_by_mean', 'mean_percentage_ratio']

    @model_validator(mode='after')
    def compound_operand_count(self):
        counts = {'percentage_ratio': 2, 'divide_by_mean': 3}
        if self.operation in counts and len(self.operands) != counts[self.operation]:
            raise ValueError('Wrong compound operand count.')
        if self.operation == 'mean_percentage_ratio' and (len(self.operands) < 2 or len(self.operands) % 2):
            raise ValueError('Use consecutive numerator,denominator pairs.')
        if any(not math.isfinite(v) for v in self.operands):
            raise ValueError('Operands must be finite.')
        return self


REPAIR_ANSWER = TypeAdapter(Answer | Abstain | FinancialCalculation)


@contextmanager
def repair_scope(study, method):
    if method not in ('finance_prompt', 'finance_tools', 'finance_parent'):
        raise ValueError('Unknown development repair method.')
    names = ('SYSTEM', 'ASSESS', 'ANSWER', 'ANSWER_OR_CALCULATION', 'calculate')
    before = {name: getattr(study, name) for name in names}
    try:
        study.SYSTEM += FINANCE
        study.ASSESS += '\nUse standard financial synonyms and fiscal years; do not invent a segment or require a derived result verbatim.'
        if method != 'finance_prompt':
            study.ANSWER_OR_CALCULATION = REPAIR_ANSWER
            study.calculate = lambda request: calculate_compound(request, before['calculate'])
            study.ANSWER += '''
The same calculator also supports these compound operations in ONE request:
- percentage_ratio: operands [numerator,denominator], result=100*numerator/denominator.
- divide_by_mean: operands [numerator,opening_balance,closing_balance], result=numerator/mean(opening,closing).
- mean_percentage_ratio: operands [numerator1,denominator1,numerator2,denominator2,...], result=mean(100*numerator_i/denominator_i).
For a multi-year average profit margin, use mean_percentage_ratio, not a ratio of averaged totals.
All operands must come from cited evidence. Preserve the normal action, citations and units fields.
'''
        yield
    finally:
        for name, value in before.items():
            setattr(study, name, value)


class QueryRetriever:
    def __init__(self, base):
        self.base = base

    def retrieve(self, query, doc_name=None, previous=None):
        return self.base.retrieve(financial_query(query), doc_name=doc_name, previous=previous)


class ParentRetriever(QueryRetriever):
    """Restore whole same-page source chunks, bounded by Qwen tokens per window."""
    def __init__(self, base, tokenizer):
        super().__init__(base)
        self.tokenizer = tokenizer
        self.pages = defaultdict(list)
        self.origins = {}
        for chunk in base.chunks:
            if chunk['kind'] == 'text':
                self.pages[(chunk['doc_name'], chunk['page0'])].append(chunk)
        for chunks in self.pages.values():
            chunks.sort(key=lambda c: int(c['id'].rsplit(':c', 1)[1]))

    def retrieve(self, query, doc_name=None, previous=None):
        original = [self.origins.get(c['id'], c) for c in previous or []]
        ranked = super().retrieve(query, doc_name=doc_name, previous=original or None)
        result, seen = [], set()
        for chunk in ranked:
            page = (chunk['doc_name'], chunk['page0'])
            if chunk['kind'] != 'text':
                result.append(chunk)
                continue
            if page in seen:
                continue
            seen.add(page)
            neighbours = self.pages[page]
            index = next(i for i, c in enumerate(neighbours) if c['id'] == chunk['id'])
            chosen = [chunk]
            for extra in [neighbours[0], *neighbours[max(0,index-1):index], *neighbours[index+1:index+2]]:
                if any(c['id'] == extra['id'] for c in chosen):
                    continue
                proposed = sorted([*chosen, extra], key=lambda c: int(c['id'].rsplit(':c',1)[1]))
                text = '\n\n'.join(c['text'] for c in proposed)
                if len(self.tokenizer.encode(text, add_special_tokens=False)) <= 900:
                    chosen = proposed
            if len(chosen) == 1:
                result.append(chunk)
                continue
            enriched = {**chunk, 'id': chunk['id']+':window',
                        'text': '\n\n'.join(c['text'] for c in chosen),
                        'source_chunk_ids': [c['id'] for c in chosen],
                        'context_kind': 'same_page_parent_window'}
            enriched['retrieved_child_embedding_tokens'] = enriched.pop('embedding_tokens', None)
            enriched['parent_window_qwen_tokens'] = len(self.tokenizer.encode(enriched['text'], add_special_tokens=False))
            self.origins[enriched['id']] = chunk
            result.append(enriched)
        return result
