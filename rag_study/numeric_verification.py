"""Versioned scalar financial-quantity checks; never full semantic grading.

No model calls, gold-page inspection, question-ID aliases or frozen-score changes.
Unsupported/ambiguous expressions are not_checked, not assumed correct.
"""
from decimal import Decimal
import re
from evaluation import precision

VERSION='financial_scalar_consistency_v1'
NUMBER=r'(?<![\w.,])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?![\d.,])'
SCALE={'thousand':Decimal('1e3'),'million':Decimal('1e6'),'billion':Decimal('1e9'),'trillion':Decimal('1e12'),'k':Decimal('1e3'),'m':Decimal('1e6'),'b':Decimal('1e9'),'bn':Decimal('1e9'),'one':Decimal(1)}
CURRENCIES={'USD':r'\bUSD\b|US\$|(?<![A-Za-z])\$|\bdollars?\b','EUR':r'\bEUR\b|€','GBP':r'\bGBP\b|£','JPY':r'\bJPY\b','CNY':r'\bCNY\b|\bRMB\b','HKD':r'\bHKD\b|HK\$','AUD':r'\bAUD\b|A\$','CAD':r'\bCAD\b|C\$'}


def parse(text):
    text=text.replace('−','-').strip()
    if re.search(r'\b(?:not|unknown|insufficient|unable|cannot)\b|\d[eE][+-]?\d',text):raise ValueError('Negation or unsupported scientific notation.')
    matches=list(re.finditer(NUMBER,text))
    if len(matches)!=1:raise ValueError('Exactly one unambiguous number is required.')
    m=matches[0];number=Decimal(m[0].replace(',',''))
    if text[:m.start()].rstrip().endswith('(') and text[m.end():].lstrip().startswith(')'):number=-abs(number)
    scales=re.findall(r'\b(thousands?|millions?|billions?|trillions?)\b',text,re.I)
    short=re.match(r'\s*(bn|[kmb])\b',text[m.end():],re.I)
    if short:scales.append(short[1])
    scales={s.lower().rstrip('s') for s in scales}
    if len(scales)>1:raise ValueError('Conflicting scales.')
    currencies={k for k,p in CURRENCIES.items() if re.search(p,text,re.I)}
    if len(currencies)>1:raise ValueError('Conflicting currencies.')
    if re.search(r'percentage.points?|percent.points?|\bpp\b',text,re.I):dimension='percentage_point'
    elif '%' in text or re.search(r'\bpercent(?:age)?s?\b',text,re.I):dimension='percent'
    elif re.search(r'\bfraction\b',text,re.I):dimension='fraction'
    else:dimension=None
    base_dollars=bool(currencies) and bool(re.search(r'\bdollars?\b',text,re.I))
    return number,next(iter(scales),'one' if base_dollars else None),next(iter(currencies),None),dimension,m


def context_units(question):
    # Only inherit an unambiguous requested scale/currency/dimension, not a numeric operand.
    fake=re.sub(r'\d+(?:\.\d+)?','',question)+' 0'
    try:return parse(fake)[1:4]
    except ValueError:return (None,None,None)


def check_numeric(candidate,reference,question=''):
    result={'version':VERSION,'status':'not_checked','reason':'','establishes_answer_correctness':False}
    try:
        rv,rs,rc,rd,match=parse(reference)
        # A prose reference is not a scalar target. Never cherry-pick its numbers.
        rest=reference.replace('−','-');rest=rest[:match.start()]+rest[match.end():]
        rest=re.sub('|'.join(CURRENCIES.values()),'',rest,flags=re.I)
        rest=re.sub(r'\b(thousands?|millions?|billions?|trillions?|bn|k|m|b|percent(?:age)?s?|percentage.points?|percent.points?|pp|fraction)\b','',rest,flags=re.I)
        if re.sub(r'[\s.,()%+\-]','',rest):raise ValueError('Reference is not scalar-only.')
        qscale,qcurrency,qdimension=context_units(question)
        rs=rs or qscale;rc=rc or qcurrency;rd=rd or qdimension or 'number'
        cv,cs,cc,cd,_=parse(candidate)
        inherited=[]
        if cs is None:cs=rs;inherited.append('scale')
        if cc is None:cc=rc;inherited.append('currency')
        if cd is None:cd=rd;inherited.append('dimension')
        rscale=SCALE[rs] if rs else Decimal(1);cscale=SCALE[cs] if cs else Decimal(1)
        # Percent values are measured in percentage points. Explicit fractions are converted.
        if rd=='fraction':rscale*=100;rd='percent'
        if cd=='fraction':cscale*=100;cd='percent'
        target=rv*rscale;got=cv*cscale
        tolerance=Decimal('0.5')*Decimal(10)**(-precision(question,rv))*rscale
        consistent=rc==cc and rd==cd and abs(target-got)<=tolerance
        return {**result,'status':'consistent' if consistent else 'inconsistent',
                'reason':'Quantity/unit comparison only; citation support and full meaning remain unverified.',
                'reference_base':str(target),'candidate_base':str(got),'tolerance_base':str(tolerance),
                'reference_currency':rc,'candidate_currency':cc,'reference_dimension':rd,'candidate_dimension':cd,
                'inherited_candidate_fields':inherited}
    except (ValueError,KeyError) as exc:return {**result,'reason':str(exc)}
