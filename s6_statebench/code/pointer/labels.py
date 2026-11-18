"""Template-derived character spans and exact token-range controls."""
import bisect
import unicodedata
from sb_labels import formatter
from sb_register import CONST, COPY, INC, NOOP

FIELDS = ('tgt', 'src', 'val')

def normal(text):
    return unicodedata.normalize('NFKC', text).strip()

def statements(prefix, name):
    config, _, _, patterns = formatter(name)
    task = config['templates']['task_description']
    block = config['layout']['block_separator']
    if not prefix.startswith(task + block): raise ValueError('unexpected task block')
    start = len(task + block); end = len(prefix)
    answer = config['templates'].get('answer_prompt')
    if answer:
        suffix = block + answer
        if not prefix.endswith(suffix): raise ValueError('unexpected answer block')
        end -= len(suffix)
    sep = '\n' if '\n' in prefix[start:end] else config['layout']['instruction_separator']
    out = {}
    while start < end:
        matches = [(key, p.match(prefix, start)) for key, p in patterns]
        matches = [(key,m) for key,m in matches if m is not None and m.end() <= end
                   and (m.end() == end or prefix.startswith(sep, m.end()))]
        if len(matches) != 1: raise ValueError(f'cannot uniquely parse at {start}: {prefix[start:start+80]!r}')
        key,m = matches[0]
        if key.startswith('assign_'):
            op={'assign_constant':CONST,'assign_reference':COPY,'assign_increment':INC}[key]
            spans={'tgt':m.span('target'), 'src':m.span('source') if op==COPY else None,
                   'val':m.span('value' if op==CONST else 'delta') if op in (CONST,INC) else None}
            out[m.end()-1]=(op, spans)
        start=m.end()
        if start<end:
            if not prefix.startswith(sep,start): raise ValueError('separator mismatch')
            start+=len(sep)
    return out

def label_reads(prefix, name, tokenizer, width=32):
    gold=statements(prefix,name)
    enc=tokenizer.encode(prefix,add_special_tokens=False)
    ids, offsets=enc.ids,enc.offsets
    starts=[x[0] for x in offsets]; ends=[x[1] for x in offsets]
    positions=[]; labels=[]; previous=-1; seen=set()
    for i,(token,(a,b)) in enumerate(zip(ids,offsets)):
        if token not in (26,13): continue
        if b<=a or prefix[b-1] != (';' if token==26 else '.'): raise ValueError('terminator offset mismatch')
        op,spans=gold.get(b-1,(NOOP,{})); left=max(0,i-width+1,previous+1)
        ranges={}; strings={}
        for field in FIELDS:
            span=spans.get(field)
            if span is None: ranges[field]=None; strings[field]=None; continue
            c,d=span
            first=bisect.bisect_right(ends,c); last=bisect.bisect_left(starts,d)-1
            if first>last or first<left or last>i: raise ValueError(f'gold span outside K={width}: {name} {field} {span} at {i}')
            expected=normal(prefix[c:d]); decoded=normal(tokenizer.decode(ids[first:last+1]))
            if decoded != expected: raise ValueError(f'gold decode mismatch {name} {field}: {decoded!r} != {expected!r}')
            ranges[field]=(first-(i-width+1),last-(i-width+1)); strings[field]=expected
        positions.append(i); labels.append({'op':op,'ranges':ranges,'strings':strings,'left':left,'char_spans':spans})
        if b-1 in gold: seen.add(b-1)
        previous=i
    if seen != set(gold): raise ValueError('statement terminator not represented')
    return ids, positions, labels
