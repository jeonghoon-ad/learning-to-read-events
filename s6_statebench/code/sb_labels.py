"""Read StateBench formatter templates into exact per-terminator gold labels."""
from functools import lru_cache
from pathlib import Path
import re
import yaml
from sb_register import CONST, COPY, INC, NOOP

ROOT = Path(__file__).resolve().parent
FORMAT_DIR = ROOT.parent / 'vendor/state-bench-5b333b8d/configs/formats'  # upstream state-bench@5b333b8d, Apache-2.0
FORMATS = ('integer-code-modular', 'spreadsheet-cells-modular', 'people-in-rooms',
           'status-lights', 'cube-painting')


def _compile(template, variables, values):
    var = '(?:' + '|'.join(re.escape(s) for s in sorted(variables, key=len, reverse=True)) + ')'
    value = '(?:' + '|'.join(re.escape(s) for s in sorted(values, key=len, reverse=True)) + ')'
    substitutions = {'target':var, 'source':var, 'variable':var, 'value':value,
                     'delta':r'\d+', 'modulus':r'\d+', 'filler_index':r'\d+', 'filler_dots':r'\.+'}
    result, seen = [], set()
    # Substitute only placeholders, preserving literal template characters.
    pos = 0
    for match in re.finditer(r'\$([A-Za-z_][A-Za-z_0-9]*)', template):
        result.append(re.escape(template[pos:match.start()]))
        key = match.group(1)
        if key not in substitutions: raise ValueError('unsupported template placeholder ' + key)
        if key in seen: result.append('(?P=' + key + ')')
        else:
            result.append('(?P<' + key + '>' + substitutions[key] + ')')
            seen.add(key)
        pos = match.end()
    result.append(re.escape(template[pos:]))
    return re.compile(''.join(result))


@lru_cache(None)
def formatter(name):
    if name not in FORMATS: raise ValueError('unknown formatter')
    config = yaml.safe_load((FORMAT_DIR / (name + '.yaml')).read_text())
    vocab = config['vocabulary']
    variables = list(vocab['variables']) + [vocab['temporary_variable']]
    values = [str(v) for v in vocab['values']]
    patterns = []
    # Consume fillers atomically, including fake semicolon-delimited assignments.
    for key in ('filler_hard', 'filler_medium', 'filler_easy', 'assign_increment', 'assign_constant', 'assign_reference'):
        template = config['templates'].get(key)
        if template: patterns.append((key, _compile(template, variables, values)))
    return config, variables, values, patterns


def parse_text(prefix, name):
    """Map exact statement terminal character offsets to 4-tuples; fillers excluded."""
    config, variables, values, patterns = formatter(name)
    task = config['templates']['task_description']
    block = config['layout']['block_separator']
    if not prefix.startswith(task + block): raise ValueError('unexpected task block')
    start = len(task + block)
    end = len(prefix)
    answer = config['templates'].get('answer_prompt')
    if answer:
        suffix = block + answer
        if not prefix.endswith(suffix): raise ValueError('unexpected answer block')
        end -= len(suffix)
    # Ai2's filler renderer switches to newline-separated instructions.
    sep = '\n' if '\n' in prefix[start:end] else config['layout']['instruction_separator']
    statements = {}
    while start < end:
        matches = [(key, pattern.match(prefix, start)) for key, pattern in patterns]
        matches = [(key, m) for key, m in matches if m is not None and m.end() <= end
                   and (m.end() == end or prefix.startswith(sep, m.end()))]
        if len(matches) != 1:
            raise ValueError(f'cannot uniquely parse instruction at {start}: {prefix[start:start+80]!r}, {[x[0] for x in matches]}')
        key, match = matches[0]
        group = match.groupdict()
        if key.startswith('assign_'):
            tgt = variables.index(group['target'])
            if key == 'assign_constant': read = (CONST, tgt, 0, values.index(group['value']))
            elif key == 'assign_reference': read = (COPY, tgt, variables.index(group['source']), 0)
            else:
                if int(group['modulus']) != 100: raise ValueError('unexpected modulus')
                read = (INC, tgt, 0, int(group['delta']))
            statements[match.end()-1] = read
        start = match.end()
        if start < end:
            if not prefix.startswith(sep, start): raise ValueError('separator mismatch')
            start += len(sep)
    return statements


def label_reads(prefix, name, tokenizer):
    """Tokenize once; identify token IDs 26/13 using source offsets, fail closed."""
    statements = parse_text(prefix, name)
    enc = tokenizer.encode(prefix, add_special_tokens=False)
    ids, offsets = enc.ids, enc.offsets
    positions, labels = [], []
    seen = set()
    for i, (token, (a, b)) in enumerate(zip(ids, offsets)):
        if token not in (26, 13): continue
        character = ';' if token == 26 else '.'
        if b <= a or prefix[b-1] != character: raise ValueError('terminator token offset mismatch')
        positions.append(i)
        labels.append(statements.get(b-1, (NOOP, 0, 0, 0)))
        if b-1 in statements: seen.add(b-1)
    if seen != set(statements):
        raise ValueError('statement terminator is not a lexical read token')
    return ids, positions, labels


def parse_choice(choice, name):
    """Parse one generated state-assertion choice into vocabulary indices."""
    config,variables,values,_=formatter(name)
    pattern=_compile(config['templates']['state_assertion'],variables,values)
    separator=config['layout']['completion_separator']
    cursor=0; state={}
    while cursor<len(choice):
        match=pattern.match(choice,cursor)
        if not match: raise ValueError('cannot parse choice assertion')
        var=variables.index(match['variable']); value=values.index(match['value'])
        if var in state: raise ValueError('duplicate variable in choice')
        state[var]=value;cursor=match.end()
        if cursor<len(choice):
            if not choice.startswith(separator,cursor): raise ValueError('choice separator mismatch')
            cursor+=len(separator)
    if not state: raise ValueError('empty choice')
    return state
