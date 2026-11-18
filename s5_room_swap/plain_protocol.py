"""Plain-lines extraction protocol shared by the open-model and local-4B EXTRACT+REG arms.

The prompt asks for one 'number|name|name' line per event. A chunk passes the count check only if it returns exactly the
chunk's line numbers, each once, each with two distinct roster names. `post` and `usage_of` are supplied by the caller
(run_tinker.plain_parse feeds a stored reply through them; no provider request is made from this module).
"""
PLAIN_INSTR = '각 줄에서 방을 바꾼 두 사람을 줄 순서대로 "번호|이름|이름" 형식으로 한 줄에 하나씩 적으세요.'
PLAIN_RE = __import__('re').compile(r'^\s*(\d+)\s*\|\s*(\S+?)\s*\|\s*(\S+?)\s*$')  # whitespace around '|' is formatting (one model writes '1|현우 | 지아')


def plain_call(lines):
    """Plain 'n|name|name' lines with a count check: exactly the chunk's line numbers, each once, two distinct roster names."""
    nums = [int(l.split('.', 1)[0]) for l in lines]
    d, wall = post({'model': MODEL, 'messages': [{'role': 'user', 'content': '\n'.join(lines) + '\n\n' + PLAIN_INSTR}],
                    'reasoning_effort': 'none', 'max_tokens': 8000}, timeout=600)
    ch = d['choices'][0]; got = {}; ok = True
    for line in (ch['message'].get('content') or '').splitlines():
        if not line.strip(): continue
        m = PLAIN_RE.match(line.strip().strip('`'))
        if not m or m.group(2) not in g.NAMES or m.group(3) not in g.NAMES or m.group(2) == m.group(3) or int(m.group(1)) in got:
            ok = False; continue
        got[int(m.group(1))] = (m.group(2), m.group(3))
    ok = ok and sorted(got) == nums
    return {'wall_s': wall, 'usage': usage_of(d), 'pairs': [list(got[n]) for n in nums if n in got], 'n_lines': len(lines), 'ok': ok,
            'finish_reason': ch.get('finish_reason'), 'resp_model': d.get('model'), 'resp_id': d.get('id')}
