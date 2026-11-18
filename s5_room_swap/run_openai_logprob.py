"""Report-only: per-field token log-probabilities on the extract+register path. Reasoning models (gpt-6.1-sol/terra,
gpt-6-luna, gpt-5.6-luna) reject logprobs (400, results/openai_logprob_probe.log); gpt-4.1-mini (non-reasoning) is used.
Per field = one name in one returned pair; field logprob = sum of logprobs of the output tokens spanning that name."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import swapgen as g, run_openai as o
HERE = Path(__file__).parent; OUT = HERE / 'results' / 'openai_logprob_4.1mini'; OUT.mkdir(parents=True, exist_ok=True)
MODEL = 'gpt-4.1-mini'


def call(lines):
    body = {'model': MODEL, 'input': '\n'.join(lines) + '\n\n' + o.INSTR, 'max_output_tokens': 4000, 'include': ['message.output_text.logprobs'],
            'top_logprobs': 1, 'text': {'format': {'type': 'json_schema', 'name': 'swaps', 'schema': o.SCHEMA, 'strict': True}}}
    resp, wall, err = o.post(body, timeout=600)
    if err: raise RuntimeError(err)
    lp = []
    for it in resp.get('output', []):
        for c in it.get('content', []) or []:
            if c.get('type') == 'output_text': lp = [(t['token'], t['logprob']) for t in c.get('logprobs') or []]
    text = ''.join(t for t, _ in lp); pairs = json.loads(o.text_of(resp))['pairs']
    # field spans: walk names in output order
    starts, pos = [], 0
    for t, _ in lp: starts.append(pos); pos += len(t)
    fields, cur = [], 0
    for p in pairs:
        for name in p:
            i = text.index('"' + name + '"', cur) + 1; j = i + len(name); cur = j
            fields.append({'name': name, 'logprob': sum(l for (t, l), s in zip(lp, starts) if s < j and s + len(t) > i)})
    return {'wall_s': wall, 'usage': o.usage_of(resp), 'pairs': pairs, 'fields': fields, 'n_lines': len(lines)}


for m in json.loads((HERE / 'manifest.json').read_text()):
    f = OUT / f"{m['id']}.json"
    if f.exists(): continue
    if o.arm_spent(True) > 8.0: print('STOP guard'); break
    text = (HERE / 'docs' / f"{m['id']}.txt").read_text(encoding='utf-8')
    lines = [l for l in text.split('\n') if '방을 바꿨습니다.' in l]
    chunks = [lines[i:i + 50] for i in range(0, len(lines), 50)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(chunks)) as ex: res = list(ex.map(call, chunks))
    pairs = [tuple(g.NAMES.index(x) for x in p) for c in res for p in c['pairs'] if len(p) == 2]
    pred = g.execute(pairs); wall = time.perf_counter() - t0
    gold_fields = [g.NAMES[x] for a, b in m['events'] for x in (a, b)]
    fields = [fd for c in res for fd in c['fields']]
    for k, fd in enumerate(fields): fd['correct'] = k < len(gold_fields) and fd['name'] == gold_fields[k]  # positional; extra fields count wrong
    ans = '\n'.join(f'{g.NAMES[i]}: {pred[i]}' for i in range(5))
    u = {k: sum((c['usage'][k] or 0) for c in res) for k in ('input_tokens', 'output_tokens')}
    rec = {'id': m['id'], 'length': m['length'], 'seed': m['seed'], 'model': MODEL, 'wall_s': wall, 'n_fields': len(fields),
           'n_fields_gold': len(gold_fields), 'wrong_fields': sum(not fd['correct'] for fd in fields),
           'min_logprob': min(fd['logprob'] for fd in fields), 'usage': u, 'fields': fields, 'answer': ans, 'truth': m['truth'], **g.score(ans, m['truth'])}
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(m['id'], f"wall={wall:.1f}s fields={len(fields)}/{len(gold_fields)} wrong={rec['wrong_fields']} min_lp={rec['min_logprob']:.4g} tok_in={u['input_tokens']} tok_out={u['output_tokens']} correct={rec['people_correct']}", flush=True)
