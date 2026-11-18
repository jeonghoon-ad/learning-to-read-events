"""OPENAI-ALONE and OPENAI-EXTRACT+REG arms. Copy for gpt-6.2-sol (2026-11-18): urllib instead of requests, 6.2 price, SWAP_IDS filter. OpenAI Responses API, key read from
~/.config/openai/api_key inside the process (never printed or stored). Accounting at US$2/US$10 per MTok (input/output);
the stop guard reserves at US$2.6/US$13 (highest listed rate) and stops the arm set at US$9.
ALONE: whole document, reasoning effort high. EXTRACT: 50 lines per call, JSON-schema swap pairs, reasoning effort low
(lowest accepted), chunks of a document concurrent; serial executor composes. Timing: wall clock per request."""
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request, urllib.error
sys.path.insert(0, str(Path(__file__).parent)); import swapgen as g
HERE = Path(__file__).parent; R = HERE / 'results'
MODEL = os.environ.get('OAI_MODEL', 'gpt-6.1-sol')
PRICES = {'gpt-6.2-sol': (2e-6, 10e-6), 'gpt-6.1-sol': (2e-6, 10e-6), 'gpt-6.1-terra': (1e-6, 5e-6), 'gpt-6-luna': (0.1e-6, 0.5e-6)}  # third-party listings
IN_P, OUT_P = PRICES[MODEL]; G_IN, G_OUT = 1.3 * IN_P, 1.3 * OUT_P; STOP = float(os.environ.get('OAI_STOP', '9.0'))
SUF = '' if MODEL == 'gpt-6.1-sol' else '_' + MODEL; D_ALONE, D_EXTRACT = 'openai' + SUF, 'openai_extract' + SUF
_KEY = open(os.path.expanduser('~/.config/openai/api_key')).read().strip()
lock = threading.Lock()
SCHEMA = {'type': 'object', 'properties': {'pairs': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string', 'enum': g.NAMES}}}},
          'required': ['pairs'], 'additionalProperties': False}
INSTR = '각 줄에서 방을 바꾼 두 사람을 줄 순서대로 적으세요. 한 줄에 한 쌍입니다.'


def post(body, timeout=2400):
    """Retries 429/5xx up to 5 times (backoff 15, 30, 60, 120 s); wall is the successful attempt's. urllib instead of requests."""
    for k in range(5):
        t0 = time.perf_counter()
        req = urllib.request.Request('https://api.openai.com/v1/responses', data=json.dumps(body).encode(), method='POST',
                                     headers={'Authorization': 'Bearer ' + _KEY, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            return data, time.perf_counter() - t0, None
        except urllib.error.HTTPError as e:
            wall = time.perf_counter() - t0; code = e.code; txt = e.read().decode(errors='replace')[:300]
            if code in (429, 500, 502, 503, 504) and k < 4:
                print('retry', code, flush=True); time.sleep(15 * 2 ** k); continue
            return None, wall, f'{code} {txt}'
        except Exception as e:
            return None, time.perf_counter() - t0, f'EXC {type(e).__name__}: {e}'



def text_of(resp):
    out = []
    for item in resp.get('output', []):
        for c in item.get('content', []) or []:
            if c.get('type') == 'output_text': out.append(c.get('text', ''))
    return ''.join(out)


def usage_of(resp):
    u = resp.get('usage') or {}
    return {'input_tokens': u.get('input_tokens', 0), 'output_tokens': u.get('output_tokens', 0),
            'reasoning_tokens': (u.get('output_tokens_details') or {}).get('reasoning_tokens')}


def cost(u, i=IN_P, o=OUT_P): return u['input_tokens'] * i + u['output_tokens'] * o


def arm_spent(guard=False):
    tot = 0.0
    for d in (D_ALONE, D_EXTRACT):
        for f in (R / d).glob('*.json') if (R / d).exists() else []:
            r = json.loads(f.read_text())
            us = [c['usage'] for c in r['chunks']] if 'chunks' in r else [r['usage']]
            tot += sum(cost(u, G_IN, G_OUT) if guard else cost(u) for u in us)
    return tot


def alone(m, effort='high', max_out=100000):
    out = R / D_ALONE; out.mkdir(parents=True, exist_ok=True); f = out / f"{m['id']}.json"
    if f.exists(): return
    text = (HERE / 'docs' / f"{m['id']}.txt").read_text(encoding='utf-8')
    if arm_spent(True) + len(text) * G_IN + max_out * G_OUT > STOP:
        print('STOP guard', m['id'], flush=True); return
    resp, wall, err = post({'model': MODEL, 'input': text, 'reasoning': {'effort': effort}, 'max_output_tokens': max_out})
    if err: print('ERR', m['id'], err, flush=True); return
    ans = text_of(resp); u = usage_of(resp)
    rec = {'id': m['id'], 'length': m['length'], 'seed': m['seed'], 'model': MODEL, 'effort': effort, 'wall_s': wall,
           'status': resp.get('status'), 'incomplete': resp.get('incomplete_details'), 'usage': u, 'thinking_tokens': u['reasoning_tokens'],
           'cost_usd': cost(u), 'answer': ans, 'truth': m['truth'], **g.score(ans, m['truth'])}
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print('ALONE', m['id'], f"wall={wall:.1f}s out={u['output_tokens']} reason={u['reasoning_tokens']} ${rec['cost_usd']:.3f} correct={rec['people_correct']} arm=${arm_spent():.3f}", flush=True)


def chunk_call(lines, effort, logprobs):
    body = {'model': MODEL, 'input': '\n'.join(lines) + '\n\n' + INSTR, 'reasoning': {'effort': effort}, 'max_output_tokens': 8000,
            'text': {'format': {'type': 'json_schema', 'name': 'swaps', 'schema': SCHEMA, 'strict': True}}}
    if logprobs: body.update({'include': ['message.output_text.logprobs'], 'top_logprobs': 1})
    resp, wall, err = post(body, timeout=600)
    if err: raise RuntimeError(err)
    lp = None
    for item in resp.get('output', []):
        for c in item.get('content', []) or []:
            if c.get('type') == 'output_text' and c.get('logprobs'): lp = [(t['token'], t['logprob']) for t in c['logprobs']]
    return {'wall_s': wall, 'usage': usage_of(resp), 'pairs': json.loads(text_of(resp))['pairs'], 'n_lines': len(lines), 'logprobs': lp}


def extract(m, effort='low', logprobs=False):
    out = R / D_EXTRACT; out.mkdir(parents=True, exist_ok=True); f = out / f"{m['id']}.json"
    if f.exists(): return
    text = (HERE / 'docs' / f"{m['id']}.txt").read_text(encoding='utf-8')
    lines = [l for l in text.split('\n') if '방을 바꿨습니다.' in l]
    chunks = [lines[i:i + 50] for i in range(0, len(lines), 50)]
    if arm_spent(True) + len(chunks) * (len(text) / len(chunks) * G_IN + 8000 * G_OUT) > STOP:
        print('STOP guard', m['id'], flush=True); return
    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(chunks)) as ex:
        res = list(ex.map(lambda c: chunk_call(c, effort, logprobs), chunks))
    pairs, bad = [], 0
    for c in res:
        bad += not (len(c['pairs']) == c['n_lines'] and all(len(p) == 2 and p[0] != p[1] for p in c['pairs']))
        pairs += [tuple(g.NAMES.index(x) for x in p) for p in c['pairs'] if len(p) == 2]
    pred = g.execute(pairs); wall = time.perf_counter() - t0
    ans = '\n'.join(f'{g.NAMES[i]}: {pred[i]}' for i in range(5))
    per = sum(int(set(a) == set(b)) for a, b in zip(pairs, m['events'])) / len(m['events']) if len(pairs) == len(m['events']) else None
    u = {k: sum((c['usage'][k] or 0) for c in res) for k in ('input_tokens', 'output_tokens', 'reasoning_tokens')}
    rec = {'id': m['id'], 'length': m['length'], 'seed': m['seed'], 'model': MODEL, 'effort': effort, 'wall_s': wall,
           'serial_chunk_wall_s': sum(c['wall_s'] for c in res), 'n_chunks': len(chunks), 'bad_chunks': bad, 'per_event_acc': per,
           'usage': u, 'thinking_tokens': u['reasoning_tokens'], 'cost_usd': cost(u), 'chunks': res, 'answer': ans, 'truth': m['truth'],
           **g.score(ans, m['truth'])}
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print('EXTRACT', m['id'], f"wall={wall:.1f}s serial={rec['serial_chunk_wall_s']:.1f}s bad={bad} per_event={per} out={u['output_tokens']} reason={u['reasoning_tokens']} ${rec['cost_usd']:.3f} correct={rec['people_correct']} arm=${arm_spent():.3f}", flush=True)


if __name__ == '__main__':
    man = json.loads((HERE / 'manifest.json').read_text())
    IDS = set(os.environ['SWAP_IDS'].split(','))
    mode = sys.argv[1]
    if mode == 'probe':
        alone(man[0]); extract(man[0])
    elif mode == 'extract':
        for m in man:
            if m['id'] in IDS: extract(m)
    elif mode == 'alone':
        for L in g.LENGTHS:
            with ThreadPoolExecutor(5) as ex: list(ex.map(alone, [m for m in man if m['length'] == L and m['id'] in IDS]))
    elif mode == 'all':
        def run_extract():
            for m in man: extract(m)
        th = threading.Thread(target=run_extract); th.start()
        for L in g.LENGTHS:
            with ThreadPoolExecutor(5) as ex: list(ex.map(alone, [m for m in man if m['length'] == L]))
        th.join(); print('DONE arm $%.3f' % arm_spent())
