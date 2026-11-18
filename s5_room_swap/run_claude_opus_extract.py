"""CLAUDE-EXTRACT+REG arm, Opus 5.5 (copy of run_claude_extract.py; only model, prices, output dir, guard and item filter differ), thinking off (`between_tools`), effort low, JSON
schema output. Claude reads 50 event lines per call and returns the swap pairs in order (no state); the shared serial
executor (swapgen.execute) composes them. Chunks of one document run concurrently (all at once); documents run one
after another. Wall = first request start to last response end + executor. Also records the serial sum of chunk walls.
Spend guard shared with run_claude_opus.py: stops at US$6.2 across both Opus arms."""
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import swapgen as g
HERE = Path(__file__).parent; OUT = HERE / 'results' / 'claude_opus_extract'; OUT.mkdir(parents=True, exist_ok=True)
MODEL, CHUNK, IN_P, OUT_P, STOP = 'claude-opus-5-5', 50, 4e-6, 20e-6, 6.2
import os; IDS = set(os.environ['SWAP_IDS'].split(','))
INSTR = '각 줄에서 방을 바꾼 두 사람을 줄 순서대로 적으세요. 한 줄에 한 쌍입니다.'
SCHEMA = {'type': 'object', 'properties': {'pairs': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string', 'enum': g.NAMES}}}},
          'required': ['pairs'], 'additionalProperties': False}


def lane_spent():
    tot = 0.0
    for d in ('claude', 'claude_extract'):
        for f in (HERE / 'results' / d).glob('*.json'):
            r = json.loads(f.read_text())
            for u in ([r['usage']] if 'usage' in r and 'chunks' not in r else [c['usage'] for c in r['chunks']]):
                tot += u['input_tokens'] * IN_P + u['output_tokens'] * OUT_P
    return tot

def opus_spent():
    tot = 0.0
    for d in ('claude_opus', 'claude_opus_extract'):
        for f in (HERE / 'results' / d).glob('*.json'):
            r = json.loads(f.read_text())
            for u in ([r['usage']] if 'chunks' not in r else [c['usage'] for c in r['chunks']]):
                tot += u['input_tokens'] * IN_P + u['output_tokens'] * OUT_P
    return tot


def chunk_call(lines):
    msg = {'role': 'user', 'content': '\n'.join(lines) + '\n\n' + INSTR}
    cmd = ['ant', 'messages', 'create', '--model', MODEL, '--max-tokens', '4000', '--thinking', '{"type": "between_tools"}',
           '--output-config', json.dumps({'effort': 'low', 'format': {'type': 'json_schema', 'schema': SCHEMA}}, ensure_ascii=False),
           '--message', json.dumps(msg, ensure_ascii=False)]
    t0 = time.perf_counter(); p = subprocess.run(cmd, capture_output=True, text=True); wall = time.perf_counter() - t0
    r = json.loads(p.stdout)
    pairs = json.loads(r['content'][0]['text'])['pairs']
    return {'wall_s': wall, 'usage': {k: r['usage'][k] for k in ('input_tokens', 'output_tokens')}, 'pairs': pairs, 'n_lines': len(lines),
            'stop_reason': r['stop_reason']}


def run_doc(m):
    f = OUT / f"{m['id']}.json"
    if f.exists(): return
    text = (HERE / 'docs' / f"{m['id']}.txt").read_text(encoding='utf-8')
    ev_lines = [l for l in text.split('\n') if '방을 바꿨습니다.' in l]
    chunks = [ev_lines[i:i + CHUNK] for i in range(0, len(ev_lines), CHUNK)]
    worst = len(chunks) * (len(text) / len(chunks) * IN_P + 4000 * OUT_P)
    if opus_spent() + worst > STOP:
        print('STOP: spend guard', m['id'], flush=True); sys.exit(1)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(chunks)) as ex:
        res = list(ex.map(chunk_call, chunks))
    pairs, bad = [], 0
    for c in res:
        ok = len(c['pairs']) == c['n_lines'] and all(len(p) == 2 and p[0] != p[1] for p in c['pairs'])
        bad += not ok
        pairs += [tuple(g.NAMES.index(x) for x in p) for p in c['pairs'] if len(p) == 2]
    pred = g.execute(pairs); wall = time.perf_counter() - t0
    ans = '\n'.join(f'{g.NAMES[i]}: {pred[i]}' for i in range(5))
    per = sum(int(set(a) == set(b)) for a, b in zip(pairs, m['events'])) / len(m['events']) if len(pairs) == len(m['events']) else None
    u = {'input_tokens': sum(c['usage']['input_tokens'] for c in res), 'output_tokens': sum(c['usage']['output_tokens'] for c in res)}
    rec = {'id': m['id'], 'length': m['length'], 'seed': m['seed'], 'model': MODEL, 'wall_s': wall,
           'serial_chunk_wall_s': sum(c['wall_s'] for c in res), 'n_chunks': len(chunks), 'bad_chunks': bad,
           'pairs_returned': len(pairs), 'per_event_acc': per, 'usage': u, 'thinking_tokens': 0,
           'cost_usd': u['input_tokens'] * IN_P + u['output_tokens'] * OUT_P, 'chunks': res, 'answer': ans, 'truth': m['truth'],
           **g.score(ans, m['truth'])}
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(m['id'], f"wall={wall:.1f}s serial={rec['serial_chunk_wall_s']:.1f}s chunks={len(chunks)} bad={bad} per_event={per} out={u['output_tokens']} ${rec['cost_usd']:.3f} correct={rec['people_correct']} opus=${opus_spent():.3f}", flush=True)


for m in json.loads((HERE / 'manifest.json').read_text()):
    if m['id'] in IDS: run_doc(m)
print('DONE opus $%.3f' % opus_spent())
