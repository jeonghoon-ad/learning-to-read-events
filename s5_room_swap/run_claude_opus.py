"""REASON-CLAUDE arm: Claude Opus 5.5 (copy of run_claude.py; only model, prices, output dir, guard and item filter differ), adaptive thinking, via the `ant` CLI (active OAuth profile; no credentials touched).
Timing: wall clock from subprocess start to stream end. Spend guard: worst-case reservation, hard stop at US$6.2 across both Opus arms (swap lane cap US$8 +20%, minus US$3.355 already spent by the Sonnet arms)."""
import json, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import swapgen as g

HERE = Path(__file__).parent; OUT = HERE / 'results' / 'claude_opus'; OUT.mkdir(parents=True, exist_ok=True)
MODEL, MAX_TOK, IN_P, OUT_P, STOP = 'claude-opus-5-5', 64000, 4e-6, 20e-6, 6.2
import os; IDS = set(os.environ['SWAP_IDS'].split(','))
lock = threading.Lock(); state = {'spent': 0.0, 'reserved': 0.0}


def spent_so_far():
    tot = 0.0
    for f in OUT.glob('*.json'):
        u = json.loads(f.read_text())['usage']; tot += u['input_tokens'] * IN_P + u['output_tokens'] * OUT_P
    return tot

def opus_spent():
    tot = 0.0
    for d in ('claude_opus', 'claude_opus_extract'):
        for f in (HERE / 'results' / d).glob('*.json'):
            r = json.loads(f.read_text())
            for u in ([r['usage']] if 'chunks' not in r else [c['usage'] for c in r['chunks']]):
                tot += u['input_tokens'] * IN_P + u['output_tokens'] * OUT_P
    return tot


def call(item):
    f = OUT / f"{item['id']}.json"
    if f.exists():
        return json.loads(f.read_text())
    text = (HERE / 'docs' / f"{item['id']}.txt").read_text(encoding='utf-8')
    worst = len(text) * IN_P + MAX_TOK * OUT_P  # chars >= tokens bound for input
    while True:
        with lock:
            if state['spent'] + state['reserved'] + worst <= STOP:
                state['reserved'] += worst; break
            if state['reserved'] == 0:
                print('STOP: spend guard', state, flush=True); return None
        time.sleep(5)
    cmd = ['ant', 'messages', 'create', '--stream', '--format', 'jsonl', '--model', MODEL, '--max-tokens', str(MAX_TOK),
           '--thinking', '{type: adaptive}', '--message', json.dumps({'role': 'user', 'content': text}, ensure_ascii=False)]
    t0 = time.perf_counter(); p = subprocess.run(cmd, capture_output=True, text=True); wall = time.perf_counter() - t0
    answer, usage, stop, inp = [], None, None, None
    for line in p.stdout.splitlines():
        try: e = json.loads(line)
        except json.JSONDecodeError: continue
        if e.get('type') == 'message_start': inp = e['message']['usage']['input_tokens']
        elif e.get('type') == 'content_block_delta' and e['delta'].get('type') == 'text_delta': answer.append(e['delta']['text'])
        elif e.get('type') == 'message_delta': usage, stop = e['usage'], e['delta'].get('stop_reason')
    with lock: state['reserved'] -= worst
    if usage is None:
        print('FAIL', item['id'], p.returncode, p.stderr[-500:], flush=True); return None
    usage['input_tokens'] = usage.get('input_tokens') or inp
    ans = ''.join(answer); sc = g.score(ans, item['truth'])
    rec = {'id': item['id'], 'length': item['length'], 'seed': item['seed'], 'model': MODEL, 'wall_s': wall, 'stop_reason': stop,
           'usage': usage, 'thinking_tokens': usage.get('output_tokens_details', {}).get('thinking_tokens'), 'answer': ans,
           'truth': item['truth'], **sc}
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    with lock: state['spent'] += usage['input_tokens'] * IN_P + usage['output_tokens'] * OUT_P
    print(item['id'], f"wall={wall:.1f}s out={usage['output_tokens']} think={rec['thinking_tokens']} stop={stop} correct={sc['people_correct']} spent=${state['spent']:.3f}", flush=True)
    return rec


if __name__ == '__main__':
    state['spent'] = opus_spent()
    man = json.loads((HERE / 'manifest.json').read_text())
    for L in g.LENGTHS:
        with ThreadPoolExecutor(5) as ex:
            list(ex.map(call, [m for m in man if m['length'] == L and m['id'] in IDS]))
    print('TOTAL spent $%.3f' % state['spent'])
