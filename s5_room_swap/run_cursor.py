"""REASON-GEMINI / REASON-GROK arms through a logged-in Cursor CLI (cursor-agent -p, --mode ask = read-only,
empty scratch workspace). Timing: wall clock from subprocess start to exit. Tokens: Cursor's usage.outputTokens (no
thinking split reported). Any tool_call event in the stream is recorded."""
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import swapgen as g
HERE = Path(__file__).parent
ARM, MODEL = sys.argv[1], sys.argv[2]
OUT = HERE / 'results' / ARM; OUT.mkdir(parents=True, exist_ok=True)
WS = Path(sys.argv[3]); WS.mkdir(parents=True, exist_ok=True)


def call(item):
    f = OUT / f"{item['id']}.json"
    if f.exists(): return
    text = (HERE / 'docs' / f"{item['id']}.txt").read_text(encoding='utf-8')
    cmd = [str(Path.home() / '.local/bin/cursor-agent'), '-p', '--output-format', 'stream-json', '--mode', 'ask', '--trust',
           '--model', MODEL, '--workspace', str(WS), text]
    attempts = []
    for attempt in range(3):
        t0 = time.perf_counter(); err = ''
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, cwd=WS); out, rc, err = p.stdout, p.returncode, p.stderr
        except subprocess.TimeoutExpired as e:
            out, rc = (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or ''), 'timeout'
        wall = time.perf_counter() - t0
        ok = rc == 0 and '"type":"result"' in out
        attempts.append({'wall_s': wall, 'rc': rc, 'stderr_tail': (err or '')[-400:]})
        if ok: break
        print('retry', item['id'], rc, (err or '')[-200:].replace('\n', ' '), flush=True); time.sleep(20)
    result, usage, tools, types = None, None, [], {}
    for line in out.splitlines():
        try: e = json.loads(line)
        except json.JSONDecodeError: continue
        t = e.get('type'); types[t] = types.get(t, 0) + 1
        if t == 'result': result, usage = e.get('result'), e.get('usage')
        if 'tool' in str(t): tools.append(line[:300])
    sc = g.score(result, item['truth'])
    rec = {'id': item['id'], 'length': item['length'], 'seed': item['seed'], 'model': MODEL, 'wall_s': wall, 'returncode': rc,
           'usage': {'input_tokens': (usage or {}).get('inputTokens'), 'output_tokens': (usage or {}).get('outputTokens'),
                     'cache_read_tokens': (usage or {}).get('cacheReadTokens')},
           'thinking_tokens': (usage or {}).get('reasoningTokens'), 'attempts': attempts, 'event_types': types, 'tool_events': tools, 'answer': result, 'truth': item['truth'], **sc}
    f.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(item['id'], f"wall={wall:.1f}s out={rec['usage']['output_tokens']} rc={rc} tools={len(tools)} correct={sc['people_correct']}", flush=True)


man = json.loads((HERE / 'manifest.json').read_text())
for L in g.LENGTHS:
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(call, [m for m in man if m['length'] == L]))
print('DONE', ARM)
