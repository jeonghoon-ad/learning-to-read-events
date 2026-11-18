"""CODE arm: the 'LLM writes a parser' path. Regex parser + serial executor over the same document text.
Timing: in-process perf_counter over parse+execute, median of 200 repeats per document (excludes writing the parser)."""
import json, re, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import swapgen as g
HERE = Path(__file__).parent; PAT = re.compile(r'(\S+?)와\(과\) (\S+?)가 방을 바꿨습니다')


def solve(text):
    room = {}
    for i, m in enumerate(re.findall(r'(\S+?)는 (\d)호실', text.split('\n')[0])):
        room[m[0]] = int(m[1])
    for a, b in PAT.findall(text):
        room[a], room[b] = room[b], room[a]
    return '\n'.join(f'{k}: {v}' for k, v in room.items())


rows = []
for m in json.loads((HERE / 'manifest.json').read_text()):
    text = (HERE / 'docs' / f"{m['id']}.txt").read_text(encoding='utf-8')
    ts = []
    for _ in range(200):
        t0 = time.perf_counter(); ans = solve(text); ts.append(time.perf_counter() - t0)
    rows.append({'id': m['id'], 'length': m['length'], 'seed': m['seed'], 'wall_s': statistics.median(ts), 'answer': ans, **g.score(ans, m['truth'])})
(HERE / 'results').mkdir(exist_ok=True); (HERE / 'results' / 'code.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
for r in rows: print(r['id'], r['people_correct'], f"{r['wall_s']*1e3:.3f} ms")
