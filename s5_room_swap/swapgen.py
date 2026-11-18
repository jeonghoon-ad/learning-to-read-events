"""Seeded Korean room-swap puzzles, serial executor, scorer. `python3 swapgen.py` regenerates docs/ and manifest.json."""
import json, random, re, sys
from pathlib import Path

NAMES = ['철수', '영희', '민수', '지아', '현우']
OPEN = '철수는 1호실, 영희는 2호실, 민수는 3호실, 지아는 4호실, 현우는 5호실에 있습니다. 다음 일이 순서대로 일어났습니다.'
NO_CODE = '코드를 실행하거나 도구를 쓰지 말고, 글을 읽고 답하세요.'
QUESTION = "지금 다섯 명은 각각 몇 호실에 있을까요? '이름: 호실' 형식으로 다섯 줄만 답하세요."
LENGTHS = (100, 300, 1000)
SEEDS = range(5)


def events(rng, n):
    return [tuple(rng.sample(range(5), 2)) for _ in range(n)]


def event_line(k, a, b):
    return f'{k}. {NAMES[a]}와(과) {NAMES[b]}가 방을 바꿨습니다.'


def document(ev, no_code=True):
    body = '\n'.join(event_line(k, a, b) for k, (a, b) in enumerate(ev, 1))
    tail = (NO_CODE + '\n' if no_code else '') + QUESTION
    return f'{OPEN}\n\n{body}\n\n{tail}'


def execute(ev):
    room = {i: i + 1 for i in range(5)}  # person -> room, serial
    for a, b in ev:
        room[a], room[b] = room[b], room[a]
    return [room[i] for i in range(5)]


def score(answer_text, truth):
    got = {}
    for name, r in re.findall(r'(철수|영희|민수|지아|현우)\s*[:：]\s*(\d)', answer_text or ''):
        got[name] = int(r)  # last occurrence wins
    correct = sum(got.get(NAMES[i]) == truth[i] for i in range(5))
    return {'parsed': {k: got.get(k) for k in NAMES}, 'people_correct': correct, 'exact': correct == 5}


def build(out):
    out = Path(out); (out / 'docs').mkdir(parents=True, exist_ok=True)
    manifest = []
    for L in LENGTHS:
        for s in SEEDS:
            ev = events(random.Random(f'eval-{L}-{s}'), L)
            name = f'swap{L}_s{s}'
            (out / 'docs' / f'{name}.txt').write_text(document(ev), encoding='utf-8')
            manifest.append({'id': name, 'length': L, 'seed': s, 'events': ev, 'truth': execute(ev)})
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    m = build(Path(__file__).parent)
    for r in m:
        print(r['id'], r['truth'])
