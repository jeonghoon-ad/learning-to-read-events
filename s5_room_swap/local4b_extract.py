"""EXTRACT+REG with an off-the-shelf 4B served by vLLM on a rented GPU. No training.
Same documents, chunks (50 lines), prompt (plain_protocol.PLAIN_INSTR), parser (run_tinker.plain_parse: whitespace around '|'
normalised, then the unchanged PLAIN_RE/plain_call checks), count check + one re-request then abstain, register
(swapgen.execute) and scorer (swapgen.score) as the open-model (Tinker) EXTRACT+REG rows. Temperature 0, thinking off
(chat_template_kwargs enable_thinking=False). Chunks of one document are sent concurrently; documents run one after another.
Modes: preflight (3 raw chunk replies printed with their parse), full (15 documents), shuffle (counterexample document)."""
import json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import swapgen as g
import run_tinker as rt  # stdlib-only module; we use source_parts/plain_parse (no provider call)

URL = 'http://127.0.0.1:8000/v1/chat/completions'
MAX_TOKENS = 1024  # same reservation as the Tinker EXTRACT rows; a 50-line reply is ~400-600 tokens
import os
OFFSET = os.environ.get('OFFSET') == '1'  # stated-offset prompt: state each chunk's line-number range


def prompt_for(lines):
    if not OFFSET:
        return rt.extract_prompt(lines)
    a, b = int(lines[0].split('.', 1)[0]), int(lines[-1].split('.', 1)[0])
    return (f'다음은 {a}번부터 {b}번까지의 줄입니다.\n\n' + '\n'.join(lines) + '\n\n' + rt.source_parts()['PLAIN_INSTR'] +
            f' 번호는 원문의 번호 {a}부터 {b}까지를 그대로 쓰고, 1부터 다시 세지 마세요.')


def ask(model, lines):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt_for(lines)}], 'temperature': 0.0,
            'max_tokens': MAX_TOKENS, 'chat_template_kwargs': {'enable_thinking': False}}
    t0 = time.perf_counter(); r = requests.post(URL, json=body, timeout=600); wall = time.perf_counter() - t0
    r.raise_for_status(); d = r.json(); ch = d['choices'][0]
    return {'text': ch['message'].get('content') or '', 'reasoning': ch['message'].get('reasoning_content'), 'finish_reason': ch.get('finish_reason'),
            'wall_s': wall, 'usage': {'input_tokens': d['usage']['prompt_tokens'], 'output_tokens': d['usage']['completion_tokens'], 'reasoning_tokens': 0}}


def checked(model, lines):
    a = ask(model, lines); p = rt.plain_parse(lines, a['text'])
    if p['ok']:
        return {**p, 'retried': False, 'first': None, 'raw': [a]}
    b = ask(model, lines); q = rt.plain_parse(lines, b['text'])
    return {**q, 'retried': True, 'first': {'pairs': p['pairs'], 'ok': p['ok']}, 'raw': [a, b]}


def run_doc(model, doc_id, lines_all, events, truth, out):
    chunks = [lines_all[i:i + 50] for i in range(0, len(lines_all), 50)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(chunks)) as ex:
        res = list(ex.map(lambda c: checked(model, c), chunks))
    wall = time.perf_counter() - t0
    abstained = not all(c['ok'] for c in res)
    pairs = [tuple(g.NAMES.index(x) for x in p) for c in res for p in c['pairs']]
    ans = '' if abstained else '\n'.join(f'{g.NAMES[i]}: {v}' for i, v in enumerate(g.execute(pairs)))
    per = sum(int(set(a) == set(b)) for a, b in zip(pairs, events)) / len(events) if len(pairs) == len(events) else None
    us = [r['usage'] for c in res for r in c['raw']]
    u = {k: sum(x[k] for x in us) for k in ('input_tokens', 'output_tokens', 'reasoning_tokens')}
    rec = {'id': doc_id, 'length': len(events), 'model': model, 'prompt_variant': 'stated offset' if OFFSET else 'plain', 'temperature': 0.0, 'thinking': 'off (enable_thinking=False)',
           'max_tokens': MAX_TOKENS, 'wall_s': wall, 'n_chunks': len(chunks), 'chunks_retried': sum(c['retried'] for c in res),
           'chunks_abstained': sum(not c['ok'] for c in res), 'abstained': abstained, 'per_event_acc': per, 'usage': u, 'thinking_tokens': 0,
           'chunks': res, 'answer': ans, 'truth': truth, **g.score(ans, truth)}
    (out / f'{doc_id}.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(model, doc_id, f"wall={wall:.1f}s retried={rec['chunks_retried']} abstained_chunks={rec['chunks_abstained']} per_event={per} correct={rec['people_correct']}", flush=True)
    return rec


def main():
    mode, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    man = json.loads((HERE / 'manifest.json').read_text())
    out = HERE / 'out' / tag; out.mkdir(parents=True, exist_ok=True)
    if mode == 'preflight':
        m = [x for x in man if x['length'] == 100][0]; chunks = rt.chunks_for(m)
        picks = [chunks[0], chunks[1], rt.chunks_for([x for x in man if x['length'] == 300][0])[3]]
        recs = []
        for lines in picks:
            a = ask(model, lines); p = rt.plain_parse(lines, a['text'])
            recs.append({'lines': [lines[0], lines[-1]], 'raw': a, 'parse_ok': p['ok'], 'n_pairs': len(p['pairs'])})
            print('=' * 30, model, 'raw reply (first 600 chars, repr):'); print(repr(a['text'][:600])); print('... tail:', repr(a['text'][-200:]))
            print('finish', a['finish_reason'], 'usage', a['usage'], '| parse ok', p['ok'], 'pairs', len(p['pairs']), 'of', len(lines), flush=True)
        (out / 'PREFLIGHT.json').write_text(json.dumps(recs, ensure_ascii=False, indent=1))
    elif mode == 'full':
        for m in man:
            if (out / f"{m['id']}.json").exists(): continue
            lines = [l for l in rt.alone_prompt(m).split('\n') if '방을 바꿨습니다.' in l]
            run_doc(model, m['id'], lines, m['events'], m['truth'], out)
    elif mode == 'shuffle':
        # Counterexample: swap1000_s1 with its 1,000 event lines in a seeded random order, renumbered 1..1000. Scored against the
        # ORIGINAL document's truth (a reader that reads should NOT get 5/5) and against the shuffled document's own truth.
        m = [x for x in man if x['id'] == 'swap1000_s1'][0]
        ev = list(m['events']); random.Random('shuffle-counterexample').shuffle(ev)
        lines = [g.event_line(k, a, b) for k, (a, b) in enumerate(ev, 1)]
        rec = run_doc(model, 'swap1000_s1_shuffled', lines, ev, g.execute(ev), out)
        vs_orig = g.score(rec['answer'], m['truth'])
        rec2 = {**rec, 'truth_original': m['truth'], 'score_vs_original_truth': vs_orig, 'truth_shuffled': g.execute(ev)}
        (out / 'swap1000_s1_shuffled.json').write_text(json.dumps(rec2, ensure_ascii=False, indent=1))
        print('SHUFFLE vs original truth', vs_orig['people_correct'], '/5 ; vs own truth', rec['people_correct'], '/5', flush=True)


if __name__ == '__main__':
    main()
