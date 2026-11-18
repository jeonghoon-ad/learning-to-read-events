#!/usr/bin/env python3
"""Diagnostic-only room-swap runner; no decision or experiment execution at import."""
import argparse
import ast
import fcntl
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import swapgen as g

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
LEDGER = RESULTS / 'tinker_spend.json'
STOP = 9.00  # spend ledger: cap US$10, stop US$9
CAP = 10.0
ALONE_MAX = 32768
EXTRACT_MAX = 1024  # a 50-line chunk reply 'n|name|name' is about 400-600 tokens; reserve against 1,024, not the 8,000 ceiling
MODELS = {
    'qwen': ('Qwen/Qwen3.5-397B-A17B:peft:262144', 'qwen3_5', 1.65, 7.50),
    'ultra': ('nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:peft:262144', 'nemotron3_ultra', 1.80, 6.225),
}


def source_parts():
    """Load exactly the shared plain-lines protocol (plain_protocol.py)."""
    tree = ast.parse((HERE / 'plain_protocol.py').read_text(encoding='utf-8'))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'PLAIN_INSTR', 'PLAIN_RE'} for t in node.targets):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == 'plain_call':
            selected.append(node)
    if len(selected) != 3:
        raise RuntimeError('plain protocol source changed; inspect before proceeding')
    ns = {'g': g, 'MODEL': 'local-parser-only'}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(HERE / 'plain_protocol.py'), 'exec'), ns)
    return ns


def alone_prompt(m):
    # run_openai.alone sends these file bytes directly.
    text = (HERE / 'docs' / (m['id'] + '.txt')).read_text(encoding='utf-8')
    if text != g.document(m['events']):
        raise RuntimeError('Existing document does not match the imported prompt construction: ' + m['id'])
    return text


def chunks_for(m):
    lines = [line for line in alone_prompt(m).split('\n') if '방을 바꿨습니다.' in line]
    return [lines[i:i + 50] for i in range(0, len(lines), 50)]


def extract_prompt(lines):
    return '\n'.join(lines) + '\n\n' + source_parts()['PLAIN_INSTR']


def load_renderer(model, arm):
    from transformers import AutoTokenizer
    from tinker_cookbook import model_info, renderers
    name, base, _, _ = MODELS[model]
    renderer_name = base if arm == 'alone' else base + '_disable_thinking'
    if renderer_name not in model_info.get_recommended_renderer_names(name):
        raise RuntimeError('Renderer is not listed by local model_info')
    tokenizer = AutoTokenizer.from_pretrained(name.split(':peft:')[0], local_files_only=True, trust_remote_code=False)
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=name)
    return tokenizer, renderer, renderer_name


def render(renderer, text):
    return renderer.build_generation_prompt([{'role': 'user', 'content': text}])


def cost(model, prompt, output):
    _, _, pi, po = MODELS[model]
    return (prompt * pi + output * po) / 1_000_000


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class Ledger:
    def __init__(self, path=LEDGER, stop=STOP):
        self.path = Path(path)
        self.stop = stop

    def transact(self, fn):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.path) + '.lock', 'a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {'cap_usd': CAP, 'stop_usd': self.stop, 'calls': {}}
            if data['cap_usd'] != CAP or data['stop_usd'] != self.stop:
                raise RuntimeError('Ledger policy mismatch')
            result = fn(data)
            atomic_json(self.path, data)
            return result

    def reserve(self, call_id, model, prompt_tokens, max_tokens, fingerprint):
        def change(data):
            calls = data['calls']
            if call_id in calls:
                item = calls[call_id]
                if item['fingerprint'] != fingerprint:
                    raise RuntimeError('Stored call does not match current prompt/configuration')
                if item['state'] != 'settled':
                    raise RuntimeError('Call has unresolved reservation; do not resend: ' + call_id)
                return item['response']
            if sum(c['state'] == 'reserved' for c in calls.values()) >= 5:
                raise RuntimeError('Five calls are already reserved; resume after they settle')
            reserve = cost(model, prompt_tokens, max_tokens)
            committed = sum(c['cost_usd'] if c['state'] == 'settled' else c['reserved_usd'] for c in calls.values())
            if committed + reserve > self.stop or committed + reserve > CAP:
                raise RuntimeError(f'STOP: committed ${committed:.6f} + reservation ${reserve:.6f} exceeds ${self.stop:.2f}')
            calls[call_id] = {'state': 'reserved', 'model': model, 'prompt_tokens': prompt_tokens, 'max_tokens': max_tokens,
                              'fingerprint': fingerprint, 'reserved_usd': reserve}
            return None
        return self.transact(change)

    def settle(self, call_id, response):
        def change(data):
            c = data['calls'][call_id]
            if c['state'] != 'reserved':
                raise RuntimeError('Call already settled')
            c.update(state='settled', response=response, cost_usd=cost(c['model'], c['prompt_tokens'], len(response['tokens'])))
        self.transact(change)

    def snapshot(self):
        def read(data):
            return json.loads(json.dumps(data))
        return self.transact(read)


def parse_tokens(tokenizer, renderer, tokens, arm):
    # Both thinking renderers prefill <think>; split on token IDs to count sampled tokens exactly.
    close = tokenizer.encode('</think>', add_special_tokens=False)
    stop = renderer.get_stop_sequences()
    if not close or len(close) != 1 or not all(isinstance(x, int) for x in stop):
        raise RuntimeError('Unexpected tokenizer boundary tokens')
    end = len(tokens)
    while end and tokens[end - 1] in stop:
        end -= 1
    body = tokens[:end]
    split = body.index(close[0]) if close[0] in body else None
    if arm == 'alone':
        thinking = len(body) if split is None else split
        answer_tokens = [] if split is None else body[split + 1:]
    else:
        thinking = 0 if split is None else split
        answer_tokens = body if split is None else body[split + 1:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    return {'answer': answer, 'thinking_tokens': thinking, 'answer_tokens': len(answer_tokens),
            'boundary_and_stop_tokens': len(tokens) - thinking - len(answer_tokens),
            'raw_text': tokenizer.decode(tokens, skip_special_tokens=False)}


def send(client, ledger, model, arm, call_id, text, tokenizer, renderer, max_tokens):
    import hashlib
    import tinker
    prompt = render(renderer, text)
    prompt_tokens = prompt.to_ints()
    config = json.dumps([MODELS[model][0], arm, prompt_tokens, max_tokens, renderer.get_stop_sequences()] + ([VARIANT, TEMPERATURE] if VARIANT else []), separators=(',', ':'))
    fingerprint = hashlib.sha256(config.encode()).hexdigest()
    saved = ledger.reserve(call_id, model, len(prompt_tokens), max_tokens, fingerprint)
    if saved is not None:
        return saved
    start = time.perf_counter()
    # No automatic retry: ambiguous failures remain reserved and cannot be resent.
    try:
        seq = client.sample(prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens, stop=renderer.get_stop_sequences(), temperature=TEMPERATURE)).result().sequences[0]
    except Exception:
        raise RuntimeError('Sampling failed; reservation remains unresolved and will not be resent') from None
    response = {'tokens': list(seq.tokens), 'stop_reason': seq.stop_reason, 'wall_s': time.perf_counter() - start}
    ledger.settle(call_id, response)
    return response


VARIANT = ''  # '_t0' = greedy (temperature 0) pass on the 1,000-event documents only
TEMPERATURE = 1.0


def call_id(model, arm, doc, chunk=None, attempt=0):
    return f'{model}/{arm}{VARIANT}/{doc}/' + ('alone' if chunk is None else f'chunk{chunk:03d}/attempt{attempt}')


def plain_parse(lines, response_text):
    # Run the unchanged upstream parser with a local response; no provider request.
    ns = source_parts()
    # Rule set after the raw check, before any scoring: whitespace around '|' is formatting, not content; Qwen writes '1|현우 | 지아'.
    response_text = '\n'.join(re.sub(r'\s*\|\s*', '|', l) for l in (response_text or '').split('\n'))
    ns['post'] = lambda body, timeout=600: ({'choices': [{'message': {'content': response_text}, 'finish_reason': 'stop'}]}, 0.0)
    ns['usage_of'] = lambda d: {'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0}
    return ns['plain_call'](lines)


def collect_doc(m, model, arm, client, ledger, tokenizer, renderer):
    if arm == 'alone':
        send(client, ledger, model, arm, call_id(model, arm, m['id']), alone_prompt(m), tokenizer, renderer, ALONE_MAX)
        return
    for i, lines in enumerate(chunks_for(m)):
        for attempt in range(2):
            response = send(client, ledger, model, arm, call_id(model, arm, m['id'], i, attempt), extract_prompt(lines), tokenizer, renderer, EXTRACT_MAX)
            parsed = parse_tokens(tokenizer, renderer, response['tokens'], arm)
            if plain_parse(lines, parsed['answer'])['ok']:
                break


def require_marker(out):
    if not (out / 'RAW_CHECKED').is_file():
        raise RuntimeError(f'Scoring refused: inspect three raw replies and create {out / "RAW_CHECKED"}')


def record_doc(m, model, arm, calls, tokenizer, renderer, renderer_name, out):
    require_marker(out)
    used = []
    if arm == 'alone':
        ids = [call_id(model, arm, m['id'])]
        if any(x not in calls or calls[x]['state'] != 'settled' for x in ids): return None
        used = [calls[x] for x in ids]
        p = parse_tokens(tokenizer, renderer, used[0]['response']['tokens'], arm)
        answer = p['answer']
        extra = {'finish_reason': used[0]['response']['stop_reason'], 'cap_hit': used[0]['response']['stop_reason'] == 'length'}
    else:
        parsed_chunks = []
        for i, lines in enumerate(chunks_for(m)):
            first_id = call_id(model, arm, m['id'], i, 0)
            if first_id not in calls or calls[first_id]['state'] != 'settled': return None
            first = calls[first_id]; used.append(first)
            p = parse_tokens(tokenizer, renderer, first['response']['tokens'], arm)
            parsed = plain_parse(lines, p['answer'])
            first_rec = None
            if not parsed['ok']:
                second_id = call_id(model, arm, m['id'], i, 1)
                if second_id not in calls or calls[second_id]['state'] != 'settled': return None
                second = calls[second_id]; used.append(second)
                first_rec = {'pairs': parsed['pairs'], 'usage': usage(first, p), 'wall_s': first['response']['wall_s']}
                p = parse_tokens(tokenizer, renderer, second['response']['tokens'], arm)
                parsed = plain_parse(lines, p['answer'])
                current = second
            else:
                current = first
            parsed.update(retried=first_rec is not None, first=first_rec, usage=usage(current, p),
                          wall_s=current['response']['wall_s'], finish_reason=current['response']['stop_reason'])
            parsed_chunks.append(parsed)
        abstained = not all(c['ok'] for c in parsed_chunks)
        pairs = [tuple(g.NAMES.index(x) for x in p) for c in parsed_chunks for p in c['pairs']]
        answer = '' if abstained else '\n'.join(f'{g.NAMES[i]}: {v}' for i, v in enumerate(g.execute(pairs)))
        per = sum(int(set(a) == set(b)) for a, b in zip(pairs, m['events'])) / len(m['events']) if len(pairs) == len(m['events']) else None
        extra = {'chunks': parsed_chunks, 'n_chunks': len(parsed_chunks), 'chunks_retried': sum(c['retried'] for c in parsed_chunks),
                 'abstained': abstained, 'per_event_acc': per,
                 'cap_hit': any(c['response']['stop_reason'] == 'length' for c in used)}
    ps = [parse_tokens(tokenizer, renderer, c['response']['tokens'], arm) for c in used]
    us = [usage(c, p) for c, p in zip(used, ps)]
    u = {k: sum(x[k] for x in us) for k in ('input_tokens', 'output_tokens', 'reasoning_tokens')}
    return {'id': m['id'], 'length': m['length'], 'seed': m['seed'], 'model': MODELS[model][0], 'renderer': renderer_name,
            'arm': arm, 'max_tokens': ALONE_MAX if arm == 'alone' else EXTRACT_MAX,
            'wall_s': sum(c['response']['wall_s'] for c in used), 'usage': u, 'thinking_tokens': u['reasoning_tokens'],
            'answer_tokens': sum(p['answer_tokens'] for p in ps), 'cost_usd': sum(c['cost_usd'] for c in used),
            'answer': answer, 'truth': m['truth'], **extra, **g.score(answer, m['truth'])}


def usage(c, p):
    return {'input_tokens': c['prompt_tokens'], 'output_tokens': len(c['response']['tokens']), 'reasoning_tokens': p['thinking_tokens']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--arm', choices=['alone', 'extract'], required=True)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--real', action='store_true')
    action.add_argument('--dry', action='store_true')
    action.add_argument('--print-raw', type=int, metavar='N')
    action.add_argument('--score', action='store_true')
    ap.add_argument('--greedy-1000', action='store_true', help='temperature 0, 1,000-event documents only (ALONE)')
    args = ap.parse_args()
    man = json.loads((HERE / 'manifest.json').read_text())
    global VARIANT, TEMPERATURE
    if args.arm == 'extract':
        TEMPERATURE = 0.0  # EXTRACT+REG at temperature 0, reasoning off
    if args.greedy_1000:
        if args.arm != 'alone': raise SystemExit('--greedy-1000 is for the ALONE arm')
        VARIANT, TEMPERATURE = '_t0', 0.0
        man = [m for m in man if m['length'] == 1000]
    tokenizer, renderer, renderer_name = load_renderer(args.model, args.arm)
    if args.dry:
        counts = [len(render(renderer, alone_prompt(m)).to_ints()) for m in man] if args.arm == 'alone' else [len(render(renderer, extract_prompt(ch)).to_ints()) for m in man for ch in chunks_for(m)]
        attempts = 1 if args.arm == 'alone' else 2
        limit = ALONE_MAX if args.arm == 'alone' else EXTRACT_MAX
        print(json.dumps({'model': args.model, 'arm': args.arm, 'renderer': renderer_name, 'documents': len(man),
            'calls_max': len(counts)*attempts, 'prompt_tokens_max': sum(counts)*attempts,
            'sample_tokens_max': len(counts)*attempts*limit,
            'upper_bound_usd': sum(cost(args.model, n, limit) for n in counts)*attempts}, indent=2))
        return
    out = RESULTS / f'tinker_{args.model}_{args.arm}{VARIANT}'
    if args.score: require_marker(out)
    ledger = Ledger()
    if args.real:
        import tinker
        key = Path('~/.config/tinker/api_key').expanduser().read_text().strip()
        try:
            service = tinker.ServiceClient(api_key=key)
            client = service.create_sampling_client(base_model=MODELS[args.model][0])
        except Exception:
            raise RuntimeError('Tinker client initialization failed') from None
        finally:
            del key
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(collect_doc, m, args.model, args.arm, client, ledger, tokenizer, renderer) for m in man]
            errors = []
            for future in futures:
                try: future.result()
                except Exception as e: errors.append(str(e))
            if errors: raise RuntimeError('\n'.join(sorted(set(errors))))
        return
    calls = ledger.snapshot()['calls']
    if args.print_raw is not None:
        if args.print_raw < 1: raise ValueError('N must be positive')
        found = 0
        for m in man:
            prefix = f'{args.model}/{args.arm}{VARIANT}/{m["id"]}/'
            for cid in sorted(k for k in calls if k.startswith(prefix)):
                c = calls[cid]
                if c['state'] != 'settled': continue
                p = parse_tokens(tokenizer, renderer, c['response']['tokens'], args.arm)
                if args.arm == 'alone':
                    parsed = {'parsed_names': dict(re.findall(r'(철수|영희|민수|지아|현우)\s*[:：]\s*(\d)', p['answer']))}
                else:
                    index = int(cid.split('/')[-2].replace('chunk', ''))
                    parsed = plain_parse(chunks_for(m)[index], p['answer'])
                print(json.dumps({'call': cid, 'stop_reason': c['response']['stop_reason'], **p, 'parse_result': parsed}, ensure_ascii=False, indent=1))
                found += 1
                if found >= args.print_raw: return
        print(f'Printed {found} stored replies; requested {args.print_raw}.')
        return
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for m in man:
        rec = record_doc(m, args.model, args.arm, calls, tokenizer, renderer, renderer_name, out)
        if rec is not None:
            atomic_json(out / (m['id'] + '.json'), rec)
            count += 1
    print(f'Scored {count}/{len(man)} complete documents in {out}')


if __name__ == '__main__':
    try: main()
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
