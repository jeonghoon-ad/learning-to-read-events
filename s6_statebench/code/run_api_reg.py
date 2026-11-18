"""IC register arms through Anthropic and OpenAI APIs, with a per-chunk line-count check.
Derived from run_ic_reg.py. Changes in the measurement path:
  (1) the prompt asks for exactly one output line per numbered input line (`<n>|none` for no change);
  (2) per chunk, the returned line numbers must be exactly the numbers sent, each once, every line parseable;
      otherwise the chunk is re-requested once, and if it fails again the item is abstained;
  (3) OpenAI models (Responses API for reasoning models; Chat Completions with logprobs for gpt-4.1-mini).
Usage: python run_api_reg.py --model M --cells solvable__tokens_32k [--limit N] --tag T
"""
import argparse, asyncio, json, math, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]  # sb_register, sb_labels and pointer/ ship next to this file
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from pointer.labels import label_reads, normal
from pointer.register import execute
from sb_register import CONST, COPY, INC, NOOP

# US$ per token (input, output). Anthropic and OpenAI published per-token prices, standard tier.
PRICES = {'claude-opus-5-5': (4.0e-6, 20.0e-6), 'claude-sonnet-5-5': (2.0e-6, 10.0e-6),
          'claude-haiku-5-5': (0.10e-6, 0.50e-6),  # prompts up to 100K tokens
          'gpt-6-sol': (2.0e-6, 10.0e-6), 'gpt-6.1-terra': (1.0e-6, 5.0e-6), 'gpt-4.1-mini': (0.4e-6, 1.6e-6),
          'gpt-6-luna': (0.10e-6, 0.50e-6)}  # standard tier
PROVIDER = lambda m: 'anthropic' if m.startswith('claude') else 'openai'
STOPS = {'anthropic': 13.0, 'openai': 22.0}
CHUNK = 200
STATE_PAIR_RE = re.compile(r"([A-Za-z_]\w*)\s*(?:\?=|=|:)\s*(-?\d+)")  # Ai2 eval_api.py, verbatim
INSTRUCTION = (
    "Below are numbered lines from a text that describes changes to the state of some entities. "
    "Write exactly one output line for every numbered line, in order:\n"
    "<n>|set|<target>|<value>    when the line gives <target> a value written in the line\n"
    "<n>|copy|<target>|<source>  when the line gives <target> the current value of <source>\n"
    "<n>|add|<target>|<amount>   when the line increases <target> by <amount> (ignore any wrap-around)\n"
    "<n>|none                    when the line changes no state\n"
    "<n>|unsure                  when you cannot tell what the line does\n"
    "Write each name and number exactly as it appears in the line. Output only these lines."
)
LINE_RE = re.compile(r'^(\d+)\|(set|copy|add)\|([^|]+)\|([^|]+)$')
BARE_RE = re.compile(r'^(\d+)\|(none|unsure)$')


class Budget:
    """Spend ledger shared by every process (one JSON line per billed attempt); checked before each call."""
    def __init__(self, ledger): self.ledger = ledger
    def total(self, prov):
        if not self.ledger.exists(): return 0.0
        return sum(json.loads(l)['cost'] for l in open(self.ledger) if l.strip() and json.loads(l)['provider'] == prov)
    @property
    def spent(self): return {p: round(self.total(p), 4) for p in STOPS}
    def check(self, prov):
        t = self.total(prov)
        if t >= STOPS[prov]: raise RuntimeError(f'{prov} hard stop at US${t:.3f} (ledger {self.ledger.name})')
    def add(self, prov, model, cost):
        with open(self.ledger, 'a') as f: f.write(json.dumps({'provider': prov, 'model': model, 'cost': cost, 't': time.time()}) + '\n')


class Clients:
    def __init__(self):
        self.a = self.o = None
    def anthropic(self):
        if self.a is None:
            import anthropic; self.a = anthropic.AsyncAnthropic(max_retries=4)
        return self.a
    def openai(self):
        if self.o is None:
            import openai
            self.o = openai.AsyncOpenAI(api_key=open(os.path.expanduser('~/.config/openai/api_key')).read().strip(), max_retries=4)
        return self.o


async def call_once(clients, model, prompt):
    """Return dict(text, stop, tin, tout, logprob_min) or raise."""
    if PROVIDER(model) == 'anthropic':
        async with clients.anthropic().messages.stream(model=model, max_tokens=16_384, thinking={'type': 'adaptive'},
                                                       output_config={'effort': 'medium'},
                                                       messages=[{'role': 'user', 'content': prompt}]) as s:
            m = await s.get_final_message()
        u = m.usage
        return {'text': ''.join(b.text for b in m.content if b.type == 'text'),
                'stop': m.stop_reason if m.stop_reason in ('refusal', 'max_tokens') else None,
                'tin': u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0),
                'tout': u.output_tokens, 'logprob_min': None}
    o = clients.openai()
    if model == 'gpt-4.1-mini':
        r = await o.chat.completions.create(model=model, max_completion_tokens=16_384, logprobs=True,
                                            messages=[{'role': 'user', 'content': prompt}])
        ch = r.choices[0]; lps = [t.logprob for t in (ch.logprobs.content or [])] if ch.logprobs else []
        return {'text': ch.message.content or '', 'stop': 'max_tokens' if ch.finish_reason == 'length' else (None if ch.finish_reason == 'stop' else ch.finish_reason),
                'tin': r.usage.prompt_tokens, 'tout': r.usage.completion_tokens,
                'logprob_min': min(lps) if lps else None}
    r = await o.responses.create(model=model, max_output_tokens=16_384, reasoning={'effort': 'medium'}, input=prompt)
    stop = None
    if r.status != 'completed':
        stop = 'max_tokens' if getattr(r.incomplete_details, 'reason', '') == 'max_output_tokens' else (r.status or 'incomplete')
    return {'text': r.output_text or '', 'stop': stop, 'tin': r.usage.input_tokens, 'tout': r.usage.output_tokens, 'logprob_min': None}


def check_chunk(text, a, n):
    """Line-count check: exactly the numbers a+1..a+n, each once, every non-empty line parseable. Returns (ok, parsed, reasons)."""
    parsed = {}; reasons = []
    for line in text.splitlines():
        line = line.strip().strip('`')
        if not line: continue
        m = LINE_RE.match(line); b = BARE_RE.match(line)
        if not (m or b): reasons.append('unparsed line'); continue
        num = int((m or b).group(1))
        if num in parsed: reasons.append('repeated line number'); continue
        parsed[num] = m.groups()[1:] if m else (b.group(2),)
    if set(parsed) != set(range(a + 1, a + n + 1)): reasons.append(f'line count {len(parsed)} != {n} sent')
    return (not reasons), parsed, reasons


async def chunk_call(clients, model, budget, sem, prompt, a, n, raw_path):
    prov = PROVIDER(model); pin, pout = PRICES[model]; attempts = []
    for attempt in (1, 2):
        async with sem:
            budget.check(prov); t0 = time.time()
            try:
                r = await call_once(clients, model, prompt); err = None
            except Exception as e:
                r = {'text': '', 'stop': None, 'tin': 0, 'tout': 0, 'logprob_min': None}; err = type(e).__name__ + ': ' + str(e)[:300]
            lat = time.time() - t0
        cost = r['tin'] * pin + r['tout'] * pout; budget.add(prov, model, cost)
        hard = bool(err or r['stop'])  # error, refusal or truncation: abstain at once, no re-request (protocol)
        ok, parsed, reasons = (False, {}, [err or r['stop']]) if hard else check_chunk(r['text'], a, n)
        attempts.append({'attempt': attempt, 'ok': ok, 'reasons': reasons, 'cost': cost, 'latency_s': lat, 'tin': r['tin'], 'tout': r['tout'],
                         'logprob_min': r['logprob_min'], 'text': r['text']})
        if ok or hard: break
    raw_path.write_text(json.dumps({'prompt': prompt, 'attempts': attempts}))
    last = attempts[-1]
    return {'ok': last['ok'], 'parsed': parsed if last['ok'] else {}, 'reasons': [] if last['ok'] else last['reasons'],
            'retried': len(attempts) > 1, 'first_reasons': attempts[0]['reasons'],
            'cost': sum(x['cost'] for x in attempts), 'latency_s': sum(x['latency_s'] for x in attempts),
            'tin': sum(x['tin'] for x in attempts), 'tout': sum(x['tout'] for x in attempts),
            'logprob_min': min([x['logprob_min'] for x in attempts if x['logprob_min'] is not None], default=None)}


def numbered_lines(doc, tok):
    """Numbering units as sent (review 20:10 fix). Every non-empty source line after the task line is one unit, so a comment
    line is never merged into the statement on the next line. A line holding several statement terminators that does not
    begin with the comment marker `//` (items rendered on one line) is split at its terminators, one unit per statement.
    A unit maps to the last statement terminator on it (comment lines with internal `;` keep their text whole).
    Returns (task_line, units, unit_to_read, n_reads)."""
    import bisect
    p = doc['prefix']; ids, pos, _ = label_reads(p, 'integer-code-modular', tok)
    off = tok.encode(p, add_special_tokens=False).offsets
    starts = []; texts = []; c = 0
    for raw in p.split('\n'):
        starts.append(c); texts.append(raw); c += len(raw) + 1
    on_line = {}
    for r, i in enumerate(pos):
        e = off[i][1] - 1; k = bisect.bisect_right(starts, e) - 1
        on_line.setdefault(k, []).append((r, e - starts[k]))
    nonempty = [k for k, t in enumerate(texts) if t.strip()]
    task_idx = nonempty[0]
    units = []; unit_to_read = {}
    for k in nonempty[1:]:
        line = texts[k]; reads = on_line.get(k, [])
        if len(reads) > 1 and not line.lstrip().startswith('//'):
            prev = 0
            for r, col in reads:
                units.append(line[prev:col + 1].strip()); unit_to_read[len(units)] = r; prev = col + 1
            if line[prev:].strip(): units.append(line[prev:].strip())
        else:
            units.append(line.strip())
            if reads: unit_to_read[len(units)] = reads[-1][0]
    mapped = set(unit_to_read.values()) | {r for r, _ in on_line.get(task_idx, [])}
    unmapped = [r for r in range(len(pos)) if r not in mapped]
    if any(not texts[bisect.bisect_right(starts, off[pos[r]][1] - 1) - 1].lstrip().startswith('//') for r in unmapped):
        raise ValueError('a statement outside a comment line has no unit')
    return texts[task_idx].strip(), units, unit_to_read, len(pos)


def to_reads(parsed_all, line_to_read, n_reads):
    reads = [{'op': NOOP, 'tgt': None, 'src': None, 'val': None} for _ in range(n_reads)]; unsure = 0; off_statement = 0
    for num, g in parsed_all.items():
        if g[0] == 'none': continue
        if g[0] == 'unsure': unsure += 1; continue
        if num not in line_to_read: off_statement += 1; continue  # a transition claimed for a non-statement line (e.g. a comment); counted, not composed
        op, t, x = g[0], normal(g[1]), normal(g[2])
        reads[line_to_read[num]] = ({'op': CONST, 'tgt': t, 'src': None, 'val': x} if op == 'set' else
                          {'op': COPY, 'tgt': t, 'src': x, 'val': None} if op == 'copy' else
                          {'op': INC, 'tgt': t, 'src': None, 'val': x})
    return reads, unsure, off_statement


def exact(reads, doc):
    state, _ = execute(reads, 'integer-code-modular', 100)
    gold = dict(STATE_PAIR_RE.findall(doc['choices'][doc['correct_choice_index']]))
    hits = sum(state.get(k) == v for k, v in gold.items())
    return hits == len(gold), hits, len(gold)


async def run_item(clients, model, budget, sem, doc, tok, raw):
    head, st, line_to_read, n_reads = numbered_lines(doc, tok)
    chunks = [(a, st[a:a + CHUNK]) for a in range(0, len(st), CHUNK)]
    async def one(k, a, lines):
        body = '\n'.join(f'{a + j + 1}. {s}' for j, s in enumerate(lines))
        return await chunk_call(clients, model, budget, sem, f"{INSTRUCTION}\n\nTask: {head}\n\n{body}", a, len(lines), raw / f"{doc['example_id']}_{k:02d}.json")
    res = await asyncio.gather(*(one(k, a, l) for k, (a, l) in enumerate(chunks)))
    parsed_all = {}
    for r in res: parsed_all.update(r['parsed'])
    reads, unsure, off_statement = to_reads(parsed_all, line_to_read, n_reads)
    reasons = sorted({x for r in res for x in r['reasons']}) + (['unsure'] if unsure else [])
    ok, hits, nvar = exact(reads, doc)
    lp = [r['logprob_min'] for r in res if r['logprob_min'] is not None]
    return {'model': model, 'provider': PROVIDER(model), 'example_id': doc['example_id'], 'exact': ok and not reasons,
            'exact_if_not_abstained': ok, 'variables_correct': hits, 'variables': nvar, 'abstained': bool(reasons),
            'abstain_reason': reasons[:5] or None, 'chunks': len(res), 'chunks_retried': sum(r['retried'] for r in res),
            'first_attempt_reasons': sorted({x for r in res for x in r['first_reasons']})[:5],
            'lines_sent': len(st), 'statements': n_reads, 'transitions_on_non_statement_lines': off_statement, 'transitions_returned': sum(r['op'] != NOOP for r in reads),
            'cost': sum(r['cost'] for r in res), 'latency_s': sum(r['latency_s'] for r in res),
            'input_tokens': sum(r['tin'] for r in res), 'output_tokens': sum(r['tout'] for r in res),
            'logprob_min': min(lp) if lp else None}


async def main():
    global CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(PRICES)); ap.add_argument('--cells', nargs='+', required=True)
    ap.add_argument('--limit', type=int); ap.add_argument('--tag', required=True); ap.add_argument('--concurrency', type=int, default=6)
    # ops options (21:06 n=30 extension): sample folder, a separate spend ledger and its stops; defaults keep the ext2 behaviour
    ap.add_argument('--samples-dir', default='ic_samples'); ap.add_argument('--ledger', default='api_spend_ledger.jsonl')
    ap.add_argument('--stop-anthropic', type=float, default=STOPS['anthropic']); ap.add_argument('--stop-openai', type=float, default=STOPS['openai'])
    ap.add_argument('--chunk', type=int, default=CHUNK)  # diagnostic variant only; default keeps 200
    a = ap.parse_args()
    STOPS['anthropic'], STOPS['openai'] = a.stop_anthropic, a.stop_openai
    if a.chunk <= 0: raise SystemExit('chunk must be positive')
    CHUNK = a.chunk
    tok = Tokenizer.from_file(snapshot_download('Qwen/Qwen3.5-0.8B-Base', revision='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68', local_files_only=True) + '/tokenizer.json')
    out = HERE / f'api_results_{a.tag}.jsonl'
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l); done.add((r['cell'], r['model'], r['example_id']))
    budget = Budget(HERE / a.ledger); sem = asyncio.Semaphore(a.concurrency); clients = Clients()
    for cell in a.cells:
        docs = [json.loads(l) for l in open(HERE / 'raw' / a.samples_dir / f'integer-code-modular--{cell}__seed0.jsonl')]
        docs = docs[:a.limit] if a.limit else docs
        raw = HERE / 'raw' / f'api_{a.tag}' / a.model / cell; raw.mkdir(parents=True, exist_ok=True)
        tasks = [run_item(clients, a.model, budget, sem, d, tok, raw) for d in docs if (cell, a.model, d['example_id']) not in done]
        with open(out, 'a') as f:
            for coro in asyncio.as_completed(tasks):
                try: r = await coro
                except RuntimeError as e: print('STOP', e, flush=True); break
                r['cell'] = cell; f.write(json.dumps(r) + '\n'); f.flush()
                print(cell, a.model, r['example_id'][:8], 'exact', r['exact'], f"vars {r['variables_correct']}/{r['variables']}", 'abst', r['abstained'],
                      'retried', r['chunks_retried'], f"${r['cost']:.3f}", f"spent {budget.spent}", flush=True)
    print('SPENT', json.dumps(budget.spent))

if __name__ == '__main__':
    asyncio.run(main())
