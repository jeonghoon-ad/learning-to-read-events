"""Hardening arms: ALONE (Ai2's generative prompt, verbatim) and CODE-TOOL (model writes a program; we run it
locally in a sandbox). Same items as the n=30 register cells. Records per item: wall time, output tokens, reasoning tokens (OpenAI), cap hit.
Usage: python run_hardening.py --arm alone|alone_large|code --model M --cells C [C...] [--limit N] --ledger L --stop-anthropic X --stop-openai Y
"""
import argparse, asyncio, json, os, re, resource, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRICES = {'claude-opus-5-5': (4.0e-6, 20.0e-6), 'gpt-6-sol': (2.0e-6, 10.0e-6), 'gpt-6-luna': (0.10e-6, 0.50e-6),
          'claude-sonnet-5-5': (2.0e-6, 10.0e-6), 'claude-haiku-5-5': (0.10e-6, 0.50e-6)}  # 07:17: same values as reviewed in run_api_reg.py
PROVIDER = lambda m: 'anthropic' if m.startswith('claude') else 'openai'
STOPS = {'anthropic': 60.0, 'openai': 40.0}
LARGE_CAP = 65_536  # alone_large: stated cap, with effort high

# --- Ai2 state-tracking-evals@6ffbc01a scripts/eval_api.py, verbatim ---
GENERATE_QUESTION = (
    "What is the final value of every variable after all of the assignments "
    "above have executed? You may reason before answering. End your reply with "
    "exactly one line of the form"
)
GENERATE_INSTRUCTION = "with each <value> replaced by that variable's final value."
STATE_PAIR_RE = re.compile(r"([A-Za-z_]\w*)\s*(?:\?=|=|:)\s*(-?\d+)")


def gold_state(doc: dict) -> dict[str, str]:
    """Final state from the correct choice, which is the dataset completion."""
    return dict(STATE_PAIR_RE.findall(doc["choices"][doc["correct_choice_index"]]))


def build_generate_prompt(doc: dict) -> str:
    template = " ".join(f"{var} ?= <value>;" for var in gold_state(doc))
    return (
        f"{doc['prefix']}\n\n{GENERATE_QUESTION}\n\n"
        f"ANSWER: {template}\n\n{GENERATE_INSTRUCTION}"
    )


def parse_state(text: str) -> dict[str, str] | None:
    """Final state from the last ANSWER line, else the last line naming values."""
    lines = text.replace("*", "").replace("`", "").splitlines()
    answer_lines = [line for line in lines if re.search(r"answer\s*:", line, re.I)]
    for line in reversed(answer_lines or lines):
        tail = re.split(r"answer\s*:", line, flags=re.I)[-1]
        pairs = STATE_PAIR_RE.findall(tail)
        if pairs:
            return dict(pairs)
    return None
# --- end Ai2 ---

CODE_INSTRUCTION = (
    "Write a Python 3 program that reads the text above from standard input, carries out every instruction in it in order, "
    "and prints exactly one line of the form\n\n{template}\n\n{instr}\n"
    "Use only the Python standard library. Reply with the program in a single ```python code block."
)
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
def sandbox_profile(workdir):
    """Deny-by-default profile (review 22:18): only the interpreter may run; reads limited to the Python install, the venv and the
    temp folder; writes only to the temp folder; no network."""
    exe = os.path.realpath(sys.executable); base = os.path.realpath(sys.base_prefix); venv = os.path.realpath(sys.prefix)
    w = os.path.realpath(workdir)
    return ('(version 1)(deny default)(import "bsd.sb")'
            f'(allow process-exec (literal "{exe}"))'
            f'(allow file-read* (subpath "{base}") (subpath "{venv}") (subpath "{w}"))'
            f'(allow file-write* (subpath "{w}"))')


def _limits():  # no imports here: preexec_fn runs between fork and exec while other threads exist (review 22:20)
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))


class Budget:
    def __init__(self, ledger): self.ledger = ledger
    def total(self, prov):
        if not self.ledger.exists(): return 0.0
        return sum(json.loads(l)['cost'] for l in open(self.ledger) if l.strip() and json.loads(l)['provider'] == prov)
    def check(self, prov):
        t = self.total(prov)
        if t >= STOPS[prov]: raise RuntimeError(f'{prov} hard stop at US${t:.3f} (ledger {self.ledger.name})')
    def add(self, prov, model, cost, note):
        with open(self.ledger, 'a') as f: f.write(json.dumps({'provider': prov, 'model': model, 'cost': cost, 't': time.time(), 'note': note}) + '\n')


_clients = {}


async def call(model, prompt, *, max_tokens, effort):
    """effort=None -> provider default (Ai2's provider sends no thinking/reasoning setting then). Returns dict."""
    if PROVIDER(model) == 'anthropic':
        import anthropic
        c = _clients.setdefault('a', anthropic.AsyncAnthropic(timeout=900, max_retries=8))  # Ai2's client settings
        req = {'model': model, 'max_tokens': max_tokens, 'messages': [{'role': 'user', 'content': prompt}]}
        if effort:
            req['thinking'] = {'type': 'adaptive'}; req['output_config'] = {'effort': effort}
        async with c.messages.stream(**req) as s:
            m = await s.get_final_message()
        u = m.usage
        return {'text': ''.join(b.text for b in m.content if b.type == 'text'), 'cap_hit': m.stop_reason == 'max_tokens',
                'refusal': m.stop_reason == 'refusal', 'tin': u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0),
                'tout': u.output_tokens, 'reasoning': None, 'resp_model': m.model, 'resp_id': m.id}
    import openai
    big = max_tokens > 16_384
    key = 'o_large' if big else 'o'
    c = _clients.setdefault(key, openai.AsyncOpenAI(api_key=open(os.path.expanduser('~/.config/openai/api_key')).read().strip(),
                                                    timeout=1800 if big else 900, max_retries=2 if big else 8))  # Ai2: 900 s / 8 retries
    req = {'model': model, 'max_output_tokens': max_tokens, 'input': prompt}
    if effort: req['reasoning'] = {'effort': effort}
    r = await c.responses.create(**req)
    det = getattr(r.usage, 'output_tokens_details', None)
    reason = getattr(r.incomplete_details, 'reason', '') if r.status == 'incomplete' else ''
    cap = reason == 'max_output_tokens'
    return {'text': r.output_text or '', 'cap_hit': cap, 'refusal': reason == 'content_filter', 'status': r.status, 'tin': r.usage.input_tokens, 'tout': r.usage.output_tokens,
            'reasoning': getattr(det, 'reasoning_tokens', None), 'resp_model': r.model, 'resp_id': r.id}


def run_code(code, program_text, timeout=120):
    with tempfile.TemporaryDirectory() as d:
        d = os.path.realpath(d); p = Path(d) / 'prog.py'; p.write_text(code)
        t0 = time.time()
        try:
            r = subprocess.run(['sandbox-exec', '-p', sandbox_profile(d), os.path.realpath(sys.executable), '-I', str(p)], input=program_text,
                               capture_output=True, text=True, timeout=timeout, cwd=d, preexec_fn=_limits, env={'PATH': '/usr/bin:/bin'})
            return {'rc': r.returncode, 'stdout': r.stdout[-4000:], 'stderr': r.stderr[-2000:], 'exec_s': time.time() - t0, 'timeout': False}
        except subprocess.TimeoutExpired:
            return {'rc': None, 'stdout': '', 'stderr': '', 'exec_s': time.time() - t0, 'timeout': True}


async def run_item(arm, model, budget, sem, doc, raw):
    prov = PROVIDER(model); pin, pout = PRICES[model]; gold = gold_state(doc)
    if arm == 'code':
        template = " ".join(f"{var} ?= <value>;" for var in gold)
        prompt = f"{doc['prefix']}\n\n" + CODE_INSTRUCTION.format(template=template, instr=GENERATE_INSTRUCTION)
        max_tokens, effort = 16_384, None
    elif arm == 'alone':
        prompt, max_tokens, effort = build_generate_prompt(doc), 16_384, None
    else:
        prompt, max_tokens, effort = build_generate_prompt(doc), LARGE_CAP, 'high'
    async with sem:
        budget.check(prov); t0 = time.time()
        try:
            r = await call(model, prompt, max_tokens=max_tokens, effort=effort); err = None
        except Exception as e:
            r = {'text': '', 'cap_hit': False, 'refusal': False, 'tin': 0, 'tout': 0, 'reasoning': None, 'status': 'error'}; err = type(e).__name__ + ': ' + str(e)[:300]
        wall = time.time() - t0
    cost = r['tin'] * pin + r['tout'] * pout; budget.add(prov, model, cost, f'hardening {arm}')
    rec = {'arm': arm, 'model': model, 'provider': prov, 'example_id': doc['example_id'], 'error': err, 'cap_hit': r['cap_hit'], 'refusal': r['refusal'],
           'status': r.get('status', 'stop_reason=' + ('max_tokens' if r['cap_hit'] else 'refusal' if r['refusal'] else 'end')),
           'cost': cost, 'wall_s': wall, 'input_tokens': r['tin'], 'output_tokens': r['tout'], 'reasoning_tokens': r['reasoning'],
           'max_tokens': max_tokens, 'effort': effort or 'provider default', 'resp_model': r.get('resp_model'), 'resp_id': r.get('resp_id')}
    if arm == 'code':
        m = CODE_RE.findall(r['text'])
        if err or not m:
            pred = None; ex = {'rc': None, 'timeout': False, 'exec_s': 0.0, 'stderr': '', 'stdout': ''}
            rec['code_status'] = 'error' if err else ('cap_hit' if r['cap_hit'] else 'no code block')
        else:
            ex = await asyncio.to_thread(run_code, m[-1], doc['prefix'])
            pred = parse_state(ex['stdout']) if ex['rc'] == 0 else None  # exit code 0 required (stated)
            rec['code_status'] = 'timeout' if ex['timeout'] else ('ran' if ex['rc'] == 0 else f"exit {ex['rc']}")
            # report-only: does the program compute from its input? Append one final assignment to a gold variable and rerun;
            # a program that executes its input must then print the new value for that variable.
            if ex['rc'] == 0 and gold:
                var = next(iter(gold)); newval = str((int(gold[var]) + 1) % 100)
                ex2 = await asyncio.to_thread(run_code, m[-1], doc['prefix'].rstrip('\n') + f'\n{var} <- {newval};\n')
                rec['output_changes_on_perturbation'] = (parse_state(ex2['stdout']) or {}).get(var) == newval
            else:
                rec['output_changes_on_perturbation'] = None
        rec['exec_s'] = ex['exec_s']; rec['wall_s_total'] = wall + ex['exec_s']
    else:
        pred = parse_state(r['text']) if not err else None
    matched = sum((pred or {}).get(k) == v for k, v in gold.items())
    rec['correct'] = (matched == len(gold)) and not err  # Ai2 scores what parses; cap hit and refusal are recorded, not forced wrong
    rec['variable_accuracy'] = matched / len(gold)
    (raw / f"{doc['example_id']}.json").write_text(json.dumps({**rec, 'prompt_len': len(prompt), 'text': r['text'], 'exec': ex if arm == 'code' else None}))
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['alone', 'alone_large', 'code']); ap.add_argument('--model', required=True, choices=list(PRICES))
    ap.add_argument('--cells', nargs='+', required=True); ap.add_argument('--limit', type=int); ap.add_argument('--samples-dir', default='ic_samples_n30')
    ap.add_argument('--ledger', required=True); ap.add_argument('--stop-anthropic', type=float, required=True); ap.add_argument('--stop-openai', type=float, required=True)
    ap.add_argument('--concurrency', type=int, default=4)
    a = ap.parse_args(); STOPS['anthropic'], STOPS['openai'] = a.stop_anthropic, a.stop_openai
    out = HERE / 'hardening_results.jsonl'; done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l); done.add((r['arm'], r['cell'], r['model'], r['example_id']))
    budget = Budget(HERE / a.ledger); sem = asyncio.Semaphore(a.concurrency)
    for cell in a.cells:
        docs = [json.loads(l) for l in open(HERE / 'raw' / a.samples_dir / f'integer-code-modular--{cell}__seed0.jsonl')]
        docs = docs[:a.limit] if a.limit else docs
        raw = HERE / 'raw' / 'hardening' / a.arm / a.model / cell; raw.mkdir(parents=True, exist_ok=True)
        tasks = [run_item(a.arm, a.model, budget, sem, d, raw) for d in docs if (a.arm, cell, a.model, d['example_id']) not in done]
        with open(out, 'a') as f:
            for coro in asyncio.as_completed(tasks):
                try: r = await coro
                except RuntimeError as e: print('STOP', e, flush=True); break
                r['cell'] = cell
                if r['error']:  # Ai2: errored rows are left out of accuracy and rerun; ours go to a side log and stay undone
                    with open(HERE / 'hardening_errors.jsonl', 'a') as g: g.write(json.dumps(r) + '\n')
                    print('ERROR (not scored, rerunnable)', r['example_id'][:8], r['error'][:120], flush=True); continue
                f.write(json.dumps(r) + '\n'); f.flush()
                print(a.arm, cell, a.model, r['example_id'][:8], 'correct', r['correct'], 'cap', r['cap_hit'], f"vars {r['variable_accuracy']:.2f}",
                      r.get('code_status', ''), f"${r['cost']:.3f}", f"wall {r['wall_s']:.0f}s", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
