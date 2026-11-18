"""Arm x length table for the room-swap task (paper section 5), from the per-document records in results/ only.

python3 make_table.py            -> the arm x length table, then the section 5 numbers re-derived from the same records.
Standard library only. Costs use the per-MTok prices stated in the PRICES table below (input, output); records that
carry their own `cost_usd` (Tinker) use it.
"""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import swapgen as g
R = Path(__file__).parent / 'results'

# (label, results folder, (US$ per input token, per output token) or None)
ARMS = [
    ('REASON-CLAUDE (Sonnet 5.5, adaptive)', 'claude', (2e-6, 10e-6)),
    ('CLAUDE-EXTRACT+REG (Sonnet 5.5, thinking off)', 'claude_extract', (2e-6, 10e-6)),
    ('REASON-CLAUDE (Opus 5.5, adaptive)', 'claude_opus', (4e-6, 20e-6)),
    ('OPENAI-ALONE (gpt-6.1-sol, effort high)', 'openai', (2e-6, 10e-6)),
    ('OPENAI-EXTRACT+REG (gpt-6.1-sol, effort low)', 'openai_extract', (2e-6, 10e-6)),
    ('OPENAI-ALONE (gpt-6.2-sol, effort high)', 'openai_gpt-6.2-sol', (2e-6, 10e-6)),
    ('OPENAI-EXTRACT+REG (gpt-6.2-sol, effort low)', 'openai_extract_gpt-6.2-sol', (2e-6, 10e-6)),
    ('OPENAI-ALONE (gpt-6.1-terra, effort high)', 'openai_gpt-6.1-terra', (1e-6, 5e-6)),
    ('OPENAI-EXTRACT+REG (gpt-6.1-terra, effort low)', 'openai_extract_gpt-6.1-terra', (1e-6, 5e-6)),
    ('OPENAI-ALONE (gpt-6-luna, effort high)', 'openai_gpt-6-luna', (0.1e-6, 0.5e-6)),
    ('OPENAI-EXTRACT+REG (gpt-6-luna, effort low)', 'openai_extract_gpt-6-luna', (0.1e-6, 0.5e-6)),
    ('LOGPROB EXTRACT+REG (gpt-4.1-mini)', 'openai_logprob_4.1mini', (0.4e-6, 1.6e-6)),
    ('REASON-GEMINI (gemini-4-carbon-high, Cursor)', 'gemini', None),
    ('REASON-GROK (grok-4.7-xhigh, Cursor)', 'grok', None),
    ('TINKER-ALONE (Qwen3.5-397B-A17B, thinking on, T=1.0)', 'tinker_qwen_alone', 'record'),
    ('TINKER-ALONE (Qwen3.5-397B-A17B, thinking on, T=0)', 'tinker_qwen_alone_t0', 'record'),
    ('TINKER-EXTRACT+REG (Qwen3.5-397B-A17B, reasoning off, T=0)', 'tinker_qwen_extract', 'record'),
    ('TINKER-ALONE (Nemotron-3-Ultra-550B-A55B, thinking on, T=1.0)', 'tinker_ultra_alone', 'record'),
    ('TINKER-ALONE (Nemotron-3-Ultra-550B-A55B, thinking on, T=0)', 'tinker_ultra_alone_t0', 'record'),
    ('TINKER-EXTRACT+REG (Nemotron-3-Ultra-550B-A55B, reasoning off, T=0)', 'tinker_ultra_extract', 'record'),
    ('LOCAL-EXTRACT+REG (DNA3.0-4B, first prompt)', 'local4b_dna3.0-4b_first_prompt', None),
    ('LOCAL-EXTRACT+REG (Nemotron-3-Nano-4B, first prompt)', 'local4b_nemotron-3-nano-4b_first_prompt', None),
    ('LOCAL-EXTRACT+REG (DNA3.0-4B, stated offset)', 'local4b_dna3.0-4b_stated_offset', None),
    ('LOCAL-EXTRACT+REG (Nemotron-3-Nano-4B, stated offset)', 'local4b_nemotron-3-nano-4b_stated_offset', None),
]


def rows_of(d):
    # Per-document records only: skip the shuffled control and the preflight file.
    return [json.loads(f.read_text()) for f in sorted((R / d).glob('swap*_s[0-9].json'))]


def cost(r, price):
    if price == 'record':
        return r.get('cost_usd', 0.0)
    if price is None:
        return None
    u = r['usage']
    return u['input_tokens'] * price[0] + u['output_tokens'] * price[1]


def load():
    man = json.loads((R.parent / 'manifest.json').read_text())
    arms = {'NULL (initial rooms)': ([{'id': m['id'], 'length': m['length'], 'wall_s': 0.0, 'answer': 'initial rooms',
            **g.score('\n'.join(f'{n}: {i+1}' for i, n in enumerate(g.NAMES)), m['truth'])} for m in man], None),
            'CODE (regex + executor)': (json.loads((R / 'code.json').read_text()), None)}
    for label, d, price in ARMS:
        rs = rows_of(d)
        if rs: arms[label] = (rs, price)
    return arms


def med(xs):
    return statistics.median(xs) if xs else None


def table(arms):
    out = ['| Arm | Events | Exact 5/5 | Answered | Mean people correct | Median wall (answered) | Median output tokens, of which thinking | US$ (cell) |',
           '|---|---|---|---|---|---|---|---|']
    for arm, (rows, price) in arms.items():
        for L in g.LENGTHS:
            rs = [r for r in rows if r['length'] == L]
            if not rs: continue
            ans = [r for r in rs if r.get('answer')]
            wall = med([r['wall_s'] for r in ans])
            w = '-' if not wall else (f'{wall*1e3:.2f} ms' if wall < 1 else f'{wall:.1f} s')
            tr = [r for r in ans if 'usage' in r and (r.get('thinking_tokens') is not None)]
            tk = f"{med([r['usage']['output_tokens'] for r in tr]):.0f}, {med([r.get('thinking_tokens') or 0 for r in tr]):.0f} (n={len(tr)})" if tr else '-'
            cs = [cost(r, price) for r in rs]
            usd = '-' if price is None else f'{sum(cs):.3f}'
            out.append(f"| {arm} | {L} | {sum(r['exact'] for r in rs)}/{len(rs)} | {len(ans)}/{len(rs)} | "
                       f"{statistics.mean(r['people_correct'] for r in rs):.1f} | {w} | {tk} | {usd} |")
    return '\n'.join(out)


def at(arms, label, L=1000):
    rows, price = arms[label]
    return [r for r in rows if r['length'] == L], price


def misreads_passed_chunks(rows, man):
    """Wrong pairs and events read, counting only chunks that passed the count check (local 4B rows)."""
    ev = {m['id']: m['events'] for m in man}
    wrong = total = 0
    for r in rows:
        for i, c in enumerate(r['chunks']):
            if not c['ok']: continue
            truth = ev[r['id']][50 * i: 50 * i + 50]
            for p, t in zip(c['pairs'], truth):
                total += 1
                wrong += set(g.NAMES.index(x) for x in p) != set(t)
    return wrong, total


def paper_numbers(arms):
    man = json.loads((R.parent / 'manifest.json').read_text())
    P = []
    nul, _ = at(arms, 'NULL (initial rooms)')
    P.append(f"5.1 null at 1,000 events: mean people right {statistics.mean(r['people_correct'] for r in nul):.1f}; exact {sum(r['exact'] for r in nul)}/5")
    for lab in ['REASON-CLAUDE (Sonnet 5.5, adaptive)', 'REASON-CLAUDE (Opus 5.5, adaptive)', 'OPENAI-ALONE (gpt-6.1-sol, effort high)',
                'OPENAI-ALONE (gpt-6.2-sol, effort high)', 'REASON-GEMINI (gemini-4-carbon-high, Cursor)', 'REASON-GROK (grok-4.7-xhigh, Cursor)',
                'OPENAI-ALONE (gpt-6.1-terra, effort high)', 'OPENAI-ALONE (gpt-6-luna, effort high)',
                'TINKER-ALONE (Qwen3.5-397B-A17B, thinking on, T=1.0)', 'TINKER-ALONE (Qwen3.5-397B-A17B, thinking on, T=0)',
                'TINKER-ALONE (Nemotron-3-Ultra-550B-A55B, thinking on, T=1.0)', 'TINKER-ALONE (Nemotron-3-Ultra-550B-A55B, thinking on, T=0)']:
        rs, _ = at(arms, lab)
        extra = ''
        if 'TINKER' in lab:
            extra = f"; cap hits {sum(bool(r.get('cap_hit')) for r in rs)}/5; answered {sum(bool(r.get('answer')) for r in rs)}/5"
        P.append(f"5.2 {lab} @1000: exact {sum(r['exact'] for r in rs)}/5, mean people right {statistics.mean(r['people_correct'] for r in rs):.1f}{extra}")
    for lab, L in [('REASON-CLAUDE (Sonnet 5.5, adaptive)', 100), ('REASON-CLAUDE (Sonnet 5.5, adaptive)', 300), ('REASON-CLAUDE (Sonnet 5.5, adaptive)', 1000),
                   ('OPENAI-ALONE (gpt-6.1-sol, effort high)', 100), ('OPENAI-ALONE (gpt-6.1-sol, effort high)', 300), ('OPENAI-ALONE (gpt-6.1-sol, effort high)', 1000),
                   ('REASON-CLAUDE (Opus 5.5, adaptive)', 1000), ('OPENAI-ALONE (gpt-6.2-sol, effort high)', 1000)]:
        rs, pr = at(arms, lab, L)
        P.append(f"5.2 median thinking tokens {lab} @{L}: {med([r['thinking_tokens'] for r in rs]):.0f}")
    rs, pr = at(arms, 'REASON-CLAUDE (Opus 5.5, adaptive)')
    P.append(f"5.2 Opus @1000: median wall {med([r['wall_s'] for r in rs]):.1f} s, US$ per document {sum(cost(r, pr) for r in rs)/5:.3f}")
    split = ['CLAUDE-EXTRACT+REG (Sonnet 5.5, thinking off)', 'OPENAI-EXTRACT+REG (gpt-6.1-sol, effort low)', 'OPENAI-EXTRACT+REG (gpt-6.2-sol, effort low)',
             'OPENAI-EXTRACT+REG (gpt-6.1-terra, effort low)', 'OPENAI-EXTRACT+REG (gpt-6-luna, effort low)', 'LOGPROB EXTRACT+REG (gpt-4.1-mini)',
             'TINKER-EXTRACT+REG (Qwen3.5-397B-A17B, reasoning off, T=0)', 'TINKER-EXTRACT+REG (Nemotron-3-Ultra-550B-A55B, reasoning off, T=0)']
    for lab in split:
        rows, _ = arms[lab]
        cells = ', '.join(f"{L}: {sum(r['exact'] for r in rows if r['length'] == L)}/{sum(1 for r in rows if r['length'] == L)}" for L in g.LENGTHS if any(r['length'] == L for r in rows))
        pe = [r.get('per_event_acc') for r in rows if r.get('per_event_acc') is not None]
        acc = f'min per-event accuracy {min(pe)}' if pe else f"wrong name fields {sum(r.get('wrong_fields', 0) for r in rows)} of {sum(r.get('n_fields', 0) for r in rows)}"
        P.append(f"5.3 {lab}: exact {cells}; {acc}; thinking tokens max {max((r.get('thinking_tokens') or 0) for r in rows)}")
    rs, _ = at(arms, 'OPENAI-EXTRACT+REG (gpt-6-luna, effort low)')
    P.append(f"5.3 Luna split @1000: median reasoning tokens {med([r['thinking_tokens'] for r in rs]):.0f}")
    for alone, ext in [('REASON-CLAUDE (Sonnet 5.5, adaptive)', 'CLAUDE-EXTRACT+REG (Sonnet 5.5, thinking off)'),
                       ('OPENAI-ALONE (gpt-6.1-sol, effort high)', 'OPENAI-EXTRACT+REG (gpt-6.1-sol, effort low)'),
                       ('OPENAI-ALONE (gpt-6.2-sol, effort high)', 'OPENAI-EXTRACT+REG (gpt-6.2-sol, effort low)')]:
        a, pa = at(arms, alone); e, pe = at(arms, ext)
        ratio = med([r['wall_s'] for r in a]) / med([r['wall_s'] for r in e])
        P.append(f"5.3 {alone.split('(')[1][:-1]} @1000: wall ratio alone/split {ratio:.1f} (unrounded medians {med([r['wall_s'] for r in a]):.4f} / {med([r['wall_s'] for r in e]):.4f}); "
                 f"US$ per document alone {sum(cost(r, pa) for r in a)/5:.3f}, split {sum(cost(r, pe) for r in e)/5:.3f}; "
                 f"median serial chunk-time sum {med([r.get('serial_chunk_wall_s') for r in e]):.1f} s; median thinking alone {med([r['thinking_tokens'] for r in a]):.0f}")
    for lab in ['LOCAL-EXTRACT+REG (DNA3.0-4B, first prompt)', 'LOCAL-EXTRACT+REG (Nemotron-3-Nano-4B, first prompt)',
                'LOCAL-EXTRACT+REG (DNA3.0-4B, stated offset)', 'LOCAL-EXTRACT+REG (Nemotron-3-Nano-4B, stated offset)']:
        rows, _ = arms[lab]
        cells = ', '.join(f"{L}: {sum(r['exact'] for r in rows if r['length'] == L)}/5" for L in g.LENGTHS)
        w, t = misreads_passed_chunks([r for r in rows if r['length'] == 1000], man)
        ab = sum(r['abstained'] for r in rows if r['length'] == 1000)
        P.append(f"5.4 {lab}: exact {cells}; @1000 abstained {ab}/5, misread {w} of {t} events in passed chunks, "
                 f"median wall {med([r['wall_s'] for r in rows if r['length'] == 1000]):.1f} s")
        d = arms[lab][0]
    for d in ['local4b_dna3.0-4b_stated_offset', 'local4b_nemotron-3-nano-4b_stated_offset']:
        s = json.loads((R / d / 'swap1000_s1_shuffled.json').read_text())
        P.append(f"5.4 shuffled control {d}: vs the shuffled document's own truth exact={s['exact']}, "
                 f"vs the original document's truth people right={s['score_vs_original_truth']['people_correct']}")
    code = [r for r in arms['CODE (regex + executor)'][0] if r['length'] == 1000]
    P.append(f"5.5 regex + executor: exact {sum(r['exact'] for r in arms['CODE (regex + executor)'][0])}/15; median time at 1000 events {med([r['wall_s'] for r in code])*1e3:.2f} ms")
    return '\n'.join(P)


if __name__ == '__main__':
    arms = load()
    print(table(arms))
    print()
    print(paper_numbers(arms))
