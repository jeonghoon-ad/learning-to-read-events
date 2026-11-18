"""Re-derive the section 6 counts from the per-item records in this folder. Standard library only.
Run: python3 rescore_s6.py
"""
import collections, csv, json, math, os

H = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(H, 'path_a'); B = os.path.join(H, 'path_b')


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def binom_cdf(k, n, p):
    if k < 0: return 0.0
    if k >= n: return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    t = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k + 1)]
    m = max(t)
    return min(1.0, math.exp(m) * sum(math.exp(x - m) for x in t))


def cp(k, n, a=0.05):
    """Two-sided Clopper-Pearson interval."""
    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if f(mid) else (mid, hi)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) >= a / 2)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) <= a / 2)
    return lower, upper


def frac(k, n):
    lo, hi = cp(k, n)
    return f'{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]'


def path_a():
    print('6.1 OLMo-Hybrid-7B after length- and density-matched training')
    for fmt in ('integer-code-modular', 'spreadsheet-cells-modular'):
        t = jl(os.path.join(A, f'olmo_hybrid_{fmt}_test_scores.jsonl'))
        print(f"  {fmt}: register exact final state {sum(bool(x['score']['register_exact']) for x in t)}/{len(t)}")
    cells = list(csv.DictReader(open(os.path.join(A, 'finetune_control', 'per_cell_table.csv'))))
    # recount both sides of the table from the per-item rows
    ft = collections.Counter(); reg = collections.Counter(); regtw = collections.Counter()
    for r in jl(os.path.join(A, 'finetune_control', 'ft_olmo_rows.jsonl')):
        if 'FT-OLMo' in r['arms']:
            ft[(r['format'], r['tier'], r['stratum'])] += bool(r['arms']['FT-OLMo']['mc_correct'])
    for r in jl(os.path.join(A, 'finetune_control', 'olmo_register_rows.jsonl')):
        e = r['arms']['E-R']; k = (r['format'], r['tier'], r['stratum'])
        reg[k] += bool(e['mc_correct']); regtw[k] += bool(e['mc_correct']) and not e.get('ties')
    mism = [c for c in cells if int(c['ft_olmo_mc']) != ft[(c['format'], c['tier'], c['stratum'])] or int(c['esl_mc']) != reg[(c['format'], c['tier'], c['stratum'])]
            or int(c['esl_mc_ties_wrong']) != regtw[(c['format'], c['tier'], c['stratum'])]]
    print(f'6.1 fine-tuning control: {len(cells)} cells in the per-cell table; cells whose counts differ from the per-item recount: {len(mism)}')
    per = [c for c in cells if c['tier'] == 'periodic' and c['stratum'] in ('tokens_4k', 'tokens_8k', 'tokens_16k')]
    sa = [c for c in cells if c['tier'] in ('solvable', 'aperiodic')]
    for col in ('esl_mc', 'esl_mc_ties_wrong'):
        led = sum(int(c[col]) > int(c['ft_olmo_mc']) for c in per)
        ge = sum(int(c['ft_olmo_mc']) >= int(c[col]) for c in sa)
        print(f'  register column {col}: register led in {led} of {len(per)} periodic cells (4k-16k); fine-tuned at or above it in {ge} of {len(sa)} solvable and aperiodic cells')


def path_b():
    res = jl(os.path.join(B, 'alone_and_code_tool_results.jsonl'))
    spl = jl(os.path.join(B, 'split_results.jsonl'))
    order = [('claude-opus-5-5', 'solvable__tokens_32k'), ('claude-opus-5-5', 'periodic__tokens_32k'), ('gpt-6-sol', 'solvable__tokens_32k'),
             ('gpt-6-sol', 'periodic__tokens_32k'), ('gpt-6-luna', 'solvable__tokens_16k'), ('gpt-6-luna', 'periodic__tokens_16k')]
    print('6.2 Path B, n = 30 per cell')
    print('| Model, cell | Alone | Cap hits | Alone, among finished | Alone, cap 65,536, effort high | Through the split | Code tool |')
    print('|---|---|---|---|---|---|---|')
    for m, c in order:
        al = [r for r in res if r['model'] == m and r['cell'] == c and r['arm'] == 'alone']
        lg = [r for r in res if r['model'] == m and r['cell'] == c and r['arm'] == 'alone_large']
        cd = [r for r in res if r['model'] == m and r['cell'] == c and r['arm'] == 'code']
        sp = [r for r in spl if r['model'] == m and r['cell'] == c]
        fin = [r for r in al if not r['cap_hit']]
        print(f"| {m}, {c} | {sum(bool(r['correct']) for r in al)}/{len(al)} | {sum(bool(r['cap_hit']) for r in al)} | "
              f"{sum(bool(r['correct']) for r in fin)}/{len(fin)} | {(str(sum(bool(r['correct']) for r in lg)) + '/' + str(len(lg))) if lg else ''} | "
              f"{sum(bool(r['exact']) for r in sp)}/{len(sp)} | {sum(bool(r['correct']) for r in cd)}/{len(cd)} |")
    capped_ok = [r['example_id'] for r in res if r['model'] == 'claude-opus-5-5' and r['arm'] == 'alone' and r['cap_hit'] and r['correct']]
    print(f'  capped replies scored correct (Opus alone): {capped_ok}')

if __name__ == '__main__':
    path_a()
    print()
    path_b()
