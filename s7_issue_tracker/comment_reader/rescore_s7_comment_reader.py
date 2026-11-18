"""Re-derive the section 7 comment-reader numbers from the released items and predictions.

Point estimates and the commit replay use the standard library only. The project-clustered bootstrap interval uses
NumPy (same function, seed and resample count as the run); it is skipped when NumPy is not installed.
Run: python3 rescore_s7_comment_reader.py
"""
import json, math, os
from collections import defaultdict

H = os.path.dirname(os.path.abspath(__file__))


def jl(p):
    return {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(H, p)) if l.strip()}


def norm(v):  # the scorer's normalisation
    return 'unknown' if v is None else str(v).strip().casefold()


def parse_reply(reply, vocab):  # a free-text reply counts only if it names exactly one vocabulary value
    if not isinstance(reply, str): return None
    m = [v for v in vocab if norm(v) == norm(reply)]
    return m[0] if len(m) == 1 else None


def bootstrap(rows, seed=20261117, n=10000):
    import numpy as np
    groups = defaultdict(lambda: [0, 0])
    for project, a, b in rows:
        groups[project][0] += int(a) - int(b); groups[project][1] += 1
    x = np.array([groups[k] for k in sorted(groups)], dtype=float)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), size=(n, len(x))); s = x[idx].sum(axis=1); v = s[:, 0] / s[:, 1]
    return float(x[:, 0].sum() / x[:, 1].sum()), [float(q) for q in np.quantile(v, [.025, .975])], len(x)


def main():
    items = jl('items.jsonl')
    vocab = {i: r['vocab'][r['field']] for i, r in items.items()}
    P = 'predictions/'
    arms = {
        'reader': {i: parse_reply(r.get('reply'), vocab[i]) for i, r in jl(P + 'reader.jsonl').items()},
        'reader, comment text withheld': {i: parse_reply(r.get('reply'), vocab[i]) for i, r in jl(P + 'reader_comment_withheld.jsonl').items()},
        'same training, creation line only (twin)': {i: parse_reply(r.get('reply'), vocab[i]) for i, r in jl(P + 'twin_creation_line_only.jsonl').items()},
        'frequency table fitted to the training projects': {i: r['prior_0'] for i, r in jl(P + 'frequency_table.jsonl').items()},
        'hand-written parser': {i: r['prediction'] for i, r in jl(P + 'parser.jsonl').items()},
    }
    terra = {}
    for i, r in jl(P + 'terra_api_transcripts.jsonl').items():
        try: terra[i] = r['parsed']['answers'][0]['value']
        except (KeyError, IndexError, TypeError): terra[i] = None
    arms['GPT-6.1 Terra, whole record'] = terra
    ok = lambda i, v: v is not None and norm(v) == norm(items[i]['gold'])
    for group in ('implicit', 'explicit'):
        ids = [i for i, r in items.items() if r['group'] == group]
        print(f'{group} items: {len(ids)}')
        for name, pred in arms.items():
            print(f'  {name}: {sum(ok(i, pred.get(i)) for i in ids) / len(ids):.3f}')
    imp = [i for i, r in items.items() if r['group'] == 'implicit']
    for base in ('frequency table fitted to the training projects', 'reader, comment text withheld'):
        rows = [(items[i]['project'], ok(i, arms['reader'][i]), ok(i, arms[base][i])) for i in imp]
        diff = sum(a - b for _, a, b in rows) / len(rows)
        try:
            d, ci, k = bootstrap(rows)
            print(f'reader minus {base}: {diff:+.3f}, project-clustered 95% interval {ci[0]:+.3f} to {ci[1]:+.3f} ({k} projects)')
        except ImportError:
            print(f'reader minus {base}: {diff:+.3f} (interval needs NumPy; not computed here)')
    # commit replay: threshold = lowest dev confidence at which dev precision >= 0.999, then applied to the 500 implicit items
    dev_t = {r['id']: r['target'] for r in map(json.loads, open(os.path.join(H, 'commit_replay', 'dev_reader_targets.jsonl')))}
    dev = sorted(((math.exp(sum(r['logprobs'])), norm(r['reply']) == norm(dev_t[r['id']]))
                  for r in map(json.loads, open(os.path.join(H, 'commit_replay', 'dev_reader_outputs.jsonl')))), key=lambda x: -x[0])
    best = None; correct = 0
    for i, (c, good) in enumerate(dev, 1):
        correct += good
        if correct / i >= 0.999 and (i == len(dev) or dev[i][0] < c): best = (c, i, correct)
    tau = best[0]
    rd = jl(P + 'reader.jsonl')
    rel = [i for i in imp if math.exp(sum(rd[i]['logprobs'])) >= tau]
    wrong = [i for i in rel if norm(rd[i]['reply']) != norm(items[i]['gold'])]
    print(f'commit replay: dev threshold {tau:.6f} ({best[1]} of {len(dev)} dev items released, {best[1] - best[2]} wrong); '
          f'implicit items committed {len(rel)} of {len(imp)} ({len(rel) / len(imp):.1%}), wrong {len(wrong)}')


if __name__ == '__main__':
    main()
