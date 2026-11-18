"""Calibrate tau_hold / tau_flag on DEV ONLY (val dialogues 21-220, REG_OFF outputs) and freeze them with SHA-256.
tau_hold = 10th percentile, tau_flag = 25th percentile of per-op confidence (min token probability) over valid ops,
per model. Refuses to overwrite an existing frozen file."""
import hashlib, json
from pathlib import Path
import numpy as np
from schema import ROOT

out = ROOT / 'thresholds.json'
if out.exists(): raise SystemExit('thresholds.json already frozen; refusing to overwrite')
res = {}
for model in ('gpt-6-luna', 'gpt-6.1-terra'):
    rows = [json.loads(l) for l in open(ROOT / 'out_policy' / 'dev' / f'REG_OFF_{model}.jsonl')]
    # 19:56 review fix: refuse to freeze on an incomplete dev run
    from run_policy import split_ids
    from schema import load_archive
    data = load_archive('2.4', 'data.json'); ids = split_ids('dev')
    expect = {(d, t) for d in ids for t in range(len(data[d]['log']) // 2)}
    have = {(r['dialogue'], r['turn']) for r in rows}
    if have != expect: raise SystemExit(f'{model}: dev run incomplete ({len(have)}/{len(expect)} turns); not freezing')
    probs = [o['prob'] for r in rows for o in r.get('ops', []) if o['valid'] and o['prob'] is not None]
    n_ops = sum(len(r.get('ops', [])) for r in rows)
    res[model] = {'tau_hold': float(np.percentile(probs, 10)), 'tau_flag': float(np.percentile(probs, 25)),
                  'dev_dialogues': len({r['dialogue'] for r in rows}), 'dev_turns': len(rows),
                  'dev_ops': n_ops, 'dev_ops_with_confidence': len(probs)}
res['_rule'] = 'tau_hold = 10th percentile, tau_flag = 25th percentile of per-op min token probability, valid ops, dev (val 21-220) REG_OFF'
out.write_text(json.dumps(res, indent=1))
d = hashlib.sha256(out.read_bytes()).hexdigest()
(ROOT / 'thresholds.json.sha256').write_text(f'{d}  thresholds.json\n')
print(json.dumps(res, indent=1)); print('FROZEN sha256', d)
