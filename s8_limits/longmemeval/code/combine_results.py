"""Rebuild main150 scores per model and arm from the persisted answer/judge rows (results file is per set and was
overwritten by later model runs). Uses the same judge label rule and score() as lme_run. No API calls."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import execution, judge, lme_run

OUT = ROOT / 'out' / 'main150'
data = json.load(open(lme_run.DATA))
sel = lme_run.select_set(data, 'main150')
J = 'gpt-4o-2024-08-06'

def rows_for(model, arm):
    jpath = OUT / f'{arm}_{J}_judge_{model}.jsonl'
    apath = OUT / f'{arm}_{model}_answer.jsonl'
    if not jpath.exists() or not apath.exists():
        return None
    jr = {json.loads(l)['custom_id']: json.loads(l) for l in jpath.read_text().splitlines()}
    ar = {json.loads(l)['custom_id']: json.loads(l) for l in apath.read_text().splitlines()}
    rows = []
    for q in sel:
        j = jr.get(execution.cid(arm, J, 'judge_' + model, q['question_id']))
        a = ar.get(execution.cid(arm, model, 'answer', q['question_id']))
        if j is None or a is None:
            return None
        rows.append({'question_id': q['question_id'], 'type': q['question_type'], 'arm': arm,
                     'correct': judge.label(j['text']), 'answer_cost_usd': a['actual_cost_usd'],
                     'judge_cost_usd': j['actual_cost_usd']})
    return rows

def extract_cost(model):
    p = OUT / f'REG_{model}_extract.jsonl'
    return sum(json.loads(l)['actual_cost_usd'] for l in p.read_text().splitlines()) if p.exists() else 0.0

report = {}
for model in ('gpt-6-luna', 'gpt-6.1-terra', J):
    arms = {arm: rows_for(model, arm) for arm in ('ALONE', 'REG')}
    present = [r for v in arms.values() if v for r in v]
    if not present:
        continue
    scores = lme_run.score(present)
    cost = sum(r['answer_cost_usd'] + r['judge_cost_usd'] for r in present) + extract_cost(model)
    report[model] = {'arms_present': [a for a, v in arms.items() if v], 'scores': scores, 'actual_cost_usd': cost}
(OUT / 'RESULTS_main150_combined.json').write_text(json.dumps(report, indent=1))
for m, r in report.items():
    s = r['scores']
    line = [m, 'cost %.2f' % r['actual_cost_usd']]
    for arm in r['arms_present']:
        line.append('%s %.3f %s' % (arm, s[arm]['overall']['accuracy'], s[arm]['overall']['bootstrap_95_ci']))
        for t in ('knowledge-update', 'temporal-reasoning'):
            line.append('  %s/%s %.3f' % (arm, t, s[arm][t]['accuracy']))
    p = s.get('paired_alone_minus_reg', {}).get('overall')
    if p:
        line.append('ALONE-REG %+.3f %s' % (p['difference'], p['bootstrap_95_ci']))
    print(' | '.join(line))
