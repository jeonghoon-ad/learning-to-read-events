"""Score the commit-policy runs (DESIGN_commit_policy_rev0.1.md). Pinned IC-DST typo_fix/evaluate unchanged.
Per op: right / wrong against the per-turn human gold state (2.4 primary, 2.1 secondary).
Primary numbers per arm: (1) wrong ops written OFF vs ON, (2) coverage released/withheld/flagged turns,
(3) JGA on released turns vs JGA OFF, (4) withhold correctness (wrong vs right counts) for the confidence policy and for
random withholding of the same number of ops (offline replay on the OFF outputs, 100 seeds)."""
import argparse, hashlib, json, random
from collections import Counter
import numpy as np
from schema import ROOT, load_archive, gold_state
from icdst_eval import typo_fix, evaluate
from score import read_ontology
from run_policy import split_ids

OUTP = ROOT / 'out_policy' / 'test'
THR_DIR = ROOT
DATA = load_archive('2.4', 'data.json'); DATA21 = load_archive('2.1', 'data.json')
ONT = {'2.4': read_ontology('2.4'), '2.1': read_ontology('2.1')}

def gold(did, t, version):
    d = DATA if version == '2.4' else DATA21
    return gold_state(d[did]['log'][2 * t + 1]['metadata'])

def op_right(op, g, version):
    if op['op'] == 'delete': return op['slot'] not in g
    v = typo_fix({op['slot']: op['value'].lower()}, ONT[version], version).get(op['slot'])
    return v is not None and op['slot'] in g and v in g[op['slot']].split('|')

def jga(state, g, version):
    pred = typo_fix({k: v.lower() for k, v in state.items()}, ONT[version], version)
    return evaluate(pred, dict(g))[0]

def load(name):
    p = OUTP / f'{name}.jsonl'
    if not p.exists(): return None
    rows = {}
    for l in p.read_text().splitlines():
        r = json.loads(l); rows[(r['dialogue'], r['turn'])] = r
    return rows

def apply_stored(state, ops, sts):
    # 19:56 review BLOCK fix: replay uses the stored per-op validity (rows do not keep evidence strings)
    new = dict(state)
    for o, s in zip(ops, sts):
        if s == 'withheld' or not o['valid']: continue
        if o['op'] == 'delete': new.pop(o['slot'], None)
        else: new[o['slot']] = o['value']
    return new

def replay(off, statuses_fn):
    """Rebuild registers from OFF outputs with given per-op statuses; returns per-turn (state, statuses)."""
    out = {}; dids = sorted({d for d, _ in off})
    for did in dids:
        state = {}; t = 0
        while (did, t) in off:
            r = off[(did, t)]; sts = statuses_fn(r)
            state = apply_stored(state, r['ops'], sts); out[(did, t)] = (dict(state), sts); t += 1
    return out

def summarize(rows_states, src_rows, version):
    """rows_states: {(did,t): (state, statuses)}; src_rows gives the ops."""
    c = Counter(); jg_all = []; jg_rel = []
    for key, (state, sts) in rows_states.items():
        did, t = key; g = gold(did, t, version); ops = src_rows[key]['ops']
        tstat = 'withheld' if 'withheld' in sts else 'flagged' if 'flagged' in sts else 'released'
        c['turns_' + tstat] += 1
        for o, s in zip(ops, sts):
            if not o['valid']: c['ops_invalid'] += 1; continue
            right = op_right(o, g, version)
            c[f'ops_{s}_{"right" if right else "wrong"}'] += 1
        j = jga(state, g, version); jg_all.append(j)
        if tstat == 'released': jg_rel.append(j)
    c['turns'] = len(rows_states)
    written_wrong = c['ops_released_wrong'] + c['ops_flagged_wrong']
    return {'counts': dict(c), 'wrong_ops_written': written_wrong, 'wrong_ops_written_flagged': c['ops_flagged_wrong'],
            'right_ops_written': c['ops_released_right'] + c['ops_flagged_right'],
            'withheld_wrong': c['ops_withheld_wrong'], 'withheld_right': c['ops_withheld_right'],
            'jga_all_turns': float(np.mean(jg_all)) if jg_all else None, 'jga_released_turns': float(np.mean(jg_rel)) if jg_rel else None,
            'released_turns': len(jg_rel)}

def online(rows):
    return {k: (r['state'], [o['status'] for o in r['ops']]) for k, r in rows.items()}

def main():
    thr_hash = hashlib.sha256((THR_DIR / 'thresholds.json').read_bytes()).hexdigest()
    if thr_hash != (THR_DIR / 'thresholds.json.sha256').read_text().split()[0]: raise SystemExit('threshold hash mismatch')
    thr = json.loads((THR_DIR / 'thresholds.json').read_text())
    report = {'thresholds_sha256': thr_hash, 'thresholds': thr, 'arms': {}}
    for model in ('gpt-6-luna', 'gpt-6.1-terra', 'claude-haiku-5-5'):
        off, on = load(f'REG_OFF_{model}'), load(f'REG_ON_{model}')
        if off is None: continue
        if on is not None:   # 19:56 review fix: compare OFF and ON on the same turns only
            common = set(off) & set(on)
            arm_cov = {'off_turns': len(off), 'on_turns': len(on), 'common_turns': len(common)}
            off = {k: v for k, v in off.items() if k in common}; on = {k: v for k, v in on.items() if k in common}
        else:
            arm_cov = {'off_turns': len(off), 'on_turns': 0, 'common_turns': len(off)}
        if model.startswith('claude'):
            conf_fn = lambda r: ['withheld' if r.get('unsure') else 'released'] * len(r['ops'])
        else:
            t = thr[model]
            conf_fn = lambda r, t=t: ['withheld' if (o['prob'] is None or o['prob'] < t['tau_hold']) else 'flagged' if o['prob'] < t['tau_flag'] else 'released' for o in r['ops']]
        arm = {}
        for version in ('2.4', '2.1'):
            res = {'OFF_online': summarize(online(off), off, version)}
            if on is not None: res['ON_online'] = summarize(online(on), on, version)
            conf_replay = replay(off, conf_fn); res['ON_offline_replay'] = summarize(conf_replay, off, version)
            # matched random withholding: same number of withheld ops (or unsure turns for Claude), 100 seeds
            n_with = sum(s.count('withheld') for _, s in conf_replay.values())
            rnd = []
            if model.startswith('claude'):
                # 19:56 review fix: Haiku control matches the number of withheld TURNS (unsure turns that have ops)
                turn_keys = sorted(k for k, r in off.items() if r['ops'])
                n_turns = sum(1 for k, (_, s) in conf_replay.items() if 'withheld' in s)
                for seed in range(100):
                    pick = set(random.Random(seed).sample(turn_keys, min(n_turns, len(turn_keys))))
                    rr = replay(off, lambda r, pick=pick: ['withheld' if (r['dialogue'], r['turn']) in pick else 'released'] * len(r['ops']))
                    rnd.append(summarize(rr, off, version))
            else:
                keys = [(k, i) for k, r in sorted(off.items()) for i in range(len(r['ops']))]
                for seed in range(100):
                    pick = set(random.Random(seed).sample(keys, min(n_with, len(keys))))
                    rr = replay(off, lambda r, pick=pick: ['withheld' if ((r['dialogue'], r['turn']), i) in pick else 'released' for i in range(len(r['ops']))])
                    rnd.append(summarize(rr, off, version))
            def band(f): v = [f(x) for x in rnd]; return {'mean': float(np.mean(v)), 'p2.5': float(np.percentile(v, 2.5)), 'p97.5': float(np.percentile(v, 97.5))}
            res['RANDOM_matched_offline'] = {'withheld_ops': n_with, 'seeds': 100,
                                             'wrong_ops_written': band(lambda x: x['wrong_ops_written']),
                                             'withheld_wrong': band(lambda x: x['withheld_wrong']), 'withheld_right': band(lambda x: x['withheld_right']),
                                             'jga_all_turns': band(lambda x: x['jga_all_turns']), 'jga_released_turns': band(lambda x: x['jga_released_turns'])}
            res['coverage_of_runs'] = arm_cov
            arm[version] = res
        report['arms'][model] = arm
    # secondary: ALONE JGA on the dialogues it covers, beside REG_OFF on the same dialogues
    sec = {}
    for model in ('gpt-6-luna', 'gpt-6.1-terra', 'claude-haiku-5-5'):
        al, off = load(f'ALONE_{model}'), load(f'REG_OFF_{model}')
        if al is None or off is None: continue
        keys = [k for k in al if k in off]
        sec[model] = {v: {'turns': len(keys), 'ALONE_jga': float(np.mean([jga(al[k]['state'], gold(*k, v), v) for k in keys])),
                          'REG_OFF_jga': float(np.mean([jga(off[k]['state'], gold(*k, v), v) for k in keys]))} for v in ('2.4', '2.1')}
    report['secondary_alone_vs_reg'] = sec
    (THR_DIR / 'RESULT_commit_policy_tables.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1)[:6000])

if __name__ == '__main__':
    main()
