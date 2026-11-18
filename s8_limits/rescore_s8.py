"""Re-derive the section 8 LongMemEval and MultiWOZ numbers from the released records. Standard library only.
Run: python3 rescore_s8.py

MultiWOZ: the counts of withheld changes are recounted here from the per-turn records and the thresholds. Whether a
change is right needs the MultiWOZ 2.4 gold states and the IC-DST scorer, which are not redistributed; those numbers
are read from the run's own summary table (multiwoz/commit_policy_tables_as_run.json), produced by code/score_policy.py.
"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
label = lambda text: 'yes' in text.lower()   # LongMemEval's own judge-reply rule (code/judge.py: label)
for name, p in (('LongMemEval 150 state questions, GPT-6.1 Terra alone', 'longmemeval/main150/ALONE_gpt-4o-2024-08-06_judge_gpt-6.1-terra.jsonl'),
                ('LongMemEval 150 state questions, GPT-6.1 Terra with the register', 'longmemeval/main150/REG_gpt-4o-2024-08-06_judge_gpt-6.1-terra.jsonl')):
    rows = [json.loads(l) for l in open(os.path.join(H, p))]
    k = sum(label(r['text']) for r in rows)
    print(f'{name}: {k}/{len(rows)} = {k / len(rows):.3f} (judge: gpt-4o-2024-08-06)')

M = os.path.join(H, 'multiwoz')
jl = lambda f: [json.loads(l) for l in open(os.path.join(M, 'test', f))]
thr = json.load(open(os.path.join(M, 'thresholds.json')))['gpt-6.1-terra']
off, on = jl('REG_OFF_gpt-6.1-terra.jsonl'), jl('REG_ON_gpt-6.1-terra.jsonl')
# replay of the policy on the outputs written without it (code/score_policy.py: conf_fn)
status = lambda o: 'withheld' if (o['prob'] is None or o['prob'] < thr['tau_hold']) else 'flagged' if o['prob'] < thr['tau_flag'] else 'released'
replay_withheld = sum(1 for r in off for o in r['ops'] if o['valid'] and status(o) == 'withheld')
live_withheld = sum(1 for r in on for o in r['ops'] if o['valid'] and o['status'] == 'withheld')
t = json.load(open(os.path.join(M, 'commit_policy_tables_as_run.json')))['arms']['gpt-6.1-terra']['2.4']
OFF, REP, LIVE, RND = t['OFF_online'], t['ON_offline_replay'], t['ON_online'], t['RANDOM_matched_offline']
print('MultiWOZ 2.4, GPT-6.1 Terra (test set)')
print(f'  recounted from records: policy replayed on the same outputs withholds {replay_withheld} changes '
      f'(run table: {RND["withheld_ops"]}); live run withheld {live_withheld} valid changes '
      f'(run table: {LIVE["withheld_wrong"] + LIVE["withheld_right"]})')
print(f'  same outputs: wrong changes written without the policy {OFF["wrong_ops_written"]}, with it {REP["wrong_ops_written"]}; '
      f'withholding the same number ({RND["withheld_ops"]}) at random, {RND["seeds"]} draws: '
      f'{RND["wrong_ops_written"]["p2.5"]:.1f} to {RND["wrong_ops_written"]["p97.5"]:.1f}  [run table]')
print(f'  live run: {LIVE["wrong_ops_written"]} wrong changes written; {LIVE["withheld_right"]:,} of '
      f'{LIVE["withheld_wrong"] + LIVE["withheld_right"]:,} withheld changes would have been right; joint goal accuracy '
      f'{OFF["jga_all_turns"]:.4f} without the policy to {LIVE["jga_all_turns"]:.4f} with it  [run table]')
