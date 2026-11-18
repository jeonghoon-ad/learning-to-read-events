"""Re-derive the section 4 numbers (and the section 2.3 preamble test) from the per-window records in this release.

Standard library only. Run: python3 rescore_certificates.py
Inputs: per_window/*.csv (one row per window, derived from the per-event dumps by tools/derive_window_csv.py),
candidate_set_replay/*.json, ../s2_3_preamble/*.csv.
"""
import csv, json, math, os

H = os.path.dirname(os.path.abspath(__file__))
SHIFT = 46.9      # largest gap change observed between two machines on the same frozen inputs (paper, section 4.2)
LATE_BAND = 1707  # last events of each window (paper, section 4.2)

ROWS = [
    ('Qwen3.5-0.8B-Base (layer 9)', 262140, 'row1_qwen3.5-0.8b-base_L9_262140.csv'),
    ('Qwen3.5-0.8B (rev 2fc06364), automatic fit', 262140, 'row2_qwen3.5-0.8b_autofit_262140.csv'),
    ('dnotitia DNA3.0-0.8B (rev ec36b145, layer 6), tuned', 262140, 'row3_dna3.0-0.8b_tuned_L6_262140.csv'),
    ('dnotitia DNA3.0-0.8B, automatic fit', 262140, 'row4_dna3.0-0.8b_autofit_262140.csv'),
    ('dnotitia DNA3.0-4B', 262140, 'row5_dna3.0-4b_262140.csv'),
    ('NVIDIA Nemotron-3-Nano-4B (layer 11)', 196608, 'row6_nemotron-3-nano-4b_L11_196608.csv'),
    ('NVIDIA Nemotron-3-Nano-4B (layer 11), fifth attempt', 262140, 'row7_nemotron-3-nano-4b_L11_262140_attempt5.csv'),
    ('NVIDIA Nemotron-3-Nano-4B (layer 11), fourth attempt, not certified', 262140, 'nemotron-3-nano-4b_L11_262140_attempt4_not_certified.csv'),
    ('Re-run of row 1, fresh instance, N = 299', 262140, 'rerun_fresh_instance_qwen3.5-0.8b-base_N299.csv'),
    ('Re-run of row 1, N = 473', 262140, 'rerun_qwen3.5-0.8b-base_N473.csv'),
]


def cp_lower(n, failures, alpha=0.05):
    """One-sided (1 - alpha) Clopper-Pearson lower bound on the pass probability, given `failures` of n windows."""
    k = n - failures
    if k == 0:
        return 0.0
    if failures == 0:
        return alpha ** (1.0 / n)

    def tail(p):  # P(X >= k | n, p)
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * math.log(p) + (n - j) * math.log1p(-p))
                   for j in range(k, n + 1))
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if tail(mid) < alpha: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def load(name):
    rows = list(csv.DictReader(open(os.path.join(H, 'per_window', name))))
    cert = [r for r in rows if r['label'] in ('certification', 'fresh_preselected')]
    return rows, cert


def main():
    print('Section 4.1: bounds at N = 299 and N = 473')
    print(f'  cp_lower(299, 0) = {cp_lower(299, 0):.6f}   cp_lower(299, 1) = {cp_lower(299, 1):.6f}   cp_lower(473, 1) = {cp_lower(473, 1):.6f}')
    print('\nSection 4.2-4.3: one line per run (certification windows only; control windows excluded)')
    print('| Run | Window tokens | N | Failing windows | Lower bound | Smallest correct gap | Windows with a correct event below 46.9 | Real events | Wrong events |')
    print('|---|---|---|---|---|---|---|---|---|')
    all_err = {}
    for label, tokens, f in ROWS:
        rows, cert = load(f)
        n = len(cert); fails = sum(1 - int(r['pass']) for r in cert)
        gaps = [float(r['smallest_correct_margin']) for r in cert if r['smallest_correct_margin']]
        exposed = sum(1 for g in gaps if g < SHIFT)
        print(f"| {label} | {tokens:,} | {n} | {fails} | {cp_lower(n, fails):.5f} | {min(gaps):.2f} | {exposed} | "
              f"{sum(int(r['real_events']) for r in cert):,} | {sum(int(r['wrong_events']) for r in cert)} |")
        all_err[f] = cert
    print('\nSection 4.2: late-band errors (last 1,707 events of each window), Nemotron-3-Nano-4B at 262,140 tokens')
    for f in ('nemotron-3-nano-4b_L11_262140_attempt4_not_certified.csv', 'row7_nemotron-3-nano-4b_L11_262140_attempt5.csv'):
        cert = all_err[f]
        late = sum(1 for r in cert for s in filter(None, r['error_real_steps'].split(';')) if int(s) > int(r['real_events']) - LATE_BAND)
        ev = LATE_BAND * len(cert)
        print(f'  {f}: {late} late-band errors in {ev:,} late-band events = {late / ev:.3g}; passing windows {sum(int(r["pass"]) for r in cert)}/{len(cert)}')
    print('\nSection 4.3: the failing window(s) of the two re-runs')
    for f in ('rerun_fresh_instance_qwen3.5-0.8b-base_N299.csv', 'rerun_qwen3.5-0.8b-base_N473.csv'):
        for r in all_err[f]:
            if r['pass'] == '0':
                step = int(r['error_real_steps'])
                print(f"  {f}: seed {r['seed']}, misread at real event {step}, gap {float(r['error_margins']):.1f}, "
                      f"positions after it {int(r['real_events']) - step + 1:,}; runner-reported wrong register positions {r.get('runner_wrong_register_positions')}")
    rs = json.load(open(os.path.join(H, 'per_window', 'rerun_N473_run_summary.json')))
    print(f"\nSection 4.5: N = 473 run, all {rs['windows_scored']} windows (473 + 3 controls) in {rs['elapsed_seconds_all_windows']:.1f} s = "
          f"{rs['elapsed_seconds_all_windows'] / rs['windows_scored']:.2f} s per window; real events per window {rs['frozen_identity']['real_events_per_full_window']:,}")
    print('\nSection 4.4: runner-up of every error, in the runs that record it (4B hosts)')
    def runner_up(files):
        flags = [int(x) for f in files for r in all_err[f] for x in filter(None, r['error_true_is_runner_up'].split(';'))]
        return sum(flags), len(flags)
    k, n = runner_up(('row5_dna3.0-4b_262140.csv', 'row6_nemotron-3-nano-4b_L11_196608.csv', 'nemotron-3-nano-4b_L11_262140_attempt4_not_certified.csv'))
    print(f'  DNA3.0-4B, Nemotron at 196,608 and Nemotron attempt 4 together: true transition was the runner-up in {k} of {n} errors')
    k, n = runner_up(('row7_nemotron-3-nano-4b_L11_262140_attempt5.csv',))
    print(f'  Nemotron attempt 5 (262,140), separately: {k} of {n}')
    rep = json.load(open(os.path.join(H, 'candidate_set_replay', 'replay_results.json')))['sets']
    errs = flagged = wrong_exact = base_wrong = 0; withheld = []
    for k, s in rep.items():
        m = s['summary']; errs += m['top1_error_count']; flagged += m['errors_flagged']
        wrong_exact += m['wrong_exact_released_events']; base_wrong += m['baseline_wrong_released_events']
        if m['real_events'] / m['windows'] > 10000:   # long windows only
            withheld.append(m['withheld_events'] / m['real_events'])
        print(f"  replay {k}: threshold {s['threshold']:.1f}, errors {m['top1_error_count']}, flagged {m['errors_flagged']}, wrong EXACT released {m['wrong_exact_released_events']}, "
              f"always-commit wrong positions {m['baseline_wrong_released_events']:,}, truth in final set {m['truth_in_final_set']}/{m['windows']}, withheld {m['withheld_events'] / m['real_events']:.1%}")
    print(f'  replay totals: {errs} errors, {flagged} flagged, {wrong_exact} wrong EXACT releases, always-commit baseline {base_wrong:,} wrong positions; '
          f'long-window share not EXACT {min(withheld):.1%} to {max(withheld):.1%}')
    a = json.load(open(os.path.join(H, 'candidate_set_replay', 'after_freeze_25th_error.json')))['per_rung']['0.01']
    e = a['error_detail'][0]
    print(f"  25th error (after the freeze): gap {e['margin']:.1f}, threshold {a['T']:.1f}, flagged {e['flagged']}, wrong EXACT released {a['wrong_exact_released_events']}")
    print('\nSection 2.3: start-of-stream preamble test (23 windows)')
    pre = os.path.join(H, '..', 's2_3_preamble')
    no = list(csv.DictReader(open(os.path.join(pre, 'without_preamble_23_windows.csv'))))
    wrong = [r for r in no if r['first_event_correct'] == '0']
    print(f"  without preamble: first event wrong in {len(wrong)} of {len(no)} windows, classes {sorted(set(r['first_event_pred'] for r in wrong))}")
    w = list(csv.DictReader(open(os.path.join(pre, 'with_preamble_3_identity_events_23_windows.csv'))))
    print(f"  with preamble: {sum(int(r['pass']) for r in w)} of {len(w)} windows read without error; wrong events {sum(int(r['wrong_events']) for r in w)} "
          f"in {sum(int(r['real_events']) for r in w):,} real events")


if __name__ == '__main__':
    main()
