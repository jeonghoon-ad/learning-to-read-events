"""Derive one per-window CSV row per certification window from the per-event dumps (.npz).

Standard library only: an .npz file is a zip archive of .npy arrays, read here with zipfile + array.
Usage:
    python3 derive_window_csv.py OUT.csv DUMP_DIR [--boot N] [--glob PATTERN]

Each dump holds one window: per-event arrays `pred` (selector top-1), `true` (correct transition index)
and `margin` (top-1 minus top-2 score). Some dumps also hold `is_boot` (preamble positions), `seed`,
`real_step` and `top2_idx`. Where `is_boot` is absent, the first N positions (--boot, default 3) are
the preamble. Preamble positions are excluded from every count.

Columns:
  window_file, seed, label, real_events, wrong_events, pass (1 if wrong_events == 0),
  smallest_correct_margin (smallest margin over correctly read real events),
  smallest_margin_any (over all real events), error_real_steps (1-based, ';'-separated),
  error_margins (';'-separated), error_true_is_runner_up (';'-separated 0/1, blank if top2_idx absent)
"""
import array, ast, csv, glob, os, re, sys, zipfile

DT = {'<i2': 'h', '<i4': 'i', '<i8': 'q', '<f4': 'f', '<f8': 'd', '|b1': 'B', '<u4': 'I'}


def read_npy(z, name):
    with z.open(name) as f:
        if f.read(6) != b'\x93NUMPY':
            raise ValueError('not an npy member: ' + name)
        major = f.read(2)[0]
        hl = int.from_bytes(f.read(2 if major == 1 else 4), 'little')
        hdr = ast.literal_eval(f.read(hl).decode('latin1'))
        data = f.read()
    if hdr['fortran_order']:
        raise ValueError('fortran order not supported')
    descr = hdr['descr']
    if descr.startswith('<U'):
        return data.decode('utf-32-le').rstrip('\x00')
    a = array.array(DT[descr])
    a.frombytes(data)
    if sys.byteorder != 'little':
        a.byteswap()
    return a


def window_row(path, boot_default):
    z = zipfile.ZipFile(path)
    names = {n[:-4] for n in z.namelist() if n.endswith('.npy')}
    pred = read_npy(z, 'pred.npy'); true = read_npy(z, 'true.npy'); margin = read_npy(z, 'margin.npy')
    n = len(pred)
    if 'is_boot' in names:
        boot = read_npy(z, 'is_boot.npy')
    else:
        boot = [1] * boot_default + [0] * (n - boot_default)
    top2 = read_npy(z, 'top2_idx.npy') if 'top2_idx' in names else None
    seed = None
    if 'seed' in names:
        s = read_npy(z, 'seed.npy'); seed = s[0] if len(s) else None
    base = os.path.basename(path)
    m = re.match(r'^(?:W_|P4_|(?P<label>[a-z_]+?)_)(?P<seed>\d+)\.npz$', base)
    label = (m.group('label') or 'certification') if m else ''
    if seed is None and m:
        seed = int(m.group('seed'))
    real = 0; wrong_steps = []; wrong_m = []; runner = []
    min_ok = float('inf'); min_any = float('inf')
    for i in range(n):
        if boot[i]:
            continue
        real += 1
        g = margin[i]
        if g < min_any: min_any = g
        if pred[i] == true[i]:
            if g < min_ok: min_ok = g
        else:
            wrong_steps.append(real); wrong_m.append(g)
            if top2 is not None:
                runner.append(int(top2[i] == true[i]))
    return {'window_file': base, 'seed': seed, 'label': label, 'real_events': real,
            'wrong_events': len(wrong_steps), 'pass': int(not wrong_steps),
            'smallest_correct_margin': repr(min_ok) if min_ok != float('inf') else '',
            'smallest_margin_any': repr(min_any),
            'error_real_steps': ';'.join(map(str, wrong_steps)),
            'error_margins': ';'.join(repr(x) for x in wrong_m),
            'error_true_is_runner_up': ';'.join(map(str, runner)) if top2 is not None else ''}


def main():
    args = sys.argv[1:]
    boot = 3; pattern = '*.npz'
    if '--boot' in args:
        i = args.index('--boot'); boot = int(args[i + 1]); del args[i:i + 2]
    if '--glob' in args:
        i = args.index('--glob'); pattern = args[i + 1]; del args[i:i + 2]
    out, src = args
    files = sorted(glob.glob(os.path.join(src, pattern)))
    rows = [window_row(f, boot) for f in files]
    cols = list(rows[0])
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f'{out}: {len(rows)} windows, {sum(r["wrong_events"] for r in rows)} wrong events, '
          f'{sum(1 - r["pass"] for r in rows)} failing windows')


if __name__ == '__main__':
    main()
