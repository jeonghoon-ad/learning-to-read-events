"""Regenerate the true transition of every event in a certification window from its seed (standard library only).

A window of n events is: a fixed preamble of k events, then the first n - k values of the seed's event row.
The event row is what `torch.randint(0, 120, (n,), generator=torch.Generator().manual_seed(seed))` returns on CPU:
for this range PyTorch draws one 32-bit MT19937 output per value and reduces it modulo 120. MT19937 is seeded
with the standard init_genrand(seed). Values are indices 0..119 into the lexicographic list of permutations of
{0,..,4} (see ../../code/register/s5_register.py).

Preambles:
  BOOT3 = (0, 0, 0)                                    three identity events (Qwen3.5-0.8B-Base package, both re-runs)
  N8    = (85, 105, 43, 79, 4, 44, 114, 47)            eight fixed elements whose product is the identity (automatic fits)

Window sizes: 262,140 tokens = 43,690 events of 6 tokens; 196,608 tokens = 32,768 events.

Usage:  python3 window_labels.py SEED N_EVENTS {boot3|n8} [--out FILE]   (prints or writes one index per line)
"""
import sys

BOOT3 = (0, 0, 0)
N8 = (85, 105, 43, 79, 4, 44, 114, 47)


class MT19937:
    def __init__(self, seed):
        self.mt = [0] * 624
        self.mt[0] = seed & 0xffffffff
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xffffffff
        self.i = 624

    def _twist(self):
        mt = self.mt
        for k in range(624):
            y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff)
            v = mt[(k + 397) % 624] ^ (y >> 1)
            if y & 1:
                v ^= 0x9908b0df
            mt[k] = v
        self.i = 0

    def u32(self):
        if self.i >= 624:
            self._twist()
        y = self.mt[self.i]; self.i += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= y >> 18
        return y & 0xffffffff


def event_row(seed, n):
    g = MT19937(seed)
    return [g.u32() % 120 for _ in range(n)]


def window_labels(seed, n_events, boot):
    k = len(boot)
    return list(boot) + event_row(seed, n_events)[:n_events - k]


if __name__ == '__main__':
    a = sys.argv[1:]
    out = None
    if '--out' in a:
        i = a.index('--out'); out = a[i + 1]; del a[i:i + 2]
    seed, n, kind = int(a[0]), int(a[1]), a[2].lower()
    labels = window_labels(seed, n, BOOT3 if kind == 'boot3' else N8)
    text = '\n'.join(map(str, labels)) + '\n'
    if out:
        open(out, 'w').write(text)
    else:
        sys.stdout.write(text)
