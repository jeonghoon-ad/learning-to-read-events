"""Checks the stdlib register against s5.py (loaded with a minimal torch stand-in, so torch is not needed) and itself.
Run: python3 test_s5_register.py"""
import os, random, sys, types
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import s5_register as r

fake = types.ModuleType('torch'); fake.int64 = 'int64'
fake.tensor = lambda data, dtype=None: data   # s5.py builds MUL with torch.tensor(list_of_lists, dtype=torch.int64)
sys.modules['torch'] = fake
src = open(os.path.join(H, 's5.py')).read()
ns = {}; exec(compile(src, 's5.py', 'exec'), ns)

assert ns['ALL_PERMS'] == r.ALL_PERMS and ns['PERM_TO_IDX'] == r.PERM_TO_IDX
assert ns['MUL'] == r.MUL, 'table differs from s5.py'
rng = random.Random(0)
for n in (1, 2, 7, 64, 1000):
    seq = [rng.randrange(120) for _ in range(n)]
    ref = ns['prefix_products']([a + 1 for a in seq])[1:]   # s5.py uses class ids 1..120 for transitions
    assert r.prefix_states(seq) == ref == r.prefix_states_scan(seq), n
for a in range(120):                                       # associativity, spot-checked
    for b in rng.sample(range(120), 10):
        c = rng.randrange(120)
        assert r.MUL[a][r.MUL[b][c]] == r.MUL[r.MUL[a][b]][c]
print('PASS: table equals s5.py; serial, scan and s5.prefix_products agree; associativity spot check')
