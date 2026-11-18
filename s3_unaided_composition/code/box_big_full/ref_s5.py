# Rev. 0.1 -- 2026-11-03
# Exact S5 arithmetic for the escapement dry look. Integers only (the float
# helpers below are a fixed 0/1 embedding table and are never used for the
# group operation itself).
#
# Convention: permutations of {0..4} are tuples g with g[i] = image of i.
# compose(a, b) = a . b means "apply b first, then a": (a.b)[i] = a[b[i]].
# Prefix products: g_0 = identity, g_t = a_t . g_{t-1}  (newest action on the left).
# Action classes: 0 = "no-op" (identity), 1..120 = ALL_PERMS[k-1].
import itertools
import torch

ALL_PERMS = list(itertools.permutations(range(5)))  # lexicographic order
PERM_TO_IDX = {p: i for i, p in enumerate(ALL_PERMS)}
IDENTITY = (0, 1, 2, 3, 4)
assert ALL_PERMS[0] == IDENTITY and len(ALL_PERMS) == 120
N_PERMS = 120
N_CLASSES = 121


def compose(a, b):
    """a . b : apply b first, then a.  (a.b)[i] = a[b[i]]"""
    return tuple(a[b[i]] for i in range(5))


def inverse(a):
    inv = [0] * 5
    for i, x in enumerate(a):
        inv[x] = i
    return tuple(inv)


def class_to_perm(c):
    return IDENTITY if c == 0 else ALL_PERMS[c - 1]


def class_to_pidx(c):
    return 0 if c == 0 else c - 1


# MUL[a, b] = index of ALL_PERMS[a] . ALL_PERMS[b]
MUL = torch.tensor(
    [[PERM_TO_IDX[compose(a, b)] for b in ALL_PERMS] for a in ALL_PERMS], dtype=torch.int64
)


def prefix_products(actions):
    """Pure python. actions: iterable of class ids (0..120). Returns list of
    perm indices of length T+1: [g_0 (=0, identity), g_1, ..., g_T]."""
    g = IDENTITY
    out = [PERM_TO_IDX[g]]
    for c in actions:
        g = compose(class_to_perm(int(c)), g)
        out.append(PERM_TO_IDX[g])
    return out


def prefix_products_torch(actions):
    """Batched exact version. actions: LongTensor [B,T] of class ids.
    Returns LongTensor [B,T]: entry t is g_{t+1} (state AFTER applying the action
    at position t), i.e. equals prefix_products(row)[1:]."""
    assert actions.dtype == torch.int64
    B, T = actions.shape
    pidx = (actions - 1).clamp(min=0)  # class 0 and class 1 both map to identity
    g = torch.zeros(B, dtype=torch.int64, device=actions.device)
    mul = MUL.to(actions.device)
    out = torch.empty(B, T, dtype=torch.int64, device=actions.device)
    for t in range(T):
        g = mul[pidx[:, t], g]
        out[:, t] = g
    return out


def perm_matrix_embedding():
    """Fixed float tensor [120,25]: flattened 5x5 matrix P with P[i, g[i]] = 1."""
    E = torch.zeros(120, 5, 5)
    for k, g in enumerate(ALL_PERMS):
        for i in range(5):
            E[k, i, g[i]] = 1.0
    return E.reshape(120, 25)


def class_matrix_embedding():
    """[121,25]: class 0 = identity matrix, class k = perm matrix of ALL_PERMS[k-1]."""
    E = perm_matrix_embedding()
    return torch.cat([E[0:1], E], dim=0)


# ---- text / token format --------------------------------------------------
# One step = arrangement digits 1-5 followed by ",", e.g. "31524,". The step
# boundary is the last token of the piece (the ","). Pieces are tokenized
# separately and ids concatenated, so boundaries are exact.
SYNTH_IDS = {"1": 16, "2": 17, "3": 18, "4": 19, "5": 20, ",": 21}  # arbitrary stand-in ids


def perm_to_str(p):
    return "".join(str(x + 1) for x in p) + ","


def synth_tokenize(piece):
    return [SYNTH_IDS[ch] for ch in piece]


def encode_steps(step_pidx, tokenize=synth_tokenize):
    """step_pidx: list of perm indices for one sequence. Returns (ids, boundaries)
    where boundaries[i] = token index of the last token of step i."""
    ids, bounds = [], []
    for k in step_pidx:
        ids.extend(tokenize(perm_to_str(ALL_PERMS[int(k)])))
        bounds.append(len(ids) - 1)
    return ids, bounds


def make_dataset(n_seq, n_steps, seed, tokenize=synth_tokenize):
    """Returns dict of tensors:
    steps   [N,n]  perm idx of each step (0..119)
    targets [N,n]  prefix-product perm idx after each step
    ids     [N,L]  token ids (L = sum of piece lengths)
    bounds  [N,n]  token index of each step boundary
    actions [N,L]  ground-truth selector classes: step perm class (idx+1) at the boundary, 0 elsewhere
    """
    gen = torch.Generator().manual_seed(seed)
    steps = torch.randint(0, 120, (n_seq, n_steps), generator=gen)
    targets = torch.empty_like(steps)
    ids, bounds = [], []
    for i in range(n_seq):
        row = steps[i].tolist()
        targets[i] = torch.tensor(prefix_products([k + 1 for k in row])[1:])
        a, b = encode_steps(row, tokenize)
        ids.append(a)
        bounds.append(b)
    ids = torch.tensor(ids, dtype=torch.int64)
    bounds = torch.tensor(bounds, dtype=torch.int64)
    actions = torch.zeros_like(ids)
    actions.scatter_(1, bounds, steps + 1)
    return dict(steps=steps, targets=targets, ids=ids, bounds=bounds, actions=actions)
