"""Exact S5 integer register, standard library only.

Mirror of the integer part of s5.py (the module the certified pipeline imports; sha256 recorded in MIRROR_OF.json):
same permutation order, same composition convention, same 120 x 120 table, same prefix products. s5.py builds the table
as a torch tensor; this file keeps it as nested lists so that the scorers in this release run without torch.

Convention: a permutation of {0..4} is a tuple g with g[i] = image of i. compose(a, b) = a . b applies b first, then a.
State after event t: g_t = a_t . g_{t-1}, starting from the identity (index 0). Transition indices are 0..119 into
ALL_PERMS (lexicographic order). The table is associative, so the serial fold and the doubling scan below agree exactly.
"""
import itertools

ALL_PERMS = list(itertools.permutations(range(5)))
PERM_TO_IDX = {p: i for i, p in enumerate(ALL_PERMS)}
IDENTITY = (0, 1, 2, 3, 4)
assert ALL_PERMS[0] == IDENTITY and len(ALL_PERMS) == 120


def compose(a, b):
    """a . b : apply b first, then a.  (a.b)[i] = a[b[i]]"""
    return tuple(a[b[i]] for i in range(5))


# MUL[a][b] = index of ALL_PERMS[a] . ALL_PERMS[b]   (120 x 120, integers only)
MUL = [[PERM_TO_IDX[compose(a, b)] for b in ALL_PERMS] for a in ALL_PERMS]


def prefix_states(transitions, start=0):
    """Serial register: transitions are indices 0..119; returns the state index after each event."""
    g, out = start, []
    for a in transitions:
        g = MUL[a][g]
        out.append(g)
    return out


def prefix_states_scan(transitions):
    """Same states by a Hillis-Steele doubling scan (the parallel form). Starts from the identity."""
    v = list(transitions)
    step = 1
    while step < len(v):
        v = v[:step] + [MUL[v[i]][v[i - step]] for i in range(step, len(v))]
        step *= 2
    return v


def final_state(transitions, start=0):
    g = start
    for a in transitions:
        g = MUL[a][g]
    return g
