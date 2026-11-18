"""Affine-copy state transitions for StateBench; diagnostic research only."""
CONST, COPY, INC, NOOP = range(4)
NREG, MODULUS = 17, 100


def execute_serial(reads, initial=None):
    state = list(initial) if initial is not None else [None] * NREG
    if len(state) != NREG:
        raise ValueError('expected 17 registers')
    for op, tgt, src, val in reads:
        if op == NOOP:
            continue
        if not 0 <= tgt < NREG:
            raise ValueError('invalid target')
        if op == CONST:
            if not 0 <= val < MODULUS: raise ValueError('invalid value')
            state[tgt] = val
        elif op == COPY:
            if not 0 <= src < NREG: raise ValueError('invalid source')
            state[tgt] = state[src]
        elif op == INC:
            if not 0 <= val < MODULUS: raise ValueError('invalid increment')
            state[tgt] = None if state[tgt] is None else (state[tgt] + val) % MODULUS
        else:
            raise ValueError('invalid operation')
    return state


def identity():
    # (source index, offset); source -1 represents a constant.
    return tuple((i, 0) for i in range(NREG))


def transform(read):
    op, tgt, src, val = read
    out = list(identity())
    if op == NOOP: return tuple(out)
    if not 0 <= tgt < NREG: raise ValueError('invalid target')
    if op == CONST:
        if not 0 <= val < MODULUS: raise ValueError('invalid value')
        out[tgt] = (-1, val)
    elif op == COPY:
        if not 0 <= src < NREG: raise ValueError('invalid source')
        out[tgt] = (src, 0)
    elif op == INC:
        if not 0 <= val < MODULUS: raise ValueError('invalid increment')
        out[tgt] = (tgt, val)
    else:
        raise ValueError('invalid operation')
    return tuple(out)


def compose(first, second):
    """Return second(first(state)); associative with constant propagation."""
    out = []
    for src, offset in second:
        if src == -1:
            out.append((-1, offset))
        else:
            prev, add = first[src]
            out.append((prev, (offset + add) % MODULUS))
    return tuple(out)


def execute_scan(reads, initial=None):
    # Balanced associative reduction, suitable for parallel scan backends.
    level = [transform(read) for read in reads]
    if not level: level = [identity()]
    while len(level) > 1:
        level = [compose(level[i], level[i+1]) if i+1 < len(level) else level[i]
                 for i in range(0, len(level), 2)]
    state = list(initial) if initial is not None else [None] * NREG
    if len(state) != NREG: raise ValueError('expected 17 registers')
    return [offset if src == -1 else None if state[src] is None else (state[src] + offset) % MODULUS
            for src, offset in level[0]]


class AffineCopyRegister:
    """Stateful serial or associative-scan execution across chunks."""
    def __init__(self, initial=None):
        self.state = list(initial) if initial is not None else [None] * NREG
        if len(self.state) != NREG: raise ValueError('expected 17 registers')

    def serial(self, reads):
        self.state = execute_serial(reads, self.state)
        return list(self.state)

    def scan(self, reads):
        self.state = execute_scan(reads, self.state)
        return list(self.state)
