# Exploratory Phase 0 component.
"""Pure S5 composition and conservative transcript parsing for Phase 0."""
import re


def _permutation(value):
    return isinstance(value, str) and len(value) == 5 and set(value) == set('12345')


def compose(event, state):
    if not _permutation(event) or not _permutation(state):
        raise ValueError('expected permutations of 12345')
    return ''.join(event[int(digit) - 1] for digit in state)


def prefixes(events):
    state, out = '12345', []
    for event in events:
        state = compose(event, state)
        out.append(state)
    return out


def score(arm, events, output):
    """Strict transcript syntax; unparseable output is incorrect."""
    events = list(events)
    if not events:
        raise ValueError('empty event window')
    states = prefixes(events)
    lines = output.strip().splitlines()
    parsed, format_ok, wrong = None, False, None
    if arm == 'A':
        if len(lines) == 1 and _permutation(lines[0].strip()):
            parsed, format_ok = lines[0].strip(), True
    elif arm == 'B':
        if len(lines) == len(events) + 1:
            format_ok = True
            for i, (line, event, expected) in enumerate(zip(lines, events, states), 1):
                match = re.fullmatch(r'\s*(\d+)\s*:\s*([1-5]{5})\s*>\s*([1-5]{5})\s*', line)
                if not match or int(match[1]) != i or match[2] != event or not _permutation(match[3]):
                    format_ok = False
                    break
                if match[3] != expected and wrong is None:
                    wrong = i
            match = re.fullmatch(r'\s*final\s*:\s*([1-5]{5})\s*', lines[-1], re.IGNORECASE)
            if not match or not _permutation(match[1]):
                format_ok = False
            else:
                parsed = match[1]
    else:
        raise ValueError('unsupported arm')
    return {'pred': parsed, 'truth': states[-1], 'exact': bool(format_ok and parsed == states[-1]),
            'first_wrong_step': wrong, 'format_ok': format_ok}


def prompt(arm, events, demos):
    """Construct a task-only prompt from three explicit demo rows."""
    if arm not in ('A', 'B') or len(demos) != 3 or not events:
        raise ValueError('expected arm A or B, three demos and a nonempty window')
    intro = ('Start with state 12345. Each event rewrites the state: replace every digit d '
             'of the state by the d-th digit of the event.\n')
    intro += ('Return only the final five-digit state.\n' if arm == 'A' else
              'Write one line per event as t: event > state, then final: state.\n')
    def render(row):
        values = list(row)
        if not values:
            raise ValueError('empty demo')
        states = prefixes(values)
        out = 'Events: ' + ''.join(e + ',' for e in values) + '\n'
        if arm == 'A':
            return out + states[-1] + '\n\n'
        out += ''.join(f'{i}: {e} > {s}\n' for i, (e, s) in enumerate(zip(values, states), 1))
        return out + 'final: ' + states[-1] + '\n\n'
    return intro + '\n' + ''.join(render(row) for row in demos) + 'Events: ' + ''.join(e + ',' for e in events) + '\n'
