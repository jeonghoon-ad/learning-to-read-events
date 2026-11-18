# Exploratory Phase 0 component.
"""Exact binomial intervals and the preregistered stop-state helper."""
from scipy.stats import beta


def bounds(successes, total):
    if (type(successes) is not int or type(total) is not int or total < 1
            or successes < 0 or successes > total):
        raise ValueError('expected integer 0 <= successes <= total and total > 0')
    k, n = successes, total
    return {
        'lower_one_sided': 0.0 if k == 0 else float(beta.ppf(.05, k, n-k+1)),
        'upper_one_sided': 1.0 if k == n else float(beta.ppf(.95, k+1, n-k)),
        'lower_two_sided': 0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
        'upper_two_sided': 1.0 if k == n else float(beta.ppf(.975, k+1, n-k)),
    }


def stop_state(history):
    """Given ordered complete (n, successes, windows), indicate confirmation.

    None for n_chance means no stop criterion met. If met, next_n is the sole
    confirmation doubling, or None if confirmation has run or is disallowed.
    Caller remains responsible for checking the approved workload's lengths.
    """
    previous = 0
    for index, (length, successes, total) in enumerate(history):
        if type(length) is not int or length <= previous:
            raise ValueError('lengths must increase')
        previous = length
        upper = bounds(successes, total)['upper_one_sided']
        if upper < .05:
            remaining = history[index + 1:]
            if remaining and (length * 2 > 256 or len(remaining) != 1 or remaining[0][0] != length * 2):
                raise ValueError('only one permitted next-doubling confirmation')
            if remaining:
                bounds(remaining[0][1], remaining[0][2])
            confirm = length * 2 if index + 1 == len(history) and length * 2 <= 256 else None
            return {'n_chance': length, 'next_n': confirm}
    return {'n_chance': None, 'next_n': None}
