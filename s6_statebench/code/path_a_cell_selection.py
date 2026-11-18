"""OLMo preregistered row ordering; does not invoke or alter the Qwen runner."""
from collections import defaultdict
import random

FORMATS = ('integer-code-modular', 'spreadsheet-cells-modular', 'people-in-rooms')
TIERS = ('periodic', 'solvable', 'aperiodic', 'r-trivial')
STRATA = ('tokens_under_2k', 'tokens_4k', 'tokens_8k', 'tokens_16k',
          'tokens_32k', 'tokens_64k', 'tokens_128k')


def ordered_cells(manifest, rows_per_cell=48, seed=2617, defer_strata=()):
    """Select deterministic nested prefixes of identity-checked manifest cells."""
    if rows_per_cell not in (32, 48):
        raise ValueError('OLMo rows_per_cell must be 32 or 48')
    groups = defaultdict(list)
    seen = set()
    for row in manifest:
        fmt, tier = row['config'].split('--', 1)
        if fmt not in FORMATS or tier not in TIERS:
            raise ValueError('unexpected format or tier')
        key = (row['config'], row['example_id'])
        if key in seen:
            raise ValueError('duplicate identity-checked row')
        seen.add(key)
        if row['stratum'] in STRATA:
            groups[(fmt, tier, row['stratum'])].append(row)
    if set(defer_strata) - set(STRATA):
        raise ValueError('unknown deferred stratum')
    rng = random.Random(seed)
    cells = []
    for fmt in FORMATS:
        for tier in TIERS:
            for stratum in STRATA:
                key = (fmt, tier, stratum)
                pool = sorted(groups.get(key, ()), key=lambda row: row['example_id'])
                if pool:
                    rng.shuffle(pool)
                    cells.append((key, pool[:rows_per_cell]))
    yield from (cell for cell in cells if cell[0][2] not in defer_strata)
    yield from (cell for cell in cells if cell[0][2] in defer_strata)
