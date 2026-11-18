"""Ai2 eval_api.py@6ffbc01a sample_docs selection rule (seed 0), applied to the public allenai/state-bench Hub at our pinned revision.
The benchmark authors' own frontier item set was not reachable, so the items are
NOT Ai2's frontier items; the rule is the same. Writes raw/ic_samples_n30/<config>__<tier>__seed0.jsonl locally;
the item texts are not redistributed in this release (see ../item_ids/ for the identifiers and text hashes)."""
import json, random, sys
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
REPO, REV = 'allenai/state-bench', 'ceb0bbf5e992d2012432a82ef28a76c44ce908ef'
COLS = ["example_id","complexity","prefix","choices","correct_choice_index","num_variables","num_assignments","max_dependency_length","token_length","token_stratum"]
out = Path(__file__).resolve().parent / 'raw' / 'ic_samples_n30'; out.mkdir(parents=True, exist_ok=True)
fs = HfFileSystem(); seed = 0; n = 30
for comp in sys.argv[1].split(','):
    for tier in sys.argv[2].split(','):
        config = f'integer-code-modular--{comp}'
        shards = sorted(fs.glob(f"datasets/{REPO}@{REV}/data/{config}/test__{tier}-*.parquet"))
        shard_ids = [pq.ParquetFile(fs.open(s)).read(columns=["example_id"])["example_id"].to_pylist() for s in shards]
        locations = [(si, row) for si, ids in enumerate(shard_ids) for row in range(len(ids))]
        order = list(range(len(locations))); random.Random(f"{seed}:{config}:{tier}").shuffle(order)
        picks = [locations[i] for i in order][:n]
        fetched = {}
        for si in sorted({s for s, _ in picks}):
            rows = [r for s, r in picks if s == si]
            table = pq.ParquetFile(fs.open(shards[si])).read(columns=COLS)
            fetched.update(zip(((si, r) for r in rows), table.take(rows).to_pylist()))
        docs = [fetched[p] for p in picks]
        (out / f'{config}__{tier}__seed0.jsonl').write_text(''.join(json.dumps(d) + '\n' for d in docs))
        print(config, tier, len(locations), 'rows in split ->', len(docs), 'picked')
