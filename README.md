# Learning to Read Events for Exact State Tracking with Frozen Language Models: data and code

Records, scoring scripts and the integer register behind the numbers in the preprint
"Learning to Read Events for Exact State Tracking with Frozen Language Models" (Jeonghoon Lee, Attractor Dynamics Inc.).

Every re-scoring script below uses the Python standard library only, except one optional interval in section 7 that
needs NumPy. Run each from its own folder. Expected outputs are the numbers quoted next to each command.

## Layout and what each folder backs

| Folder | Paper section | Backs |
|---|---|---|
| `code/register/` | 2.1 | The exact S5 register: `s5.py` is the module the certified pipeline imports, unchanged (sha256 `9b7b890c…`, the hash recorded in every certificate manifest); `s5_register.py` is a standard-library mirror of its integer part (120 x 120 table, composition, serial and doubling-scan prefix states); `test_s5_register.py` checks they agree. |
| `s2_3_preamble/` | 2.3 | Start-of-stream test, 23 windows of 43,690 events: without the preamble 21 of 23 windows read the first event as the same wrong class (`without_preamble_23_windows.csv`); with three identity events first, 23 of 23 windows and 0 errors in 1,004,801 real events (`with_preamble_3_identity_events_23_windows.csv`). |
| `s3_unaided_composition/` | 3 | Every generation of the prompted chain-of-thought baseline, one row per window (`data/*/out*/rows.jsonl`, with `summary.json`), and the evaluation code that ran on each machine (`code/<machine>/`). Qwen3.5-0.8B-Base (H1) 296/299 at n = 1 and 2/299 at n = 2; Qwen3.8-27B (H3) 3/100 at n = 2 without reasoning; with reasoning 96/100, 99/100, 100/100, 50/50, 50/50, 23/25 (24 final states right), 6/8 and 1/4 at n = 2 to 256; about 230 to 250 generated tokens per event from n = 8 to 32; no run reached its token budget. |
| `s4_certificates/` | 4 | One CSV row per certification window for every row of the certificate table, the fourth Nemotron attempt and both re-runs (`per_window/`): seed, pass or fail, real events, wrong events, smallest correct gap and the step and gap of every error. `candidate_set_replay/` holds the per-window replay records behind section 4.4. `tools/` regenerates every window's true transitions from its seed and derives the CSVs from the per-event dumps. |
| `s5_room_swap/` | 5 | Document generator and the 15 documents (`swapgen.py`, `docs/`, `manifest.json` with seeds, events and truth), every harness that called a model, one JSON record per document per arm (`results/`), the regular-expression baseline and the table script. |
| `s6_statebench/` | 6 | Item identifiers and selection code (`item_ids/`, `code/select_ai2_rule_n30.py`, `code/path_a_cell_selection.py`); OLMo-Hybrid per-item scores on the integer-code and spreadsheet formats, and the fine-tuning control's per-cell table with both sides' per-item rows (`path_a/`); API per-item results and transcripts for the alone, large-cap, code-tool and split arms (`path_b/`). |
| `s7_issue_tracker/` | 7 | The 600 comment items, per-item predictions of the reader, the comment-withheld reader, the creation-line twin, the frequency table, the hand-written parser and GPT-6.1 Terra, and the commit-replay inputs (`comment_reader/`). |
| `s8_limits/` | 8 | LongMemEval answers, judge replies and register extractions for the 150 state questions (`longmemeval/main150/`); MultiWOZ 2.4 per-turn records for GPT-6.1 Terra with and without the commit policy, the thresholds and the run's summary table (`multiwoz/`). |

## Re-scoring

```
cd code/register && python3 test_s5_register.py
cd s4_certificates && python3 rescore_certificates.py
cd s3_unaided_composition && python3 analysis_tables.py
cd s5_room_swap && python3 make_table.py
cd s6_statebench && python3 rescore_s6.py
cd s7_issue_tracker/comment_reader && python3 rescore_s7_comment_reader.py
cd s8_limits && python3 rescore_s8.py
```

What each prints, against the paper:

- **Section 4** (`rescore_certificates.py`): N, failing windows, one-sided 95% Clopper-Pearson lower bound, smallest correct gap and the number of windows holding a correct event with gap below 46.9, for every certificate row: 0.99003 / 10.46 / 5; 0.99003 / 8.97 / 175; 0.99003 / 23.17 / 18; 5 failing, 0.965; 2 failing, 0.97909 / 56.74 / 0; 1 failing at 196,608 tokens, 0.98423; 3 failing, 0.97427 / 16.28 / 6. The re-runs: 298 of 299 (0.98423; one misread at event 278, gap 38.0, 43,410 positions after it) and 472 of 473 (0.990010; smallest correct gap 6.52). Late-band errors 30 and 2 in 510,393 events; 6.82 s per window over the 476 windows of the N = 473 run; the true transition was the runner-up in 34 of 34 errors (DNA3.0-4B, Nemotron at 196,608 and the fourth Nemotron attempt) and in 3 of 3 in the fifth attempt; the candidate-set replay's 24 errors, 0 wrong EXACT releases against 89,964 for always-commit, and 74% to 99.7% of long-window positions not EXACT; the 25th error at gap 260.7 above its threshold 188.3.
- **Section 5** (`make_table.py`): the arm x length table, then every section 5 number: wall-time ratios 13.0, 11.2 and 17.0 from unrounded medians, cost per 1,000-event document, median thinking tokens, the local 4B misreads (1 of 4,950 and 2 of 5,000), the shuffled control and the parser's 0.26 ms.
- **Section 6** (`rescore_s6.py`): 576/576 and 575/576; the fine-tuning control's 5 of 6 and 19 of 24 (recounted from both sides' per-item rows); every count in the API table of section 6.2, including the code-tool counts.
- **Section 7** (`rescore_s7_comment_reader.py`): 0.688 against 0.610, 0.568, 0.046 and 0.096, a 0.174 drop without comment text, 0.92 against 0.85 on explicit items, and 22 of 500 committed with none wrong. The project-clustered interval (+0.028 to +0.126) is printed when NumPy is installed.
- **Section 8** (`rescore_s8.py`): 0.947 against 0.860 on the 150 LongMemEval state questions (gpt-4o-2024-08-06 judge, LongMemEval's own yes-rule). MultiWOZ 2.4, GPT-6.1 Terra: on the same outputs, 575 wrong changes written without the policy and 474 with it, against 497.5 to 524.0 for withholding the same number (970) at random over 100 draws; in the live run 509 written, and 1,162 of the 1,288 withheld changes would have been right; joint goal accuracy 0.6175 without the policy and 0.5042 with it. The withheld counts (970 and 1,288) are recounted from the per-turn records; whether a change is right needs the MultiWOZ 2.4 gold states and the IC-DST scorer, so those numbers are read from the run's own table (`multiwoz/commit_policy_tables_as_run.json`, produced by `multiwoz/code/score_policy.py`).

The section 3 generations and the section 4 dumps were produced on rented GPUs; the scripts above only re-score the
records. Re-running a model needs the model weights and, for the API arms, a provider key, which the harnesses read
from a file in the user's home directory at run time (no key is included).

## Regenerating the inputs

- **Certification windows (section 4).** A window is a fixed preamble followed by the first events of the row drawn
  from its seed. `s4_certificates/tools/window_labels.py SEED N_EVENTS {boot3|n8}` regenerates every true transition
  of any window with the standard library (MT19937, as PyTorch's CPU generator draws them); it was checked against the
  stored dumps of five runs. The seeds are in the CSVs; the N = 473 seed block was fixed and hashed before the run
  (`per_window/rerun_N473_preregistered_seeds.json`).
- **Room-swap documents (section 5).** `python3 swapgen.py` rebuilds `docs/` and `manifest.json` byte for byte.
- **StateBench items (section 6).** Items come from the public `allenai/state-bench` release at revision `ceb0bbf5`.
  Path B items: `code/select_ai2_rule_n30.py` applies the benchmark's own `sample_docs` rule (seed 0, first 30 per cell);
  `item_ids/path_b_items_seed0_n30.jsonl` lists the selected identifiers with the sha256 of each item text. Path A
  items: `item_ids/path_a_items.jsonl` (identifier, config, stratum, prefix hash) for every item in the released Path A
  records, and `code/path_a_cell_selection.py` (48 rows per cell).
- **Issue-tracker items (section 7).** Drawn from the public Apache Jira by a rule fixed before selection;
  `comment_reader/code/jcv2_draw_items.py` is that rule and `comment_reader/draw_report.json` its output summary. It
  needs a local export of the public Jira data to run.

## What the records are

- API reasoning arms store the final answer text and the provider's token counts; providers do not return hidden
  reasoning text. Extraction arms store every parsed pair per chunk; the open-model (Tinker) arms keep every sampled
  token in `s5_room_swap/results/tinker_ledger_raw_tokens.json`; local 4B arms keep every raw reply.
- StateBench split transcripts keep every model reply. The prompt, which quotes the item, is replaced by its sha256
  and length; `code/run_api_reg.py` rebuilds it from the item.
- In the issue-tracker items, authors and assignees were already `user_N`. For this release e-mail addresses and
  phone-number-like strings are replaced by `[email:<hash>]` and `[phone:<hash>]` (first 10 hex characters of the
  sha256 of the lower-cased string, so equal values stay equal and scoring is unchanged), and home-directory paths by
  `~`. The same tokens are used in the LongMemEval records. All re-scored numbers above were checked after this step.
- Per-statement read arrays were removed from the OLMo rows to keep the package small; every field that enters a
  reported number is kept.

## Withheld, and why

- **Selector checkpoints, selector training code and data generators, and the automatic fitting pipeline.** Not
  part of this release (section 11). The certificate records, the register and the window generator are included,
  so every certificate number can be re-derived without them.
- **Per-event dumps of the certification runs** (about 2 GB of `.npz`, one file per window with the per-event
  prediction, truth and gap). The per-window CSVs in `s4_certificates/per_window/` were derived from them with
  `tools/derive_window_csv.py`; the dumps themselves are not in this folder.
- **Transcripts of the one unnamed commercial model in section 10**, withheld so that the model stays unnamed.
- **StateBench, LongMemEval and MultiWOZ item texts.** Shipped as identifiers and hashes (see `LICENSE_SCOPE.md`).
- **Operations material**: machine set-up, launch and monitoring scripts, logs of rented machines and spend ledgers
  other than the Tinker ledger, which is also the token-level transcript of those calls.
- **Preregistration documents.** The comment-reader comparison of section 7 was preregistered on 14 November 2026
  (document sha256 `7f1fe4ea50ae115a5a9de43d53163b0e5eb3b5a58713a1ac0ea231af5cdaf474`; commit-replay addendum
  `611616ec5ade5cd0ec44079721cb2f256f1f74b4ada727108bb872e64571d826`). The documents are internal notes and are not
  included.

## Licence

See `LICENSE_SCOPE.md` and `LICENSE`. Company-authored code and records: CC BY-NC 4.0. Third-party data keep their
own licences. No patent licence is granted by this release.

`MANIFEST.json` and `SHA256SUMS` list every file with its size and SHA-256 (`shasum -a 256 -c SHA256SUMS`).
