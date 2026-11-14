# sag2-rs

Rust port of the SAG2 retrieval core. This workspace lives outside the
`SAG-Benchmark` reference repository and is verified against it: the reference
checkout stays read-only, and every observable output must match the Python
implementation byte for byte.

Design authority: `../SAG2-Rust-移植设计.md` (§1.0 comparison functions,
§2 crate layout, §3 index contract, §13 hard constraints).

| crate | purpose |
|---|---|
| `sag2-json` | Python-compatible compact JSON: insertion-ordered values, `py_repr(f64)` identical to CPython 3.12 `repr(float)` |
| `sag2-rank-sys` | vendored `sag_rank.c` (FTS5 `sag_stats`/`sag_bm25`) built with `-DSQLITE_CORE -DSAG_STATIC` into a static archive, plus a safe `register` wrapper |
| `sag2-linktest` | independent downstream package proving the static archive links against `rusqlite`'s bundled SQLite 3.50.2 |
| `sag2-index` | index access and construction: `metadata.signature`, the §3.4 open decision table, the Python schema/statistics/provenance build, cached statistics and `retrieve` with the reference pruning, LRU and epoch semantics |
| `sag2` | orchestration: `rank_dpp` (§9 R5 compensated sums), the judge payload builder and `search_sag2` (§1.3, §8) |
| `sag2-cli` | `verify --gate A|B|C|D`: replay of the golden retrieve/search cases and the tokenizer/chunk/index gates |

```sh
cargo build --workspace
cargo test --workspace
cargo run -p sag2-linktest
```

## Gate A (design §1.1)

```sh
/tmp/refenv/bin/python tools/export_golden.py --all   # CPython >= 3.12 only
cargo run -p sag2-cli -- verify --gate A --golden golden/
```

`tools/export_golden.py` imports the reference implementation read-only and
writes `golden/retrieve.jsonl` (ordered ids, document ids and the raw
`struct.pack('<d', score)` bits for every case), `golden/index_dump.jsonl`
(`chunks` rows and `metadata.signature`) plus the Python-built indexes the cases
replay against. `verify` exits `0` when every case matches, `2` on a comparison
failure (including any `ulp_flips`), and `3` when the golden cannot be trusted.

## Gate B (design §1.2, §5.5, §6.2-§6.3)

```sh
cargo run -p sag2-cli -- verify --gate B --fixtures sortkeys,expansion --golden golden/
```

* `sortkeys` — pyuca 1.2's `sort_key` over the vocabulary, every code point and
  200,000 ordered comparison pairs (§5.5).
* `expansion` — `golden/expansion.jsonl`, one record per `(query, hop)` of a
  frozen `search_sag2` run driven by the reference's fixed `OracleJev`
  (`tests/test_sag2.py:31-62`): C0's four queries, C4's twenty heading queries
  over the repository prose, C3's fifty generated queries, and C7's direct
  `retrieve_expansion`/`repair_names` sub-cases (design §6.2's alias matrix and
  the `title_phrases` grounding branch, neither of which `search_sag2` can
  reach on a corpus that contains its own terms). Records compare the ordered
  `terms`, `term_scores` (every `f64` by bits), `queries[].ranking_expression`
  byte for byte, `selected[]` and the interleaved `hits`.

## Gate C (design §1.3, §8, §9 R5)

```sh
cargo run -p sag2-cli -- verify --gate C --fixtures search --golden golden/
```

`golden/search.jsonl.gz` holds one case per `search_sag2` run: C0's four frozen
queries and options at `index_workers` 1 **and** 4, C4's twenty heading queries
and C3's fifty generated queries at 1. Each case carries both halves §1.3
compares:

* **C-1** every request payload as the exact `compact_json` string plus the
  `sha256(compact_json([endpoint, payload]))` the journal keys on, compared
  byte for byte (a failure reports the first differing byte and its JSON
  pointer); and
* **C-2** `ranked_ids`/`ranked_scores_bits`, `baseline_ids`, every trace record
  (stage sequence, candidate/document ids, `terms`, `term_scores`, expansion
  queries, `evidence_ids`, per-round scores and batches) and `jev_usage`
  (counters plus each batch's `estimated_input_tokens`/`candidate_count`).

`elapsed_ms` is the only excluded field and is declared in `excluded_fields`.
The indexes are built by this crate from `golden/search_corpus.jsonl`, so the
`index_workers` dimension proves the parallel build is equivalent. The judge is
the reference's fixed `OracleJev` (`tests/test_sag2.py:31-62`), so no network is
involved; `tiktoken-rs`'s embedded `o200k_base` produces the token budget.

## Gate D (design §1.4, §4.4)

```sh
cargo run -p sag2-cli -- verify --gate D --fixtures tokenizer,chunks,index --golden golden/
```

`--fixtures` selects any subset:

* `tokenizer` — the frozen tokenizer lines, the ASCII cross product, the
  exhaustive per-code-point Unicode tables (§4.4) and the `freq` table (below).
* `freq` — rule 5's short (`<= 3` code point) `TOKENIZER.FREQ` key set against
  `golden/freq_keys.txt.gz`: exact set equality (no missing, no extra key), an
  explicit anchor-position tally (every 64th entry of the compiled table is a
  binary-search anchor, and a one-sided window drops exactly those), and
  `tokenize(key)` for the `golden/freq_probes.jsonl` sample, which by
  construction contains every anchor key plus every short key the C0/C4/C7
  corpora contain.
* `chunks` - `golden/chunks.jsonl`: every fixture document plus the desktop
  reference cases and the `tests/test_chunking.py` scenarios, compared on
  heading, spans (code point offsets) and content bytes.  Both §9 R13 defaults
  are recorded: `chunk_path` is what the index passes (`document.txt` when the
  row has no path) while a `null` `chunk_path` is `chunk_document`'s
  `document.md`.
* `index` - builds every fixture with this crate and compares it against the
  Python-built index: D1 `sqlite_master` (name/type/whitespace-normalised sql,
  including the `content=''` flag and column order), D2 the signature and
  `index_statistics.data`, D3 every `chunks` row and every `chunk_fts` row, and
  D4 the 24 `(rowid, bm25)` probes replayed with the reference ranking SQL.  The
  `C6-source` fixture is built in `source_dir` mode against
  `golden/sources/C6-source`, so the contentless schema, `start_char`/`end_char`
  and `source_files` are gate-covered too.

`--index-dir <dir>` also lets Gate D build into a chosen directory instead of a
temporary one. Gate D *creates* that directory and builds every fixture there;
Gate A only *reads* it as-is (never falling back to `golden/indexes`; a fixture
missing from it is an exit-`3` tooling failure). `report.json` records each
fixture's path and `sha256` plus whether it matches the manifest digest — so a
Gate A run against rebuilds is self-describing.

The codec fixtures under `crates/sag2-json/tests/fixtures/` are generated from
CPython 3.12 with `python3.12 crates/sag2-json/tests/fixtures/generate.py`.
