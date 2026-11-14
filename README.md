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
| `sag2-cli` | `verify --gate A|D`: replay of the golden retrieve cases and the tokenizer/chunk/index gates |

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

## Gate D (design §1.4, §4.4)

```sh
cargo run -p sag2-cli -- verify --gate D --fixtures tokenizer,chunks,index --golden golden/
```

`--fixtures` selects any subset:

* `tokenizer` - the frozen tokenizer lines, the ASCII cross product and the
  exhaustive per-code-point Unicode tables (§4.4).
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

`--index-dir <dir>` keeps the Rust-built indexes; replay Gate A against them by
copying `golden/` aside, replacing `indexes/*.sqlite` with the built files and
updating each fixture's `index_sha256` in the copy's `manifest.json`.

The codec fixtures under `crates/sag2-json/tests/fixtures/` are generated from
CPython 3.12 with `python3.12 crates/sag2-json/tests/fixtures/generate.py`.
