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
| `sag2-index` | read-only index access: `metadata.signature`, the §3.4 open decision table, cached statistics and `retrieve` with the reference pruning, LRU and epoch semantics (Gate A) |
| `sag2-cli` | `verify --gate A`: replay of the golden retrieve cases |

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

The codec fixtures under `crates/sag2-json/tests/fixtures/` are generated from
CPython 3.12 with `python3.12 crates/sag2-json/tests/fixtures/generate.py`.
