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

```sh
cargo build --workspace
cargo test --workspace
cargo run -p sag2-linktest
```

The codec fixtures under `crates/sag2-json/tests/fixtures/` are generated from
CPython 3.12 with `python3.12 crates/sag2-json/tests/fixtures/generate.py`.
