# sag2-rs

Rust port of the SAG2 retrieval core, usable as a library and verified against
the reference implementation. The reference checkout stays read-only, and every
observable output must match the Python implementation byte for byte.

Design authority: `../SAG2-Rust-移植设计.md` (§1.0 comparison functions,
§2 crate layout, §3 index contract, §13 hard constraints).

## Use it as a library

The crate is not published yet; depend on it by path (or git) for now:

```toml
[dependencies]
sag2 = { path = "../sag2-rs/crates/sag2" }
```

sag2 re-exports everything a consumer needs, so no `sag2-index` or `sag2-json`
dependency is required.

### BM25-only recall

No judge, no payload builder, no async runtime. This path also builds with
`default-features = false`:

```rust
let index = sag2::open_existing("index.sqlite")?;      // read-only, signature-checked
let expression = index.match_query("anchor birthplace depicts")?;
let hits = index.retrieve(&expression, &Default::default(), 5, None)?;
for hit in hits {
    println!("{} {} {:.6}", hit.id, hit.title, hit.score);
}
```

`examples/bm25_only.rs` runs this against the committed C0 fixture:

```sh
cargo run --example bm25_only
```

### Full SAG2 search

Add a judge. `BatchedJudge<T: Transport>` builds the byte-exact payloads and
applies the 32000/64000-token budget; `OracleJudge` is the deterministic
fixture transport the gates replay (a real deployment implements `Judge` or
`Transport`; HTTP arrives in P3):

```rust
let index = sag2::open_existing("index.sqlite")?;
let judge = sag2::BatchedJudge::new(sag2::OracleJudge::new());
let result = sag2::search(&index, &judge, "anchor birthplace depicts", &SearchOptions::default())?;
for candidate in result.candidates.iter().take(5) {
    println!("{} {} {}", candidate.hit.id, candidate.hit.title, candidate.jev_score);
}
```

```sh
cargo run --example oracle_search
```

### Async

Enable the `async` feature to run retrieval and the search loop on tokio's
blocking pool. `AsyncIndex` caps in-flight work at the index's query-worker
count, so overload becomes async waiting instead of unbounded blocking threads:

```rust
let index = sag2::AsyncIndex::open("index.sqlite").await?;
let judge: std::sync::Arc<dyn sag2::Judge> =
    std::sync::Arc::new(sag2::BatchedJudge::new(sag2::OracleJudge::new()));
let result = index.search(judge, "anchor birthplace depicts", SearchOptions::default()).await?;
```

`Judge` stays synchronous on purpose. The search loop holds no async state and
the crate must not force one executor on its consumers, so a judge that needs to
await (an HTTP provider, say) wraps itself — `tokio::task::block_in_place` plus
`Handle::block_on` on a multi-thread runtime, or its own runtime. That keeps the
async decision in the layer that performs I/O.

### Feature matrix

| feature | default | enabled by it | without it |
|---|---|---|---|
| `tiktoken` | on | `BatchedJudge`, `Payload`, `Question`, `Instructions`, `Criteria`, `Answer`, `Transport`, `Response`, `OracleJudge`, the token-budget constants (`CONTEXT_BUDGET`, `BATCH_BUDGET`, `STATE_OVERHEAD`, `QUESTION_OVERHEAD` and the prompt notes) | only a custom `Judge` can score; `search`, retrieval and expansion still work. The accounting types (`BatchDetail`, `RoundUsage`, `UsageBatch`, `Usage`) stay: `Judge::score` and `SearchResult` return them |
| `async` | off | `AsyncIndex`, `AsyncError`, `async_index::search`; pulls in tokio (`rt`, `sync`) and futures-core | no tokio, no futures-core, no async code is compiled |

```sh
cargo build -p sag2                        # BM25 + full SAG2
cargo build -p sag2 --no-default-features  # BM25 only
cargo build -p sag2 --features async       # + async wrappers
```

### SQLite integration modes

`sag2-index` is the only crate that depends on `rusqlite`; the facade never
exposes a `rusqlite` type. There are three integration modes:

1. **No SQLite at all** — depend on `sag2-core` for the tokenizer, chunker and
   UCA collation, as a host with its own storage does (`chunking.py:1`).
2. **Bundled SQLite (default)** — `sag2`/`sag2-index` link `rusqlite`'s bundled
   amalgamation, SQLite 3.50.2. This is the only configuration the gates
   certify, and there is deliberately no "use the system SQLite" feature.
3. **A host that already links SQLite** — for example via `sqlx` or `diesel`.
   `libsqlite3-sys` declares `links = "sqlite3"`, so cargo refuses any dependency
   graph with two versions of it (`multiple packages link to native library
   'sqlite3'`). The host must unify on the `libsqlite3-sys` 0.35.x line that
   `rusqlite` 0.37 uses; no crate here declares `links = "sqlite3"`, and adding
   a second SQLite wrapper is unsupported.

## Workspace crates

| crate | purpose |
|---|---|
| `sag2-json` | Python-compatible compact JSON: insertion-ordered values, `py_repr(f64)` identical to CPython 3.12 `repr(float)` |
| `sag2-rank-sys` | vendored `sag_rank.c` (FTS5 `sag_stats`/`sag_bm25`) built with `-DSQLITE_CORE -DSAG_STATIC` into a static archive, plus a safe `register` wrapper |
| `sag2-linktest` | independent downstream package proving the static archive links against `rusqlite`'s bundled SQLite 3.50.2 |
| `sag2-index` | index access and construction: `metadata.signature`, the §3.4 open decision table, the Python schema/statistics/provenance build, cached statistics and `retrieve` with the reference pruning, LRU and epoch semantics |
| `sag2` | library facade: `rank_dpp` (§9 R5 compensated sums), the judge payload builder, `search_sag2` (§1.3, §8) and the optional async wrappers |
| `sag2-cli` | `verify --gate A\|B\|C\|D`: replay of the golden retrieve/search cases and the tokenizer/chunk/index gates |

```sh
cargo build --workspace
cargo test --workspace
cargo run -p sag2-linktest
```

CI (`.github/workflows/ci.yml`) runs `fmt --check`, `clippy -D warnings`, the
test suite on Linux/macOS/Windows, the `--no-default-features` build of every
crate, and all four gates against the committed `golden/` fixtures.


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
and C3's fifty generated queries at 1, plus C0 variants that cover the gate's
blind spots - `max_hops` 0 and 1 (the zero-hop case must still score once) and
one multi-turn `context` (evidence + `search_history` + `previous_decisions`,
which turns on the judge's history state). Each case carries both halves §1.3
compares:

* **C-1** every request payload as the exact `compact_json` string plus the
  `sha256(compact_json([endpoint, payload]))` the journal keys on, compared
  byte for byte (a failure reports the first differing byte and its JSON
  pointer); and
* **C-2** `ranked_ids`/`ranked_scores_bits`, `baseline_ids`, the full returned
  candidates (id, document id, hop, fusion score bits, `jev_model`, title,
  heading, content, spans, characters, source id and `full_doc`), every trace
  record (stage sequence, candidate/document ids, `terms`, `term_scores`,
  expansion queries, `evidence_ids`, per-round scores and batches) and
  `jev_usage` (counters plus each batch's
  `estimated_input_tokens`/`candidate_count`).

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

* `tokenizer` — the frozen tokenizer lines, the pinned mixed-script segments
  (`4s店`, `1号店`, `江南style` and mixed sentences), every title/text/chunk
  segment of the Gate A/D fixtures plus C4's repository prose and C7's alias
  corpus, the ASCII cross product, the exhaustive per-code-point Unicode tables
  (§4.4) and the `freq` table (below).  C4/C7 are here because they are the
  corpora whose
  mixed CJK/ASCII blocks reach the HMM-buffer branch.
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

## Licence

MIT — see [`LICENSE`](LICENSE). The port tracks the MIT-licensed
`SAG-Benchmark` reference (`sag2/pyproject.toml`: `license = { text = "MIT" }`);
every crate in this workspace declares `license = "MIT"`.
