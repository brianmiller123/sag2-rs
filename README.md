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
let index = sag2::open_existing("index.sqlite", sag2::Profile::Sag2ZhV1)?;  // read-only, signature- and profile-checked
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
let index = sag2::open_existing("index.sqlite", sag2::Profile::Sag2ZhV1)?;
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
let index = sag2::AsyncIndex::open("index.sqlite", sag2::Profile::Sag2ZhV1).await?;
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
| `lindera` | off | `multilingual-v1-ja` segmentation over a runtime dictionary directory (`sag2_core::DictionaryAnalysis`, `sag2_core::analysis::segment::lindera_seg::LinderaSegmenter`); pulls in `lindera` 6.2 with its bundled dictionary and mmap features off | the Japanese profile fails closed with `AnalysisError::DictionaryUnsupported`; every other profile is unaffected |

```sh
cargo build -p sag2                        # BM25 + full SAG2
cargo build -p sag2 --no-default-features  # BM25 only
cargo build -p sag2 --features async       # + async wrappers
cargo build -p sag2 --features lindera     # + the Japanese dictionary
```

### Dictionaries (feature `lindera`, `SAG2_DICT_DIR`)

`multilingual-v1-ja` segments with a morphological dictionary. Compiled ipadic
(≈ 10.5 MB) is far too large to ship, so the dictionary is a **runtime
directory**: it is not in this repository, not vendored, and never downloaded
at build time.  Korean needs no dictionary: its eojeol are space-delimited, so
the word-run pipeline already yields word-level tokens.

The library takes the directory through `IndexOptions::dictionary_dir`; the
CLI takes `--dict-dir` (or the `SAG2_DICT_DIR` environment variable) on both
`build` and `search`; the `lindera` feature has to be on (it requires Rust
1.88, while the default build keeps the workspace's 1.85):

```rust
let options = sag2::IndexOptions {
    profile: sag2::Profile::Multilingual { language: sag2::Language::Ja },
    dictionary_dir: Some("/path/to/ipadic".into()),
    ..sag2::IndexOptions::default()
};
let (index, outcome) = sag2::CorpusIndex::open(&path, &corpus, "fingerprint", options)?;
```

The directory is lindera's compiled on-disk dictionary format — the layout its
`embed-ipadic` feature compiles in, produced with lindera's dictionary builder
from the upstream mecab-ipadic sources. Every file the filesystem loader reads
must be present:

```
metadata.json  char_def.bin  unk.bin  matrix.mtx
dict.trie  dict.valsidx  dict.vals  dict.wordsidx  dict.words
```

The dictionary is provenance, not a hint:

* a Japanese profile **without** a directory is a typed error from
  both build and open (`AnalysisError::DictionaryMissing`, surfaced as
  `OpenError::Analysis`); nothing silently falls back to the character pipeline;
* a directory handed to a profile that does not segment with one is
  `AnalysisError::DictionaryUnused` naming the profile, never a silently
  ignored argument: an index records the pipeline that produced its tokens;
* the build records `rust_index_metadata.analysis_dict` = `"none"` or the
  dictionary fingerprint (`sha256` over `metadata.json` and the sorted file
  digests, path-independent); opening the index with another dictionary is
  `OpenError::IncompatibleIndex` and never rebuilds;
* a missing, unreadable or malformed directory is `AnalysisError::Dictionary`,
  never a panic;
* `SAG2_DICT_DIR` is ambient state: the CLI reads it only for a profile that
  segments with a dictionary, so a stale export never fails an unrelated
  `sag2-zh-v1` build.  An explicit `--dict-dir` is an instruction about the
  build in hand and still errors on a dictionary-free profile.

The offline end-to-end test loads a real dictionary only when
`SAG2_DICT_DIR` points at one (it is `#[ignore]`d in CI):

```sh
SAG2_DICT_DIR=/path/to/ipadic \
  cargo test --offline -p sag2-core --features lindera --test lindera_dictionary -- --ignored
```

## Languages

`Profile::Sag2ZhV1` is the certified pipeline and the default everywhere, and a
default index carries none of the plan's provenance rows: an absent `analysis`
reads back as `sag2-zh-v1` and an absent `analysis_dict` as `none`, so the key
set stays exactly the ten keys design §3.3 fixes (Gate D compares it exactly).
`Profile::Multilingual { language }` normalises per language and takes its word
boundaries from the multilingual word character set - a letter or a digit, a
mark with `combining_class == 0`, and the two joiner classes (ccc 7 nukta,
ccc 9 virama) - instead of the reference classifier's ASCII + `U+4E00-U+9FD5`
set; Han text still goes through jieba.  A word run the reference pipeline would
have shredded emits its code points *and* the whole run, so a word-level index
does not cost the code-point recall the non-ASCII packs rely on.  The profile
persists `rust_index_metadata.analysis` (`multilingual-v1-<lang>`),
`analysis_version` (`ml-2`, the multilingual pipeline revision — bumped
whenever the tokenizer or normaliser changes, so an index from another revision
is rebuilt rather than read with a tokenizer that did not produce its tokens)
and, only for a profile that actually loaded a dictionary, `analysis_dict`, so
an index always reads back through the pipeline that built it.

| language | profile | analysis | word rate | Phrase3 R@10 gain | state |
|---|---|---|---|---|---|
| Russian | `multilingual-v1-ru` | normalise, word runs + code points | 1.00 | +0.00 | deferred |
| Arabic | `multilingual-v1-ar` | normalise (hamza/alef/tatweel), word runs + code points | 1.00 | +0.00 | deferred |
| Hindi | `multilingual-v1-hi` | word runs + code points (nukta and virama join a run) | 1.00 | +0.00 | measured |
| Greek | `multilingual-v1-el` | normalise (strip tonos/dialytika), word runs + code points | 1.00 | +0.00 | deferred |
| Vietnamese | `multilingual-v1-vi` | normalise (shared Latin fold; tone marks kept); syllables space-separated | 0.17 | +0.00 | deferred |
| Polish | `multilingual-v1-pl` | normalise (Latin fold) + word runs | 1.00 | +0.00 | measured |
| Thai | `multilingual-v1-th` | `newmm` words (nlpo3 + CC0 word list) | 0.92 | +0.00 | measured |
| Japanese | `multilingual-v1-ja` | `lindera` words (runtime dictionary) | - | - | dictionary required |
| Korean | `multilingual-v1-ko` | word runs (no dictionary, no morphological analysis) | 1.00 | +0.00 | measured |
| German | `multilingual-v1-de` | normalise (shared Latin fold + `ä/ö/ü`→`ae/oe/ue`) + word runs with both ASCII spellings | 1.00 | +0.00 | measured |
| French | `multilingual-v1-fr` | normalise (shared fold) + word runs | 1.00 | +0.00 | measured |
| Chinese | `sag2-zh-v1` | jieba 0.42.1 words (the reference pipeline) | 0.75 | +0.00 | measured |
| English | `sag2-zh-v1` | ASCII words (the reference pipeline) | 1.00 | +0.00 | measured |

The numbers are measured by `crates/sag2-core/tests/multilingual.rs` on the
self-authored fixture in `crates/sag2-core/tests/data/multilingual.jsonl`
(thirteen packs, ten documents and ten labelled queries each).  `word rate` is
the share of the pack's content words that survive as one whole token: a
multilingual profile emits a run's code points *and* the run, so "exactly one
token in the stream" would score a correctly segmented `bibliothèque` as a
miss.

Typing without diacritics does not need a fold for French, Polish or Vietnamese:
FTS5 `unicode61` removes diacritics by default, and the whole-word token the
word run stores is what puts the accented word in the body, so `bibliotheque`
reaches `bibliothèque` and `truong dai hoc` reaches `trường đại học`.  Stripping
the marks in `normalize` would only merge homographs (`cote`/`côté`/`côte`,
`ma`/`má`/`mà`/`mả`/`mã`/`mạ`) for recall the word run already banked.

German is the exception the fold exists for: an ASCII keyboard spells `Müller`
as the transliteration `mueller`, which FTS cannot bridge from the
diacritic-free `muller`, so `multilingual-v1-de` expands `ä/ö/ü` to `ae/oe/ue`.
The fold alone would trade one spelling for the other, so the German word run
emits its code points and *both* ASCII spellings (`müller` → `mueller` and
`muller`); no other language gets that emission.  Both rules are pinned by
`crates/sag2-index/tests/profile_queries.rs`.

The `state` column is what the fixture asserts:

* **measured** - the profile has a word-level pipeline, so the word rate has a
  pinned target and the `Sag2ZhV1` rate is pinned as the baseline; a change on
  either side fails the test.
* **deferred** - the profile needs the stemmer plan §5 batch 2 owns (Snowball
  for ru/el, an Arabic stemmer, Vietnamese word boundaries).  A whole word is
  one token for them, but an inflected form is still a different term, which is
  what the stemmer has to fix.  The word rate is reported but **not** asserted,
  because a target would only restate the gap.
* **dictionary required** - `ja` cannot tokenise without a runtime dictionary
  and CI has none, so there is nothing to measure offline.  It returns
  `AnalysisError::DictionaryMissing` instead of a character-level stream, which
  is asserted for every content word.

`Phrase3 R@10 gain` is `recall@10(Phrase3) - recall@10(Lexical)` on the same
pack; it never trails Lexical in any pack.  Both modes now reach 1.000 on every
pack, so the fixture no longer separates them: Lexical used to trail on the
character-level packs (ru 0.90, ar/el 0.80) because an `OR` of single
characters retrieves the word-salad decoys as readily as the document, and the
word token L1 adds outranks them.  `Phrase3`'s advantage is still the PUD-scale
result in the plan's §4; this fixture cannot reproduce ranking effects of that
size on ten documents.

What the table says, plainly: two of the plan's §5.1 L1 criteria are met -
French, Korean and Polish all reach 1.00 (from 0.50, n/a and 0.58), and no R@10
number drops.  Hindi reaches 1.00 as well: nukta (ccc 7) and virama (ccc 9) are
joiners, so `सड़क` and `ट्रेन` are one token each and the word rate moved from
0.42 to 1.00.  Vietnamese stays at 0.17 because its words are space-separated
syllables that need their own word boundary, not a character class; that
remains §5 batch 2 work.

## Query modes

`QueryMode` decides how the query becomes an FTS5 `MATCH` expression; the
index is never touched.

| mode | expression | notes |
|---|---|---|
| `Lexical` (default) | `"term" OR "term" ...`, the de-duplicated unigram `OR` (`sag2.py:95-97`) | byte-identical to the reference; every gate replays this |
| `Phrase3` | whole-chunk phrases, then every three-token window, then the same unigrams as a fallback | query-side only; the 80-term budget reserves eight slots for the unigram fallback |

`Phrase3` is what a token stream without word boundaries needs: without them the
`OR` of single characters matches almost every chunk, while a three-character
window has to exist.  The unigram fallback is not optional - exact phrases
alone lose Korean recall (90.5% to 49.4% R@10 in the plan's measurements),
because the character unigrams act as an accidental stemmer for the
agglutinative particles.  On the fixture above Phrase3 never trails Lexical;
both reach 1.000 on every pack since the `multilingual-v1` profiles gained word
tokens, so the fixture no longer shows the separation the plan's PUD numbers do.

Use it through the library (`SearchOptions::query_mode`,
`CorpusIndex::match_query_with`) or the CLI (`--query-mode phrase3`).

## SQLite integration modes

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

## Command line

```sh
sag2-cli build --input <jsonl|dir> --output <index> \
               [--profile sag2-zh-v1|multilingual-v1-<lang>] [--dict-dir <path>] [--chunk-chars <n>]
sag2-cli search --index <path> --query <text> \
                [--query-mode lexical|phrase3] [--judge oracle] [--limit <n>] [--dict-dir <path>]
```

`build` takes either a JSONL file (one `{"title", "text", "path"?, "idx"?}`
object per line) or a directory of UTF-8 files (title = file stem, `path` =
relative path), and fingerprints the corpus: `sha256` of the file, or of the
sorted `relpath\0sha256` list for a directory.  `--profile` defaults to
`sag2-zh-v1`; the identifier is recorded in the index.  `search` opens the
index **as the profile it records** and tokenises the query with that same
pipeline.  Exit codes: `0` success, `3` a usage, build or open failure (the
message is the typed error, never a panic).

```
$ cargo run --offline -q -p sag2-cli -- build \
    --input corpus.jsonl --output demo.sqlite --profile multilingual-v1-de
index:       demo.sqlite
profile:     multilingual-v1-de
fingerprint: d3e032288125e2d6ee3308b499699136d37a0313820766979198d7f8452eea0e
documents:   2
chunks:      2
action:      Built

$ cargo run --offline -q -p sag2-cli -- search --index demo.sqlite --query STRASSE
index:      demo.sqlite
profile:    multilingual-v1-de
mode:       lexical
expression: "strasse"
judge:      none (BM25 retrieval only)
hits:       1
1	doc=1	score=-0.000001	Hafen

$ cargo run --offline -q -p sag2-cli -- search --index demo.sqlite --query STRASSE --query-mode phrase3
index:      demo.sqlite
profile:    multilingual-v1-de
mode:       phrase3
expression: "strasse"
judge:      none (BM25 retrieval only)
hits:       1
1	doc=1	score=-0.000001	Hafen
```

`corpus.jsonl` was:

```json
{"title":"Hafen","text":"Die Straße führt zum alten Hafen. Der Hafen ist ruhig am Morgen."}
{"title":"Bibliothek","text":"Die Bibliothek öffnet um neun. Studenten lesen dort jeden Abend."}
```

`--judge oracle` runs the full SAG2 loop with the deterministic fixture judge
(no network) and prints `jev`/`fusion` scores; omit it for BM25 retrieval only.
A Japanese index needs its dictionary on both commands:

```sh
cargo build -p sag2-cli --features lindera
SAG2_DICT_DIR=/path/to/ipadic cargo run -p sag2-cli --features lindera -- \
    search --index ja.sqlite --query 東京
```

Every other profile refuses a `--dict-dir`/`SAG2_DICT_DIR` value with
`AnalysisError::DictionaryUnused` rather than ignoring it: a Korean index is
built from word runs, and silently accepting a dictionary directory would make
it look as though a morphological one had been used.

## Workspace crates

| crate | purpose |
|---|---|
| `sag2-json` | Python-compatible compact JSON: insertion-ordered values, `py_repr(f64)` identical to CPython 3.12 `repr(float)` |
| `sag2-rank-sys` | vendored `sag_rank.c` (FTS5 `sag_stats`/`sag_bm25`) built with `-DSQLITE_CORE -DSAG_STATIC` into a static archive, plus a safe `register` wrapper |
| `sag2-linktest` | independent downstream package proving the static archive links against `rusqlite`'s bundled SQLite 3.50.2 |
| `sag2-index` | index access and construction: `metadata.signature`, the §3.4 open decision table, the Python schema/statistics/provenance build, cached statistics and `retrieve` with the reference pruning, LRU and epoch semantics |
| `sag2` | library facade: `rank_dpp` (§9 R5 compensated sums), the judge payload builder, `search_sag2` (§1.3, §8) and the optional async wrappers |
| `sag2-cli` | `verify --gate A\|B\|C\|D` (replay of the golden retrieve/search cases and the tokenizer/chunk/index gates), `build` (corpus to index) and `search` (query one index) |

```sh
cargo build --workspace
cargo test --workspace
cargo run -p sag2-linktest
```

CI (`.github/workflows/ci.yml`) runs `fmt --check`, `clippy -D warnings`, the
test suite on Linux/macOS/Windows, the `--no-default-features` build of every
crate, the `multilingual` job (the fixture above plus the `lindera` build) and
all four gates against the committed `golden/` fixtures.


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
