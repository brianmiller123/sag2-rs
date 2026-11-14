# Vendor provenance

These files are verbatim copies of the read-only reference checkout
`SAG-Benchmark/sag2/scripts/native/`. They are vendored, not generated, so the
C code and the SQLite headers used to build it are pinned by hash rather than
by whatever happens to be installed.

Source commit: `bb7f82751ba1576219f9d9ebd021128603d130c3`
(`docs: align bilingual README headers with SAG-Wiki style`).

| file | sha256 |
|---|---|
| `sag_rank.c` | `a87ab48d4c6fecd57f9d9b4b86109a20917451766942a88d1a040f593025726b` |
| `sqlite3.h` | `7db44ac3e95c465c30eea8d45d81474c7cda49ab76e8a681e752e535d1560f2e` |
| `sqlite3ext.h` | `b184dd1586d935133d37ad76fa353faf0a1021ff2fdedeedcc3498fff74bbb94` |

`sag_rank_sha256` (design §3.3 provenance key):
`a87ab48d4c6fecd57f9d9b4b86109a20917451766942a88d1a040f593025726b`.

`sqlite3.h` reports `SQLITE_VERSION "3.50.2"` (line 149), the same version the
`rusqlite`/`libsqlite3-sys` `bundled` amalgamation provides. Design §2.3 rule 4
requires the vendored headers to match the runtime amalgamation: `fts5_api`
layout mismatches are silent memory corruption, not a link error.

## Refresh procedure

```sh
cd /home/pop/AI/code/sag2
cp SAG-Benchmark/sag2/scripts/native/{sag_rank.c,sqlite3.h,sqlite3ext.h} \
   sag2-rs/crates/sag2-rank-sys/vendor/
sha256sum sag2-rs/crates/sag2-rank-sys/vendor/*
git -C SAG-Benchmark rev-parse HEAD   # record the new source commit above
```

Any hash change must be accompanied by an updated `SAG_RANK_SHA256` in the
index provenance later (P0-T2) and by a Gate A rerun: a drifted `sag_rank.c`
changes every BM25 score.
