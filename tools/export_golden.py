#!/usr/bin/env python3
"""Export the Gate A golden fixtures for the Rust port of SAG2 (design §1.1, §7.2).

The exporter imports the reference implementation **read-only** and records for
every Gate A case the ordered chunk ids, the ordered document ids, the raw
``f64`` bit pattern of every score and the complete ``chunks`` row of every
returned hit (title/heading/content/spans/characters/source_id, design §1.1).
Scores are stored as ``struct.pack('<d', score).hex()`` so the comparison never
round-trips through a decimal representation (design §1.0: layer S uses
``eq_f64_bits``).

Requires CPython >= 3.12.  Python 3.12 is the first release whose ``sum()``
uses Neumaier compensated summation; the pruning bound in
``CorpusIndex._retrieve_snapshot`` is such a sum (design §11 D5).  A 3.11 run
silently produces different goldens, so the version gate is a hard exit that
writes nothing.

Outputs (all under ``--out``, default ``golden/``):

``retrieve.jsonl``
    one Gate A case per line, with every hit's §1.1 fields in rank order
``index_dump.jsonl``
    every ``chunks`` row (``SELECT *``) and the ``metadata.signature`` of every
    fixture index
``indexes/<fixture>.sqlite``
    the Python-built index each case runs against (Gate A runs on these)
``chunks.jsonl``
    every fixture document, the desktop reference cases and the
    ``tests/test_chunking.py`` scenarios with heading, spans and content
``index_expected.jsonl``
    per fixture: ``metadata.signature``, every ``chunks`` row, the
    ``chunk_fts`` vocabulary per column, ``index_statistics`` and the
    ``rust_index_metadata`` key set (Gate D's Rust-vs-Python baseline)
``manifest.json``
    generator/python/sqlite/jieba/pyuca/unicode versions, the reference source
    digests and commit, corpus digest, file digests and the fixture -> index
    mapping
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 12):  # pragma: no cover - exercised by the refusal test
    sys.stderr.write(
        "export_golden.py requires CPython >= 3.12 (got "
        f"{sys.version_info.major}.{sys.version_info.minor}); sum() is not "
        "compensated on older interpreters and the pruning bound would differ.\n"
    )
    raise SystemExit(1)

import argparse
import gzip
import hashlib
import io
import itertools
import json
import os
import random
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import regex

# Never leave bytecode (or any other artefact) inside the read-only reference
# checkout (design §7.1).
sys.dont_write_bytecode = True
if sys.pycache_prefix is None:
    sys.pycache_prefix = str(Path(tempfile.gettempdir()) / "sag2-pyc")

# `_load_statistics` builds its term frequencies from a `set`, so the cached
# `diversity_idf` key order — and with it the index file bytes — depends on
# hash randomisation.  Re-exec with a fixed seed so re-exporting produces a
# byte-identical golden tree (design §7.5 `golden-regen`).
if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(
        sys.executable,
        [sys.executable, *sys.argv],
        {**os.environ, "PYTHONHASHSEED": "0"},
    )

DEFAULT_REFERENCE = Path("../SAG-Benchmark/sag2/scripts")
GENERATOR_VERSION = "0.4.0"

# Design §3.3: the v1 `rust_index_metadata` key set.  A Rust-built index must
# carry exactly these keys; the values are compared against the runtime.
RUST_INDEX_METADATA_KEYS = [
    "format_version",
    "tokenizer_version",
    "chunk_policy_version",
    "jieba_dict_sha256",
    "unicode_version",
    "uca_version",
    "uca_keys_sha256",
    "sqlite_version",
    "sag_rank_sha256",
    "sag2_rs_version",
]

# The reference modules whose identity the manifest records (design §7.1).
REFERENCE_FILES = {
    "sag2_py_sha256": "sag2.py",
    "sag2_rank_py_sha256": "sag2_rank.py",
    "chunking_py_sha256": "chunking.py",
}

# Placeholder used by the reference tests for synthetic corpora
# (tests/test_sag2.py:74 ``sag2.CorpusIndex(..., "fixture")``).
C0_FINGERPRINT = "fixture"

# Corpus constructions copied verbatim from the reference tests.  The helpers
# there are local closures, so they cannot be imported; the citations are the
# contract for what "C1"/"C2" mean.
#   C1: tests/test_sag2_rank.py:10-36 (parametrized rare_documents 80 / 140)
#   C2: tests/test_sag2.py:124-147 (limit/term/tie paths)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex_bits(value: float) -> str:
    return struct.pack("<d", value).hex()


def _grid(case_prefix, expression, limits=(1, 7, 100), terms=(None,), excluded=()):
    """Cartesian product of (limit, term) for one expression."""
    cases = []
    for limit in limits:
        for term in terms:
            label = "none" if term is None else term
            cases.append(
                {
                    "case_id": f"{case_prefix}/limit{limit}/term{label}",
                    "expression": expression,
                    "excluded": sorted(excluded),
                    "limit": limit,
                    "term": term,
                }
            )
    return cases


class Fixture:
    """One corpus + the Gate A cases that run against its index."""

    def __init__(self, name, fingerprint, corpus, chunk_chars, case_builder):
        self.name = name
        self.fingerprint = fingerprint
        self.corpus = corpus
        self.chunk_chars = chunk_chars
        self.case_builder = case_builder

    def spec_digest_input(self):
        return {
            "fixture": self.name,
            "fingerprint": self.fingerprint,
            "chunk_chars": self.chunk_chars,
            "corpus": self.corpus,
        }


# --------------------------------------------------------------------------
# C0 - tests/sag2_reference.json: 23 documents, 4 frozen queries.
# --------------------------------------------------------------------------


def fixture_c0(reference, sag2):
    corpus = reference["corpus"]

    def cases(_db):
        exprs = [
            (f"query/{index}", sag2.match_query(case["query"]))
            for index, case in enumerate(reference["cases"])
        ]
        exprs += [
            ("literal/pair", '"alice paul" OR "bob smith"'),
            ("column/title", 'title : "anchor"'),
        ]
        out = []
        for label, expression in exprs:
            out += _grid(f"C0/{label}", expression, terms=(None, "bridge"))
        main = exprs[0][1]
        all_ids = _matching_ids(_db, main)
        for label, excluded in (
            ("known", {1, 3}),
            ("unknown", {9999}),
            ("all", set(all_ids)),
        ):
            out.append(
                {
                    "case_id": f"C0/excluded/{label}",
                    "expression": main,
                    "excluded": sorted(excluded),
                    "limit": 7,
                    "term": None,
                }
            )
        out.append(
            {
                "case_id": "C0/error/unterminated-quote",
                "expression": '"alice',
                "excluded": [],
                "limit": 7,
                "term": None,
            }
        )
        return out

    return Fixture("C0", C0_FINGERPRINT, corpus, 2200, cases)


# --------------------------------------------------------------------------
# C1 - pruning path (tests/test_sag2_rank.py:10-36).
# --------------------------------------------------------------------------


def fixture_c1(rare_documents, sag2):
    rare = [f"rare{i}" for i in range(5)]
    common = [f"common{i}" for i in range(4)]
    corpus = [
        {
            "title": f"Document {i}",
            "text": " ".join(common + rare * 8 if i < rare_documents else common),
        }
        for i in range(300)
    ]
    main = " OR ".join(map(sag2.quote_term, rare + common))  # 9 phrases: prunes
    short = " OR ".join(map(sag2.quote_term, common))  # 4 phrases: no prune

    def cases(_db):
        prefix = f"C1/rare{rare_documents}"
        out = _grid(f"{prefix}/main", main, excluded={1, 3})
        # Pruning needs limit >= 100 and >= 8 phrases; cover both refusals and
        # the boundary limit itself (design §1.1 family 2).
        for limit in (99, 100):
            out.append(
                {
                    "case_id": f"{prefix}/boundary-limit{limit}",
                    "expression": main,
                    "excluded": [1, 3],
                    "limit": limit,
                    "term": None,
                }
            )
        out += _grid(f"{prefix}/short", short, limits=(100,), excluded={1, 3})
        out += _grid(f"{prefix}/term", main, terms=("rare0",), excluded={1, 3})
        return out

    return Fixture(f"C1-rare{rare_documents}", "prune", corpus, 10000, cases)


# --------------------------------------------------------------------------
# C2 - limit / term / tie path (tests/test_sag2.py:124-147).
# --------------------------------------------------------------------------


def fixture_c2(sag2):
    corpus = [
        {"title": "Document", "text": "target " * (i // 3 + 1) + ("bridge" if i % 2 else "filler")}
        for i in range(240)
    ]
    expression = '"target" OR "bridge"'
    reference_excluded = {1, 2, 3, 232, 233, 234}

    def cases(db):
        out = _grid("C2/main", expression, terms=(None, "bridge"),
                    excluded=reference_excluded)
        all_ids = _matching_ids(db, expression)
        for label, excluded in (
            ("none", set()),
            ("unknown", {9999}),
            ("all", set(all_ids)),
        ):
            out.append(
                {
                    "case_id": f"C2/excluded/{label}",
                    "expression": expression,
                    "excluded": sorted(excluded),
                    "limit": 7,
                    "term": None,
                }
            )
        return out

    return Fixture("C2", "limit", corpus, 10000, cases)


# --------------------------------------------------------------------------
# C3 - generated corpus, scale path.  2000 documents x 1 chunk = 2000 chunks.
# --------------------------------------------------------------------------

C3_SEED = 20261114
C3_DOCUMENTS = 2000
C3_QUERIES = 50
C3_VOCABULARY = 60


def fixture_c3(sag2):
    rng = random.Random(C3_SEED)
    pool = [f"lex{i}" for i in range(C3_VOCABULARY)]
    corpus = []
    for i in range(C3_DOCUMENTS):
        words = [rng.choice(pool) for _ in range(12)] + [f"topic{i % 40}"]
        corpus.append({"title": f"Doc {i}", "text": " ".join(words)})
    queries = [tuple(rng.sample(pool, 2)) for _ in range(C3_QUERIES)]

    def cases(_db):
        out = []
        for index, (left, right) in enumerate(queries):
            expression = f"{sag2.quote_term(left)} OR {sag2.quote_term(right)}"
            excluded = {1, 2, 3} if index % 5 == 0 else set()
            out += _grid(f"C3/q{index:02d}", expression, limits=(7, 100), excluded=excluded)
        return out

    return Fixture("C3", "scale", corpus, 10000, cases)


# --------------------------------------------------------------------------
# C5 - Unicode stress: emoji, combining marks, astral planes, CJK ext-B,
#      percent URLs, A.H.G. and half-ounce (design §1.0, §4.2).
# --------------------------------------------------------------------------

C5_DOCUMENTS = [
    ("Emoji", "Rocket 🚀 launch and globe 🌍 marker near the 🧪 lab."),
    ("Combining", "Cafe\u0301 society meets the nai\u0308ve re\u0301sume\u0301 owner."),
    ("Astral Music", "The 𝄞 clef and the 𠀀 glyph appear in one line."),
    ("CJK Extension B", "𠀀𠀁𠀂 rare ideographs beside 汉字 text."),
    ("Percent URL", "See https://example.com/a%20b?q=1%2F2 for the mirrored copy."),
    ("Initials", "A.H.G. wrote the note; A.H.G. signed it twice."),
    ("Hyphen", "A half-ounce and a half-ounce differ from half ounce."),
    ("Dots", "The U.S.A. delegation met the U.K. minister."),
    ("Accents", "Zürich, Ångström and naïve café all appear here."),
    ("Greek", "αβγ symbols and Σigma totals appear in the appendix."),
    ("Math", "f(x) = x² + 1 and 3 × 4 = 12 appear in the sheet."),
    ("Quotes", 'He said "quote me" and left the room.'),
    ("CJK", "北京 上海 广州 深圳 成都 杭州 武汉 西安 南京 重庆"),
    ("Mixed", "Tokyo 東京 and New York 纽约 both appear in the list."),
    ("Astral Emoji", "👍🏽 and 👨‍👩‍👧‍👦 appear in the reaction bar."),
    ("Long Dash", "Em—dash and en–dash and hyphen-joined-word appear."),
]

C5_QUERIES = [
    "rocket launch",
    "cafe\u0301 society",
    "𝄞 clef",
    "𠀀𠀁 ideographs",
    "https://example.com/a%20b?q=1%2F2",
    "A.H.G.",
    "half-ounce",
    "U.S.A. delegation",
    "Zürich Ångström",
    "αβγ Σigma",
    "f(x) = x² + 1",
    '"quote me"',
    "北京 上海",
    "Tokyo 東京",
    "👨‍👩‍👧‍👦 reaction",
    "hyphen-joined-word",
]


def fixture_c5(sag2):
    corpus = [{"title": title, "text": text} for title, text in C5_DOCUMENTS]

    def cases(_db):
        out = []
        for index, query in enumerate(C5_QUERIES):
            expression = sag2.match_query(query)
            out += _grid(f"C5/q{index:02d}", expression, terms=(None,))
        for index in (0, 6):
            expression = sag2.match_query(C5_QUERIES[index])
            out += _grid(
                f"C5/q{index:02d}/term",
                expression,
                terms=("rocket" if index == 0 else "half",),
            )
        return out

    return Fixture("C5", "unicode", corpus, 10000, cases)


def _matching_ids(db, expression):
    return [row[0] for row in db.execute(
        "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?", (expression,)
    )]


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def export_fixture(fixture, sag2, work_dir):
    """Build the fixture index, run every case, return (db_path, cases, rows)."""
    db_path = work_dir / f"{fixture.name}.sqlite"
    index = sag2.CorpusIndex(
        db_path, fixture.corpus, fixture.fingerprint, chunk_chars=fixture.chunk_chars
    )
    try:
        specs = fixture.case_builder(index.db)
        cases = []
        for spec in specs:
            case = dict(spec)
            try:
                hits = index.retrieve(
                    spec["expression"], set(spec["excluded"]), spec["limit"], spec["term"]
                )
            except sqlite3.Error as error:
                case.update(
                    {
                        "error": "sqlite",
                        "error_message": str(error),
                        "ids": [],
                        "document_ids": [],
                        "scores_bits": [],
                        "hits": [],
                    }
                )
                cases.append(case)
                continue
            hits = [
                {
                    "id": hit["id"],
                    "document_id": hit["document_id"],
                    "title": hit["title"],
                    "heading": hit["heading"],
                    "content": hit["content"],
                    "spans": hit["spans"],
                    "characters": hit["characters"],
                    "source_id": hit["source_id"],
                    "score_bits": _hex_bits(hit["score"]),
                }
                for hit in hits
            ]
            case.update(
                {
                    "error": None,
                    "ids": [hit["id"] for hit in hits],
                    "document_ids": [hit["document_id"] for hit in hits],
                    "scores_bits": [hit["score_bits"] for hit in hits],
                    "hits": hits,
                }
            )
            cases.append(case)
        signature = index.db.execute("SELECT signature FROM metadata").fetchone()[0]
        rows = [
            dict(row) for row in index.db.execute("SELECT * FROM chunks ORDER BY id")
        ]
        expectation = index_expectation(fixture, index)
        return db_path, signature, cases, rows, expectation
    finally:
        index.close()


# --------------------------------------------------------------------------
# Gate D (chunks) - design §1.4 D3, §7.2 `chunks`: every fixture document plus
# the desktop reference cases and the `tests/test_chunking.py` scenarios.  The
# record carries the *raw* corpus path (`path`, may be null) and the argument
# `_prepare_index_document` actually passes to `chunk_document`
# (`chunk_path`, `doc.get("path", "document.txt")`).  Keeping both makes the
# §9 R13 double default observable: `chunk_document(path=None)` means
# `document.md`, the index default means `document.txt`.
# --------------------------------------------------------------------------


def _chunk_case(doc_id, fixture, title, source, path, chunk_path, target, chunks, idx=None):
    return {
        "doc_id": doc_id,
        "fixture": fixture,
        "title": title,
        "source": source,
        "path": path,
        "idx": idx,
        "chunk_path": chunk_path,
        "target": target,
        "chunks": [
            {
                "heading": chunk["heading"],
                "spans": [list(span) for span in chunk["spans"]],
                "content": chunk["content"],
            }
            for chunk in chunks
        ],
    }


def export_chunk_cases(fixtures, reference_dir, sag2):
    """`chunks.jsonl`: fixture documents, desktop cases and chunking scenarios."""
    cases = []
    for fixture in fixtures:
        for position, document in enumerate(fixture.corpus):
            chunk_path = document.get("path", "document.txt")
            chunks = sag2.chunk_document(
                document["text"], fixture.chunk_chars, chunk_path
            )
            cases.append(
                _chunk_case(
                    f"{fixture.name}/{position}",
                    fixture.name,
                    document.get("title", ""),
                    document["text"],
                    document.get("path"),
                    chunk_path,
                    fixture.chunk_chars,
                    chunks,
                    document.get("idx"),
                )
            )

    desktop_path = reference_dir.parent / "tests" / "desktop_chunks.json"
    desktop = json.loads(desktop_path.read_text())
    for case in desktop["cases"]:
        chunks = sag2.chunk_document(case["source"], case["target"], case["path"])
        actual = [
            {
                "heading": chunk["heading"],
                "spans": [list(span) for span in chunk["spans"]],
                "content": chunk["content"],
            }
            for chunk in chunks
        ]
        if actual != case["expected"]:
            raise SystemExit(
                f"desktop chunk case {case['name']} disagrees with the reference file"
            )
        cases.append(
            _chunk_case(
                f"chunking/desktop/{case['name']}",
                "chunking",
                "",
                case["source"],
                case["path"],
                case["path"],
                case["target"],
                chunks,
            )
        )

    first = "完整条件" * 100 + "。"
    long = "超长句子" * 150 + "。"
    last = "只有满足上述条件才能发布。"
    scenarios = [
        (
            "chunking/long-sentence-overlap",
            "# 发布指南\n## 生产环境\n" + first + long + last,
            None,
            None,
            512,
        ),
        (
            "chunking/plain-text-offsets",
            "完整句子。" * 200,
            "document.txt",
            "document.txt",
            512,
        ),
        (
            "chunking/crlf-file-index",
            "# 发布指南\r\n## 生产环境\r\n" + ("授权规则必须完整保留。\r\n\r\n" * 60),
            "dsid_1__Guide.md",
            "dsid_1__Guide.md",
            512,
        ),
        (
            "chunking/large-gap-utf16",
            "# A\nfirst.\n" + "\n" * 10000 + "second.",
            "guide.md",
            "guide.md",
            2200,
        ),
        (
            "chunking/heading-retention",
            "# 发布指南\n## 生产环境\n" + "发布前必须核对授权。" * 100,
            "guide.md",
            "guide.md",
            512,
        ),
        (
            "chunking/function-default-document-md",
            "# not a heading in plain text\n\nsecond paragraph",
            None,
            None,
            512,
        ),
        (
            "chunking/index-default-document-txt",
            "# not a heading in plain text\n\nsecond paragraph",
            None,
            "document.txt",
            512,
        ),
    ]
    for doc_id, source, path, chunk_path, target in scenarios:
        chunks = sag2.chunk_document(source, target, chunk_path)
        cases.append(
            _chunk_case(doc_id, "chunking", "", source, path, chunk_path, target, chunks)
        )
    return cases


# --------------------------------------------------------------------------
# Gate D (index) - design §1.4 D1-D3: the signature, every `chunks` row, the
# `chunk_fts` vocabulary per column, the cached statistics and the provenance
# key set.  The reference index itself is the P0 `indexes/<fixture>.sqlite`.
# --------------------------------------------------------------------------


def index_expectation(fixture, index):
    """Everything Gate D compares about a Python-built index."""
    metadata_rows = index.db.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    signature = index.db.execute("SELECT signature FROM metadata").fetchone()[0]
    chunks = [
        dict(row) for row in index.db.execute("SELECT * FROM chunks ORDER BY id")
    ]
    index.db.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS temp.vocab "
        "USING fts5vocab(main,chunk_fts,'col')"
    )
    vocabulary = {}
    for column in ("title", "heading", "body"):
        vocabulary[column] = [
            [row["term"], row["doc"]]
            for row in index.db.execute(
                "SELECT term, doc FROM temp.vocab WHERE col=? ORDER BY term",
                (column,),
            )
        ]
    statistics_rows = index.db.execute(
        "SELECT signature, data FROM index_statistics"
    ).fetchall()
    if len(statistics_rows) != 1:
        raise SystemExit(
            f"{fixture.name}: expected one index_statistics row, "
            f"found {len(statistics_rows)}"
        )
    return {
        "fixture": fixture.name,
        "index": f"{fixture.name}.sqlite",
        "fingerprint": fixture.fingerprint,
        "chunk_chars": fixture.chunk_chars,
        "metadata_rows": metadata_rows,
        "signature": signature,
        "chunks": chunks,
        "fts_vocabulary": vocabulary,
        "statistics_signature": statistics_rows[0][0],
        "statistics": statistics_rows[0][1],
        "rust_index_metadata_keys": RUST_INDEX_METADATA_KEYS,
    }


# --------------------------------------------------------------------------
# Gate D (tokenizer) - design §4.4: frozen tokenizer lines, every corpus
# segment, exhaustive per-codepoint tables and the ASCII-run cross product.
# --------------------------------------------------------------------------

# `[a-zA-Z0-9%._-]` - the connected-ASCII alphabet of design §4.2.
ASCII_RUN_POOL = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789%._-"
ASCII_RUN_PAIR_COUNT = 20_000
ASCII_RUN_SEED = 20261114
CODEPOINTS = 0x110000
# Lone surrogates are not Unicode scalar values; `str.encode("utf-8")` (and
# Rust's `char`) refuse them, so the golden never samples them.
SCALARS = [cp for cp in range(CODEPOINTS) if not 0xD800 <= cp <= 0xDFFF]

# Fixed-width binary layouts.  Every file starts with an 8-byte magic tag and
# then carries exactly one record per Unicode scalar value (0..=0x10FFFF), so a
# reader can index record `cp` directly at `8 + cp * RECORD_BYTES`.
UNICODE_TABLES = {
    "lower": {
        "magic": b"SAG2LOWR",
        "record_bytes": 17,  # cp:u32le, len:u8, cps:[u32le;3] (zero padded)
        "fields": ["cp:u32le", "len:u8", "cps:[u32le;3]"],
        "path": "unicode_lower.bin.gz",
    },
    "casefold": {
        "magic": b"SAG2CASE",
        "record_bytes": 17,
        "fields": ["cp:u32le", "len:u8", "cps:[u32le;3]"],
        "path": "unicode_casefold.bin.gz",
    },
    "combining": {
        "magic": b"SAG2COMB",
        "record_bytes": 5,  # cp:u32le, class:u8
        "fields": ["cp:u32le", "combining_class:u8"],
        "path": "unicode_combining.bin.gz",
    },
    "categories": {
        "magic": b"SAG2CATE",
        "record_bytes": 5,  # cp:u32le, mask:u8 (bit0 L, bit1 N, bit2 Lu, bit3 M)
        "fields": ["cp:u32le", "mask:u8 (1<<0 L, 1<<1 N, 1<<2 Lu, 1<<3 M)"],
        "path": "unicode_categories.bin.gz",
    },
}

CATEGORY_PATTERNS = (
    (0b0001, regex.compile(r"\p{L}")),
    (0b0010, regex.compile(r"\p{N}")),
    (0b0100, regex.compile(r"\p{Lu}")),
    (0b1000, regex.compile(r"\p{M}")),
)


# Sparse, variable-width tables (NFKD/NFKC mappings and the canonical
# composition pairs).  `nfkd`/`nfkc` carry `u32 count, u8 width` after their
# magic tag and then `count` records of `cp:u32le, len:u8, cps:[u32le;width]`.
# `compose` carries `u32 count` and then `count` records of
# `a:u32le, b:u32le, composite:u32le`, sorted by `(a, b)`.
SPARSE_UNICODE_TABLES = {
    "nfkd": {
        "magic": b"SAG2NFKD",
        "path": "unicode_nfkd.bin.gz",
        "fields": ["cp:u32le", "len:u8", "cps:[u32le;width]"],
        "header": "count:u32le, width:u8",
    },
    "nfkc": {
        "magic": b"SAG2NFKC",
        "path": "unicode_nfkc.bin.gz",
        "fields": ["cp:u32le", "len:u8", "cps:[u32le;width]"],
        "header": "count:u32le, width:u8",
    },
    "compose": {
        "magic": b"SAG2COMP",
        "path": "unicode_compose.bin.gz",
        "fields": ["a:u32le", "b:u32le", "composite:u32le"],
        "header": "count:u32le",
    },
}

HANGUL_FIRST = 0xAC00
HANGUL_LAST = 0xD7A3


def _sparse_mapping(transform):
    """Non-identity single-codepoint results, Hangul excluded (it is algorithmic)."""
    records = []
    width = 0
    for cp in range(CODEPOINTS):
        if 0xD800 <= cp <= 0xDFFF or HANGUL_FIRST <= cp <= HANGUL_LAST:
            continue
        mapped = transform(chr(cp))
        if mapped == chr(cp):
            continue
        records.append((cp, mapped))
        width = max(width, len(mapped))
    return records, width


def _sparse_mapping_bytes(magic, records, width):
    buffer = bytearray(magic)
    buffer += struct.pack("<IB", len(records), width)
    for cp, mapped in records:
        buffer += struct.pack("<IB", cp, len(mapped))
        buffer += b"".join(struct.pack("<I", ord(c)) for c in mapped.ljust(width, "\0"))
    return bytes(buffer)


def _compose_pairs():
    """Canonical composition pairs that Python's NFC actually applies."""
    pairs = []
    for cp in range(CODEPOINTS):
        if 0xD800 <= cp <= 0xDFFF or HANGUL_FIRST <= cp <= HANGUL_LAST:
            continue
        decomposition = unicodedata.decomposition(chr(cp))
        if not decomposition or decomposition.startswith("<"):
            continue
        parts = [int(part, 16) for part in decomposition.split()]
        if len(parts) != 2:
            continue
        left, right = parts
        if unicodedata.normalize("NFC", chr(left) + chr(right)) == chr(cp):
            pairs.append((left, right, cp))
    pairs.sort()
    return pairs


def export_sparse_unicode_tables():
    nfkd, nfkd_width = _sparse_mapping(lambda char: unicodedata.normalize("NFKD", char))
    nfkc, nfkc_width = _sparse_mapping(lambda char: unicodedata.normalize("NFKC", char))
    pairs = _compose_pairs()
    blobs = {
        "nfkd": _sparse_mapping_bytes(b"SAG2NFKD", nfkd, nfkd_width),
        "nfkc": _sparse_mapping_bytes(b"SAG2NFKC", nfkc, nfkc_width),
    }
    buffer = bytearray(b"SAG2COMP")
    buffer += struct.pack("<I", len(pairs))
    for left, right, composite in pairs:
        buffer += struct.pack("<III", left, right, composite)
    blobs["compose"] = bytes(buffer)
    counts = {
        "nfkd": {"records": len(nfkd), "width": nfkd_width},
        "nfkc": {"records": len(nfkc), "width": nfkc_width},
        "compose": {"records": len(pairs)},
    }
    metadata = {
        name: {
            "path": SPARSE_UNICODE_TABLES[name]["path"],
            "magic": SPARSE_UNICODE_TABLES[name]["magic"].decode(),
            "fields": SPARSE_UNICODE_TABLES[name]["fields"],
            "header": SPARSE_UNICODE_TABLES[name]["header"],
            **counts[name],
        }
        for name in SPARSE_UNICODE_TABLES
    }
    return {name: _gzip_bytes(blob) for name, blob in blobs.items()}, metadata


def _gzip_bytes(data: bytes) -> bytes:
    """Deterministic gzip (mtime 0) so re-exports stay byte-identical."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def fixture_segments(fixture, rows):
    """Every corpus title/text plus every chunk content, in a stable order."""
    segments = []
    for index, document in enumerate(fixture.corpus):
        segments.append((f"{fixture.name}/title/{index}", document.get("title", "")))
        segments.append((f"{fixture.name}/text/{index}", document.get("text", "")))
    for row in rows:
        segments.append((f"{fixture.name}/chunk/{row['id']}/content", row["content"]))
    return segments


def export_tokenizer(reference, fixture_segments_by_name, sag2):
    """`tokenizer.jsonl`: segment id, normalised text and the reference tokens."""
    segments = []
    for index, entry in enumerate(reference["tokenizer"]):
        tokens = sag2.tokenize(entry["text"])
        if tokens != entry["tokens"]:
            raise SystemExit(
                f"frozen tokenizer line {index} disagrees with the reference implementation"
            )
        segments.append((f"C0/tokenizer/{index}", entry["text"], tokens))
    for name, entries in fixture_segments_by_name.items():
        for segment_id, text in entries:
            segments.append((segment_id, text, sag2.tokenize(text)))
    lines = [
        {"segment_id": segment_id, "text": text, "tokens": tokens}
        for segment_id, text, tokens in segments
    ]
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), len(lines)


def export_unicode_tables():
    """Exhaustive per-codepoint lower/casefold/combining/category results."""
    buffers = {name: bytearray(spec["magic"]) for name, spec in UNICODE_TABLES.items()}
    max_lower = max_casefold = 0
    for cp in range(CODEPOINTS):
        char = chr(cp)
        low = char.lower()
        fold = char.casefold()
        max_lower = max(max_lower, len(low))
        max_casefold = max(max_casefold, len(fold))
        if len(low) > 3 or len(fold) > 3:
            raise SystemExit(f"U+{cp:04X}: mapping wider than the fixed-width layout")
        for name, value in (("lower", low), ("casefold", fold)):
            buffer = buffers[name]
            buffer += struct.pack("<IB", cp, len(value))
            buffer += b"".join(struct.pack("<I", ord(c)) for c in value.ljust(3, "\0"))
        buffers["combining"] += struct.pack("<IB", cp, unicodedata.combining(char))
        mask = 0
        for bit, pattern in CATEGORY_PATTERNS:
            if pattern.match(char):
                mask |= bit
        buffers["categories"] += struct.pack("<IB", cp, mask)
    for name, spec in UNICODE_TABLES.items():
        expected = 8 + CODEPOINTS * spec["record_bytes"]
        if len(buffers[name]) != expected:
            raise SystemExit(f"{name} table has {len(buffers[name])} bytes, want {expected}")
    return {name: _gzip_bytes(bytes(buffer)) for name, buffer in buffers.items()}


def export_ascii_runs(sag2):
    """`ascii_runs.jsonl`: the ASCII alphabet cross product + combining pairs."""
    runs = []
    for width in (2, 3):
        for combination in itertools.product(ASCII_RUN_POOL, repeat=width):
            runs.append("".join(combination))
    marks = [cp for cp in range(CODEPOINTS) if unicodedata.combining(chr(cp))]
    rng = random.Random(ASCII_RUN_SEED)
    for _ in range(ASCII_RUN_PAIR_COUNT):
        cp = SCALARS[rng.randrange(len(SCALARS))]
        runs.append(chr(cp) + chr(marks[rng.randrange(len(marks))]))
    lines = [{"text": text, "tokens": sag2.tokenize(text)} for text in runs]
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), len(lines)


def reference_provenance(reference_dir):
    """Identify the exact reference source set the goldens came from."""
    provenance = {}
    for key, name in REFERENCE_FILES.items():
        path = reference_dir / name
        if not path.is_file():
            raise SystemExit(f"reference {path} is missing; cannot record golden provenance")
        provenance[key] = _sha256_bytes(path.read_bytes())
    result = subprocess.run(
        ["git", "-C", str(reference_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"cannot determine the reference commit under {reference_dir}: "
            f"{result.stderr.strip()}"
        )
    provenance["commit"] = result.stdout.strip()
    return provenance


def build_manifest(
    fixtures,
    stats,
    files,
    reference,
    tokenizer,
    ascii_runs,
    unicode_tables,
    sparse_unicode,
    chunks,
    index_expected,
):
    corpus_digest = hashlib.sha256()
    for fixture in fixtures:
        corpus_digest.update(
            json.dumps(fixture.spec_digest_input(), ensure_ascii=False, sort_keys=True).encode()
        )
    return {
        "generator_version": GENERATOR_VERSION,
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "jieba": stats["jieba"],
        "pyuca": stats["pyuca"],
        "unicode": stats["unicode"],
        "regex": stats["regex"],
        "corpus_sha256": corpus_digest.hexdigest(),
        "reference": reference,
        "files": files,
        "tokenizer": tokenizer,
        "ascii_runs": ascii_runs,
        "unicode_tables": unicode_tables,
        "unicode_sparse": sparse_unicode,
        "chunks": chunks,
        "index_expected": index_expected,
        "fixtures": [
            {
                "fixture": entry["fixture"],
                "index": entry["index"],
                "index_sha256": entry["index_sha256"],
                "chunks": entry["chunks"],
                "cases": entry["cases"],
                "passed": entry["passed"],
                "errors": entry["errors"],
                "signature": entry["signature"],
            }
            for entry in stats["fixtures"]
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="export every fixture (C0-C3, C5)")
    parser.add_argument("--out", type=Path, default=Path("golden"), help="output directory")
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="read-only reference checkout scripts directory",
    )
    args = parser.parse_args(argv)
    if not args.all:
        parser.error("only '--all' is implemented in P0; pass --all")

    reference_dir = args.reference.resolve()
    sys.path.insert(0, str(reference_dir))
    import jieba  # noqa: E402 - import after the version guard and sys.path update
    import pyuca  # noqa: E402

    import sag2  # noqa: E402

    reference = json.loads((reference_dir.parent / "tests" / "sag2_reference.json").read_text())
    provenance = reference_provenance(reference_dir)

    fixtures = [
        fixture_c0(reference, sag2),
        fixture_c1(80, sag2),
        fixture_c1(140, sag2),
        fixture_c2(sag2),
        fixture_c3(sag2),
        fixture_c5(sag2),
    ]

    out_dir = args.out
    staging = Path(tempfile.mkdtemp(prefix="sag2-golden-"))
    work_dir = staging / "indexes"
    work_dir.mkdir(parents=True)
    try:
        retrieve_lines, dump_lines, fixture_stats = [], [], []
        index_expectations = []
        segments_by_fixture = {}
        for fixture in fixtures:
            db_path, signature, cases, rows, expectation = export_fixture(
                fixture, sag2, work_dir
            )
            index_expectations.append(expectation)
            segments_by_fixture[fixture.name] = fixture_segments(fixture, rows)
            fixture_stats.append(
                {
                    "fixture": fixture.name,
                    "index": db_path.name,
                    "index_sha256": _sha256_bytes(db_path.read_bytes()),
                    "chunks": len(rows),
                    "cases": len(cases),
                    "passed": sum(1 for case in cases if case["error"] is None),
                    "errors": sum(1 for case in cases if case["error"] is not None),
                    "signature": signature,
                }
            )
            for case in cases:
                line = {
                    "case_id": case["case_id"],
                    "fixture": fixture.name,
                    "expression": case["expression"],
                    "excluded": case["excluded"],
                    "limit": case["limit"],
                    "term": case["term"],
                    "ids": case["ids"],
                    "document_ids": case["document_ids"],
                    "scores_bits": case["scores_bits"],
                    "hits": case["hits"],
                }
                if case["error"] is not None:
                    line["error"] = case["error"]
                retrieve_lines.append(line)
            dump_lines.append({"type": "metadata", "fixture": fixture.name, "signature": signature})
            for row in rows:
                dump_lines.append({"type": "chunk", "fixture": fixture.name, **row})
            print(
                f"{fixture.name}: {len(fixture.corpus)} documents, {len(rows)} chunks, "
                f"{len(cases)} cases ({fixture_stats[-1]['errors']} expected errors)"
            )

        if len(retrieve_lines) < 200:
            raise SystemExit(
                f"only {len(retrieve_lines)} cases exported; Gate A needs at least 200"
            )

        retrieve_text = "".join(
            json.dumps(line, ensure_ascii=False) + "\n" for line in retrieve_lines
        )
        dump_text = "".join(
            json.dumps(line, ensure_ascii=False) + "\n" for line in dump_lines
        )
        chunk_cases = export_chunk_cases(fixtures, reference_dir, sag2)
        chunk_text = "".join(
            json.dumps(line, ensure_ascii=False) + "\n" for line in chunk_cases
        )
        index_expected_text = "".join(
            json.dumps(line, ensure_ascii=False) + "\n"
            for line in index_expectations
        )
        tokenizer_text, tokenizer_lines = export_tokenizer(
            reference, segments_by_fixture, sag2
        )
        ascii_text, ascii_lines = export_ascii_runs(sag2)
        unicode_blobs = export_unicode_tables()
        sparse_blobs, sparse_metadata = export_sparse_unicode_tables()

        files = [
            {"path": "retrieve.jsonl", "sha256": _sha256_bytes(retrieve_text.encode()),
             "lines": len(retrieve_lines)},
            {"path": "index_dump.jsonl", "sha256": _sha256_bytes(dump_text.encode()),
             "lines": len(dump_lines)},
            {"path": "tokenizer.jsonl", "sha256": _sha256_bytes(tokenizer_text.encode()),
             "lines": tokenizer_lines},
            {"path": "ascii_runs.jsonl", "sha256": _sha256_bytes(ascii_text.encode()),
             "lines": ascii_lines},
            {"path": "chunks.jsonl", "sha256": _sha256_bytes(chunk_text.encode()),
             "lines": len(chunk_cases)},
            {"path": "index_expected.jsonl",
             "sha256": _sha256_bytes(index_expected_text.encode()),
             "lines": len(index_expectations)},
        ]
        for name, spec in UNICODE_TABLES.items():
            blob = unicode_blobs[name]
            files.append(
                {
                    "path": spec["path"],
                    "sha256": _sha256_bytes(blob),
                    "records": CODEPOINTS,
                    "record_bytes": spec["record_bytes"],
                    "uncompressed_bytes": 8 + CODEPOINTS * spec["record_bytes"],
                    "gzip_bytes": len(blob),
                }
            )
        for name, spec in SPARSE_UNICODE_TABLES.items():
            blob = sparse_blobs[name]
            files.append(
                {
                    "path": spec["path"],
                    "sha256": _sha256_bytes(blob),
                    "records": sparse_metadata[name]["records"],
                    "gzip_bytes": len(blob),
                }
            )
        manifest = build_manifest(
            fixtures,
            {
                "jieba": jieba.__version__,
                "pyuca": pyuca.Collator.UCA_VERSION,
                "unicode": unicodedata.unidata_version,
                "regex": regex.__version__,
                "fixtures": fixture_stats,
            },
            files,
            provenance,
            {"segments": tokenizer_lines, "golden": "tokenizer.jsonl"},
            {
                "runs": ascii_lines,
                "golden": "ascii_runs.jsonl",
                "pair_count": ASCII_RUN_PAIR_COUNT,
                "pair_seed": ASCII_RUN_SEED,
                "pool": ASCII_RUN_POOL,
            },
            {
                name: {
                    "path": spec["path"],
                    "magic": spec["magic"].decode(),
                    "record_bytes": spec["record_bytes"],
                    "fields": spec["fields"],
                    "codepoints": CODEPOINTS,
                }
                for name, spec in UNICODE_TABLES.items()
            },
            sparse_metadata,
            {
                "documents": len(chunk_cases),
                "chunks": sum(len(case["chunks"]) for case in chunk_cases),
                "scenarios": sum(
                    1 for case in chunk_cases if case["fixture"] == "chunking"
                ),
                "golden": "chunks.jsonl",
            },
            {
                "fixtures": len(index_expectations),
                "golden": "index_expected.jsonl",
            },
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        index_dir = out_dir / "indexes"
        index_dir.mkdir(exist_ok=True)
        for entry in fixture_stats:
            shutil.copyfile(work_dir / entry["index"], index_dir / entry["index"])
        (out_dir / "retrieve.jsonl").write_text(retrieve_text)
        (out_dir / "index_dump.jsonl").write_text(dump_text)
        (out_dir / "tokenizer.jsonl").write_text(tokenizer_text)
        (out_dir / "ascii_runs.jsonl").write_text(ascii_text)
        (out_dir / "chunks.jsonl").write_text(chunk_text)
        (out_dir / "index_expected.jsonl").write_text(index_expected_text)
        for name, spec in UNICODE_TABLES.items():
            (out_dir / spec["path"]).write_bytes(unicode_blobs[name])
        for name, spec in SPARSE_UNICODE_TABLES.items():
            (out_dir / spec["path"]).write_bytes(sparse_blobs[name])
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            f"wrote {len(retrieve_lines)} retrieve cases, {len(dump_lines)} index rows, "
            f"{tokenizer_lines} tokenizer segments, {ascii_lines} ASCII runs, "
            f"{len(chunk_cases)} chunk cases, {len(index_expectations)} index expectations, "
            f"{len(UNICODE_TABLES)} dense + {len(SPARSE_UNICODE_TABLES)} sparse unicode "
            f"tables and {len(fixture_stats)} indexes "
            f"to {out_dir}"
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
