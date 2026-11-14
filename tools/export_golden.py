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
``tokenizer.jsonl``
    every tokenizer segment: the four frozen reference lines plus every
    document title/text and every chunk content of the Gate A/D fixtures and
    of C4 (the repository prose) and C7 (the alias/grounding corpus), so the
    mixed CJK/ASCII HMM branch is covered by the tokenizer gate itself
``index_expected.jsonl``
    per fixture: the ``sqlite_master`` schema, ``metadata.signature``, every
    ``chunks`` row, every ``chunk_fts`` row, the 24 ``(rowid, bm25)`` probes,
    the ``chunk_fts`` vocabulary per column, ``index_statistics`` and the
    ``rust_index_metadata`` key set (Gate D's Rust-vs-Python baseline)
``sources/<fixture>/``
    the source tree of a ``source_dir`` fixture (``content=''``)
``manifest.json``
    generator/python/sqlite/jieba/pyuca/unicode versions, the reference source
    digests and commit, corpus digest, file digests and the fixture -> index
    mapping
``unicode_nfd.bin.gz``
    canonical NFD of every code point (pyuca's `sort_key` starts with it; the
    tokenizer's tables only cover NFKD/NFKC)
``sk_vocab.jsonl``, ``sk_cp.bin.gz``, ``sk_pairs.bin.gz``
    Gate B's sort-key goldens (design §5.5): the vocabulary keys, the key of
    every single-code-point string and 200,000 ordered comparison pairs
``expansion.jsonl``
    Gate B's expansion goldens (design §1.2): one line per ``(query, hop)``,
    carrying the replay inputs (seeds, base expression, excluded ids, limits)
    and the ordered ``terms`` / ``term_scores`` / ``queries`` / ``hits`` the
    reference produced for that hop
``search.jsonl.gz``, ``search_corpus.jsonl``
    Gate C's frozen judge trace (design §1.3): one line per search case with
    the byte-exact request payloads, their journal digests, the ranked
    candidates, the trace records and the judge usage; and the corpus + chunk
    parameters each case's index is built from (C0 at ``index_workers`` 1 and
    4, C4 and C3 at 1)
``freq_keys.txt.gz``, ``freq_probes.jsonl``
    Gate D's `freq` section (design §4.1 rule 5): every short (<= 3 code point)
    positive ``TOKENIZER.FREQ`` key in UTF-8 byte order, and sampled keys with
    the reference ``tokenize`` output.  The sample includes every anchor
    position and every short key the gate corpora contain, so the anchor
    boundary of the compiled table cannot go untested again.
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
import asyncio
import gzip
import hashlib
import io
import itertools
import json
import os
import random
import re
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
GENERATOR_VERSION = "0.8.1"

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

    def __init__(
        self,
        name,
        fingerprint,
        corpus,
        chunk_chars,
        case_builder,
        probes=(),
        source_relative=None,
        queries=(),
    ):
        self.name = name
        self.fingerprint = fingerprint
        self.corpus = corpus
        self.chunk_chars = chunk_chars
        self.case_builder = case_builder
        # `(expression, term)` pairs replayed by Gate D4 (design §1.4).
        self.probes = list(probes)
        # Relative path of the checked-in source tree; non-`None` builds in
        # `source_dir` mode (`content=''`, `start_char`/`end_char`).
        self.source_relative = source_relative
        self.source_dir = None
        # The corpus' own natural queries, used by the Gate B expansion export
        # (design §1.2 sample corpora).
        self.queries = list(queries)

    def spec_digest_input(self):
        return {
            "fixture": self.name,
            "fingerprint": self.fingerprint,
            "chunk_chars": self.chunk_chars,
            "corpus": self.corpus,
            "source_dir": self.source_relative,
        }


# --------------------------------------------------------------------------
# C0 - tests/sag2_reference.json: 23 documents, 4 frozen queries.
# --------------------------------------------------------------------------


def fixture_c0(reference, sag2):
    corpus = reference["corpus"]
    exprs = [
        (f"query/{index}", sag2.match_query(case["query"]))
        for index, case in enumerate(reference["cases"])
    ]
    exprs += [
        ("literal/pair", '"alice paul" OR "bob smith"'),
        ("column/title", 'title : "anchor"'),
    ]

    def cases(_db):
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

    # Gate D4 (design §1.4): 12 probes, every C0 expression in its plain and
    # its `term`-qualified form.  Each term is chosen to co-occur with its
    # expression, so no probe is vacuous (an empty probe asserts nothing).
    qualified_terms = {
        '"anchor" OR "birthplace" OR "depicts"': "alice",
        '"anchor"': "alice",
        '"alston" OR "born"': "born",
        '"rene" OR "descartes" OR "born"': "born",
        '"alice paul" OR "bob smith"': "alice",
        'title : "anchor"': "alice",
    }
    probes = [(expression, None) for _, expression in exprs]
    probes += [
        (expression, qualified_terms[expression]) for _, expression in exprs
    ]
    return Fixture("C0", C0_FINGERPRINT, corpus, 2200, cases, probes=probes)


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

    return Fixture(
        "C3",
        "scale",
        corpus,
        10000,
        cases,
        queries=[f"{left} {right}" for left, right in queries],
    )


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

    # Gate D4: 12 of the 16 C5 query expressions (design §1.4 counts C0 12 +
    # C5 12 = 24 probes).
    probes = [(sag2.match_query(query), None) for query in C5_QUERIES[:12]]
    return Fixture("C5", "unicode", corpus, 10000, cases, probes=probes)


def _matching_ids(db, expression):
    return [row[0] for row in db.execute(
        "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?", (expression,)
    )]


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def export_fixture(fixture, sag2, work_dir):
    """Build the fixture index, run every case, return its Gate A/D payload."""
    db_path = work_dir / f"{fixture.name}.sqlite"
    index = sag2.CorpusIndex(
        db_path,
        fixture.corpus,
        fixture.fingerprint,
        chunk_chars=fixture.chunk_chars,
        source_dir=fixture.source_dir,
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
        expectation = index_expectation(fixture, index, sag2)
        rows = [
            dict(row) for row in index.db.execute("SELECT * FROM chunks ORDER BY id")
        ]
        return db_path, cases, rows, expectation
    finally:
        index.close()


# --------------------------------------------------------------------------
# C6 - `source_dir` mode (design §1.4 sample corpora): contentless FTS5
#      (`content=''`), `start_char`/`end_char` instead of `content`/`full_doc`,
#      the `source_files` provenance table and a statistics corpus read back
#      from it.  The source tree is exported to `golden/sources/C6-source`
#      because `sag2.py:203` reads the text from the tree, not from the corpus
#      row (F-005).
# --------------------------------------------------------------------------

C6_CHUNK_CHARS = 512
C6_SOURCE_RELATIVE = "sources/C6-source"
C6_GUIDE = (
    "# Release gate\n\n"
    "Does the gate compare spans? Yes, and it compares them byte for byte. "
    "Is the answer short? No, the gate compares whole rows.\n\n"
    "## Evidence\n" + "The gate replays every probe against the rebuilt index. " * 12
)
C6_NOTES = "\n".join(
    f"Why does line {line} matter? Because the source tree is the corpus."
    for line in range(1, 25)
)
C6_TOML = "\n".join(
    ["[build]", 'target = "release"', "jobs = 8", "[gate]", 'fixture = "C6-source"']
)


def fixture_c6(sag2):
    documents = [
        ("Release gate", "guide.md", C6_GUIDE),
        ("Field notes", "notes.txt", C6_NOTES),
        ("Build settings", "config.toml", C6_TOML),
        # Same stripped title and content as `guide.md`: the build dedups it
        # before writing `source_files` or the statistics corpus
        # (`sag2.py:269-290`).
        ("Release gate", "guide-copy.md", C6_GUIDE),
    ]
    corpus = [
        {
            "title": title,
            "text": text,
            "path": path,
            "sha256": _sha256_bytes(text.encode("utf-8")),
        }
        for title, path, text in documents
    ]

    def cases(_db):
        return []

    # Gate D4 must also exercise the contentless index: these three
    # expressions bind real rows there (the FTS5 `content=''` table stores the
    # term index even though it returns NULL column values).
    probes = [
        (sag2.match_query(query), None)
        for query in ("gate", "corpus", "release")
    ]
    return Fixture(
        "C6-source",
        "source",
        corpus,
        C6_CHUNK_CHARS,
        cases,
        probes=probes,
        source_relative=C6_SOURCE_RELATIVE,
    )


def probe_expressions(fixtures):
    """The design §1.4 D4 probe count (C0 12 + C5 12)."""
    return sum(len(fixture.probes) for fixture in fixtures)


# --------------------------------------------------------------------------
# C4 - real text, mixed scripts (design §1.0, §1.2): the repository's own
#      prose, with the queries taken from the chunk headings.  C4 runs only in
#      Gate B's expansion replay; it is not a Gate A/D fixture, so it stays out
#      of `retrieve.jsonl`, `index_expected.jsonl` and the sort-key vocabulary.
# --------------------------------------------------------------------------

C4_CHUNK_CHARS = 350
C4_DOCUMENTS = (
    "README-CN.md",
    "README.md",
    "sag2/docs/agentic.md",
    "sag2/docs/local-rerank.md",
)
C4_QUERY_LIMIT = 20


def fixture_c4(reference_dir):
    root = reference_dir.parent.parent
    corpus = []
    for relative in C4_DOCUMENTS:
        text = (root / relative).read_text(encoding="utf-8")
        title = next(
            (line[2:].strip() for line in text.splitlines() if line.startswith("# ")),
            Path(relative).name,
        )
        corpus.append({"title": title, "text": text, "path": relative})
    return Fixture("C4", "realmix", corpus, C4_CHUNK_CHARS, lambda _db: [])


def c4_queries(fixture, sag2):
    """Twenty queries, one per distinct tokenizable chunk heading."""
    import chunking  # noqa: PLC0415 - imported after `sys.path` gains the reference

    queries, seen = [], set()
    for document in fixture.corpus:
        for chunk in chunking.chunk_document(
            document["text"], fixture.chunk_chars, document["path"]
        ):
            heading = chunk["heading"].strip()
            if not heading or heading in seen or not sag2.tokenize(heading):
                continue
            seen.add(heading)
            queries.append(heading)
            if len(queries) == C4_QUERY_LIMIT:
                return queries
    raise SystemExit(f"C4 offers only {len(queries)} heading queries; need {C4_QUERY_LIMIT}")


# --------------------------------------------------------------------------
# Gate B (expansion) - design §1.2.  The Python `search_sag2` run is the only
# thing that can produce the `(query, hop)` inputs: hop `h`'s seeds are hop
# `h-1`'s ranked frontier, and `seen` is the set of already-scored ids.  So the
# export drives the real search with the tests' fixed `OracleJev`
# (`tests/test_sag2.py:31-62`) and records the arguments and results of the two
# calls design §1.2 makes accountable: `expansion_terms` and
# `retrieve_expansion`.
# --------------------------------------------------------------------------


def oracle_jev(sag2):
    """`tests/test_sag2.py:31-62` `OracleJev`, byte for byte."""

    class OracleJev(sag2.JevClient):
        """Fixed judgments used by the independent frozen SAG2 reference run."""

        def __init__(self):
            super().__init__(None, "http://localhost/decisions", "jev-latest", "local-test")
            self.payloads = []

        async def request(self, payload):
            self.payloads.append(payload)
            state = json.loads(payload["state"])
            evidence = [row["title"] for row in state["previously_retrieved_evidence"]]
            answers = {}
            for key, question in payload["questions"].items():
                title = question["instructions"]["candidate_title"]
                score = 0.01
                if title == "Anchor":
                    score = 0.8
                elif title == "Multiple":
                    score = 0.95 if "Anchor" in evidence else 0.1
                elif title == "Bridge":
                    score = 0.99 if "Multiple" in evidence else 0.05
                elif title == "High":
                    score = 0.7
                elif title == "Series":
                    score = 0.08
                answers[key] = {"type": "noul", "noul": score}
            return {
                "model": "oracle-jev",
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "request_attempts": 1,
            }

    return OracleJev()


def _serialise_term_scores(terms):
    """`expansion_terms`' output, with every `f64` kept as raw bits (§1.0)."""
    scores = []
    for term in terms:
        record = {
            "term": term["term"],
            "score_bits": _hex_bits(term["score"]),
            "unweighted_score_bits": _hex_bits(term["unweighted_score"]),
            "seed_weight_bits": _hex_bits(term["seed_weight"]),
            "seed_jev_score_bits": _hex_bits(term["seed_jev_score"]),
            "term_frequency": term["term_frequency"],
            "document_frequency": term["document_frequency"],
            "query_context_score_bits": _hex_bits(term["query_context_score"]),
            "seed_document_id": term["seed_document_id"],
            "kind": term["kind"],
        }
        if "name" in term:
            record["name"] = term["name"]
        scores.append(record)
    return scores


def _serialise_queries(queries):
    """`retrieve_expansion`'s per-term records, `f64`s as raw bits (§1.0)."""
    return [
        {
            "term": entry["term"],
            "term_rank": entry["term_rank"],
            "kind": entry["kind"],
            "match_expression": entry["match_expression"],
            "ranking_expression": entry["ranking_expression"],
            "retrieved_count": entry["retrieved_count"],
            "selected": [
                {
                    "document_id": selected["document_id"],
                    "bm25_rank": selected["bm25_rank"],
                    "bm25_score_bits": _hex_bits(selected["bm25_score"]),
                }
                for selected in entry["selected"]
            ],
        }
        for entry in queries
    ]


def _serialise_hits(hits):
    """The ordered interleaved hits, with only the fields §1.2 compares."""
    return [
        {
            "id": hit["id"],
            "document_id": hit["document_id"],
            "score_bits": _hex_bits(hit["score"]),
            "expansion_term": hit["expansion_term"],
            "expansion_term_rank": hit["expansion_term_rank"],
            "expansion_bm25_rank": hit["expansion_bm25_rank"],
        }
        for hit in hits
    ]


def _expansion_lines(index, sag2, name, position, query, options):
    """One `(query, hop)` line per expansion round `search_sag2` performs."""
    client = oracle_jev(sag2)
    state = {"pending": None, "calls": []}
    original_terms = index.expansion_terms
    original_retrieve = index.retrieve_expansion

    def expansion_terms(text, seeds, limit):
        terms = original_terms(text, seeds, limit)
        # `retrieve_expansion` copies each term before grounding it
        # (`sag2.py:668`), so these values are the ones the trace records.
        state["pending"] = {
            "seeds": [
                {key: seed[key] for key in ("document_id", "jev_score", "content")}
                for seed in seeds
            ],
            "term_limit": limit,
            "terms": [term["term"] for term in terms],
            "term_scores": _serialise_term_scores(terms),
        }
        return terms

    def retrieve_expansion(expression, terms, excluded, limit):
        hits, queries = original_retrieve(expression, terms, excluded, limit)
        pending = state["pending"]
        state["pending"] = None
        state["calls"].append(
            {
                **pending,
                "expression": expression,
                "excluded": sorted(excluded),
                "limit": limit,
                "queries": _serialise_queries(queries),
                "hits": _serialise_hits(hits),
            }
        )
        return hits, queries

    index.expansion_terms = expansion_terms
    index.retrieve_expansion = retrieve_expansion
    try:
        asyncio.run(sag2.search_sag2(index, client, query, **(options or {})))
    finally:
        del index.expansion_terms
        del index.retrieve_expansion
    if not state["calls"]:
        raise SystemExit(f"{name}: query {query!r} produced no expansion round")
    return [
        {
            "case_id": f"{name}/q{position:02d}/hop{hop}",
            "stage": "expansion",
            "fixture": name,
            "query": query,
            "hop": hop,
            **call,
        }
        for hop, call in enumerate(state["calls"], start=1)
    ]


def export_expansion(fixture, queries, sag2, work_dir):
    """Export Gate B's `(query, hop)` lines for one fixture's index."""
    db_path = work_dir / f"{fixture.name}.sqlite"
    index = sag2.CorpusIndex(
        db_path, fixture.corpus, fixture.fingerprint, chunk_chars=fixture.chunk_chars
    )
    try:
        signature = index.db.execute("SELECT signature FROM metadata").fetchone()[0]
        chunks = index.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        lines = []
        for position, (query, options) in enumerate(queries):
            lines.extend(_expansion_lines(index, sag2, fixture.name, position, query, options))
    finally:
        index.close()
    return (
        {
            "fixture": fixture.name,
            "index": db_path.name,
            "index_sha256": _sha256_bytes(db_path.read_bytes()),
            "chunks": chunks,
            "signature": signature,
            "records": len(lines),
        },
        lines,
    )


# --------------------------------------------------------------------------
# C7 - `repair_names` and `title_phrases` grounding (design §6.2, §6.3).
#
# Neither branch is reachable through `search_sag2` on a corpus whose own
# terms are present in it: a term cut out of a seed sentence is by construction
# an adjacent body phrase, so `has_match` is always true.  The reference tests
# the two branches by calling `retrieve_expansion` with explicit terms
# (`tests/test_sag2.py:150-165`), so Gate B does the same here, on an index
# whose titles create real aliases ("Ångström"/"Zürich" fold to different
# strings, "Alice Paul Smith" is a title phrase).
# --------------------------------------------------------------------------

C7_CHUNK_CHARS = 2200
C7_FINGERPRINT = "names"
C7_CORPUS = (
    ("Ångström", "Ångström measures light at the guild."),
    ("Zürich", "Zürich guild hall stands by the zurich plaza."),
    ("Alice Paul Smith", "Alice Paul Smith spoke to the guild."),
)

# Direct `retrieve_expansion` calls: (input terms, exclusions, limit).
C7_RETRIEVALS = (
    (
        "repair-changes-the-term",
        [{"term": "angstrom unit", "kind": "word"}],
        [],
        5,
    ),
    (
        "repair-keeps-a-present-name",
        [{"term": "zurich plaza", "kind": "word"}],
        [],
        5,
    ),
    (
        "grounds-a-name-phrase-on-a-title",
        [{"term": "alice paul smith jones", "kind": "name_phrase"}],
        [],
        5,
    ),
    (
        "present-term-skips-grounding",
        [{"term": "alice paul smith", "kind": "name_phrase"}],
        [],
        5,
    ),
    (
        "interleaves-two-terms",
        [
            {"term": "alice paul smith", "kind": "name_phrase"},
            {"term": "guild", "kind": "word"},
        ],
        [],
        4,
    ),
    (
        "exclusion-blocks-the-original",
        [{"term": "alice paul smith jones", "kind": "name_phrase"}],
        [3],
        3,
    ),
)

# `repair_names` case matrix (design §6.2: 别名 × 大小写变体).
C7_REPAIR_TEXTS = (
    "Ångström",
    "ångström",
    "ÅNGSTRÖM",
    "Angstrom",
    "ANGSTROM",
    "angstrom",
    "aNgStRöM",
    "Zürich",
    "zurich",
    "ZURICH",
    "Zurich",
    "Alice Paul Smith",
    "alice paul smith",
    "ALICE PAUL SMITH",
    "tokens angstrom and zurich",
    "no alias here",
)
C7_REPAIR_EXCLUSIONS = ([], [1])


def c7_fixture():
    """The C7 alias/grounding corpus (shared by the expansion and freq exports)."""
    return Fixture(
        "C7",
        C7_FINGERPRINT,
        [{"title": title, "text": text} for title, text in C7_CORPUS],
        C7_CHUNK_CHARS,
        lambda _db: [],
    )


def export_named_expansion(sag2, work_dir):
    """The C7 sub-cases: direct `retrieve_expansion` and `repair_names` calls."""
    fixture = c7_fixture()
    db_path = work_dir / f"{fixture.name}.sqlite"
    index = sag2.CorpusIndex(
        db_path, fixture.corpus, fixture.fingerprint, chunk_chars=fixture.chunk_chars
    )
    try:
        signature = index.db.execute("SELECT signature FROM metadata").fetchone()[0]
        chunks = index.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        lines = []
        for label, terms, excluded, limit in C7_RETRIEVALS:
            hits, queries = index.retrieve_expansion(
                sag2.match_query("guild"), [dict(term) for term in terms], set(excluded), limit
            )
            lines.append(
                {
                    "case_id": f"{fixture.name}/retrieve/{label}",
                    "stage": "retrieve",
                    "fixture": fixture.name,
                    "expression": sag2.match_query("guild"),
                    "excluded": sorted(excluded),
                    "limit": limit,
                    "input_terms": [dict(term) for term in terms],
                    "queries": _serialise_queries(queries),
                    "hits": _serialise_hits(hits),
                }
            )
        for position, text in enumerate(C7_REPAIR_TEXTS):
            for excluded in C7_REPAIR_EXCLUSIONS:
                lines.append(
                    {
                        "case_id": f"{fixture.name}/repair/{position:02d}/excluded{len(excluded)}",
                        "stage": "repair",
                        "fixture": fixture.name,
                        "text": text,
                        "excluded": sorted(excluded),
                        "repaired": index.repair_names(text, set(excluded)),
                    }
                )
    finally:
        index.close()
    return (
        {
            "fixture": fixture.name,
            "index": db_path.name,
            "index_sha256": _sha256_bytes(db_path.read_bytes()),
            "chunks": chunks,
            "signature": signature,
            "records": len(lines),
        },
        lines,
    )


# --------------------------------------------------------------------------
# Gate C (search) - design §1.3.  The reference `search_sag2` under the fixed
# `OracleJev` is the only thing that can produce the payload bytes and the
# trace; the export records both halves the gate compares:
#
#   C-1  every request payload as the exact `compact_json` string, plus the
#        `sha256(compact_json([endpoint, payload]))` the journal keys on; and
#   C-2  the ranked ids/scores, the trace records and the `jev_usage` counters.
#
# The corpora are shipped alongside (`search_corpus.jsonl`) because the
# verifier builds the indexes itself: C0 at `index_workers` 1 and 4 (design
# §2.5: a parallel build must be equivalent), C4 and C3 at 1.
# --------------------------------------------------------------------------

SEARCH_ENDPOINT = "http://localhost/decisions"

# The defaults of `search_sag2` (`sag2.py:930-940`), recorded explicitly so the
# verifier never has to guess what a fixture ran with.
SEARCH_DEFAULTS = {
    "candidate_k": 100,
    "seed_k": 10,
    "max_hops": 2,
    "expansion_term_limit": 20,
}


def _serialise_batch(batch):
    """One `jev_usage.batches[]` entry (`sag2.py:922-927`), field for field."""
    return {
        "candidate_count": batch["candidate_count"],
        "estimated_input_tokens": batch["estimated_input_tokens"],
        "request_attempts": batch["request_attempts"],
        "http_attempts": batch["http_attempts"],
        "failed_attempts": dict(batch["failed_attempts"]),
        "input_tokens": batch["input_tokens"],
        "output_tokens": batch["output_tokens"],
        "model": batch["model"],
    }


def _serialise_usage(usage):
    """`jev_usage` (`sag2.py:915-921`); batches carry their hop (`:946`)."""
    return {
        "requests": usage["requests"],
        "judgments": usage["judgments"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "batches": [
            {"hop": batch["hop"], **_serialise_batch(batch)}
            for batch in usage["batches"]
        ],
    }


def _serialise_search_trace(trace):
    """The two trace record shapes (`sag2.py:969-980`, `:1014-1031`)."""
    records = []
    for entry in trace:
        if entry["stage"] in ("bm25", "expansion"):
            records.append(
                {
                    "stage": entry["stage"],
                    "hop": entry["hop"],
                    "count": entry["count"],
                    "candidate_document_ids": entry["candidate_document_ids"],
                    "candidate_ids": entry["candidate_ids"],
                    "terms": entry["terms"],
                    "term_scores": _serialise_term_scores(entry["term_scores"]),
                    "expansion_queries": _serialise_queries(entry["expansion_queries"]),
                }
            )
        elif entry["stage"] == "jev":
            records.append(
                {
                    "stage": entry["stage"],
                    "hop": entry["hop"],
                    "count": entry["count"],
                    "carried_seed_count": entry["carried_seed_count"],
                    "protected_bm25_count": entry["protected_bm25_count"],
                    "evidence_ids": entry["evidence_ids"],
                    "scores": [
                        {"id": score["id"], "score_bits": _hex_bits(score["score"])}
                        for score in entry["scores"]
                    ],
                    "batches": [_serialise_batch(batch) for batch in entry["batches"]],
                }
            )
        else:
            raise SystemExit(f"search: unknown trace stage {entry['stage']!r}")
    return records


def _payload_digest(payload):
    """`sha256(compact_json([endpoint, payload]))` (`model_api.py:48`)."""
    return _sha256_bytes(
        json.dumps([SEARCH_ENDPOINT, payload], ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _search_case(name, workers, position, query, options, client, result):
    return {
        "case_id": f"{name}/w{workers}/q{position:02d}",
        "fixture": name,
        "index_workers": workers,
        "query": query,
        "options": options,
        "endpoint": SEARCH_ENDPOINT,
        "payloads": [json.dumps(payload, ensure_ascii=False, separators=(",", ":")) for payload in client.payloads],
        "payloads_sha256": [_payload_digest(payload) for payload in client.payloads],
        "ranked_ids": [candidate["id"] for candidate in result["candidates"]],
        "ranked_scores_bits": [
            _hex_bits(candidate["jev_score"]) for candidate in result["candidates"]
        ],
        "baseline_ids": [candidate["id"] for candidate in result["baseline"]],
        "trace": _serialise_search_trace(result["trace"]),
        "jev_usage": _serialise_usage(result["jev_usage"]),
        "algorithm": result["algorithm"],
    }


def _search_corpus(fixture, chunk_chars):
    documents = []
    for document in fixture.corpus:
        record = {"title": document.get("title", ""), "text": document["text"]}
        if "path" in document:
            record["path"] = document["path"]
        if "idx" in document:
            record["idx"] = document["idx"]
        documents.append(record)
    return {
        "fixture": fixture.name,
        "fingerprint": fixture.fingerprint,
        "chunk_chars": chunk_chars,
        "corpus": documents,
    }


def export_search(reference, fixtures, c4, sag2, work_dir):
    """Gate C: the frozen `search_sag2` case set (design §1.3)."""
    c0 = next(fixture for fixture in fixtures if fixture.name == "C0")
    c3 = next(fixture for fixture in fixtures if fixture.name == "C3")
    specs = [
        (
            c0,
            c0.chunk_chars,
            (1, 4),
            [(case["query"], case["options"]) for case in reference["cases"]],
        ),
        (c4, c4.chunk_chars, (1,), [(query, None) for query in c4_queries(c4, sag2)]),
        (c3, c3.chunk_chars, (1,), [(query, None) for query in c3.queries]),
    ]
    corpora, cases = [], []
    for fixture, chunk_chars, worker_counts, queries in specs:
        corpora.append(_search_corpus(fixture, chunk_chars))
        for workers in worker_counts:
            db_path = work_dir / f"{fixture.name}-search-w{workers}.sqlite"
            index = sag2.CorpusIndex(
                db_path,
                fixture.corpus,
                fixture.fingerprint,
                chunk_chars=chunk_chars,
                index_workers=workers,
            )
            try:
                for position, (query, options) in enumerate(queries):
                    resolved = {**SEARCH_DEFAULTS, **(options or {})}
                    client = oracle_jev(sag2)
                    result = asyncio.run(
                        sag2.search_sag2(index, client, query, **resolved)
                    )
                    cases.append(
                        _search_case(
                            fixture.name, workers, position, query, resolved, client, result
                        )
                    )
            finally:
                index.close()
        print(
            f"{fixture.name}: {len(queries)} search queries at index_workers "
            f"{list(worker_counts)}"
        )
    return corpora, cases


# --------------------------------------------------------------------------
# Gate D `freq` - rule 5's decision table (design §4.1).  The compiled table
# stores one entry per short positive `TOKENIZER.FREQ` key and finds it with a
# binary search over anchors every `ANCHOR_EVERY` entries.  A lookup that scans
# only one side of an anchor silently drops 1/ANCHOR_EVERY of the dictionary
# (P2b found exactly that: 4,017 of 257,096 entries), and the P1 tokenizer
# goldens contained no anchor key at all, so Gate D could not see it.  This
# export ships the whole key set and a probe sample that covers every anchor
# position by construction.
# --------------------------------------------------------------------------

FREQ_PROBE_COUNT = 5000
FREQ_PROBE_SEED = 20261115


def freq_constants():
    """`FREQ_MAX_CHARS` / `ANCHOR_EVERY`, read from the table generator."""
    source = (
        Path(__file__).resolve().parent.parent
        / "crates"
        / "sag2-core"
        / "build.rs"
    ).read_text()
    values = {}
    for name in ("FREQ_MAX_CHARS", "ANCHOR_EVERY"):
        match = re.search(rf"const {name}: usize = (\d+);", source)
        if match is None:
            raise SystemExit(f"cannot find {name} in sag2-core/build.rs")
        values[name] = int(match.group(1))
    return values["FREQ_MAX_CHARS"], values["ANCHOR_EVERY"]


def export_freq(sag2, fixtures):
    """The short `TOKENIZER.FREQ` key set and deterministic tokenizer probes."""
    max_chars, stride = freq_constants()
    # `TOKENIZER.FREQ` is built lazily on the first cut (`jieba.Tokenizer`).
    sag2.tokenize("")
    keys = sorted(
        key
        for key, value in sag2.TOKENIZER.FREQ.items()
        if len(key) <= max_chars and value > 0
    )
    key_set = set(keys)
    # Every anchor position, so the probe list cannot miss one by sampling.
    probes = {key for position, key in enumerate(keys) if position % stride == 0}
    anchors = len(probes)
    # Every short key the gate corpora contain, so real text is covered too.
    for fixture in fixtures:
        for document in fixture.corpus:
            for text in (document["title"], document["text"]):
                probes.update(
                    token for token in sag2.tokenize(text) if token in key_set
                )
    corpus_probes = len(probes) - anchors
    remaining = [key for key in keys if key not in probes]
    random.Random(FREQ_PROBE_SEED).shuffle(remaining)
    probes.update(remaining[: max(0, FREQ_PROBE_COUNT - len(probes))])
    ordered = sorted(probes)
    lines = [{"key": key, "tokens": sag2.tokenize(key)} for key in ordered]
    text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    blob = _gzip_bytes(("\n".join(keys) + "\n").encode("utf-8"))
    return blob, text, {
        "keys": len(keys),
        "max_chars": max_chars,
        "anchor_every": stride,
        "anchor_entries": anchors,
        "probes": len(lines),
        "corpus_probes": corpus_probes,
        "golden": "freq_keys.txt.gz",
        "probes_golden": "freq_probes.jsonl",
    }


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
    # F-001: `?` is a prose sentence boundary (`chunking.py:29-39`).  This
    # document is far longer than the chunk target and mixes the three shapes
    # the greedy skip handles: `?` + space, `?` + closer and `?` + no space.
    questions = (
        "# Gate review\n\n"
        "Why does the gate compare spans? Because spans are the chunk contract. "
        'Does it compare bytes? "Yes", byte for byte. '
        "Is the answer short?No, the gate compares whole rows.\n\n"
        + "Does every probe replay the ranking SQL? It does, row for row. " * 20
    )
    # F-002: `.toml` is one of the 45 `CODE_SUFFIXES`, so this document is
    # chunked on newlines (code mode), not as prose.
    toml_lines = ["[build]", 'target = "release"', "jobs = 8", "", "[gate]"]
    toml_lines += [f'check_{index} = "the gate keeps the binding"' for index in range(45)]
    scenarios = [
        (
            "chunking/long-sentence-overlap",
            "# 发布指南\n## 生产环境\n" + first + long + last,
            None,
            None,
            512,
        ),
        (
            "chunking/prose-question-boundary",
            questions,
            "guide.md",
            "guide.md",
            512,
        ),
        (
            "chunking/code-suffix-toml",
            "\n".join(toml_lines) + "\n",
            "config.toml",
            "config.toml",
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
# Gate D (index) - design §1.4 D1-D4: the schema, the signature, every
# `chunks` row, every `chunk_fts` row, the 24 `(rowid, bm25)` probes, the
# per-column vocabulary, the cached statistics and the provenance key set.  The
# reference index itself is the P0 `indexes/<fixture>.sqlite`.
# --------------------------------------------------------------------------


def _normalise_sql(sql):
    """Design §1.4 D1 compares `sqlite_master.sql` with whitespace normalised."""
    return None if sql is None else " ".join(sql.split())


def _normalise_source_signature(fixture, signature, sag2):
    """Record the source-dir slot relative to the golden, not the temp tree.

    `sag2.py:203` signs `source_dir.resolve()`, so a `source_dir` build's
    signature carries the exporter's temporary absolute path.  The golden must
    stay reproducible across machines, so the slot records the exported
    relative tree; the verifier rebuilds the absolute form from its own
    resolution (F-007).
    """
    if fixture.source_relative is None:
        return signature
    fields = json.loads(signature)
    if fields[3] != str(fixture.source_dir):
        raise SystemExit(
            f"{fixture.name}: signature source_dir {fields[3]!r} is not the "
            f"build tree {str(fixture.source_dir)!r}"
        )
    fields[3] = fixture.source_relative
    return sag2.compact_json(fields)


def index_expectation(fixture, index, sag2):
    """Everything Gate D compares about a Python-built index."""
    metadata_rows = index.db.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    signature = _normalise_source_signature(
        fixture,
        index.db.execute("SELECT signature FROM metadata").fetchone()[0],
        sag2,
    )
    chunks = [
        dict(row) for row in index.db.execute("SELECT * FROM chunks ORDER BY id")
    ]
    schema = [
        {"type": kind, "name": name, "sql": _normalise_sql(sql)}
        for kind, name, sql in index.db.execute(
            "SELECT type,name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' AND name <> 'rust_index_metadata' "
            "ORDER BY name"
        )
    ]
    # D3: under `content=''` FTS5 stores no column values, so `body` is
    # dropped and the term binding is left to D4 (design §1.4 D3).
    fts_columns = (
        ("rowid", "title", "heading")
        if fixture.source_dir is not None
        else ("rowid", "title", "heading", "body")
    )
    fts_rows = [
        list(row)
        for row in index.db.execute(
            f"SELECT {','.join(fts_columns)} FROM chunk_fts ORDER BY rowid"
        )
    ]
    limit = index.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    source_files = (
        [
            {"document_id": document_id, "path": path, "sha256": sha256}
            for document_id, path, sha256 in index.db.execute(
                "SELECT document_id, path, sha256 FROM source_files ORDER BY document_id"
            )
        ]
        if fixture.source_dir is not None
        else []
    )
    probes = []
    for expression, term in fixture.probes:
        query = (
            f"({expression}) AND {sag2.quote_term(term)}"
            if term is not None
            else expression
        )
        # The reference path, not a re-typed SQL string: `retrieve` is the
        # Gate A entry point and ranks through `sag_bm25` (`sag2.py:477-521`).
        hits = index.retrieve(expression, set(), limit, term)
        if not hits:
            raise SystemExit(
                f"{fixture.name}: D4 probe {query!r} binds no rows; "
                "a vacuous probe asserts nothing"
            )
        probes.append(
            {
                "expression": expression,
                "query": query,
                "limit": limit,
                "rows": [[hit["id"], _hex_bits(hit["score"])] for hit in hits],
            }
        )
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
        "source_dir": fixture.source_relative,
        "schema": schema,
        "chunks": chunks,
        "fts_rows": fts_rows,
        "source_files": source_files,
        "fts_vocabulary": vocabulary,
        "probes": probes,
        "statistics_signature": _normalise_source_signature(
            fixture, statistics_rows[0][0], sag2
        ),
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
    # Canonical decomposition (design §5.4 step 1): pyuca's `sort_key` starts
    # with `unicodedata.normalize("NFD", s)`, and the tokenizer tables only
    # cover NFKD/NFKC, so Gate B needs this table of its own.
    "nfd": {
        "magic": b"SAG2NFDD",
        "path": "unicode_nfd.bin.gz",
        "fields": ["cp:u32le", "len:u8", "cps:[u32le;width]"],
        "header": "count:u32le, width:u8",
    },
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
    nfd, nfd_width = _sparse_mapping(lambda char: unicodedata.normalize("NFD", char))
    nfkd, nfkd_width = _sparse_mapping(lambda char: unicodedata.normalize("NFKD", char))
    nfkc, nfkc_width = _sparse_mapping(lambda char: unicodedata.normalize("NFKC", char))
    pairs = _compose_pairs()
    blobs = {
        "nfd": _sparse_mapping_bytes(b"SAG2NFDD", nfd, nfd_width),
        "nfkd": _sparse_mapping_bytes(b"SAG2NFKD", nfkd, nfkd_width),
        "nfkc": _sparse_mapping_bytes(b"SAG2NFKC", nfkc, nfkc_width),
    }
    buffer = bytearray(b"SAG2COMP")
    buffer += struct.pack("<I", len(pairs))
    for left, right, composite in pairs:
        buffer += struct.pack("<III", left, right, composite)
    blobs["compose"] = bytes(buffer)
    counts = {
        "nfd": {"records": len(nfd), "width": nfd_width},
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
    if fixture.source_dir is not None:
        # A contentless index stores `start_char`/`end_char`, not `content`
        # (`sag2.py:236-252`).  The chunk text is a span slice of the corpus
        # text already covered by the `text` segment above.
        return segments
    for row in rows:
        segments.append((f"{fixture.name}/chunk/{row['id']}/content", row["content"]))
    return segments


def tokenizer_segments_for(fixture, sag2, work_dir):
    """Build `fixture` and return its title/text/chunk segments.

    C4 (reference prose) and C7 (aliases and grounding) are not Gate A/D
    fixtures, so `export_fixture` never produces their rows.  Building their
    index here is what puts their text in `tokenizer.jsonl`; without it the
    mixed CJK/ASCII HMM branch would only be exercised end to end by Gate C.
    """
    db_path = work_dir / f"{fixture.name}.sqlite"
    index = sag2.CorpusIndex(
        db_path,
        fixture.corpus,
        fixture.fingerprint,
        chunk_chars=fixture.chunk_chars,
    )
    try:
        rows = [
            dict(row)
            for row in index.db.execute("SELECT id,content FROM chunks ORDER BY id")
        ]
    finally:
        index.close()
    return fixture_segments(fixture, rows)


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


# --------------------------------------------------------------------------
# Gate B (collation) - design §5.5: the pyuca 1.2 sort-key goldens.
#
# The open term set is the reason this fixture exists: it is the one gate that
# must fail even when no current fixture changes, because a term that appears
# tomorrow may use a code point that collates differently.
# --------------------------------------------------------------------------

SK_CP_MAGIC = b"SAG2SKCP"
SK_PAIRS_MAGIC = b"SAG2SKPR"
SK_PAIR_COUNT = 200_000
SK_PAIR_SEED = 20261114

# Design §5.5 item 1's adversarial set: case pairs, composed and decomposed
# accents, `ß`, Greek final sigma, kana variants, CJK and CJK ext-B, combining
# sequences (including one that needs canonical reordering), emoji, single
# punctuation and the empty string.
SK_ADVERSARIAL_TERMS = [
    "",
    "alice", "Alice", "ALICE",
    "cafe", "café", "cafe\u0301",
    "straße", "strasse",
    "ΑΣ", "ας", "ασ",
    "ア", "あ", "ｱ",
    "一", "二", "一二三", "一二年",
    "\U00020000", "\U00020001", "\U0002A6D6",
    "e\u0301\u0327", "e\u0327\u0301", "\u0301",
    "👍", "👨‍👩‍👧‍👦", "𝄞",
    "%", "_", ".", "-", "A.H.G.", "half-ounce",
    "ｶﾀｶﾅ", "ガ", "か\u3099",
    # DUCET contractions the fixture corpora would never produce: a
    # two-code-point key with a starter second code point (`l·`) and the only
    # three-code-point keys in the table (Kannada, Sinhala, Tibetan stacks).
    "l·l", "L·L", "ೋ", "ෝ", "ྲཱྀ",
    # A code point that only reaches its two-code-point entry after NFD:
    # `й` decomposes to `и` + U+0306 and DUCET folds the pair into one entry.
    "й", "Й",
    # Non-starter folding across an intervening mark of a lower class: pyuca
    # folds U+0306 / U+0654 into the preceding segment even though it is not
    # adjacent (`и` + U+0316 + U+0306, `ا` + U+0650 + U+0654).
    "и\u0316\u0306", "ا\u0650\u0654",
    # A mark sequence blocked from folding by an equal combining class.
    "a\u0301\u0301",
]


def _sortkey_vocabulary(fixtures, index_expectations, retrieve_lines, chunk_cases, sag2):
    """Every term the gates could order (design §5.5 item 1).

    The `frequencies` keys are the terms `expansion_terms` ranks; the corpus
    tokens, name phrases and title windows cover the terms `retrieve_expansion`
    derives from them (`name_phrase` joins and `repair_names` grounding); the
    query expressions cover the C0/C3/C4/C5 query terms; the adversarial list
    covers the code points no fixture contains.
    """
    terms = set(SK_ADVERSARIAL_TERMS)
    for expectation in index_expectations:
        statistics = json.loads(expectation["statistics"])
        terms.update(statistics.get("frequencies", {}))
        # `title_phrases` is a list of token tuples; the sorted term is the
        # space-joined phrase (`sag2.py:686-688`).
        terms.update(" ".join(phrase) for phrase in statistics.get("title_phrases", []))
        aliases = statistics.get("aliases", {})
        terms.update(aliases)
        terms.update(aliases.values())

    def absorb(text):
        terms.update(sag2.tokenize(text))
        for match in sag2.NAMES.finditer(text):
            terms.add(" ".join(sag2.tokenize(match[0])))

    for fixture in fixtures:
        for document in fixture.corpus:
            absorb(document.get("title", ""))
            absorb(document.get("text", ""))
    for case in chunk_cases:
        absorb(case["source"])
        for chunk in case["chunks"]:
            absorb(chunk["content"])
    for line in retrieve_lines:
        absorb(line["expression"])
    # `title_phrases` and the `repair_names` fallback ground a phrase to a
    # contiguous token window of a title (`sag2.py:686-688`), so every 2..4
    # window is a term the sorter may have to place.
    for fixture in fixtures:
        for document in fixture.corpus:
            tokens = sag2.tokenize(document.get("title", ""))
            for start in range(len(tokens)):
                for width in (2, 3, 4):
                    window = tokens[start : start + width]
                    if len(window) == width:
                        terms.add(" ".join(window))
    return terms


def _sortkey_codepoints(collator):
    """`sk_cp.bin.gz`: the key of every single-code-point string.

    Layout: `magic, u32 count`, then one variable-length record per code point
    in ascending order: `len:u8, weights:[u16le;len]`.  A fixed-width record
    costs > 4 MB gzipped because the longest key is 57 weights; the design's
    ≤ 3 MB target needs the length-prefixed form.  Every weight is asserted to
    fit `u16` so the narrowing is checked, not assumed.
    """
    buffer = bytearray(SK_CP_MAGIC)
    buffer += struct.pack("<I", CODEPOINTS)
    max_weights = 0
    for cp in range(CODEPOINTS):
        key = collator.sort_key(chr(cp))
        if len(key) > 0xFF:
            raise SystemExit(f"U+{cp:04X}: sort key wider than 255 weights")
        for weight in key:
            if weight > 0xFFFF:
                raise SystemExit(f"U+{cp:04X}: sort key weight {weight} does not fit u16")
        max_weights = max(max_weights, len(key))
        buffer += struct.pack("<B", len(key))
        buffer += b"".join(struct.pack("<H", weight) for weight in key)
    return bytes(buffer), max_weights


def _sortkey_pairs(input_order, keys_by_term, collator):
    """`sk_pairs.bin.gz`: 200,000 ordered pairs and their expected sign.

    Layout: `magic, u32 count`, then `kind:u8, a:u32le, b:u32le, expected:i8`
    where `kind == 0` means code points and `kind == 1` indexes `sk_vocab.jsonl`
    (file order).  Key equality alone cannot prove the comparison semantics
    (design §5.5), so this fixture checks the sign independently.
    """
    rng = random.Random(SK_PAIR_SEED)
    index_of = {term: position for position, term in enumerate(input_order)}
    records = []

    def sign(left, right):
        return (left > right) - (left < right)

    def add_vocab(left, right):
        records.append(
            (
                1,
                index_of[left],
                index_of[right],
                sign(keys_by_term[left], keys_by_term[right]),
            )
        )

    def add_codepoints(left, right):
        records.append(
            (
                0,
                left,
                right,
                sign(collator.sort_key(chr(left)), collator.sort_key(chr(right))),
            )
        )

    # Every ordered pair of the adversarial set: case, accent, sigma, kana,
    # CJK and combining-sequence relations are all represented exactly.
    for left in SK_ADVERSARIAL_TERMS:
        for right in SK_ADVERSARIAL_TERMS:
            add_vocab(left, right)
    # Equal terms must compare 0 (the sorter is stable and ties are real).
    for term in input_order[:20_000]:
        add_vocab(term, term)
    # Adjacent code points, the densest place where implicit weights meet
    # explicit entries.
    for cp in range(0, CODEPOINTS - 1, 28):
        if 0xD800 <= cp <= 0xDFFF or 0xD800 <= cp + 1 <= 0xDFFF:
            continue
        add_codepoints(cp, cp + 1)
    # Category boundaries: CJK vs ASCII, ext-B vs BMP, marks vs bases, emoji.
    boundaries = [
        (0x0041, 0x4E00), (0x4E00, 0x9FCC), (0x9FCC, 0xA000),
        (0x3400, 0x4DB5), (0x4DB5, 0x20000), (0x2A6D6, 0x2B820),
        (0x2B820, 0x2CEAF), (0x2CEA2, 0x2B820), (0x17000, 0x18AFF),
        (0x0301, 0x0061), (0x0065, 0x0301), (0x0391, 0x03B1),
        (0x30A2, 0x3042), (0xFF71, 0x30A2), (0x1F44D, 0x0041),
        (0x0000, 0x0020), (0x0020, 0x0021), (0x007E, 0x00A0),
        (0x00E9, 0x0065), (0x10FFFF, 0x0041),
    ]
    for index in range(20_000):
        left, right = boundaries[index % len(boundaries)]
        add_codepoints(left, right)
    # Random code points and random vocabulary terms, then fill to the exact
    # declared count with vocabulary pairs (deterministic under the seed).
    for _ in range(40_000):
        add_codepoints(
            SCALARS[rng.randrange(len(SCALARS))], SCALARS[rng.randrange(len(SCALARS))]
        )
    for _ in range(60_000):
        add_vocab(
            input_order[rng.randrange(len(input_order))],
            input_order[rng.randrange(len(input_order))],
        )
    while len(records) < SK_PAIR_COUNT:
        add_vocab(
            input_order[rng.randrange(len(input_order))],
            input_order[rng.randrange(len(input_order))],
        )
    if len(records) != SK_PAIR_COUNT:
        raise SystemExit(f"built {len(records)} sort-key pairs, want {SK_PAIR_COUNT}")
    buffer = bytearray(SK_PAIRS_MAGIC)
    buffer += struct.pack("<I", len(records))
    for kind, left, right, expected in records:
        buffer += struct.pack("<BIIb", kind, left, right, expected)
    return bytes(buffer), len(records)


def export_sortkeys(
    fixtures,
    index_expectations,
    retrieve_lines,
    chunk_cases,
    sag2,
    collator,
):
    """Gate B's three goldens plus their manifest metadata (design §5.5).

    The two binary goldens are returned already gzipped with a zero mtime, so
    re-exporting produces byte-identical files (design §7.5 `golden-regen`).
    """
    terms = _sortkey_vocabulary(
        fixtures, index_expectations, retrieve_lines, chunk_cases, sag2
    )
    input_order = sorted(terms)
    keys_by_term = {term: list(collator.sort_key(term)) for term in input_order}
    ordered = sorted(input_order, key=lambda term: tuple(keys_by_term[term]))
    rank = {term: position for position, term in enumerate(ordered)}
    lines = [
        {"term": term, "key": keys_by_term[term], "rank": rank[term]}
        for term in input_order
    ]
    vocab_text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    codepoints, max_weights = _sortkey_codepoints(collator)
    pairs, pair_count = _sortkey_pairs(input_order, keys_by_term, collator)
    metadata = {
        "vocab": {
            "golden": "sk_vocab.jsonl",
            "terms": len(lines),
            # `term` order is code point order; `rank` is the position in
            # Python's stable `sorted(terms, key=collator.sort_key)`.
            "order": "term",
        },
        "codepoints": {
            "golden": "sk_cp.bin.gz",
            "magic": SK_CP_MAGIC.decode(),
            "codepoints": CODEPOINTS,
            "layout": "u32 count, then len:u8 + weights:[u16le;len] per code point",
            "max_weights": max_weights,
            "surrogates": "present in the golden; lone surrogates are not scalar values",
        },
        "pairs": {
            "golden": "sk_pairs.bin.gz",
            "magic": SK_PAIRS_MAGIC.decode(),
            "pairs": pair_count,
            "layout": "u32 count, then kind:u8 + a:u32le + b:u32le + expected:i8",
            "seed": SK_PAIR_SEED,
            "kinds": {"0": "code points", "1": "sk_vocab.jsonl file order"},
        },
    }
    return (
        vocab_text,
        len(lines),
        _gzip_bytes(codepoints),
        _gzip_bytes(pairs),
        metadata,
    )


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
    sortkeys,
    expansion,
    freq,
    search,
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
        "sortkeys": sortkeys,
        "expansion": expansion,
        "freq": freq,
        "search": search,
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


def verify_code_suffixes(chunking):
    """Refuse to export when the Rust `CODE_SUFFIXES` drifts from the reference.

    `chunk.rs` picks the code chunking mode from this table, so a missing entry
    silently turns a code/config document into prose (F-002).  The check parses
    the literal out of the Rust source and compares the *set* with
    `chunking.CODE_SUFFIXES`.
    """
    rust_source = (
        Path(__file__).resolve().parent.parent
        / "crates"
        / "sag2-core"
        / "src"
        / "chunk.rs"
    ).read_text()
    match = re.search(
        r"CODE_SUFFIXES: \[&str; (\d+)\] = \[(.*?)\];", rust_source, re.S
    )
    if match is None:
        raise SystemExit(f"cannot find the CODE_SUFFIXES literal in {rust_source}")
    entries = re.findall(r'"(\.[^"]+)"', match.group(2))
    declared = int(match.group(1))
    reference = sorted(chunking.CODE_SUFFIXES)
    if declared != len(entries) or sorted(entries) != reference:
        missing = sorted(set(reference) - set(entries))
        extra = sorted(set(entries) - set(reference))
        raise SystemExit(
            f"chunk.rs CODE_SUFFIXES drifts from chunking.py: declared {declared}, "
            f"found {len(entries)} entries, missing {missing}, extra {extra}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="export every fixture (C0-C3, C5, C6)")
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

    # The reference module-level singleton (`sag2.py:50`); Gate B's goldens are
    # its output, so the exporter must use the same class the reference picks
    # (`pyuca.Collator` is the UCA 9.0.0 collator under Python >= 3.6).
    collator = pyuca.Collator()

    reference = json.loads((reference_dir.parent / "tests" / "sag2_reference.json").read_text())
    provenance = reference_provenance(reference_dir)
    import chunking  # noqa: E402 - the reference `CODE_SUFFIXES` literal
    verify_code_suffixes(chunking)

    fixtures = [
        fixture_c0(reference, sag2),
        fixture_c1(80, sag2),
        fixture_c1(140, sag2),
        fixture_c2(sag2),
        fixture_c3(sag2),
        fixture_c5(sag2),
        fixture_c6(sag2),
    ]
    # Gate B's real-text corpus (design §1.2): built and shipped, but not a
    # Gate A/D fixture, so it does not enter those goldens.
    c4 = fixture_c4(reference_dir)

    out_dir = args.out
    staging = Path(tempfile.mkdtemp(prefix="sag2-golden-"))
    work_dir = staging / "indexes"
    work_dir.mkdir(parents=True)
    try:
        # `source_dir` fixtures read their text from a tree, so materialise it
        # before the build (and copy it next to the golden afterwards).
        for fixture in fixtures:
            if fixture.source_relative is None:
                continue
            fixture.source_dir = staging / fixture.source_relative
            for document in fixture.corpus:
                target = fixture.source_dir / document["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(document["text"].encode("utf-8"))

        retrieve_lines, dump_lines, fixture_stats = [], [], []
        index_expectations = []
        segments_by_fixture = {}
        for fixture in fixtures:
            db_path, cases, rows, expectation = export_fixture(fixture, sag2, work_dir)
            signature = expectation["signature"]
            index_expectations.append(expectation)
            segments_by_fixture[fixture.name] = fixture_segments(fixture, rows)
            # A `source_dir` index embeds the absolute build tree in its
            # metadata and statistics rows, so its bytes are not reproducible
            # and it is not shipped as a reference index.
            shipped = fixture.source_relative is None
            fixture_stats.append(
                {
                    "fixture": fixture.name,
                    "index": db_path.name,
                    "index_sha256": (
                        _sha256_bytes(db_path.read_bytes()) if shipped else None
                    ),
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

        # The tokenizer gate covers every corpus whose text can reach the
        # tokenizer, including the two that are not Gate A/D fixtures.
        for extra in (c4, c7_fixture()):
            segments_by_fixture[extra.name] = tokenizer_segments_for(
                extra, sag2, work_dir
            )
            print(f"{extra.name}: {len(segments_by_fixture[extra.name])} tokenizer segments")

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
        sortkey_text, sortkey_lines, sk_cp_blob, sk_pairs_blob, sortkey_metadata = (
            export_sortkeys(
                fixtures,
                index_expectations,
                retrieve_lines,
                chunk_cases,
                sag2,
                collator,
            )
        )

        # Gate B (expansion, design §1.2): C0's four frozen queries with their
        # own options, C3's fifty generated queries and C4's twenty heading
        # queries.  C4's index is built here and shipped with the goldens.
        c3 = next(fixture for fixture in fixtures if fixture.name == "C3")
        expansion_specs = [
            (
                next(fixture for fixture in fixtures if fixture.name == "C0"),
                [(case["query"], case["options"]) for case in reference["cases"]],
            ),
            (c3, [(query, None) for query in c3.queries]),
            (c4, [(query, None) for query in c4_queries(c4, sag2)]),
        ]
        expansion_fixtures, expansion_lines = [], []
        for fixture, queries in expansion_specs:
            entry, lines = export_expansion(fixture, queries, sag2, work_dir)
            expansion_fixtures.append(entry)
            expansion_lines.extend(lines)
            print(
                f"{fixture.name}: {len(queries)} expansion queries, "
                f"{entry['records']} (query, hop) records, {entry['chunks']} chunks"
            )
        entry, lines = export_named_expansion(sag2, work_dir)
        expansion_fixtures.append(entry)
        expansion_lines.extend(lines)
        print(
            f"{entry['fixture']}: {entry['records']} repair/grounding sub-cases, "
            f"{entry['chunks']} chunks"
        )
        expansion_text = "".join(
            json.dumps(line, ensure_ascii=False) + "\n" for line in expansion_lines
        )

        # Gate C (search, design §1.3): the frozen `search_sag2` run with its
        # byte-exact payloads, its trace and its judge usage.
        search_corpora, search_cases = export_search(
            reference, fixtures, c4, sag2, work_dir
        )
        search_text = "".join(
            json.dumps(line, ensure_ascii=False) + "\n" for line in search_cases
        )
        search_corpus_text = "".join(
            json.dumps(line, ensure_ascii=False) + "\n" for line in search_corpora
        )
        search_blob = _gzip_bytes(search_text.encode())
        payload_count = sum(len(case["payloads"]) for case in search_cases)
        print(
            f"search: {len(search_cases)} cases over {len(search_corpora)} corpora, "
            f"{payload_count} payloads"
        )

        # Gate D `freq` (design §4.1 rule 5): the table's whole short key set
        # plus tokenizer probes covering every anchor position and the corpus.
        freq_blob, freq_text, freq_metadata = export_freq(
            sag2,
            [
                next(fixture for fixture in fixtures if fixture.name == "C0"),
                c4,
                c7_fixture(),
            ],
        )
        print(
            f"freq: {freq_metadata['keys']} short keys, "
            f"{freq_metadata['anchor_entries']} anchor positions, "
            f"{freq_metadata['probes']} probes "
            f"({freq_metadata['corpus_probes']} corpus keys)"
        )

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
        files.append(
            {
                "path": "sk_vocab.jsonl",
                "sha256": _sha256_bytes(sortkey_text.encode()),
                "lines": sortkey_lines,
            }
        )
        files.append(
            {
                "path": "sk_cp.bin.gz",
                "sha256": _sha256_bytes(sk_cp_blob),
                "records": sortkey_metadata["codepoints"]["codepoints"],
                "gzip_bytes": len(sk_cp_blob),
            }
        )
        files.append(
            {
                "path": "sk_pairs.bin.gz",
                "sha256": _sha256_bytes(sk_pairs_blob),
                "records": sortkey_metadata["pairs"]["pairs"],
                "gzip_bytes": len(sk_pairs_blob),
            }
        )
        files.append(
            {
                "path": "expansion.jsonl",
                "sha256": _sha256_bytes(expansion_text.encode()),
                "lines": len(expansion_lines),
            }
        )
        files.append(
            {
                "path": "freq_keys.txt.gz",
                "sha256": _sha256_bytes(freq_blob),
                "records": freq_metadata["keys"],
                "gzip_bytes": len(freq_blob),
            }
        )
        files.append(
            {
                "path": "freq_probes.jsonl",
                "sha256": _sha256_bytes(freq_text.encode()),
                "lines": freq_metadata["probes"],
            }
        )
        files.append(
            {
                "path": "search.jsonl.gz",
                "sha256": _sha256_bytes(search_blob),
                "lines": len(search_cases),
                "gzip_bytes": len(search_blob),
            }
        )
        files.append(
            {
                "path": "search_corpus.jsonl",
                "sha256": _sha256_bytes(search_corpus_text.encode()),
                "lines": len(search_corpora),
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
                "probes": probe_expressions(fixtures),
                "golden": "index_expected.jsonl",
            },
            sortkey_metadata,
            {
                "cases": len(expansion_lines),
                "fixtures": expansion_fixtures,
                "golden": "expansion.jsonl",
            },
            freq_metadata,
            {
                "cases": len(search_cases),
                "corpora": len(search_corpora),
                "payloads": payload_count,
                "endpoint": SEARCH_ENDPOINT,
                "defaults": SEARCH_DEFAULTS,
                "golden": "search.jsonl.gz",
                "corpus_golden": "search_corpus.jsonl",
            },
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        index_dir = out_dir / "indexes"
        index_dir.mkdir(exist_ok=True)
        for entry, fixture in zip(fixture_stats, fixtures):
            if entry["index_sha256"] is not None:
                shutil.copyfile(work_dir / entry["index"], index_dir / entry["index"])
            else:
                (index_dir / entry["index"]).unlink(missing_ok=True)
            if fixture.source_relative is not None:
                shutil.rmtree(out_dir / fixture.source_relative, ignore_errors=True)
                shutil.copytree(
                    staging / fixture.source_relative,
                    out_dir / fixture.source_relative,
                    dirs_exist_ok=True,
                )
        (out_dir / "retrieve.jsonl").write_text(retrieve_text)
        (out_dir / "index_dump.jsonl").write_text(dump_text)
        (out_dir / "tokenizer.jsonl").write_text(tokenizer_text)
        (out_dir / "ascii_runs.jsonl").write_text(ascii_text)
        (out_dir / "chunks.jsonl").write_text(chunk_text)
        (out_dir / "index_expected.jsonl").write_text(index_expected_text)
        (out_dir / "expansion.jsonl").write_text(expansion_text)
        (out_dir / "search.jsonl.gz").write_bytes(search_blob)
        (out_dir / "search_corpus.jsonl").write_text(search_corpus_text)
        (out_dir / "freq_keys.txt.gz").write_bytes(freq_blob)
        (out_dir / "freq_probes.jsonl").write_text(freq_text)
        for entry in expansion_fixtures:
            shutil.copyfile(work_dir / entry["index"], index_dir / entry["index"])
        for name, spec in UNICODE_TABLES.items():
            (out_dir / spec["path"]).write_bytes(unicode_blobs[name])
        for name, spec in SPARSE_UNICODE_TABLES.items():
            (out_dir / spec["path"]).write_bytes(sparse_blobs[name])
        (out_dir / "sk_vocab.jsonl").write_text(sortkey_text)
        (out_dir / "sk_cp.bin.gz").write_bytes(sk_cp_blob)
        (out_dir / "sk_pairs.bin.gz").write_bytes(sk_pairs_blob)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            f"wrote {len(retrieve_lines)} retrieve cases, {len(dump_lines)} index rows, "
            f"{tokenizer_lines} tokenizer segments, {ascii_lines} ASCII runs, "
            f"{len(chunk_cases)} chunk cases, {len(index_expectations)} index expectations, "
            f"{len(UNICODE_TABLES)} dense + {len(SPARSE_UNICODE_TABLES)} sparse unicode "
            f"tables, {sortkey_lines} sort-key terms, "
            f"{sortkey_metadata['codepoints']['codepoints']} sort-key code points, "
            f"{sortkey_metadata['pairs']['pairs']} sort-key pairs, "
            f"{len(expansion_lines)} expansion records, "
            f"{len(search_cases)} search cases ({payload_count} payloads), "
            f"{freq_metadata['keys']} short FREQ keys, "
            f"{freq_metadata['probes']} FREQ probes and "
            f"{len(fixture_stats)} indexes to {out_dir}"
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
