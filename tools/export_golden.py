#!/usr/bin/env python3
"""Export the Gate A golden fixtures for the Rust port of SAG2 (design §1.1, §7.2).

The exporter imports the reference implementation **read-only** and records for
every Gate A case the ordered chunk ids, the ordered document ids and the raw
``f64`` bit pattern of every score.  Scores are stored as
``struct.pack('<d', score).hex()`` so the comparison never round-trips through a
decimal representation (design §1.0: layer S uses ``eq_f64_bits``).

Requires CPython >= 3.12.  Python 3.12 is the first release whose ``sum()``
uses Neumaier compensated summation; the pruning bound in
``CorpusIndex._retrieve_snapshot`` is such a sum (design §11 D5).  A 3.11 run
silently produces different goldens, so the version gate is a hard exit that
writes nothing.

Outputs (all under ``--out``, default ``golden/``):

``retrieve.jsonl``
    one Gate A case per line
``index_dump.jsonl``
    the ``chunks`` rows and the ``metadata.signature`` of every fixture index
``indexes/<fixture>.sqlite``
    the Python-built index each case runs against (Gate A runs on these)
``manifest.json``
    generator/python/sqlite/jieba/pyuca/unicode versions, corpus digest, file
    digests and the fixture -> index mapping
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
import hashlib
import json
import os
import random
import shutil
import sqlite3
import struct
import tempfile
import unicodedata
from pathlib import Path

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

DEFAULT_REFERENCE = Path("/home/pop/AI/code/sag2/SAG-Benchmark/sag2/scripts")
GENERATOR_VERSION = "0.1.0"

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
                    }
                )
                cases.append(case)
                continue
            case.update(
                {
                    "error": None,
                    "ids": [hit["id"] for hit in hits],
                    "document_ids": [hit["document_id"] for hit in hits],
                    "scores_bits": [_hex_bits(hit["score"]) for hit in hits],
                }
            )
            cases.append(case)
        signature = index.db.execute("SELECT signature FROM metadata").fetchone()[0]
        rows = [
            dict(row)
            for row in index.db.execute(
                "SELECT id, document_id, heading, spans, characters, source_id "
                "FROM chunks ORDER BY id"
            )
        ]
        return db_path, signature, cases, rows
    finally:
        index.close()


def build_manifest(fixtures, stats, files):
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
        "corpus_sha256": corpus_digest.hexdigest(),
        "files": files,
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
        for fixture in fixtures:
            db_path, signature, cases, rows = export_fixture(fixture, sag2, work_dir)
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
        files = [
            {"path": "retrieve.jsonl", "sha256": _sha256_bytes(retrieve_text.encode()),
             "lines": len(retrieve_lines)},
            {"path": "index_dump.jsonl", "sha256": _sha256_bytes(dump_text.encode()),
             "lines": len(dump_lines)},
        ]
        manifest = build_manifest(
            fixtures,
            {
                "jieba": jieba.__version__,
                "pyuca": pyuca.Collator.UCA_VERSION,
                "unicode": unicodedata.unidata_version,
                "fixtures": fixture_stats,
            },
            files,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        index_dir = out_dir / "indexes"
        index_dir.mkdir(exist_ok=True)
        for entry in fixture_stats:
            shutil.copyfile(work_dir / entry["index"], index_dir / entry["index"])
        (out_dir / "retrieve.jsonl").write_text(retrieve_text)
        (out_dir / "index_dump.jsonl").write_text(dump_text)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            f"wrote {len(retrieve_lines)} retrieve cases, {len(dump_lines)} index rows "
            f"and {len(fixture_stats)} indexes to {out_dir}"
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
