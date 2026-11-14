//! `sag2-cli verify --gate A|D`: replay the golden gate cases (design §1.1,
//! §4.4, §7.3).
//!
//! Gate A replays the retrieved hits; Gate D compares the tokenizer against the
//! frozen reference lines, every corpus segment, the ASCII cross product and
//! the exhaustive per-code-point Unicode tables.
//!
//! Comparison uses only the §1.0 tolerance functions: the ordered hit rows are
//! compared field by field (`eq_ids` on `id`/`document_id`, `eq_str` on the
//! §1.1 materialisation fields, `eq_f64_bits` on the scores).  Exit codes follow
//! §1.0: `0` passed, `2` a comparison failed, `3` the tooling/data cannot be
//! used.  A gate run outside the certified toolchain, or against a reference
//! checkout that is not the one the goldens came from, compares nothing and
//! exits `3` (design §1.0 rule 2, §7.1).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rusqlite::{Connection, OpenFlags};
use sag2_index::{open_readonly, CorpusIndex, QueryError, QueryOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Generator version this verifier understands.
const GENERATOR_VERSION: &str = "0.4.0";

/// Minimum interpreter the goldens may come from (design §11 divergence D5).
const REQUIRED_PYTHON: (u64, u64) = (3, 12);

/// SQLite versions Gate A accepts (design §1.0 layer S: 3.50.2 vs 3.50.4).
const SQLITE_ALLOWLIST: [&str; 2] = ["3.50.2", "3.50.4"];

/// The reference environment's `jieba`; the goldens must come from it.
const CERTIFIED_JIEBA: &str = "0.42.1";

/// The design's documented sibling checkout of the reference implementation.
const DEFAULT_REFERENCE: &str = "../SAG-Benchmark/sag2/scripts";

/// `eq_f64(a, b) := |a-b| <= 1e-12 * max(1.0, |a|, |b|)` (design §1.0).
const RELATIVE_TOLERANCE: f64 = 1e-12;

/// Failures are buffered so the report shows the full picture (§1.0 rule 1).
const MAX_FAILURES: usize = 200;

#[derive(Deserialize)]
struct Manifest {
    generator_version: String,
    python: String,
    sqlite: String,
    jieba: String,
    pyuca: String,
    unicode: String,
    reference: ManifestReference,
    files: Vec<ManifestFile>,
    fixtures: Vec<ManifestFixture>,
    #[serde(default)]
    tokenizer: Option<TokenizerSpec>,
    #[serde(default)]
    ascii_runs: Option<AsciiRunsSpec>,
    #[serde(default)]
    unicode_tables: BTreeMap<String, UnicodeTableSpec>,
    #[serde(default)]
    unicode_sparse: BTreeMap<String, UnicodeSparseSpec>,
    #[serde(default)]
    chunks: Option<ChunksSpec>,
    #[serde(default)]
    index_expected: Option<IndexExpectedSpec>,
}

/// Gate D's chunk-golden document count (`chunks.jsonl`).
#[derive(Deserialize)]
struct ChunksSpec {
    documents: usize,
    chunks: usize,
}

/// Gate D's Rust-vs-Python index baseline count (`index_expected.jsonl`).
#[derive(Deserialize)]
struct IndexExpectedSpec {
    fixtures: usize,
}

/// Gate D's segment count (`tokenizer.jsonl`).
#[derive(Deserialize)]
struct TokenizerSpec {
    segments: usize,
}

/// Gate D's ASCII cross-product count (`ascii_runs.jsonl`).
#[derive(Deserialize)]
struct AsciiRunsSpec {
    runs: usize,
}

/// One exhaustive Unicode table: fixed-width records, one per code point.
#[derive(Deserialize)]
struct UnicodeTableSpec {
    path: String,
    magic: String,
    record_bytes: usize,
    codepoints: usize,
}

/// One sparse Unicode table: only the code points whose result is not identity.
#[derive(Deserialize)]
struct UnicodeSparseSpec {
    path: String,
    magic: String,
    records: usize,
}

/// Identity of the reference source set the goldens were exported from (§7.1).
#[derive(Deserialize)]
struct ManifestReference {
    sag2_py_sha256: String,
    sag2_rank_py_sha256: String,
    chunking_py_sha256: String,
    commit: String,
}

#[derive(Deserialize)]
struct ManifestFile {
    path: String,
    sha256: String,
    #[serde(default)]
    lines: Option<usize>,
    #[serde(default)]
    records: Option<usize>,
}

#[derive(Deserialize)]
struct ManifestFixture {
    fixture: String,
    index: String,
    index_sha256: String,
    chunks: usize,
    signature: String,
}

#[derive(Deserialize)]
struct Case {
    case_id: String,
    fixture: String,
    expression: String,
    excluded: Vec<i64>,
    limit: usize,
    term: Option<String>,
    ids: Vec<i64>,
    document_ids: Vec<i64>,
    scores_bits: Vec<String>,
    /// The §1.1 materialisation fields, in ranked order.
    hits: Vec<HitGolden>,
    #[serde(default)]
    error: Option<String>,
}

/// One expected hit: every field design §1.1 compares, in serialised form.
#[derive(Deserialize)]
struct HitGolden {
    id: i64,
    document_id: i64,
    title: String,
    heading: String,
    content: String,
    spans: String,
    characters: i64,
    source_id: String,
    score_bits: String,
}

struct Options {
    gate: String,
    golden: PathBuf,
    cache: PathBuf,
    report: PathBuf,
    filter: Option<String>,
    reference: PathBuf,
    fixtures: Option<Vec<String>>,
    /// Keep the indexes `--gate D --fixtures index` builds here.
    index_dir: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            gate: "A".to_owned(),
            golden: PathBuf::from("golden"),
            cache: std::env::temp_dir().join("sag2-verify-cache"),
            report: PathBuf::from("report.json"),
            filter: None,
            reference: PathBuf::from(DEFAULT_REFERENCE),
            fixtures: None,
            index_dir: None,
        }
    }
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Options::default();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            let value = args
                .get(index + 1)
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))?;
            match flag {
                "--gate" => options.gate = value,
                "--golden" => options.golden = PathBuf::from(value),
                "--index-cache" => options.cache = PathBuf::from(value),
                "--report" => options.report = PathBuf::from(value),
                "--filter" => options.filter = Some(value),
                "--reference" => options.reference = PathBuf::from(value),
                "--index-dir" => options.index_dir = Some(PathBuf::from(value)),
                "--fixtures" => {
                    options.fixtures = Some(
                        value
                            .split(',')
                            .map(|fixture| fixture.trim().to_owned())
                            .filter(|fixture| !fixture.is_empty())
                            .collect(),
                    )
                }
                other => return Err(format!("unknown argument {other}")),
            }
            index += 2;
        }
        Ok(options)
    }
}

#[derive(Serialize)]
struct Failure {
    case_id: String,
    path: String,
    expected: String,
    actual: String,
    delta: Option<f64>,
}

pub fn verify(args: &[String]) -> ExitCode {
    let options = match Options::parse(args) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("verify: {message}");
            eprintln!("{}", usage());
            return ExitCode::from(3);
        }
    };
    match run(&options) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("verify: {message}");
            let report = serde_json::json!({
                "golden": {"dir": options.golden.display().to_string()},
                "tolerances": {"level": "S", "f64": "bits", "ids": "exact"},
                "gates": {},
                "failures": [],
                "ulp_flips": [],
                "unsupported": [message],
                "excluded_fields": [],
                "excluded_checks": ["db_physical_bytes"],
            });
            write_report(&options.report, &report);
            ExitCode::from(3)
        }
    }
}

pub fn usage() -> &'static str {
    "usage: sag2-cli verify --gate A|D --golden golden/ \
     [--fixtures tokenizer,chunks,index] [--reference <scripts>] \
     [--index-cache <dir>] [--index-dir <dir>] [--filter <substring>] \
     [--report report.json]"
}

/// The toolchain values this build certifies for the reference environment.
struct Toolchain {
    jieba: &'static str,
    uca: &'static str,
    unicode: &'static str,
}

fn certified_toolchain() -> Result<Toolchain, String> {
    Ok(Toolchain {
        jieba: CERTIFIED_JIEBA,
        uca: sag2_index::UCA_VERSION,
        unicode: sag2_index::UNICODE_VERSION,
    })
}

/// Design §7.1: recompute the reference digests the manifest claims.
fn verify_reference(reference: &ManifestReference, dir: &Path) -> Result<(), String> {
    for (name, expected) in [
        ("sag2.py", &reference.sag2_py_sha256),
        ("sag2_rank.py", &reference.sag2_rank_py_sha256),
        ("chunking.py", &reference.chunking_py_sha256),
    ] {
        let path = dir.join(name);
        let bytes = fs::read(&path).map_err(|error| {
            format!(
                "unsupported: cannot read reference {name} at {} (pass --reference): {error}",
                path.display()
            )
        })?;
        if sha256_hex(&bytes) != *expected {
            return Err(format!(
                "unsupported: reference {name} at {} does not match the golden manifest",
                path.display()
            ));
        }
    }
    Ok(())
}

fn run(options: &Options) -> Result<ExitCode, String> {
    match options.gate.to_ascii_uppercase().as_str() {
        "A" => run_gate_a(options),
        "D" => run_gate_d(options),
        other => Err(format!(
            "gate {other} is not implemented; only gates A and D have goldens"
        )),
    }
}

/// Manifest self-consistency and the certified-toolchain checks shared by every
/// gate (design §1.0 rule 2, §7.1).
fn preflight(options: &Options) -> Result<Manifest, String> {
    let manifest_path = options.golden.join("manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("cannot read {}: {error}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&manifest_text)
        .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
    if manifest.generator_version != GENERATOR_VERSION {
        return Err(format!(
            "golden generator {} does not match verifier {GENERATOR_VERSION}",
            manifest.generator_version
        ));
    }
    let python = parse_version(&manifest.python)
        .ok_or_else(|| format!("manifest python {:?} is not a version", manifest.python))?;
    if python < REQUIRED_PYTHON {
        return Err(format!(
            "golden was generated by Python {} (< 3.12): sum() semantics differ, re-export",
            manifest.python
        ));
    }

    // Design §1.0 rule 2: outside the certified environment the gate compares
    // nothing.  The SQLite allowlist is the measured layer-S pair; jieba/UCA/
    // Unicode must be the reference toolchain the goldens were exported with.
    let runtime_sqlite = sag2_index::sqlite_version();
    if !SQLITE_ALLOWLIST.contains(&manifest.sqlite.as_str())
        || !SQLITE_ALLOWLIST.contains(&runtime_sqlite)
    {
        return Err(format!(
            "unsupported: sqlite {} (golden) / {runtime_sqlite} (runtime) is outside the \
             allowlist {SQLITE_ALLOWLIST:?}",
            manifest.sqlite
        ));
    }
    let toolchain = certified_toolchain()?;
    if manifest.jieba != toolchain.jieba
        || manifest.pyuca != toolchain.uca
        || manifest.unicode != toolchain.unicode
    {
        return Err(format!(
            "unsupported: golden toolchain jieba {} / pyuca {} / unicode {} differs from the \
             certified {}/{}/{}",
            manifest.jieba,
            manifest.pyuca,
            manifest.unicode,
            toolchain.jieba,
            toolchain.uca,
            toolchain.unicode
        ));
    }

    // Design §7.1: the golden must name the exact reference source set.
    verify_reference(&manifest.reference, &options.reference)?;
    Ok(manifest)
}

/// Verify the manifest's hash (and, for JSONL, line count) for the files one
/// gate actually reads.
fn verify_manifest_files(
    options: &Options,
    manifest: &Manifest,
    required: &[&str],
) -> Result<(), String> {
    for name in required {
        let file = manifest
            .files
            .iter()
            .find(|file| file.path == *name)
            .ok_or_else(|| format!("manifest does not list {name}"))?;
        let path = options.golden.join(&file.path);
        let bytes =
            fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        if sha256_hex(&bytes) != file.sha256 {
            return Err(format!("{} does not match the manifest sha256", file.path));
        }
        if let Some(expected_lines) = file.lines {
            let lines = bytes.iter().filter(|byte| **byte == b'\n').count();
            if lines != expected_lines {
                return Err(format!(
                    "{} has {lines} lines, manifest says {expected_lines}",
                    file.path
                ));
            }
        }
    }
    Ok(())
}

fn run_gate_a(options: &Options) -> Result<ExitCode, String> {
    let manifest = preflight(options)?;
    let runtime_sqlite = sag2_index::sqlite_version();
    verify_manifest_files(options, &manifest, &["retrieve.jsonl", "index_dump.jsonl"])?;

    let signatures = dump_signatures(options)?;
    for fixture in &manifest.fixtures {
        match signatures.get(&fixture.fixture) {
            Some(signature) if signature == &fixture.signature => {}
            _ => {
                return Err(format!(
                    "golden inconsistency: manifest/index_dump signatures differ for {}",
                    fixture.fixture
                ))
            }
        }
    }

    let mut cases = read_cases(options)?;
    if let Some(filter) = &options.filter {
        cases.retain(|case| case.case_id.contains(filter.as_str()));
    }
    if cases.is_empty() {
        return Err(format!(
            "no Gate A cases selected from {}",
            options.golden.join("retrieve.jsonl").display()
        ));
    }

    let mut indexes: BTreeMap<String, CorpusIndex> = BTreeMap::new();
    for fixture in &manifest.fixtures {
        if !cases.iter().any(|case| case.fixture == fixture.fixture) {
            continue;
        }
        let source = options.golden.join("indexes").join(&fixture.index);
        let bytes = fs::read(&source)
            .map_err(|error| format!("cannot read {}: {error}", source.display()))?;
        if sha256_hex(&bytes) != fixture.index_sha256 {
            return Err(format!(
                "{} does not match the manifest sha256",
                fixture.index
            ));
        }
        fs::create_dir_all(&options.cache)
            .map_err(|error| format!("cannot create {}: {error}", options.cache.display()))?;
        let cached = options.cache.join(&fixture.index);
        fs::write(&cached, &bytes)
            .map_err(|error| format!("cannot write {}: {error}", cached.display()))?;
        let signature = signatures[&fixture.fixture].clone();
        let index = open_readonly(&cached, &signature, QueryOptions::default())
            .map_err(|error| format!("cannot open fixture {}: {error}", fixture.fixture))?;
        if index.statistics().total as usize != fixture.chunks {
            return Err(format!(
                "fixture {} holds {} chunks, manifest says {}",
                fixture.fixture,
                index.statistics().total,
                fixture.chunks
            ));
        }
        indexes.insert(fixture.fixture.clone(), index);
    }

    let mut failures: Vec<Failure> = Vec::new();
    let mut ulp_flips: Vec<serde_json::Value> = Vec::new();
    let mut per_fixture: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for case in &cases {
        let index = indexes
            .get(&case.fixture)
            .ok_or_else(|| format!("fixture {} was not opened", case.fixture))?;
        let excluded: BTreeSet<i64> = case.excluded.iter().copied().collect();
        let hits = index.retrieve(
            &case.expression,
            &excluded,
            case.limit,
            case.term.as_deref(),
        );
        let before = failures.len();
        check(case, hits, &mut failures, &mut ulp_flips);
        let entry = per_fixture.entry(case.fixture.clone()).or_insert((0, 0));
        entry.0 += 1;
        if failures.len() == before {
            entry.1 += 1;
        }
    }

    println!(
        "{:<12} {:>7} {:>7} {:>7}",
        "fixture", "cases", "passed", "failed"
    );
    for (fixture, (total, passed)) in &per_fixture {
        println!(
            "{:<12} {:>7} {:>7} {:>7}",
            fixture,
            total,
            passed,
            total - passed
        );
    }
    let passed = per_fixture
        .values()
        .map(|(_, passed)| passed)
        .sum::<usize>();
    println!(
        "gate A: {} cases, {} passed, {} failed, {} ulp_flips",
        cases.len(),
        passed,
        cases.len() - passed,
        ulp_flips.len()
    );
    if !failures.is_empty() {
        for failure in failures.iter().take(10) {
            println!(
                "  {} {}: expected {}, got {}{}",
                failure.case_id,
                failure.path,
                failure.expected,
                failure.actual,
                failure
                    .delta
                    .map(|delta| format!(" (delta {delta:e})"))
                    .unwrap_or_default()
            );
        }
    }

    let report = serde_json::json!({
        "golden": {
            "dir": options.golden.display().to_string(),
            "generated_by": format!("export_golden.py@{}", manifest.generator_version),
            "python": manifest.python,
            "sqlite": manifest.sqlite,
            "runtime_sqlite": runtime_sqlite,
            "reference": {
                "dir": options.reference.display().to_string(),
                "commit": manifest.reference.commit,
            },
        },
        "toolchain": {
            "sqlite_allowlist": SQLITE_ALLOWLIST,
            "jieba": manifest.jieba,
            "pyuca": manifest.pyuca,
            "unicode": manifest.unicode,
        },
        "tolerances": {"level": "S", "f64": "bits", "ids": "exact"},
        "gates": {"A": {
            "cases": cases.len(),
            "passed": passed,
            "failed": cases.len() - passed,
            "ulp_flips": ulp_flips.len(),
            "fixtures": per_fixture.iter().map(|(fixture, (total, passed))| {
                serde_json::json!({"fixture": fixture, "cases": total, "passed": passed})
            }).collect::<Vec<_>>(),
        }},
        "excluded_fields": [],
        "excluded_checks": ["db_physical_bytes"],
        "failures": failures,
        "ulp_flips": ulp_flips,
        "unsupported": [],
    });
    write_report(&options.report, &report);

    Ok(if failures_empty(&report) {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}

/// Gate D `tokenizer`: segment sequences, exhaustive Unicode tables and the
/// ASCII cross product (design §4.4).  `token_positions_diff` must be exactly
/// zero - a tolerance is not allowed here.
fn gate_d_tokenizer(
    options: &Options,
    manifest: &Manifest,
    failures: &mut Vec<Failure>,
) -> Result<serde_json::Value, String> {
    verify_manifest_files(
        options,
        manifest,
        &[
            "tokenizer.jsonl",
            "ascii_runs.jsonl",
            "unicode_lower.bin.gz",
            "unicode_casefold.bin.gz",
            "unicode_combining.bin.gz",
            "unicode_categories.bin.gz",
            "unicode_nfkd.bin.gz",
            "unicode_nfkc.bin.gz",
            "unicode_compose.bin.gz",
        ],
    )?;
    let failures_before = failures.len();

    let segments = compare_token_lines(&options.golden.join("tokenizer.jsonl"), failures)?;
    let expected_segments = manifest
        .tokenizer
        .as_ref()
        .ok_or_else(|| "manifest has no tokenizer section".to_owned())?
        .segments;
    if segments.total != expected_segments {
        return Err(format!(
            "golden inconsistency: tokenizer.jsonl holds {} segments, manifest says {}",
            segments.total, expected_segments
        ));
    }
    let runs = compare_token_lines(&options.golden.join("ascii_runs.jsonl"), failures)?;
    let expected_runs = manifest
        .ascii_runs
        .as_ref()
        .ok_or_else(|| "manifest has no ascii_runs section".to_owned())?
        .runs;
    if runs.total != expected_runs {
        return Err(format!(
            "golden inconsistency: ascii_runs.jsonl holds {} runs, manifest says {}",
            runs.total, expected_runs
        ));
    }

    let mut tables: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut codepoints_checked = 0_usize;
    let mut codepoint_diffs = 0_usize;
    for spec in manifest.unicode_tables.values() {
        let file = manifest
            .files
            .iter()
            .find(|file| file.path == spec.path)
            .ok_or_else(|| format!("manifest does not list {}", spec.path))?;
        if file.records != Some(spec.codepoints) {
            return Err(format!(
                "golden inconsistency: {} records disagree with the unicode table spec",
                spec.path
            ));
        }
    }
    for name in ["lower", "casefold", "combining", "categories"] {
        let spec = manifest
            .unicode_tables
            .get(name)
            .ok_or_else(|| format!("manifest has no unicode table {name}"))?;
        let path = options.golden.join(&spec.path);
        let bytes =
            fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let data = gunzip(&path, &bytes)?;
        let counts = gate_d_table(name, spec, &data, failures)?;
        codepoints_checked += counts.codepoints;
        codepoint_diffs += counts.diffs;
        tables.insert(
            name.to_owned(),
            serde_json::json!({"codepoints": counts.codepoints, "diffs": counts.diffs}),
        );
    }

    let nfkc = manifest
        .unicode_sparse
        .get("nfkc")
        .ok_or_else(|| "manifest has no sparse nfkc table".to_owned())?;
    let nfkc_path = options.golden.join(&nfkc.path);
    let nfkc_bytes = fs::read(&nfkc_path)
        .map_err(|error| format!("cannot read {}: {error}", nfkc_path.display()))?;
    let nfkc_data = gunzip(&nfkc_path, &nfkc_bytes)?;
    let nfkc_counts = gate_d_sparse_table("nfkc", nfkc, &nfkc_data, failures)?;
    codepoints_checked += nfkc_counts.codepoints;
    codepoint_diffs += nfkc_counts.diffs;
    tables.insert(
        "nfkc".to_owned(),
        serde_json::json!({"codepoints": nfkc_counts.codepoints, "diffs": nfkc_counts.diffs}),
    );

    println!(
        "{:<12} {:>9} {:>9} {:>10} {:>6}",
        "fixture", "segments", "equal", "positions", "diff"
    );
    println!(
        "{:<12} {:>9} {:>9} {:>10} {:>6}",
        "tokenizer", segments.total, segments.equal, segments.positions, segments.diff
    );
    println!(
        "{:<12} {:>9} {:>9} {:>10} {:>6}",
        "ascii_runs", runs.total, runs.equal, runs.positions, runs.diff
    );
    println!(
        "unicode: {} tables, {} code points, {} diffs",
        tables.len(),
        codepoints_checked,
        codepoint_diffs
    );
    let passed = failures_before == failures.len()
        && segments.diff == 0
        && segments.equal == segments.total
        && runs.diff == 0
        && runs.equal == runs.total
        && codepoint_diffs == 0;
    Ok(serde_json::json!({
        "segments_total": segments.total,
        "segments_equal": segments.equal,
        "token_positions_total": segments.positions,
        "token_positions_diff": segments.diff,
        "ascii_runs_total": runs.total,
        "ascii_runs_equal": runs.equal,
        "ascii_run_positions_diff": runs.diff,
        "codepoints_checked": codepoints_checked,
        "codepoint_diffs": codepoint_diffs,
        "tables": tables,
        "passed": passed,
    }))
}

/// One `chunks.jsonl` line (`export_golden.py` `_chunk_case`).
#[derive(Deserialize)]
struct ChunkGolden {
    doc_id: String,
    fixture: String,
    title: String,
    source: String,
    path: Option<String>,
    #[serde(default)]
    idx: Option<String>,
    chunk_path: Option<String>,
    target: usize,
    chunks: Vec<ChunkExpectation>,
}

#[derive(Deserialize)]
struct ChunkExpectation {
    heading: String,
    spans: Vec<(usize, usize)>,
    content: String,
}

/// One `index_expected.jsonl` line (`export_golden.py` `index_expectation`).
#[derive(Deserialize)]
struct IndexGolden {
    fixture: String,
    index: String,
    fingerprint: String,
    chunk_chars: usize,
    metadata_rows: usize,
    signature: String,
    chunks: Vec<ChunkRowGolden>,
    fts_vocabulary: BTreeMap<String, Vec<(String, i64)>>,
    statistics_signature: String,
    statistics: String,
    rust_index_metadata_keys: Vec<String>,
}

/// `SELECT * FROM chunks` in column order (`sag2.py:236-245`).
#[derive(Deserialize)]
struct ChunkRowGolden {
    id: i64,
    document_id: i64,
    title: String,
    heading: String,
    content: String,
    full_doc: String,
    spans: String,
    characters: i64,
    source_id: String,
}

/// Gate D `chunks`: `chunk_document` over every fixture document, the desktop
/// reference cases and the `tests/test_chunking.py` scenarios.
fn gate_d_chunks(
    options: &Options,
    manifest: &Manifest,
    failures: &mut Vec<Failure>,
) -> Result<serde_json::Value, String> {
    verify_manifest_files(options, manifest, &["chunks.jsonl"])?;
    let entries = read_chunk_goldens(options)?;
    let mut documents = 0_usize;
    let mut chunks_total = 0_usize;
    let mut chunks_equal = 0_usize;
    let mut chunk_diffs = 0_usize;

    for entry in &entries {
        documents += 1;
        let actual =
            sag2_core::chunk_document(&entry.source, entry.target, entry.chunk_path.as_deref());
        if actual.len() != entry.chunks.len() {
            chunk_diffs += 1;
            push_gate_failure(
                failures,
                &entry.doc_id,
                "chunk_count",
                entry.chunks.len().to_string(),
                actual.len().to_string(),
            );
        }
        let positions = entry.chunks.len().max(actual.len());
        for position in 0..positions {
            chunks_total += 1;
            let expected = entry.chunks.get(position);
            let got = actual.get(position);
            let mut equal = true;
            match (expected, got) {
                (Some(expected), Some(got)) => {
                    if expected.heading != got.heading {
                        equal = false;
                        push_gate_failure(
                            failures,
                            &entry.doc_id,
                            format!("chunks[{position}].heading"),
                            display_text(&expected.heading),
                            display_text(&got.heading),
                        );
                    }
                    if expected.spans != got.spans {
                        equal = false;
                        push_gate_failure(
                            failures,
                            &entry.doc_id,
                            format!("chunks[{position}].spans"),
                            serialise_spans_usize(&expected.spans),
                            serialise_spans_usize(&got.spans),
                        );
                    }
                    if expected.content != got.content {
                        equal = false;
                        push_gate_failure(
                            failures,
                            &entry.doc_id,
                            format!("chunks[{position}].content"),
                            display_text(&expected.content),
                            display_text(&got.content),
                        );
                    }
                }
                (Some(expected), None) => {
                    equal = false;
                    push_gate_failure(
                        failures,
                        &entry.doc_id,
                        format!("chunks[{position}]"),
                        display_text(&expected.content),
                        "<missing>",
                    );
                }
                (None, Some(got)) => {
                    equal = false;
                    push_gate_failure(
                        failures,
                        &entry.doc_id,
                        format!("chunks[{position}]"),
                        "<missing>",
                        display_text(&got.content),
                    );
                }
                (None, None) => unreachable!("position < max(len)"),
            }
            if equal {
                chunks_equal += 1;
            } else {
                chunk_diffs += 1;
            }
        }
    }

    let spec = manifest
        .chunks
        .as_ref()
        .ok_or_else(|| "manifest has no chunks section".to_owned())?;
    if documents != spec.documents || chunks_total != spec.chunks {
        return Err(format!(
            "golden inconsistency: chunks.jsonl holds {documents} documents / \
             {chunks_total} chunks, manifest says {} / {}",
            spec.documents, spec.chunks
        ));
    }
    let passed = chunk_diffs == 0;
    println!(
        "{:<12} {:>9} {:>9} {:>9} {:>7}",
        "fixture", "documents", "chunks", "equal", "diff"
    );
    println!(
        "{:<12} {:>9} {:>9} {:>9} {:>7}",
        "chunks", documents, chunks_total, chunks_equal, chunk_diffs
    );
    Ok(serde_json::json!({
        "documents": documents,
        "chunks": chunks_total,
        "chunks_equal": chunks_equal,
        "chunk_diffs": chunk_diffs,
        "passed": passed,
    }))
}

/// A failure while reading or comparing a built index (tooling error, exit 3).
#[derive(Debug)]
struct GateError(String);

impl From<rusqlite::Error> for GateError {
    fn from(error: rusqlite::Error) -> Self {
        GateError(format!("sqlite: {error}"))
    }
}

impl From<String> for GateError {
    fn from(message: String) -> Self {
        GateError(message)
    }
}

/// A failure buffer that always counts, even after the report buffer is full.
struct GateFailures<'a> {
    failures: &'a mut Vec<Failure>,
    diffs: usize,
}

impl<'a> GateFailures<'a> {
    fn new(failures: &'a mut Vec<Failure>) -> Self {
        GateFailures { failures, diffs: 0 }
    }

    fn push(
        &mut self,
        case_id: &str,
        path: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) {
        self.diffs += 1;
        if self.failures.len() < MAX_FAILURES {
            self.failures.push(Failure {
                case_id: case_id.to_owned(),
                path: path.into(),
                expected: expected.into(),
                actual: actual.into(),
                delta: None,
            });
        }
    }
}

/// Gate D `index`: build the same corpus with this crate and compare the
/// observable index contents against the Python build (design §1.4 D1-D3).
fn gate_d_index(
    options: &Options,
    manifest: &Manifest,
    failures: &mut Vec<Failure>,
) -> Result<serde_json::Value, GateError> {
    verify_manifest_files(options, manifest, &["index_expected.jsonl", "chunks.jsonl"])?;
    let spec = manifest
        .index_expected
        .as_ref()
        .ok_or_else(|| "manifest has no index_expected section".to_owned())?;
    let chunks = read_chunk_goldens(options)?;
    let goldens = read_index_goldens(options)?;
    if goldens.len() != spec.fixtures {
        return Err(format!(
            "golden inconsistency: index_expected.jsonl holds {} fixtures, manifest says {}",
            goldens.len(),
            spec.fixtures
        )
        .into());
    }
    for golden in &goldens {
        let fixture = manifest
            .fixtures
            .iter()
            .find(|fixture| fixture.fixture == golden.fixture)
            .ok_or_else(|| {
                format!(
                    "golden inconsistency: {} is absent from the manifest fixtures",
                    golden.fixture
                )
            })?;
        if fixture.signature != golden.signature
            || fixture.index != golden.index
            || fixture.chunks != golden.chunks.len()
        {
            return Err(format!(
                "golden inconsistency: manifest and index_expected disagree for {}",
                golden.fixture
            )
            .into());
        }
    }

    let mut gate = GateFailures::new(failures);
    let temporary;
    let directory: &Path = match &options.index_dir {
        Some(directory) => {
            fs::create_dir_all(directory)
                .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
            directory
        }
        None => {
            temporary = tempfile::tempdir()
                .map_err(|error| format!("cannot create the index build directory: {error}"))?;
            temporary.path()
        }
    };
    let expected_metadata = expected_rust_metadata();
    let mut fixtures_equal = 0_usize;
    let mut fixtures_total = 0_usize;
    for golden in &goldens {
        fixtures_total += 1;
        let before = gate.diffs;
        let documents: Vec<sag2_index::Document> = chunks
            .iter()
            .filter(|entry| entry.fixture == golden.fixture)
            .map(|entry| sag2_index::Document {
                title: &entry.title,
                text: &entry.source,
                path: entry.path.as_deref(),
                idx: entry.idx.as_deref(),
            })
            .collect();
        if documents.is_empty() {
            gate.push(&golden.fixture, "corpus", "at least one document", "none");
            continue;
        }
        let path = directory.join(&golden.index);
        let opened = sag2_index::CorpusIndex::open(
            &path,
            &documents,
            &golden.fingerprint,
            sag2_index::IndexOptions {
                chunk_chars: golden.chunk_chars,
                source_dir: None,
                rebuild: sag2_index::RebuildPolicy::Always,
            },
        );
        let outcome = match opened {
            Ok((index, outcome)) => {
                drop(index);
                outcome
            }
            Err(error) => {
                gate.push(
                    &golden.fixture,
                    "build",
                    "a built index",
                    format!("{error}"),
                );
                continue;
            }
        };
        if outcome.action == sag2_index::OpenAction::Reused {
            gate.push(
                &golden.fixture,
                "build/action",
                "Built or Rebuilt",
                format!("{:?}", outcome.action),
            );
        }
        let db = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|error| format!("cannot reopen {}: {error}", path.display()))?;
        compare_index(
            &db,
            golden,
            &expected_metadata,
            &golden.rust_index_metadata_keys,
            &mut gate,
        )?;
        if gate.diffs == before {
            fixtures_equal += 1;
        }
    }
    println!(
        "{:<12} {:>9} {:>9} {:>9}",
        "fixture", "fixtures", "equal", "failed"
    );
    println!(
        "{:<12} {:>9} {:>9} {:>9}",
        "index",
        fixtures_total,
        fixtures_equal,
        fixtures_total - fixtures_equal
    );
    Ok(serde_json::json!({
        "fixtures": fixtures_total,
        "fixtures_equal": fixtures_equal,
        "passed": gate.diffs == 0,
    }))
}

fn read_chunk_goldens(options: &Options) -> Result<Vec<ChunkGolden>, String> {
    let path = options.golden.join("chunks.jsonl");
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut entries = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        entries.push(
            serde_json::from_str(line)
                .map_err(|error| format!("{}:{}: {error}", path.display(), line_index + 1))?,
        );
    }
    Ok(entries)
}

fn read_index_goldens(options: &Options) -> Result<Vec<IndexGolden>, String> {
    let path = options.golden.join("index_expected.jsonl");
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut entries = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        entries.push(
            serde_json::from_str(line)
                .map_err(|error| format!("{}:{}: {error}", path.display(), line_index + 1))?,
        );
    }
    Ok(entries)
}

/// The `rust_index_metadata` rows this build writes (design §3.3).
fn expected_rust_metadata() -> BTreeMap<&'static str, &'static str> {
    let versions = &sag2_index::RUNTIME_VERSIONS;
    [
        ("format_version", sag2_index::FORMAT_VERSION_STR),
        ("tokenizer_version", versions.tokenizer_version),
        ("chunk_policy_version", versions.chunk_policy_version),
        ("jieba_dict_sha256", versions.jieba_dict_sha256),
        ("unicode_version", versions.unicode_version),
        ("uca_version", versions.uca_version),
        ("uca_keys_sha256", versions.uca_keys_sha256),
        ("sqlite_version", versions.sqlite_version),
        ("sag_rank_sha256", versions.sag_rank_sha256),
        ("sag2_rs_version", versions.sag2_rs_version),
    ]
    .into_iter()
    .collect()
}

/// Compare a Rust-built index with the Python build.
fn compare_index(
    db: &Connection,
    golden: &IndexGolden,
    expected_metadata: &BTreeMap<&'static str, &'static str>,
    golden_keys: &[String],
    gate: &mut GateFailures<'_>,
) -> Result<(), GateError> {
    let fixture = golden.fixture.as_str();
    let metadata_rows: i64 = db.query_row("SELECT COUNT(*) FROM metadata", [], |row| row.get(0))?;
    if metadata_rows != golden.metadata_rows as i64 {
        gate.push(
            fixture,
            "metadata/rows",
            golden.metadata_rows.to_string(),
            metadata_rows.to_string(),
        );
    }
    let signature: String = db.query_row("SELECT signature FROM metadata LIMIT 1", [], |row| {
        row.get(0)
    })?;
    if signature != golden.signature {
        gate.push(
            fixture,
            "metadata/signature",
            golden.signature.clone(),
            signature,
        );
    }

    let mut statement = db.prepare(
        "SELECT id,document_id,title,heading,content,full_doc,spans,characters,source_id \
         FROM chunks ORDER BY id",
    )?;
    let mut rows = statement.query([])?;
    let mut position = 0_usize;
    while let Some(row) = rows.next()? {
        let actual = ChunkRowGolden {
            id: row.get(0)?,
            document_id: row.get(1)?,
            title: row.get(2)?,
            heading: row.get(3)?,
            content: row.get(4)?,
            full_doc: row.get(5)?,
            spans: row.get(6)?,
            characters: row.get(7)?,
            source_id: row.get(8)?,
        };
        match golden.chunks.get(position) {
            Some(expected) => compare_chunk_row(fixture, position, expected, &actual, gate),
            None => gate.push(
                fixture,
                format!("chunks[{position}]"),
                "<missing>",
                format!("id {}", actual.id),
            ),
        }
        position += 1;
    }
    if position != golden.chunks.len() {
        gate.push(
            fixture,
            "chunks/length",
            golden.chunks.len().to_string(),
            position.to_string(),
        );
    }
    drop(rows);
    drop(statement);

    db.execute(
        "CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main,chunk_fts,'col')",
        [],
    )?;
    for (column, expected) in &golden.fts_vocabulary {
        let mut statement =
            db.prepare("SELECT term, doc FROM temp.vocab WHERE col=? ORDER BY term")?;
        let actual: Vec<(String, i64)> = statement
            .query_map([column], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<_, _>>()?;
        if &actual != expected {
            let first = expected
                .iter()
                .zip(&actual)
                .position(|(left, right)| left != right)
                .unwrap_or(expected.len().min(actual.len()));
            let expected_at = expected.get(first).map(|value| format!("{value:?}"));
            let actual_at = actual.get(first).map(|value| format!("{value:?}"));
            gate.push(
                fixture,
                format!("chunk_fts/{column}"),
                format!(
                    "{} terms, first difference at {first}: {}",
                    expected.len(),
                    expected_at.unwrap_or_else(|| "<missing>".to_owned())
                ),
                format!(
                    "{} terms: {}",
                    actual.len(),
                    actual_at.unwrap_or_else(|| "<missing>".to_owned())
                ),
            );
        }
    }

    let (statistics_signature, statistics): (String, String) =
        db.query_row("SELECT signature, data FROM index_statistics", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
    if statistics_signature != golden.statistics_signature {
        gate.push(
            fixture,
            "index_statistics/signature",
            golden.statistics_signature.clone(),
            statistics_signature,
        );
    }
    let expected: serde_json::Value = serde_json::from_str(&golden.statistics)
        .map_err(|error| format!("{}: golden statistics is not JSON: {error}", golden.fixture))?;
    let actual: serde_json::Value = serde_json::from_str(&statistics)
        .map_err(|error| format!("{}: built statistics is not JSON: {error}", golden.fixture))?;
    compare_statistics(fixture, "index_statistics.data.", &expected, &actual, gate);

    let mut actual_metadata: BTreeMap<String, String> = BTreeMap::new();
    {
        let mut statement = db.prepare("SELECT key, value FROM rust_index_metadata")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            actual_metadata.insert(row.get(0)?, row.get(1)?);
        }
    }
    let mut expected_keys: Vec<&str> = expected_metadata.keys().copied().collect();
    expected_keys.sort_unstable();
    let mut actual_keys: Vec<&str> = actual_metadata.keys().map(String::as_str).collect();
    actual_keys.sort_unstable();
    let mut golden_keys: Vec<&str> = golden_keys.iter().map(String::as_str).collect();
    golden_keys.sort_unstable();
    if actual_keys != expected_keys || expected_keys != golden_keys {
        gate.push(
            fixture,
            "rust_index_metadata/keys",
            format!("{golden_keys:?}"),
            format!("{actual_keys:?}"),
        );
    }
    for (key, expected) in expected_metadata {
        match actual_metadata.get(*key) {
            Some(value) if value == expected => {}
            Some(value) => gate.push(
                fixture,
                format!("rust_index_metadata/{key}"),
                (*expected).to_owned(),
                value.clone(),
            ),
            None => gate.push(
                fixture,
                format!("rust_index_metadata/{key}"),
                (*expected).to_owned(),
                "<missing>",
            ),
        }
    }
    Ok(())
}

fn compare_chunk_row(
    fixture: &str,
    position: usize,
    expected: &ChunkRowGolden,
    actual: &ChunkRowGolden,
    gate: &mut GateFailures<'_>,
) {
    let field = |name: &str| format!("chunks[{position}].{name}");
    if expected.id != actual.id {
        gate.push(
            fixture,
            field("id"),
            expected.id.to_string(),
            actual.id.to_string(),
        );
    }
    if expected.document_id != actual.document_id {
        gate.push(
            fixture,
            field("document_id"),
            expected.document_id.to_string(),
            actual.document_id.to_string(),
        );
    }
    for (name, left, right) in [
        ("title", &expected.title, &actual.title),
        ("heading", &expected.heading, &actual.heading),
        ("content", &expected.content, &actual.content),
        ("full_doc", &expected.full_doc, &actual.full_doc),
        ("spans", &expected.spans, &actual.spans),
        ("source_id", &expected.source_id, &actual.source_id),
    ] {
        if left != right {
            gate.push(
                fixture,
                field(name),
                display_text(left),
                display_text(right),
            );
        }
    }
    if expected.characters != actual.characters {
        gate.push(
            fixture,
            field("characters"),
            expected.characters.to_string(),
            actual.characters.to_string(),
        );
    }
}

/// Compare the seven-key `index_statistics.data` structure (design §1.4 D2).
fn compare_statistics(
    fixture: &str,
    prefix: &str,
    expected: &serde_json::Value,
    actual: &serde_json::Value,
    gate: &mut GateFailures<'_>,
) {
    let (Some(expected), Some(actual)) = (expected.as_object(), actual.as_object()) else {
        gate.push(
            fixture,
            prefix.trim_end_matches('.'),
            "object",
            format!("{actual}"),
        );
        return;
    };
    let mut keys: BTreeSet<&str> = expected.keys().map(String::as_str).collect();
    keys.extend(actual.keys().map(String::as_str));
    for key in keys {
        let path = format!("{prefix}{key}");
        match (expected.get(key), actual.get(key)) {
            (None, Some(value)) => gate.push(fixture, path, "<missing>", format!("{value}")),
            (Some(value), None) => gate.push(fixture, path, format!("{value}"), "<missing>"),
            (Some(left), Some(right)) if key == "diversity_idf" => {
                compare_float_map(fixture, &path, left, right, gate)
            }
            (Some(left), Some(right)) => compare_json_exact(fixture, &path, left, right, gate),
            (None, None) => {}
        }
    }
}

fn compare_float_map(
    fixture: &str,
    prefix: &str,
    expected: &serde_json::Value,
    actual: &serde_json::Value,
    gate: &mut GateFailures<'_>,
) {
    let (Some(expected), Some(actual)) = (expected.as_object(), actual.as_object()) else {
        gate.push(fixture, prefix, "object", format!("{actual}"));
        return;
    };
    let mut keys: BTreeSet<&str> = expected.keys().map(String::as_str).collect();
    keys.extend(actual.keys().map(String::as_str));
    for key in keys {
        let path = format!("{prefix}.{key}");
        match (
            expected.get(key).and_then(serde_json::Value::as_f64),
            actual.get(key).and_then(serde_json::Value::as_f64),
        ) {
            (Some(left), Some(right)) if within_tolerance(left, right) => {}
            (Some(left), Some(right)) => {
                gate.push(fixture, path, format!("{left:?}"), format!("{right:?}"))
            }
            _ => gate.push(
                fixture,
                path,
                expected
                    .get(key)
                    .map_or("<missing>".to_owned(), |value| format!("{value}")),
                actual
                    .get(key)
                    .map_or("<missing>".to_owned(), |value| format!("{value}")),
            ),
        }
    }
}

fn compare_json_exact(
    fixture: &str,
    prefix: &str,
    expected: &serde_json::Value,
    actual: &serde_json::Value,
    gate: &mut GateFailures<'_>,
) {
    if expected == actual {
        return;
    }
    match (expected, actual) {
        (serde_json::Value::Object(expected), serde_json::Value::Object(actual)) => {
            let mut keys: BTreeSet<&str> = expected.keys().map(String::as_str).collect();
            keys.extend(actual.keys().map(String::as_str));
            for key in keys {
                let path = format!("{prefix}.{key}");
                match (expected.get(key), actual.get(key)) {
                    (None, Some(value)) => {
                        gate.push(fixture, path, "<missing>", format!("{value}"))
                    }
                    (Some(value), None) => {
                        gate.push(fixture, path, format!("{value}"), "<missing>")
                    }
                    (Some(left), Some(right)) => {
                        compare_json_exact(fixture, &path, left, right, gate)
                    }
                    (None, None) => {}
                }
            }
        }
        (serde_json::Value::Array(expected), serde_json::Value::Array(actual))
            if expected.len() == actual.len() =>
        {
            for (position, (left, right)) in expected.iter().zip(actual).enumerate() {
                compare_json_exact(fixture, &format!("{prefix}[{position}]"), left, right, gate);
            }
        }
        _ => gate.push(fixture, prefix, format!("{expected}"), format!("{actual}")),
    }
}

fn push_gate_failure(
    failures: &mut Vec<Failure>,
    case_id: &str,
    path: impl Into<String>,
    expected: impl Into<String>,
    actual: impl Into<String>,
) {
    if failures.len() < MAX_FAILURES {
        failures.push(Failure {
            case_id: case_id.to_owned(),
            path: path.into(),
            expected: expected.into(),
            actual: actual.into(),
            delta: None,
        });
    }
}

/// Spans are code point offsets (`chunking.py:153-178`); the report only needs
/// the serialised form.
fn serialise_spans_usize(spans: &[(usize, usize)]) -> String {
    let spans: Vec<(u64, u64)> = spans
        .iter()
        .map(|&(start, end)| (start as u64, end as u64))
        .collect();
    serialise_spans(&spans)
}

fn run_gate_d(options: &Options) -> Result<ExitCode, String> {
    let fixtures = options
        .fixtures
        .clone()
        .unwrap_or_else(|| vec!["tokenizer".to_owned()]);
    for fixture in &fixtures {
        if !matches!(fixture.as_str(), "tokenizer" | "chunks" | "index") {
            return Err(format!(
                "gate D fixture {fixture} is not implemented; use tokenizer, chunks or index"
            ));
        }
    }
    let manifest = preflight(options)?;
    let mut failures: Vec<Failure> = Vec::new();
    let mut sections = serde_json::Map::new();
    let mut passed = true;

    if fixtures.iter().any(|fixture| fixture == "tokenizer") {
        let section = gate_d_tokenizer(options, &manifest, &mut failures)?;
        passed &= section["passed"].as_bool().unwrap_or(false);
        sections.insert("tokenizer".to_owned(), section);
    }
    if fixtures.iter().any(|fixture| fixture == "chunks") {
        let section = gate_d_chunks(options, &manifest, &mut failures)?;
        passed &= section["passed"].as_bool().unwrap_or(false);
        sections.insert("chunks".to_owned(), section);
    }
    if fixtures.iter().any(|fixture| fixture == "index") {
        let section = gate_d_index(options, &manifest, &mut failures).map_err(|error| error.0)?;
        passed &= section["passed"].as_bool().unwrap_or(false);
        sections.insert("index".to_owned(), section);
    }
    passed &= failures.is_empty();
    println!(
        "gate D ({}): {} failures",
        fixtures.join(","),
        failures.len()
    );
    for failure in failures.iter().take(10) {
        println!(
            "  {} {}: expected {}, got {}",
            failure.case_id, failure.path, failure.expected, failure.actual
        );
    }

    let report = serde_json::json!({
        "golden": {
            "dir": options.golden.display().to_string(),
            "generated_by": format!("export_golden.py@{}", manifest.generator_version),
            "python": manifest.python,
            "sqlite": manifest.sqlite,
            "runtime_sqlite": sag2_index::sqlite_version(),
            "reference": {
                "dir": options.reference.display().to_string(),
                "commit": manifest.reference.commit,
            },
        },
        "toolchain": {
            "sqlite_allowlist": SQLITE_ALLOWLIST,
            "jieba": manifest.jieba,
            "pyuca": manifest.pyuca,
            "unicode": manifest.unicode,
        },
        "tolerances": {"level": "S", "f64": "bits", "ids": "exact"},
        "gates": {"D": {
            "fixtures": fixtures,
            "sections": sections,
        }},
        "excluded_fields": [],
        "excluded_checks": ["db_physical_bytes"],
        "failures": failures,
        "ulp_flips": [],
        "unsupported": [],
    });
    write_report(&options.report, &report);
    Ok(if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}

#[derive(Deserialize)]
struct TokenLine {
    #[serde(default)]
    segment_id: Option<String>,
    text: String,
    tokens: Vec<String>,
}

#[derive(Default)]
struct TokenCounts {
    total: usize,
    equal: usize,
    positions: usize,
    diff: usize,
}

/// Compare one JSONL token golden against `sag2_core::tokenize`.  The first
/// differing token per segment is reported with its position.
fn compare_token_lines(path: &Path, failures: &mut Vec<Failure>) -> Result<TokenCounts, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut counts = TokenCounts::default();
    for (line_index, line) in text.lines().enumerate() {
        let entry: TokenLine = serde_json::from_str(line)
            .map_err(|error| format!("{}:{}: {error}", path.display(), line_index + 1))?;
        let case_id = entry
            .segment_id
            .clone()
            .unwrap_or_else(|| format!("{}:{}", path.display(), line_index + 1));
        let actual = sag2_core::tokenize(&entry.text);
        counts.total += 1;
        let positions = entry.tokens.len().max(actual.len());
        counts.positions += positions;
        let mut mismatches = 0_usize;
        let mut first: Option<(usize, String, String)> = None;
        for position in 0..positions {
            let expected = entry.tokens.get(position);
            let got = actual.get(position);
            if expected == got {
                continue;
            }
            mismatches += 1;
            if first.is_none() {
                first = Some((
                    position,
                    expected
                        .map(|token| format!("{token:?}"))
                        .unwrap_or_else(|| "<missing>".to_owned()),
                    got.map(|token| format!("{token:?}"))
                        .unwrap_or_else(|| "<missing>".to_owned()),
                ));
            }
        }
        if mismatches == 0 {
            counts.equal += 1;
        } else {
            counts.diff += mismatches;
            if failures.len() < MAX_FAILURES {
                let (position, expected, got) = first.expect("mismatch position");
                failures.push(Failure {
                    case_id,
                    path: format!("tokens[{position}]"),
                    expected,
                    actual: got,
                    delta: None,
                });
            }
        }
    }
    Ok(counts)
}

fn gunzip(path: &Path, compressed: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Read as _;
    let mut data = Vec::new();
    flate2::read::GzDecoder::new(compressed)
        .read_to_end(&mut data)
        .map_err(|error| format!("cannot decompress {}: {error}", path.display()))?;
    Ok(data)
}

#[derive(Default)]
struct TableCounts {
    codepoints: usize,
    diffs: usize,
}

/// Compare one exhaustive table against `sag2_core`.  Lone surrogates are not
/// scalar values, so the golden's identity record is asserted instead.
fn gate_d_table(
    name: &str,
    spec: &UnicodeTableSpec,
    data: &[u8],
    failures: &mut Vec<Failure>,
) -> Result<TableCounts, String> {
    if data.len() < 8 || &data[..8] != spec.magic.as_bytes() {
        return Err(format!(
            "{} does not start with the manifest magic",
            spec.path
        ));
    }
    let expected_len = 8 + spec.codepoints * spec.record_bytes;
    if data.len() != expected_len {
        return Err(format!(
            "{} holds {} bytes, manifest layout says {expected_len}",
            spec.path,
            data.len()
        ));
    }
    let mut counts = TableCounts::default();
    for cp in 0..spec.codepoints {
        counts.codepoints += 1;
        let scalar = cp as u32;
        let base = 8 + cp * spec.record_bytes;
        let surrogate = (0xD800..=0xDFFF).contains(&cp);
        let (expected, actual) = match name {
            "lower" | "casefold" => {
                let length = data[base + 4] as usize;
                let mapped: Vec<u32> = (0..length)
                    .map(|index| {
                        let at = base + 5 + index * 4;
                        u32::from_le_bytes(data[at..at + 4].try_into().expect("mapping slot"))
                    })
                    .collect();
                if surrogate {
                    let identity = mapped == [scalar];
                    if !identity {
                        counts.diffs += 1;
                        push_table_failure(
                            failures,
                            name,
                            cp,
                            format!("{mapped:?}"),
                            "<surrogate>".to_owned(),
                        );
                    }
                    continue;
                }
                let character = char::from_u32(scalar).expect("scalar value");
                let produced: Vec<u32> = if name == "lower" {
                    sag2_core::lower(&character.to_string())
                } else {
                    sag2_core::casefold(&character.to_string())
                }
                .chars()
                .map(|mapped| mapped as u32)
                .collect();
                (format!("{mapped:?}"), format!("{produced:?}"))
            }
            "combining" => {
                let class = data[base + 4];
                if surrogate {
                    if class != 0 {
                        counts.diffs += 1;
                        push_table_failure(
                            failures,
                            name,
                            cp,
                            class.to_string(),
                            "<surrogate>".to_owned(),
                        );
                    }
                    continue;
                }
                (
                    class.to_string(),
                    sag2_core::combining_class(scalar).to_string(),
                )
            }
            "categories" => {
                let mask = data[base + 4];
                if surrogate {
                    if mask != 0 {
                        counts.diffs += 1;
                        push_table_failure(
                            failures,
                            name,
                            cp,
                            mask.to_string(),
                            "<surrogate>".to_owned(),
                        );
                    }
                    continue;
                }
                let mut actual = 0_u8;
                if sag2_core::is_letter(scalar) {
                    actual |= 0b0001;
                }
                if sag2_core::is_number(scalar) {
                    actual |= 0b0010;
                }
                if sag2_core::is_uppercase(scalar) {
                    actual |= 0b0100;
                }
                if sag2_core::is_mark(scalar) {
                    actual |= 0b1000;
                }
                (mask.to_string(), actual.to_string())
            }
            other => return Err(format!("unknown unicode table {other}")),
        };
        if expected != actual {
            counts.diffs += 1;
            push_table_failure(failures, name, cp, expected, actual);
        }
    }
    Ok(counts)
}

/// Compare one sparse table (`cp:u32le, len:u8, cps`), treating missing code
/// points as identity.  Used for the exhaustive NFKC check.
fn gate_d_sparse_table(
    name: &str,
    spec: &UnicodeSparseSpec,
    data: &[u8],
    failures: &mut Vec<Failure>,
) -> Result<TableCounts, String> {
    if data.len() < 13 || &data[..8] != spec.magic.as_bytes() {
        return Err(format!(
            "{} does not start with the manifest magic",
            spec.path
        ));
    }
    let count = u32::from_le_bytes(data[8..12].try_into().expect("sparse count")) as usize;
    let width = data[12] as usize;
    let stride = 5 + 4 * width;
    let expected_len = 13 + count * stride;
    if count != spec.records || data.len() != expected_len {
        return Err(format!(
            "{} holds {} records / {} bytes, manifest says {} records with {stride}-byte stride",
            spec.path,
            count,
            data.len(),
            spec.records
        ));
    }
    let records = &data[13..];
    let mut counts = TableCounts::default();
    for cp in 0..0x11_0000_usize {
        if (0xD800..=0xDFFF).contains(&cp) {
            continue;
        }
        counts.codepoints += 1;
        // Binary search the sparse record for this code point.
        let mut low = 0_usize;
        let mut high = count;
        let mut mapped: Option<(usize, usize)> = None;
        while low < high {
            let middle = (low + high) / 2;
            let base = middle * stride;
            let found = u32::from_le_bytes(
                records[base..base + 4]
                    .try_into()
                    .expect("sparse record cp"),
            );
            match found.cmp(&(cp as u32)) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => {
                    mapped = Some((base, records[base + 4] as usize));
                    break;
                }
            }
        }
        let expected: Vec<u32> = match mapped {
            Some((base, length)) => (0..length)
                .map(|index| {
                    let at = base + 5 + index * 4;
                    u32::from_le_bytes(records[at..at + 4].try_into().expect("sparse cps"))
                })
                .collect(),
            None => vec![cp as u32],
        };
        let character = char::from_u32(cp as u32).expect("scalar value");
        let actual: Vec<u32> = sag2_core::nfkc(&character.to_string())
            .chars()
            .map(|mapped| mapped as u32)
            .collect();
        if expected != actual {
            counts.diffs += 1;
            push_table_failure(
                failures,
                name,
                cp,
                format!("{expected:?}"),
                format!("{actual:?}"),
            );
        }
    }
    Ok(counts)
}

fn push_table_failure(
    failures: &mut Vec<Failure>,
    table: &str,
    cp: usize,
    expected: String,
    actual: String,
) {
    if failures.len() < MAX_FAILURES {
        failures.push(Failure {
            case_id: format!("unicode/{table}"),
            path: format!("U+{cp:04X}"),
            expected,
            actual,
            delta: None,
        });
    }
}

/// `failures` and `ulp_flips` both make the gate fail (design §1.1).
fn failures_empty(report: &serde_json::Value) -> bool {
    report["failures"].as_array().is_some_and(Vec::is_empty)
        && report["ulp_flips"].as_array().is_some_and(Vec::is_empty)
}

fn check(
    case: &Case,
    hits: Result<Vec<sag2_index::Hit>, QueryError>,
    failures: &mut Vec<Failure>,
    ulp_flips: &mut Vec<serde_json::Value>,
) {
    let hits = match (hits, &case.error) {
        (Err(_), Some(_)) => return,
        (Err(error), None) => {
            push_failure(
                failures,
                case,
                "retrieve",
                "results".into(),
                format!("error: {error}"),
                None,
            );
            return;
        }
        (Ok(hits), Some(_)) => {
            push_failure(
                failures,
                case,
                "retrieve",
                "error".into(),
                format!("{} hits", hits.len()),
                None,
            );
            return;
        }
        (Ok(hits), None) => hits,
    };

    if hits.len() != case.hits.len() {
        push_failure(
            failures,
            case,
            "length",
            format!("{} hits", case.hits.len()),
            format!("{} hits", hits.len()),
            None,
        );
    }
    for (position, expected) in case.hits.iter().enumerate() {
        let Some(actual) = hits.get(position) else {
            break;
        };
        let field = |name: &str| format!("hits[{position}].{name}");
        if actual.id != expected.id {
            push_failure(
                failures,
                case,
                field("id"),
                expected.id.to_string(),
                actual.id.to_string(),
                None,
            );
        }
        if actual.document_id != expected.document_id {
            push_failure(
                failures,
                case,
                field("document_id"),
                expected.document_id.to_string(),
                actual.document_id.to_string(),
                None,
            );
        }
        compare_text(
            failures,
            case,
            &field("title"),
            &expected.title,
            &actual.title,
        );
        compare_text(
            failures,
            case,
            &field("heading"),
            &expected.heading,
            &actual.heading,
        );
        compare_text(
            failures,
            case,
            &field("content"),
            &expected.content,
            &actual.content,
        );
        compare_text(
            failures,
            case,
            &field("spans"),
            &expected.spans,
            &serialise_spans(&actual.spans),
        );
        if i64::from(actual.characters) != expected.characters {
            push_failure(
                failures,
                case,
                field("characters"),
                expected.characters.to_string(),
                actual.characters.to_string(),
                None,
            );
        }
        compare_text(
            failures,
            case,
            &field("source_id"),
            &expected.source_id,
            &actual.source_id,
        );

        let Some(score) = bits_to_f64(&expected.score_bits) else {
            push_failure(
                failures,
                case,
                field("score_bits"),
                expected.score_bits.clone(),
                "unparseable golden bits".into(),
                None,
            );
            continue;
        };
        if actual.score.to_bits() == score.to_bits() {
            continue;
        }
        let delta = actual.score - score;
        push_failure(
            failures,
            case,
            field("score"),
            format!("{} ({score:e})", expected.score_bits),
            format!("{:016x} ({:e})", actual.score.to_bits(), actual.score),
            Some(delta),
        );
        if within_tolerance(score, actual.score) {
            ulp_flips.push(serde_json::json!({
                "case_id": case.case_id,
                "position": position,
                "expected": expected.score_bits,
                "actual": format!("{:016x}", actual.score.to_bits()),
                "delta": delta,
            }));
        }
    }
}

/// `eq_str` (design §1.0): the serialised form must match byte for byte.
fn compare_text(
    failures: &mut Vec<Failure>,
    case: &Case,
    path: &str,
    expected: &str,
    actual: &str,
) {
    if expected == actual {
        return;
    }
    push_failure(
        failures,
        case,
        path,
        display_text(expected),
        display_text(actual),
        None,
    );
}

/// Keep a report readable: full strings can be a whole chunk.
fn display_text(text: &str) -> String {
    const LIMIT: usize = 60;
    let mut chars = text.chars();
    let shortened: String = chars.by_ref().take(LIMIT).collect();
    if chars.next().is_none() {
        format!("{shortened:?}")
    } else {
        format!(
            "{shortened:?}… ({} bytes, sha256 {})",
            text.len(),
            &sha256_hex(text.as_bytes())[..12]
        )
    }
}

/// The compact JSON the reference stores in `chunks.spans`
/// (`sag2.py:302` `compact_json(spans)`).
fn serialise_spans(spans: &[(u64, u64)]) -> String {
    let mut text = String::with_capacity(spans.len() * 12 + 2);
    text.push('[');
    for (position, (start, end)) in spans.iter().enumerate() {
        if position > 0 {
            text.push(',');
        }
        text.push('[');
        text.push_str(&start.to_string());
        text.push(',');
        text.push_str(&end.to_string());
        text.push(']');
    }
    text.push(']');
    text
}

fn push_failure(
    failures: &mut Vec<Failure>,
    case: &Case,
    path: impl Into<String>,
    expected: String,
    actual: String,
    delta: Option<f64>,
) {
    if failures.len() < MAX_FAILURES {
        failures.push(Failure {
            case_id: case.case_id.clone(),
            path: path.into(),
            expected,
            actual,
            delta,
        });
    }
}

/// `eq_f64` (design §1.0), used only to classify a bit difference as an ulp flip.
fn within_tolerance(left: f64, right: f64) -> bool {
    (left - right).abs() <= RELATIVE_TOLERANCE * left.abs().max(1.0).max(right.abs())
}

/// Decode the raw little-endian `struct.pack('<d', score)` hex from the golden.
fn bits_to_f64(hex: &str) -> Option<f64> {
    if hex.len() != 16 {
        return None;
    }
    let mut bytes = [0_u8; 8];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(hex.get(index * 2..index * 2 + 2)?, 16).ok()?;
    }
    Some(f64::from_le_bytes(bytes))
}

fn dump_signatures(options: &Options) -> Result<BTreeMap<String, String>, String> {
    let path = options.golden.join("index_dump.jsonl");
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut signatures = BTreeMap::new();
    for line in text.lines() {
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|error| format!("{}: {error}", path.display()))?;
        if value["type"] == "metadata" {
            let fixture = value["fixture"]
                .as_str()
                .ok_or_else(|| format!("{} metadata row without fixture", path.display()))?;
            let signature = value["signature"]
                .as_str()
                .ok_or_else(|| format!("{} metadata row without signature", path.display()))?;
            signatures.insert(fixture.to_owned(), signature.to_owned());
        }
    }
    Ok(signatures)
}

fn read_cases(options: &Options) -> Result<Vec<Case>, String> {
    let path = options.golden.join("retrieve.jsonl");
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut cases = Vec::new();
    for line in text.lines() {
        let case: Case =
            serde_json::from_str(line).map_err(|error| format!("{}: {error}", path.display()))?;
        validate_case(&path, &case)?;
        cases.push(case);
    }
    Ok(cases)
}

/// The golden carries both ordered projections and the per-hit fields; they
/// must agree, or the golden itself is broken (exit 3, not a gate failure).
fn validate_case(path: &Path, case: &Case) -> Result<(), String> {
    let broken = |detail: String| format!("{}: case {} {detail}", path.display(), case.case_id);
    if case.ids.len() != case.hits.len()
        || case.document_ids.len() != case.hits.len()
        || case.scores_bits.len() != case.hits.len()
    {
        return Err(broken("has inconsistent golden arrays".to_owned()));
    }
    for (position, hit) in case.hits.iter().enumerate() {
        if hit.id != case.ids[position] {
            return Err(broken(format!("hit {position} id disagrees with ids")));
        }
        if hit.document_id != case.document_ids[position] {
            return Err(broken(format!(
                "hit {position} document_id disagrees with document_ids"
            )));
        }
        if hit.score_bits != case.scores_bits[position] {
            return Err(broken(format!(
                "hit {position} score_bits disagrees with scores_bits"
            )));
        }
    }
    Ok(())
}

fn parse_version(text: &str) -> Option<(u64, u64)> {
    let mut parts = text.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn write_report(path: &Path, report: &serde_json::Value) {
    let rendered = match serde_json::to_string_pretty(report) {
        Ok(rendered) => rendered,
        Err(error) => {
            eprintln!("verify: cannot serialise the report: {error}");
            return;
        }
    };
    if let Err(error) = fs::write(path, rendered + "\n") {
        eprintln!("verify: cannot write {}: {error}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::{check, display_text, serialise_spans, Case, Failure, HitGolden};
    use sag2_index::Hit;

    fn bits(value: f64) -> String {
        value
            .to_le_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn golden_hit() -> HitGolden {
        HitGolden {
            id: 1,
            document_id: 1,
            title: "Anchor".to_owned(),
            heading: "".to_owned(),
            content: "body".to_owned(),
            spans: "[[0,4]]".to_owned(),
            characters: 4,
            source_id: "0".to_owned(),
            score_bits: bits(-1.5),
        }
    }

    fn hit() -> Hit {
        Hit {
            id: 1,
            document_id: 1,
            title: "Anchor".to_owned(),
            heading: "".to_owned(),
            content: "body".to_owned(),
            full_doc: Some("body".to_owned()),
            spans: vec![(0, 4)],
            characters: 4,
            source_id: "0".to_owned(),
            score: -1.5,
        }
    }

    fn case(hits: Vec<HitGolden>) -> Case {
        Case {
            case_id: "C0/test".to_owned(),
            fixture: "C0".to_owned(),
            expression: "\"anchor\"".to_owned(),
            excluded: Vec::new(),
            limit: 7,
            term: None,
            ids: hits.iter().map(|hit| hit.id).collect(),
            document_ids: hits.iter().map(|hit| hit.document_id).collect(),
            scores_bits: hits.iter().map(|hit| hit.score_bits.clone()).collect(),
            hits,
            error: None,
        }
    }

    fn failures_for(expected: Case, actual: Vec<Hit>) -> Vec<Failure> {
        let mut failures = Vec::new();
        let mut ulp_flips = Vec::new();
        check(&expected, Ok(actual), &mut failures, &mut ulp_flips);
        assert!(ulp_flips.is_empty());
        failures
    }

    #[test]
    fn matching_materialised_fields_pass() {
        assert!(failures_for(case(vec![golden_hit()]), vec![hit()]).is_empty());
    }

    #[test]
    fn content_difference_fails_on_the_content_field() {
        let mut expected = golden_hit();
        expected.content = "other body".to_owned();
        let failures = failures_for(case(vec![expected]), vec![hit()]);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].path, "hits[0].content");
    }

    #[test]
    fn title_difference_fails_on_the_title_field() {
        let mut expected = golden_hit();
        expected.title = "Elsewhere".to_owned();
        let failures = failures_for(case(vec![expected]), vec![hit()]);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].path, "hits[0].title");
    }

    #[test]
    fn spans_difference_fails_on_the_spans_field() {
        let mut expected = golden_hit();
        expected.spans = "[[0,5]]".to_owned();
        let failures = failures_for(case(vec![expected]), vec![hit()]);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].path, "hits[0].spans");
    }

    #[test]
    fn characters_difference_fails_on_the_characters_field() {
        let mut expected = golden_hit();
        expected.characters = 5;
        let failures = failures_for(case(vec![expected]), vec![hit()]);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].path, "hits[0].characters");
    }

    #[test]
    fn score_bit_difference_fails_on_the_score_field() {
        let mut expected = golden_hit();
        expected.score_bits = bits(-1.25);
        let failures = failures_for(case(vec![expected]), vec![hit()]);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].path, "hits[0].score");
        assert!(failures[0].delta.is_some());
    }

    #[test]
    fn spans_serialise_like_the_reference_compact_json() {
        assert_eq!(serialise_spans(&[(0, 4)]), "[[0,4]]");
        assert_eq!(serialise_spans(&[(0, 4), (6, 9)]), "[[0,4],[6,9]]");
        assert_eq!(serialise_spans(&[]), "[]");
    }

    #[test]
    fn long_content_is_shortened_in_the_report_only() {
        let text = "x".repeat(500);
        let rendered = display_text(&text);
        assert!(rendered.contains("500 bytes"));
        assert!(rendered.len() < text.len());
    }
}
