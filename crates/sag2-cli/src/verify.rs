//! `sag2-cli verify --gate A|B|D`: replay the golden gate cases (design §1.1,
//! §4.4, §5.5, §7.3).
//!
//! Gate A replays the retrieved hits; Gate B compares pyuca's sort keys over
//! the vocabulary, every code point and 200,000 comparison pairs; Gate D
//! compares the tokenizer against the frozen reference lines, every corpus
//! segment, the ASCII cross product and the exhaustive per-code-point Unicode
//! tables.
//!
//! Gate D `index` implements design §1.4's sub-checks: D1 (`sqlite_master`
//! name/type/sql, whitespace-normalised), D2 (signature and statistics), D3
//! (every `chunks` row and every `chunk_fts` row) and D4 (24 `(rowid, bm25)`
//! probes replayed with the reference ranking SQL).
//!
//! Comparison uses only the §1.0 tolerance functions: the ordered hit rows are
//! compared field by field (`eq_ids` on `id`/`document_id`, `eq_str` on the
//! §1.1 materialisation fields, `eq_f64_bits` on the scores and on
//! `diversity_idf`).  Exit codes follow §1.0: `0` passed, `2` a comparison
//! failed, `3` the tooling/data cannot be used.  A gate run outside the
//! certified toolchain, or against a reference checkout that is not the one
//! the goldens came from, compares nothing and exits `3` (design §1.0 rule 2,
//! §7.1).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rusqlite::{Connection, OpenFlags, OptionalExtension};
use sag2_index::{
    open_readonly, CorpusIndex, ExpansionQuery, ExpansionTerm, QueryError, QueryOptions, Seed,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Generator version this verifier understands.
const GENERATOR_VERSION: &str = "0.8.1";

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
    #[serde(default)]
    sortkeys: Option<SortkeysSpec>,
    #[serde(default)]
    expansion: Option<ExpansionSpec>,
    #[serde(default)]
    freq: Option<FreqSpec>,
    #[serde(default)]
    search: Option<SearchSpec>,
}

/// Gate B's sort-key goldens (design §5.5).
#[derive(Deserialize)]
struct SortkeysSpec {
    vocab: SortkeysVocab,
    codepoints: SortkeysCodepoints,
    pairs: SortkeysPairs,
}

/// `sk_vocab.jsonl`: one term per line in code point order.
#[derive(Deserialize)]
struct SortkeysVocab {
    terms: usize,
}

/// `sk_cp.bin.gz`: `magic, u32 count`, then `len:u8 + weights:[u16le;len]`.
#[derive(Deserialize)]
struct SortkeysCodepoints {
    magic: String,
    codepoints: usize,
    max_weights: usize,
}

/// `sk_pairs.bin.gz`: `magic, u32 count`, then
/// `kind:u8 + a:u32le + b:u32le + expected:i8`.
#[derive(Deserialize)]
struct SortkeysPairs {
    magic: String,
    pairs: usize,
}

/// Gate D's chunk-golden document count (`chunks.jsonl`).
#[derive(Deserialize)]
struct ChunksSpec {
    documents: usize,
    chunks: usize,
}

/// Gate D's `freq` section: rule 5's short-key table (design §4.1).
#[derive(Deserialize)]
struct FreqSpec {
    /// Short positive `TOKENIZER.FREQ` keys (`freq_keys.txt.gz`).
    keys: usize,
    /// `FREQ_MAX_CHARS` of the table generator.
    max_chars: usize,
    /// `ANCHOR_EVERY` of the table generator.
    anchor_every: usize,
    /// How many of those keys sit on an anchor position.
    anchor_entries: usize,
    /// `freq_probes.jsonl` line count.
    probes: usize,
    #[serde(default)]
    corpus_probes: Option<usize>,
}

/// One `freq_probes.jsonl` line: a key and the reference's `tokenize` output.
#[derive(Deserialize)]
struct FreqProbe {
    key: String,
    tokens: Vec<String>,
}

/// Gate B's expansion goldens (design §1.2).
#[derive(Deserialize)]
struct ExpansionSpec {
    /// Total `(query, hop)` records in `expansion.jsonl`.
    cases: usize,
    fixtures: Vec<ExpansionFixture>,
}

/// One expansion fixture: the Python index its records replay against.
#[derive(Deserialize)]
struct ExpansionFixture {
    fixture: String,
    index: String,
    #[serde(default)]
    index_sha256: Option<String>,
    chunks: usize,
    signature: String,
    records: usize,
}

/// One `expansion.jsonl` line.  `stage` decides which fields are meaningful:
/// `expansion` replays `search_sag2`'s hop, `retrieve` calls
/// `retrieve_expansion` directly, `repair` calls `repair_names`.
#[derive(Deserialize)]
struct ExpansionLine {
    case_id: String,
    stage: String,
    fixture: String,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    seeds: Vec<SeedGolden>,
    #[serde(default)]
    terms: Vec<String>,
    #[serde(default)]
    term_scores: Vec<TermScoreGolden>,
    #[serde(default)]
    expression: Option<String>,
    #[serde(default)]
    excluded: Vec<i64>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    term_limit: Option<usize>,
    #[serde(default)]
    input_terms: Vec<InputTermGolden>,
    #[serde(default)]
    queries: Vec<QueryGolden>,
    #[serde(default)]
    hits: Vec<ExpansionHitGolden>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    repaired: Option<String>,
}

/// A seed's replayed fields (`seed["document_id"]`, `["jev_score"]`,
/// `["content"]`).  `jev_score` is a JSON number, and the exporter writes
/// Python's shortest round-tripping `repr`, so it parses back bit-identically.
#[derive(Deserialize)]
struct SeedGolden {
    document_id: i64,
    jev_score: f64,
    content: String,
}

/// One `term_scores[]` record, every `f64` as raw `struct.pack('<d')` bits.
#[derive(Deserialize)]
struct TermScoreGolden {
    term: String,
    score_bits: String,
    unweighted_score_bits: String,
    seed_weight_bits: String,
    seed_jev_score_bits: String,
    term_frequency: i64,
    document_frequency: i64,
    query_context_score_bits: String,
    seed_document_id: i64,
    kind: String,
    #[serde(default)]
    name: Option<String>,
}

/// A `retrieve` record's input term (`term`, `kind`, optional `name`).
#[derive(Deserialize)]
struct InputTermGolden {
    term: String,
    kind: String,
    #[serde(default)]
    name: Option<String>,
}

/// One `queries[]` record (`sag2.py:723-733`).
#[derive(Deserialize)]
struct QueryGolden {
    term: String,
    term_rank: usize,
    kind: String,
    match_expression: String,
    ranking_expression: String,
    retrieved_count: usize,
    selected: Vec<SelectedGolden>,
}

/// One `selected[]` entry.
#[derive(Deserialize)]
struct SelectedGolden {
    document_id: i64,
    bm25_rank: usize,
    bm25_score_bits: String,
}

/// One `hits[]` record: the interleaved order is the comparison.
#[derive(Deserialize)]
struct ExpansionHitGolden {
    id: i64,
    document_id: i64,
    score_bits: String,
    expansion_term: String,
    expansion_term_rank: usize,
    expansion_bm25_rank: usize,
}

/// Gate D's Rust-vs-Python index baseline count (`index_expected.jsonl`).
#[derive(Deserialize)]
struct IndexExpectedSpec {
    fixtures: usize,
    /// Design §1.4 D4 probe count (24).
    probes: usize,
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
    /// `null` for a `source_dir` fixture: its bytes embed the absolute build
    /// tree, so no reference index is shipped.
    #[serde(default)]
    index_sha256: Option<String>,
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
    "usage: sag2-cli verify --gate A|B|C|D --golden golden/ \
     [--fixtures tokenizer,chunks,index,freq|sortkeys|expansion|search] [--reference <scripts>] \
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
        "B" => run_gate_b(options),
        "C" => run_gate_c(options),
        "D" => run_gate_d(options),
        other => Err(format!(
            "gate {other} is not implemented; gates A, B, C and D have goldens"
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
            // A gzip-compressed JSONL golden is counted after decompression;
            // the compressed byte stream has no meaningful newline count.
            let data = if file.path.ends_with(".gz") {
                gunzip(&path, &bytes)?
            } else {
                bytes.clone()
            };
            let lines = data.iter().filter(|byte| **byte == b'\n').count();
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

    // Design §7.3: Gate A may replay against indexes built by this crate
    // (`--index-dir`) instead of the Python reference bytes.  There is never a
    // silent fallback: a fixture missing from `--index-dir` is a tooling
    // failure, and the report records the bytes each case actually ran against.
    let index_directory = options
        .index_dir
        .clone()
        .unwrap_or_else(|| options.golden.join("indexes"));
    let mut indexes: BTreeMap<String, CorpusIndex> = BTreeMap::new();
    let mut index_provenance: Vec<serde_json::Value> = Vec::new();
    for fixture in &manifest.fixtures {
        if !cases.iter().any(|case| case.fixture == fixture.fixture) {
            continue;
        }
        let source = index_directory.join(&fixture.index);
        let bytes = fs::read(&source).map_err(|error| {
            if options.index_dir.is_some() {
                format!(
                    "fixture {} has no index at {} (--index-dir): {error}",
                    fixture.fixture,
                    source.display()
                )
            } else {
                format!("cannot read {}: {error}", source.display())
            }
        })?;
        let digest = sha256_hex(&bytes);
        let manifest_verified = fixture.index_sha256.as_deref() == Some(digest.as_str());
        // The reference bytes must match the manifest exactly; a rebuild
        // legitimately differs, so its digest is recorded as provenance.
        if options.index_dir.is_none() && !manifest_verified {
            return Err(match &fixture.index_sha256 {
                Some(_) => format!("{} does not match the manifest sha256", fixture.index),
                None => format!(
                    "{} is used by Gate A but has no manifest sha256",
                    fixture.index
                ),
            });
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
        index_provenance.push(serde_json::json!({
            "fixture": fixture.fixture,
            "path": source.display().to_string(),
            "sha256": digest,
            "manifest_sha256": fixture.index_sha256,
            "manifest_verified": manifest_verified,
        }));
        indexes.insert(fixture.fixture.clone(), index);
    }
    let verified = index_provenance
        .iter()
        .filter(|entry| entry["manifest_verified"] == serde_json::Value::Bool(true))
        .count();
    println!(
        "indexes: {} ({} fixtures, {verified}/{} match the manifest sha256)",
        index_directory.display(),
        index_provenance.len(),
        index_provenance.len()
    );

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
            "indexes": {
                "dir": index_directory.display().to_string(),
                "source": if options.index_dir.is_some() { "index_dir" } else { "golden" },
                "fixtures": index_provenance,
            },
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
    /// Relative source tree when the fixture was built in `source_dir` mode
    /// (`content=''` + `start_char`/`end_char` + `source_files`).
    #[serde(default)]
    source_dir: Option<String>,
    /// D1: `SELECT type,name,sql FROM sqlite_master` (whitespace-normalised).
    schema: Vec<SchemaGolden>,
    chunks: Vec<ChunkRowGolden>,
    /// D3: every `chunk_fts` row; `[rowid,title,heading,body]`, or
    /// `[rowid,title,heading]` under `content=''` (design §1.4 D3).
    fts_rows: Vec<Vec<serde_json::Value>>,
    /// D3: `source_files` rows, empty for a materialised index.
    #[serde(default)]
    source_files: Vec<SourceFileGolden>,
    fts_vocabulary: BTreeMap<String, Vec<(String, i64)>>,
    /// D4: `(rowid, bm25)` rows of the reference ranking SQL.
    probes: Vec<ProbeGolden>,
    statistics_signature: String,
    statistics: String,
    rust_index_metadata_keys: Vec<String>,
}

/// One `sqlite_master` entry (design §1.4 D1).
#[derive(Deserialize, PartialEq, Eq)]
struct SchemaGolden {
    #[serde(rename = "type")]
    kind: String,
    name: String,
    sql: Option<String>,
}

/// One `source_files` row (`sag2.py:246`, `content=''` schema only).
#[derive(Deserialize, PartialEq, Eq, Debug)]
struct SourceFileGolden {
    document_id: i64,
    path: String,
    sha256: String,
}

/// One D4 probe: the reference `MATCH` expression and its ranked rows.
#[derive(Deserialize)]
struct ProbeGolden {
    /// The expression whose `sag_stats` blob the ranking SQL is bound to
    /// (`sag2.py:487-492`).
    expression: String,
    /// The expression the ranking SQL actually matches, i.e. `expression`
    /// composed with `term` exactly as `sag2.py:484` does.
    query: String,
    limit: usize,
    /// `(rowid, bm25 bits as little-endian hex)`, ordered by `(score, rowid)`.
    rows: Vec<(i64, String)>,
}

/// `SELECT * FROM chunks` in column order (`sag2.py:236-252`).
///
/// The materialised schema stores `content`/`full_doc`; the `source_dir`
/// schema stores `start_char`/`end_char`.  Exactly one pair is present.
#[derive(Deserialize)]
struct ChunkRowGolden {
    id: i64,
    document_id: i64,
    title: String,
    heading: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    full_doc: Option<String>,
    #[serde(default)]
    start_char: Option<i64>,
    #[serde(default)]
    end_char: Option<i64>,
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
    let probes: usize = goldens.iter().map(|golden| golden.probes.len()).sum();
    if probes != spec.probes {
        return Err(format!(
            "golden inconsistency: index_expected.jsonl holds {probes} D4 probes, \
             manifest says {}",
            spec.probes
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
    let mut subchecks = SubChecks::default();
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
        // `source_dir` fixtures are built against the checked-in source tree
        // (`golden/sources/<fixture>`), so the contentless branch, the
        // `start_char`/`end_char` layout and `source_files` are gate-covered
        // (design §1.4 sample corpora, F-005).
        let source_dir = match golden.source_dir.as_deref() {
            None => None,
            Some(relative) => {
                let path = options.golden.join(relative);
                let resolved = fs::canonicalize(&path).map_err(|error| {
                    format!("cannot resolve source tree {}: {error}", path.display())
                })?;
                Some(resolved)
            }
        };
        let path = directory.join(&golden.index);
        let opened = sag2_index::CorpusIndex::open(
            &path,
            &documents,
            &golden.fingerprint,
            sag2_index::IndexOptions {
                chunk_chars: golden.chunk_chars,
                index_workers: 1,
                source_dir: source_dir.clone(),
                rebuild: sag2_index::RebuildPolicy::Always,
            },
        );
        let outcome = match opened {
            Ok((index, outcome)) => {
                drop(index);
                outcome
            }
            // The P0 reader refuses the contentless `chunks` layout
            // (`OpenError::SourceDirUnsupported`) *after* the build wrote the
            // file; Gate D compares that file, so the refusal is expected.
            Err(sag2_index::OpenError::SourceDirUnsupported) if source_dir.is_some() => {
                sag2_index::OpenOutcome {
                    action: sag2_index::OpenAction::Built,
                    rebuilt: None,
                }
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
        sag2_index::register_rank(&db)
            .map_err(|error| format!("cannot register the rank extension: {error}"))?;
        compare_index(
            &db,
            golden,
            source_dir.as_deref(),
            &expected_metadata,
            &golden.rust_index_metadata_keys,
            &mut subchecks,
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
    // The tallies must show work (N-003): an empty count means the comparison
    // silently covered nothing, so a green report could advertise checks that
    // never ran.
    for (name, count) in [
        ("D1 sqlite_master rows", subchecks.d1_entries),
        ("D3 chunk_fts rows", subchecks.d3_rows),
        ("D3 source_files rows", subchecks.d3_source_files),
        ("D4 probes", subchecks.d4_probes),
        ("D4 ranked rows", subchecks.d4_rows),
    ] {
        if count == 0 {
            gate.push("subchecks", name.to_string(), ">= 1", count.to_string());
        }
    }
    println!(
        "sub-checks: D1 {} fixtures / {} sqlite_master rows, D3 {} fixtures / {} chunk_fts rows \
         / {} source_files rows, D4 {} fixtures / {} probes / {} ranked rows",
        subchecks.d1_fixtures,
        subchecks.d1_entries,
        subchecks.d3_fixtures,
        subchecks.d3_rows,
        subchecks.d3_source_files,
        subchecks.d4_fixtures,
        subchecks.d4_probes,
        subchecks.d4_rows
    );
    Ok(serde_json::json!({
        "fixtures": fixtures_total,
        "fixtures_equal": fixtures_equal,
        "subchecks": {
            "D1_sqlite_master": {
                "fixtures": subchecks.d1_fixtures,
                "rows": subchecks.d1_entries,
            },
            "D3_chunk_fts_rows": {
                "fixtures": subchecks.d3_fixtures,
                "rows": subchecks.d3_rows,
            },
            "D3_source_files_rows": {
                "fixtures": subchecks.d3_fixtures,
                "rows": subchecks.d3_source_files,
            },
            "D4_probes": {
                "fixtures": subchecks.d4_fixtures,
                "probes": subchecks.d4_probes,
                "ranked_rows": subchecks.d4_rows,
            },
        },
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

/// Counts of the D1/D3/D4 sub-checks, so the gate report shows they ran.
#[derive(Default)]
struct SubChecks {
    d1_fixtures: usize,
    d1_entries: usize,
    d3_fixtures: usize,
    d3_rows: usize,
    d3_source_files: usize,
    d4_fixtures: usize,
    d4_probes: usize,
    d4_rows: usize,
}

/// A `sqlite_master` row reduced to what design §1.4 D1 compares.
type NormalisedSchema = Vec<(String, String, Option<String>)>;

/// Design §1.4 D1: `sqlite_master.sql` is compared after whitespace
/// normalisation (runs of whitespace collapse to one space, ends trimmed) so
/// that a reformatted `CREATE` statement is not a schema difference.
fn normalise_sql(sql: &str) -> String {
    sql.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalised_schema(entries: &[SchemaGolden]) -> NormalisedSchema {
    entries
        .iter()
        .map(|entry| {
            (
                entry.kind.clone(),
                entry.name.clone(),
                entry.sql.as_deref().map(normalise_sql),
            )
        })
        .collect()
}

fn describe_schema_entry(entry: Option<&(String, String, Option<String>)>) -> String {
    match entry {
        None => "<missing>".to_owned(),
        Some((kind, name, sql)) => format!("{kind} {name}: {}", sql.as_deref().unwrap_or("<null>")),
    }
}

/// Compare a Rust-built index with the Python build (design §1.4 D1-D4).
///
/// `source_dir` is the resolved source tree for a contentless fixture, `None`
/// for a materialised one.  A golden fixture's signatures carry the
/// machine-independent *relative* source path; the expected forms are rebuilt
/// here from the path this run actually resolved (F-007).
fn compare_index(
    db: &Connection,
    golden: &IndexGolden,
    source_dir: Option<&Path>,
    expected_metadata: &BTreeMap<&'static str, &'static str>,
    golden_keys: &[String],
    subchecks: &mut SubChecks,
    gate: &mut GateFailures<'_>,
) -> Result<(), GateError> {
    let fixture = golden.fixture.as_str();
    let signed = |stored: &str| match source_dir {
        None => stored.to_owned(),
        Some(path) => sag2_index::signature(&golden.fingerprint, golden.chunk_chars, Some(path)),
    };
    subchecks.d1_fixtures += 1;
    subchecks.d3_fixtures += 1;
    subchecks.d4_fixtures += 1;

    // D1 — every `sqlite_master` row (name, type, normalised sql) except
    // SQLite's internals and `rust_index_metadata`, which design §3.2 adds on
    // the Rust side only (its key set is compared below).
    {
        let mut statement = db.prepare(
            "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' \
             AND name <> 'rust_index_metadata' ORDER BY name",
        )?;
        let actual: NormalisedSchema = normalised_schema(
            &statement
                .query_map([], |row| {
                    Ok(SchemaGolden {
                        kind: row.get(0)?,
                        name: row.get(1)?,
                        sql: row.get(2)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?,
        );
        let expected = normalised_schema(&golden.schema);
        subchecks.d1_entries += actual.len();
        if actual != expected {
            let at = expected
                .iter()
                .zip(&actual)
                .position(|(left, right)| left != right)
                .unwrap_or(expected.len().min(actual.len()));
            let name = expected
                .get(at)
                .or_else(|| actual.get(at))
                .map(|entry| entry.1.clone())
                .unwrap_or_else(|| "<missing>".to_owned());
            gate.push(
                fixture,
                format!("sqlite_master/{name}"),
                describe_schema_entry(expected.get(at)),
                describe_schema_entry(actual.get(at)),
            );
        }
    }

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
    let expected_signature = signed(&golden.signature);
    if signature != expected_signature {
        gate.push(fixture, "metadata/signature", expected_signature, signature);
    }

    // D3 (first half) — every `chunks` row, in the schema variant the fixture
    // declares (`sag2.py:236-252`).
    let contentless = source_dir.is_some();
    let mut statement = db.prepare(if contentless {
        "SELECT id,document_id,title,heading,start_char,end_char,spans,characters,source_id \
         FROM chunks ORDER BY id"
    } else {
        "SELECT id,document_id,title,heading,content,full_doc,spans,characters,source_id \
         FROM chunks ORDER BY id"
    })?;
    let mut rows = statement.query([])?;
    let mut position = 0_usize;
    while let Some(row) = rows.next()? {
        let actual = ChunkRowGolden {
            id: row.get(0)?,
            document_id: row.get(1)?,
            title: row.get(2)?,
            heading: row.get(3)?,
            content: (!contentless).then(|| row.get(4)).transpose()?,
            full_doc: (!contentless).then(|| row.get(5)).transpose()?,
            start_char: contentless.then(|| row.get(4)).transpose()?,
            end_char: contentless.then(|| row.get(5)).transpose()?,
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

    // D3 (second half) — every `chunk_fts` row.  Under `content=''` FTS5
    // stores no column values, so the design drops `body`; the rowid↔term
    // binding is then covered by D4, which must bind rows for that fixture.
    {
        let mut statement = db.prepare(if contentless {
            "SELECT rowid,title,heading FROM chunk_fts ORDER BY rowid"
        } else {
            "SELECT rowid,title,heading,body FROM chunk_fts ORDER BY rowid"
        })?;
        let mut rows = statement.query([])?;
        let mut position = 0_usize;
        let width = if contentless { 3 } else { 4 };
        while let Some(row) = rows.next()? {
            let mut actual = vec![serde_json::Value::from(row.get::<_, i64>(0)?)];
            for column in 1..width {
                actual.push(match row.get::<_, Option<String>>(column)? {
                    Some(text) => serde_json::Value::String(text),
                    None => serde_json::Value::Null,
                });
            }
            subchecks.d3_rows += 1;
            match golden.fts_rows.get(position) {
                Some(expected) if *expected == actual => {}
                Some(expected) => gate.push(
                    fixture,
                    format!("chunk_fts[{position}]"),
                    format!("{expected:?}"),
                    format!("{actual:?}"),
                ),
                None => gate.push(
                    fixture,
                    format!("chunk_fts[{position}]"),
                    "<missing>",
                    format!("{actual:?}"),
                ),
            }
            position += 1;
        }
        if position != golden.fts_rows.len() {
            gate.push(
                fixture,
                "chunk_fts/length",
                golden.fts_rows.len().to_string(),
                position.to_string(),
            );
        }
    }

    // D3 (third half) — the `source_files` provenance table of a `source_dir`
    // build (`sag2.py:246`, `:284-290`), which is what the contentless
    // statistics corpus is read back from.
    if contentless {
        let mut statement =
            db.prepare("SELECT document_id, path, sha256 FROM source_files ORDER BY document_id")?;
        let actual: Vec<SourceFileGolden> = statement
            .query_map([], |row| {
                Ok(SourceFileGolden {
                    document_id: row.get(0)?,
                    path: row.get(1)?,
                    sha256: row.get(2)?,
                })
            })?
            .collect::<Result<_, _>>()?;
        subchecks.d3_source_files += actual.len();
        if actual != golden.source_files {
            let at = golden
                .source_files
                .iter()
                .zip(&actual)
                .position(|(left, right)| left != right)
                .unwrap_or(golden.source_files.len().min(actual.len()));
            gate.push(
                fixture,
                format!(
                    "source_files[{at}]/{}",
                    golden
                        .source_files
                        .get(at)
                        .map_or("<missing>", |entry| entry.path.as_str())
                ),
                format!(
                    "{} rows, first difference at {at}: {:?}",
                    golden.source_files.len(),
                    golden.source_files.get(at)
                ),
                format!("{} rows: {:?}", actual.len(), actual.get(at)),
            );
        }
    }

    // D4 — replay the ranking SQL for every probe and compare
    // `(rowid, bm25 bits)` row for row (`sag2.py:484-492`).
    for (probe_index, probe) in golden.probes.iter().enumerate() {
        let blob: Option<Vec<u8>> = db
            .query_row(sag2_index::SAG_STATS_SQL, [&probe.expression], |row| {
                row.get(0)
            })
            .optional()?
            .flatten();
        let mut statement = db.prepare(sag2_index::BASE_SQL)?;
        let actual: Vec<(i64, String)> = statement
            .query_map(
                rusqlite::params![
                    blob.unwrap_or_default(),
                    probe.query,
                    "[]",
                    probe.limit as i64
                ],
                |row| Ok((row.get::<_, i64>(0)?, hex_bits(row.get::<_, f64>(1)?))),
            )?
            .collect::<Result<_, _>>()?;
        subchecks.d4_probes += 1;
        subchecks.d4_rows += actual.len();
        if actual.is_empty() {
            gate.push(
                fixture,
                format!("probe[{probe_index}]/{}", probe.query),
                "at least one ranked row",
                "none (vacuous probe)",
            );
        }
        if actual != probe.rows {
            let at = probe
                .rows
                .iter()
                .zip(&actual)
                .position(|(left, right)| left != right)
                .unwrap_or(probe.rows.len().min(actual.len()));
            gate.push(
                fixture,
                format!("probe[{probe_index}]/{}", probe.query),
                format!(
                    "{} rows, first difference at {at}: {:?}",
                    probe.rows.len(),
                    probe.rows.get(at)
                ),
                format!("{} rows: {:?}", actual.len(), actual.get(at)),
            );
        }
    }

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

    // D2 — `index_statistics` holds exactly one row (F-008).
    let statistics_rows: i64 =
        db.query_row("SELECT COUNT(*) FROM index_statistics", [], |row| {
            row.get(0)
        })?;
    if statistics_rows != 1 {
        gate.push(
            fixture,
            "index_statistics/rows",
            "1".to_owned(),
            statistics_rows.to_string(),
        );
    }
    let expected_statistics_signature = signed(&golden.statistics_signature);
    let (statistics_signature, statistics): (String, String) =
        db.query_row("SELECT signature, data FROM index_statistics", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
    if statistics_signature != expected_statistics_signature {
        gate.push(
            fixture,
            "index_statistics/signature",
            expected_statistics_signature,
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
    for (name, left, right) in [
        ("content", &expected.content, &actual.content),
        ("full_doc", &expected.full_doc, &actual.full_doc),
    ] {
        if left != right {
            gate.push(
                fixture,
                field(name),
                display_option(left),
                display_option(right),
            );
        }
    }
    for (name, left, right) in [
        ("start_char", expected.start_char, actual.start_char),
        ("end_char", expected.end_char, actual.end_char),
        (
            "characters",
            Some(expected.characters),
            Some(actual.characters),
        ),
    ] {
        if left != right {
            gate.push(
                fixture,
                field(name),
                display_count(left),
                display_count(right),
            );
        }
    }
}

fn display_option(value: &Option<String>) -> String {
    match value {
        Some(text) => display_text(text),
        None => "<null>".to_owned(),
    }
}

fn display_count(value: Option<i64>) -> String {
    match value {
        Some(number) => number.to_string(),
        None => "<null>".to_owned(),
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

/// Compare `diversity_idf` at `eq_f64_bits` (design §1.4 D2: the values feed
/// the DPP vector weights, so the strict layer applies; the report's
/// `"f64": "bits"` claim is exactly what runs here).
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
            (Some(left), Some(right)) if left.to_bits() == right.to_bits() => {}
            (Some(left), Some(right)) => gate.push(
                fixture,
                path,
                format!("{left:?} (bits {:016x})", left.to_bits()),
                format!("{right:?} (bits {:016x})", right.to_bits()),
            ),
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

/// Gate B `sortkeys`: pyuca 1.2's `sort_key` over the vocabulary, every code
/// point and 200,000 comparison pairs (design §5.5).
///
/// This is the one gate that must fail even when no existing fixture changes:
/// the term set is open, so a code point that appears tomorrow may collate
/// differently.  Keys are compared element-wise (`eq_json` on integer arrays),
/// then the sorted order is compared as a sequence, then the comparison signs
/// are checked independently.
fn run_gate_b(options: &Options) -> Result<ExitCode, String> {
    let fixtures = options
        .fixtures
        .clone()
        .unwrap_or_else(|| vec!["sortkeys".to_owned()]);
    for fixture in &fixtures {
        if fixture != "sortkeys" && fixture != "expansion" {
            return Err(format!(
                "gate B fixture {fixture} is not implemented; sortkeys and expansion have goldens"
            ));
        }
    }
    let manifest = preflight(options)?;

    let mut failures: Vec<Failure> = Vec::new();
    let mut ulp_flips: Vec<serde_json::Value> = Vec::new();
    let mut sections = serde_json::Map::new();
    let mut passed = true;
    for fixture in &fixtures {
        let section = match fixture.as_str() {
            "sortkeys" => {
                verify_manifest_files(
                    options,
                    &manifest,
                    &[
                        "sk_vocab.jsonl",
                        "sk_cp.bin.gz",
                        "sk_pairs.bin.gz",
                        // The port's NFD must exist for `sort_key` to be
                        // pyuca's; the digest check makes the table part of
                        // this gate's inputs.
                        "unicode_nfd.bin.gz",
                    ],
                )?;
                let spec = manifest
                    .sortkeys
                    .as_ref()
                    .ok_or_else(|| "manifest has no sortkeys section".to_owned())?;
                gate_b_sortkeys(options, spec, &mut failures)?
            }
            "expansion" => gate_b_expansion(options, &manifest, &mut failures, &mut ulp_flips)?,
            _ => unreachable!("fixture names were validated above"),
        };
        passed &= section["passed"].as_bool().unwrap_or(false);
        sections.insert(fixture.clone(), section);
    }
    passed &= failures.is_empty() && ulp_flips.is_empty();

    println!(
        "gate B ({}): {} failures",
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
        "gates": {"B": {
            "fixtures": fixtures,
            "sections": sections,
        }},
        "excluded_fields": [],
        "excluded_checks": ["db_physical_bytes"],
        "failures": failures,
        "ulp_flips": ulp_flips,
        "unsupported": [],
    });
    write_report(&options.report, &report);
    Ok(if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}

/// One `sk_vocab.jsonl` line.
#[derive(Deserialize)]
struct SortkeyLine {
    term: String,
    key: Vec<u32>,
    /// Position in Python's stable `sorted(terms, key=collator.sort_key)`.
    rank: usize,
}

fn gate_b_sortkeys(
    options: &Options,
    spec: &SortkeysSpec,
    failures: &mut Vec<Failure>,
) -> Result<serde_json::Value, String> {
    let failures_before = failures.len();

    // sk_vocab: element-wise keys, then the sorted sequence.
    let vocab_path = options.golden.join("sk_vocab.jsonl");
    let vocab_text = fs::read_to_string(&vocab_path)
        .map_err(|error| format!("cannot read {}: {error}", vocab_path.display()))?;
    let mut lines: Vec<SortkeyLine> = Vec::new();
    for (number, line) in vocab_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: SortkeyLine = serde_json::from_str(line)
            .map_err(|error| format!("{}:{}: {error}", vocab_path.display(), number + 1))?;
        if parsed.rank >= spec.vocab.terms {
            return Err(format!(
                "{}:{}: rank {} is outside the manifest's {} terms",
                vocab_path.display(),
                number + 1,
                parsed.rank,
                spec.vocab.terms
            ));
        }
        lines.push(parsed);
    }
    if lines.len() != spec.vocab.terms {
        return Err(format!(
            "golden inconsistency: sk_vocab.jsonl holds {} terms, manifest says {}",
            lines.len(),
            spec.vocab.terms
        ));
    }
    let mut vocab_key_diffs = 0_usize;
    let mut rust_keys: Vec<Vec<u32>> = Vec::with_capacity(lines.len());
    for line in &lines {
        let actual = sag2_core::sort_key(&line.term);
        rust_keys.push(actual.clone());
        if actual != line.key {
            vocab_key_diffs += 1;
            if failures.len() < MAX_FAILURES {
                failures.push(Failure {
                    case_id: format!("sortkey/vocab/{}", line.term),
                    path: "key".to_owned(),
                    expected: format!("{:?}", line.key),
                    actual: format!("{actual:?}"),
                    delta: None,
                });
            }
        }
    }
    // Rust's `Vec<u32>` ordering is Python's tuple ordering: lexicographic,
    // and a shorter prefix sorts first.
    let mut order: Vec<usize> = (0..lines.len()).collect();
    order.sort_by(|left, right| rust_keys[*left].cmp(&rust_keys[*right]));
    let mut expected_by_rank: Vec<&str> = vec![""; lines.len()];
    for line in &lines {
        expected_by_rank[line.rank] = line.term.as_str();
    }
    let mut vocab_order_diffs = 0_usize;
    for (position, index) in order.iter().enumerate() {
        if lines[*index].rank != position {
            vocab_order_diffs += 1;
            if failures.len() < MAX_FAILURES {
                failures.push(Failure {
                    case_id: "sortkey/vocab/order".to_owned(),
                    path: format!("position {position}"),
                    expected: expected_by_rank[position].to_owned(),
                    actual: lines[*index].term.clone(),
                    delta: None,
                });
            }
        }
    }

    // sk_cp: every code point.  Lone surrogates are not scalar values, so the
    // golden records them but `sort_key` cannot be handed one (the same
    // boundary Gate D draws for its exhaustive tables).
    let cp_path = options.golden.join("sk_cp.bin.gz");
    let compressed = fs::read(&cp_path)
        .map_err(|error| format!("cannot read {}: {error}", cp_path.display()))?;
    let data = gunzip(&cp_path, &compressed)?;
    if data.len() < 12 || &data[..8] != spec.codepoints.magic.as_bytes() {
        return Err(format!(
            "{} does not start with the manifest magic",
            cp_path.display()
        ));
    }
    let count =
        u32::from_le_bytes(data[8..12].try_into().expect("sortkey codepoint count")) as usize;
    if count != spec.codepoints.codepoints {
        return Err(format!(
            "golden inconsistency: {} holds {count} code points, manifest says {}",
            cp_path.display(),
            spec.codepoints.codepoints
        ));
    }
    let mut codepoints_checked = 0_usize;
    let mut surrogates_skipped = 0_usize;
    let mut codepoint_diffs = 0_usize;
    let mut offset = 12_usize;
    for cp in 0..count {
        if offset >= data.len() {
            return Err(format!("{} ends before code point {cp}", cp_path.display()));
        }
        let length = data[offset] as usize;
        offset += 1;
        if length > spec.codepoints.max_weights || offset + 2 * length > data.len() {
            return Err(format!(
                "{}: U+{cp:04X} declares {length} weights, manifest allows {}",
                cp_path.display(),
                spec.codepoints.max_weights
            ));
        }
        let expected: Vec<u32> = (0..length)
            .map(|index| {
                let at = offset + 2 * index;
                u32::from(u16::from_le_bytes(
                    data[at..at + 2].try_into().expect("sortkey weight"),
                ))
            })
            .collect();
        offset += 2 * length;
        if (0xD800..=0xDFFF).contains(&cp) {
            surrogates_skipped += 1;
            continue;
        }
        codepoints_checked += 1;
        let character = char::from_u32(cp as u32).expect("scalar value");
        let actual = sag2_core::sort_key(&character.to_string());
        if actual != expected {
            codepoint_diffs += 1;
            if failures.len() < MAX_FAILURES {
                failures.push(Failure {
                    case_id: "sortkey/codepoints".to_owned(),
                    path: format!("U+{cp:04X}"),
                    expected: format!("{expected:?}"),
                    actual: format!("{actual:?}"),
                    delta: None,
                });
            }
        }
    }
    if offset != data.len() {
        return Err(format!(
            "{} holds {} trailing bytes after {count} records",
            cp_path.display(),
            data.len() - offset
        ));
    }

    // sk_pairs: the sign of the comparison, independent of the key arrays.
    let pairs_path = options.golden.join("sk_pairs.bin.gz");
    let compressed = fs::read(&pairs_path)
        .map_err(|error| format!("cannot read {}: {error}", pairs_path.display()))?;
    let data = gunzip(&pairs_path, &compressed)?;
    if data.len() < 12 || &data[..8] != spec.pairs.magic.as_bytes() {
        return Err(format!(
            "{} does not start with the manifest magic",
            pairs_path.display()
        ));
    }
    let count = u32::from_le_bytes(data[8..12].try_into().expect("sortkey pair count")) as usize;
    if count != spec.pairs.pairs {
        return Err(format!(
            "golden inconsistency: {} holds {count} pairs, manifest says {}",
            pairs_path.display(),
            spec.pairs.pairs
        ));
    }
    if data.len() != 12 + count * 10 {
        return Err(format!(
            "{} holds {} bytes, {count} 10-byte records need {}",
            pairs_path.display(),
            data.len(),
            12 + count * 10
        ));
    }
    let mut pairs_checked = 0_usize;
    let mut pair_diffs = 0_usize;
    for index in 0..count {
        let base = 12 + index * 10;
        let kind = data[base];
        let left = u32::from_le_bytes(data[base + 1..base + 5].try_into().expect("pair a"));
        let right = u32::from_le_bytes(data[base + 5..base + 9].try_into().expect("pair b"));
        let expected = data[base + 9] as i8;
        let (left_term, right_term) = match kind {
            0 => (
                char::from_u32(left)
                    .ok_or_else(|| format!("sortkey pair {index}: U+{left:04X} is not scalar"))?
                    .to_string(),
                char::from_u32(right)
                    .ok_or_else(|| format!("sortkey pair {index}: U+{right:04X} is not scalar"))?
                    .to_string(),
            ),
            1 => (
                lines
                    .get(left as usize)
                    .ok_or_else(|| {
                        format!("sortkey pair {index}: index {left} is outside sk_vocab")
                    })?
                    .term
                    .clone(),
                lines
                    .get(right as usize)
                    .ok_or_else(|| {
                        format!("sortkey pair {index}: index {right} is outside sk_vocab")
                    })?
                    .term
                    .clone(),
            ),
            other => {
                return Err(format!(
                    "{}: pair {index} has unknown kind {other}",
                    pairs_path.display()
                ))
            }
        };
        pairs_checked += 1;
        let actual = match sag2_core::sort_key(&left_term).cmp(&sag2_core::sort_key(&right_term)) {
            std::cmp::Ordering::Less => -1_i8,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        };
        if actual != expected {
            pair_diffs += 1;
            if failures.len() < MAX_FAILURES {
                failures.push(Failure {
                    case_id: "sortkey/pairs".to_owned(),
                    path: format!("pair {index}"),
                    expected: format!("{expected} ({left_term:?} vs {right_term:?})"),
                    actual: actual.to_string(),
                    delta: None,
                });
            }
        }
    }

    println!(
        "sortkeys: {} terms ({} key diffs, {} order diffs), \
         {codepoints_checked} code points ({codepoint_diffs} diffs, {surrogates_skipped} surrogates), \
         {pairs_checked} pairs ({pair_diffs} diffs)",
        lines.len(),
        vocab_key_diffs,
        vocab_order_diffs
    );
    let passed = failures_before == failures.len()
        && vocab_key_diffs == 0
        && vocab_order_diffs == 0
        && codepoint_diffs == 0
        && pair_diffs == 0;
    Ok(serde_json::json!({
        "vocab_terms": lines.len(),
        "vocab_key_diffs": vocab_key_diffs,
        "vocab_order_diffs": vocab_order_diffs,
        "codepoints_checked": codepoints_checked,
        "surrogates_skipped": surrogates_skipped,
        "codepoint_diffs": codepoint_diffs,
        "pairs_checked": pairs_checked,
        "pair_diffs": pair_diffs,
        "passed": passed,
    }))
}

/// Gate B `expansion`: replay `expansion_terms` + `retrieve_expansion` from
/// the recorded `(query, hop)` inputs (design §1.2).
///
/// Every ordered field is compared: `terms`, `term_scores` (with `f64`s
/// compared by bits), `queries[].ranking_expression` byte for byte,
/// `queries[].term_rank`/`retrieved_count`/`selected[]`, and the interleaved
/// `hits` sequence.  `repair` records add `repair_names` and `retrieve`
/// records add direct `retrieve_expansion` calls, which is how design §6.2's
/// alias matrix and the `title_phrases` grounding branch stay covered.
fn gate_b_expansion(
    options: &Options,
    manifest: &Manifest,
    failures: &mut Vec<Failure>,
    ulp_flips: &mut Vec<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    verify_manifest_files(options, manifest, &["expansion.jsonl"])?;
    let spec = manifest
        .expansion
        .as_ref()
        .ok_or_else(|| "manifest has no expansion section".to_owned())?;
    let path = options.golden.join("expansion.jsonl");
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut lines: Vec<ExpansionLine> = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        lines.push(
            serde_json::from_str(line)
                .map_err(|error| format!("{}:{}: {error}", path.display(), number + 1))?,
        );
    }
    if lines.len() != spec.cases {
        return Err(format!(
            "golden inconsistency: expansion.jsonl holds {} records, manifest says {}",
            lines.len(),
            spec.cases
        ));
    }

    // Design §7.3: the reference indexes are the default; `--index-dir`
    // replays against indexes this crate built, and a missing fixture there is
    // a tooling failure rather than a silent fallback.
    let index_directory = options
        .index_dir
        .clone()
        .unwrap_or_else(|| options.golden.join("indexes"));
    let mut indexes: BTreeMap<String, CorpusIndex> = BTreeMap::new();
    let mut provenance: Vec<serde_json::Value> = Vec::new();
    for fixture in &spec.fixtures {
        let counted = lines
            .iter()
            .filter(|line| line.fixture == fixture.fixture)
            .count();
        if counted != fixture.records {
            return Err(format!(
                "golden inconsistency: {} has {counted} records, manifest says {}",
                fixture.fixture, fixture.records
            ));
        }
        let source = index_directory.join(&fixture.index);
        let bytes = fs::read(&source).map_err(|error| {
            if options.index_dir.is_some() {
                format!(
                    "fixture {} has no index at {} (--index-dir): {error}",
                    fixture.fixture,
                    source.display()
                )
            } else {
                format!("cannot read {}: {error}", source.display())
            }
        })?;
        let digest = sha256_hex(&bytes);
        let manifest_verified = fixture.index_sha256.as_deref() == Some(digest.as_str());
        if options.index_dir.is_none() && !manifest_verified {
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
        let index = open_readonly(&cached, &fixture.signature, QueryOptions::default())
            .map_err(|error| {
                format!("cannot open expansion fixture {}: {error}", fixture.fixture)
            })?;
        if index.statistics().total as usize != fixture.chunks {
            return Err(format!(
                "fixture {} holds {} chunks, manifest says {}",
                fixture.fixture,
                index.statistics().total,
                fixture.chunks
            ));
        }
        provenance.push(serde_json::json!({
            "fixture": fixture.fixture,
            "path": source.display().to_string(),
            "sha256": digest,
            "manifest_sha256": fixture.index_sha256,
            "manifest_verified": manifest_verified,
        }));
        indexes.insert(fixture.fixture.clone(), index);
    }

    let mut per_fixture: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut stages: BTreeMap<String, usize> = BTreeMap::new();
    for line in &lines {
        let index = indexes
            .get(&line.fixture)
            .ok_or_else(|| format!("fixture {} was not opened", line.fixture))?;
        let before = failures.len();
        match line.stage.as_str() {
            "expansion" => check_expansion_record(index, line, failures, ulp_flips)?,
            "retrieve" => check_retrieve_record(index, line, failures, ulp_flips)?,
            "repair" => check_repair_record(index, line, failures)?,
            other => {
                return Err(format!("{}: unknown stage {other}", line.case_id));
            }
        }
        let entry = per_fixture.entry(line.fixture.clone()).or_insert((0, 0));
        entry.0 += 1;
        if failures.len() == before {
            entry.1 += 1;
        }
        *stages.entry(line.stage.clone()).or_insert(0) += 1;
    }

    let passed_cases = per_fixture.values().map(|(_, passed)| passed).sum::<usize>();
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
    println!(
        "gate B expansion: {} records ({}), {} passed, {} failed",
        lines.len(),
        stages
            .iter()
            .map(|(stage, count)| format!("{stage} {count}"))
            .collect::<Vec<_>>()
            .join(", "),
        passed_cases,
        lines.len() - passed_cases
    );

    Ok(serde_json::json!({
        "cases": lines.len(),
        "passed_cases": passed_cases,
        "failed": lines.len() - passed_cases,
        "stages": stages,
        "indexes": {
            "dir": index_directory.display().to_string(),
            "source": if options.index_dir.is_some() { "index_dir" } else { "golden" },
            "fixtures": provenance,
        },
        "fixtures": per_fixture.iter().map(|(fixture, (total, passed))| {
            serde_json::json!({"fixture": fixture, "cases": total, "passed": passed})
        }).collect::<Vec<_>>(),
        "passed": failures.is_empty() && ulp_flips.is_empty(),
    }))
}

/// Replay one `expansion` record: the seed sentences in, the §1.2 fields out.
fn check_expansion_record(
    index: &CorpusIndex,
    line: &ExpansionLine,
    failures: &mut Vec<Failure>,
    ulp_flips: &mut Vec<serde_json::Value>,
) -> Result<(), String> {
    let query = line
        .query
        .as_deref()
        .ok_or_else(|| format!("{}: expansion record without query", line.case_id))?;
    let term_limit = line
        .term_limit
        .ok_or_else(|| format!("{}: expansion record without term_limit", line.case_id))?;
    let expression = line
        .expression
        .as_deref()
        .ok_or_else(|| format!("{}: expansion record without expression", line.case_id))?;
    let limit = line
        .limit
        .ok_or_else(|| format!("{}: expansion record without limit", line.case_id))?;
    let seeds: Vec<Seed> = line
        .seeds
        .iter()
        .map(|seed| Seed {
            document_id: seed.document_id,
            jev_score: seed.jev_score,
            content: seed.content.clone(),
        })
        .collect();
    let terms = index
        .expansion_terms(query, &seeds, term_limit)
        .map_err(|error| format!("{}: expansion_terms: {error}", line.case_id))?;
    compare_terms(failures, &line.case_id, "terms", &line.terms, &terms);
    compare_term_scores(failures, ulp_flips, &line.case_id, &line.term_scores, &terms);
    let excluded: BTreeSet<i64> = line.excluded.iter().copied().collect();
    let (hits, queries) = index
        .retrieve_expansion(expression, &terms, &excluded, limit)
        .map_err(|error| format!("{}: retrieve_expansion: {error}", line.case_id))?;
    compare_queries(failures, ulp_flips, &line.case_id, &line.queries, &queries);
    compare_expansion_hits(failures, ulp_flips, &line.case_id, &line.hits, &hits);
    Ok(())
}

/// Replay one `retrieve` record: explicit input terms, the same comparisons.
fn check_retrieve_record(
    index: &CorpusIndex,
    line: &ExpansionLine,
    failures: &mut Vec<Failure>,
    ulp_flips: &mut Vec<serde_json::Value>,
) -> Result<(), String> {
    let expression = line
        .expression
        .as_deref()
        .ok_or_else(|| format!("{}: retrieve record without expression", line.case_id))?;
    let limit = line
        .limit
        .ok_or_else(|| format!("{}: retrieve record without limit", line.case_id))?;
    let mut terms: Vec<ExpansionTerm> = Vec::with_capacity(line.input_terms.len());
    for input in &line.input_terms {
        terms.push(ExpansionTerm {
            term: input.term.clone(),
            score: 0.0,
            unweighted_score: 0.0,
            seed_weight: 0.0,
            seed_jev_score: 0.0,
            term_frequency: 0,
            document_frequency: 0,
            query_context_score: 0.0,
            seed_document_id: 0,
            kind: expansion_kind(&line.case_id, &input.kind)?,
            name: input.name.clone(),
        });
    }
    let excluded: BTreeSet<i64> = line.excluded.iter().copied().collect();
    let (hits, queries) = index
        .retrieve_expansion(expression, &terms, &excluded, limit)
        .map_err(|error| format!("{}: retrieve_expansion: {error}", line.case_id))?;
    compare_queries(failures, ulp_flips, &line.case_id, &line.queries, &queries);
    compare_expansion_hits(failures, ulp_flips, &line.case_id, &line.hits, &hits);
    Ok(())
}

/// Replay one `repair` record: `repair_names(text, excluded)`.
fn check_repair_record(
    index: &CorpusIndex,
    line: &ExpansionLine,
    failures: &mut Vec<Failure>,
) -> Result<(), String> {
    let text = line
        .text
        .as_deref()
        .ok_or_else(|| format!("{}: repair record without text", line.case_id))?;
    let expected = line
        .repaired
        .as_deref()
        .ok_or_else(|| format!("{}: repair record without repaired", line.case_id))?;
    let excluded: BTreeSet<i64> = line.excluded.iter().copied().collect();
    let actual = index
        .repair_names(text, &excluded)
        .map_err(|error| format!("{}: repair_names: {error}", line.case_id))?;
    if actual != expected {
        push_gate_failure(
            failures,
            &line.case_id,
            "repaired",
            display_text(expected),
            display_text(&actual),
        );
    }
    Ok(())
}

fn expansion_kind(case_id: &str, kind: &str) -> Result<sag2_index::ExpansionKind, String> {
    match kind {
        "name_phrase" => Ok(sag2_index::ExpansionKind::NamePhrase),
        "word" => Ok(sag2_index::ExpansionKind::Word),
        other => Err(format!("{case_id}: unknown expansion kind {other}")),
    }
}

/// `terms`: the ordered term strings, byte for byte (§1.2).
fn compare_terms(
    failures: &mut Vec<Failure>,
    case_id: &str,
    prefix: &str,
    expected: &[String],
    actual: &[ExpansionTerm],
) {
    if expected.len() != actual.len() {
        push_gate_failure(
            failures,
            case_id,
            format!("{prefix}.len"),
            expected.len().to_string(),
            actual.len().to_string(),
        );
    }
    for (position, expected) in expected.iter().enumerate() {
        let Some(actual) = actual.get(position) else {
            break;
        };
        if *expected != actual.term {
            push_gate_failure(
                failures,
                case_id,
                format!("{prefix}[{position}]"),
                display_text(expected),
                display_text(&actual.term),
            );
        }
    }
}

/// `term_scores`: every ordered field, `f64` by bits (§1.0 layer S).
fn compare_term_scores(
    failures: &mut Vec<Failure>,
    ulp_flips: &mut Vec<serde_json::Value>,
    case_id: &str,
    expected: &[TermScoreGolden],
    actual: &[ExpansionTerm],
) {
    if expected.len() != actual.len() {
        push_gate_failure(
            failures,
            case_id,
            "term_scores.len",
            expected.len().to_string(),
            actual.len().to_string(),
        );
    }
    for (position, expected) in expected.iter().enumerate() {
        let Some(actual) = actual.get(position) else {
            break;
        };
        let field = |name: &str| format!("term_scores[{position}].{name}");
        if expected.term != actual.term {
            push_gate_failure(
                failures,
                case_id,
                field("term"),
                display_text(&expected.term),
                display_text(&actual.term),
            );
        }
        let kind = expansion_kind(case_id, &expected.kind);
        match kind {
            Ok(kind) if kind == actual.kind => {}
            Ok(_) => push_gate_failure(
                failures,
                case_id,
                field("kind"),
                expected.kind.clone(),
                actual.kind.as_str().to_owned(),
            ),
            Err(message) => push_gate_failure(
                failures,
                case_id,
                field("kind"),
                expected.kind.clone(),
                message,
            ),
        }
        if expected.name != actual.name {
            push_gate_failure(
                failures,
                case_id,
                field("name"),
                format!("{:?}", expected.name),
                format!("{:?}", actual.name),
            );
        }
        if expected.term_frequency != actual.term_frequency {
            push_gate_failure(
                failures,
                case_id,
                field("term_frequency"),
                expected.term_frequency.to_string(),
                actual.term_frequency.to_string(),
            );
        }
        if expected.document_frequency != actual.document_frequency {
            push_gate_failure(
                failures,
                case_id,
                field("document_frequency"),
                expected.document_frequency.to_string(),
                actual.document_frequency.to_string(),
            );
        }
        if expected.seed_document_id != actual.seed_document_id {
            push_gate_failure(
                failures,
                case_id,
                field("seed_document_id"),
                expected.seed_document_id.to_string(),
                actual.seed_document_id.to_string(),
            );
        }
        for (name, expected_bits, actual_value) in [
            ("score", &expected.score_bits, actual.score),
            (
                "unweighted_score",
                &expected.unweighted_score_bits,
                actual.unweighted_score,
            ),
            ("seed_weight", &expected.seed_weight_bits, actual.seed_weight),
            (
                "seed_jev_score",
                &expected.seed_jev_score_bits,
                actual.seed_jev_score,
            ),
            (
                "query_context_score",
                &expected.query_context_score_bits,
                actual.query_context_score,
            ),
        ] {
            compare_bits(
                failures,
                ulp_flips,
                case_id,
                field(name),
                expected_bits,
                actual_value,
            );
        }
    }
}

/// `queries[]`: `ranking_expression` byte for byte, then rank/count/selected.
fn compare_queries(
    failures: &mut Vec<Failure>,
    ulp_flips: &mut Vec<serde_json::Value>,
    case_id: &str,
    expected: &[QueryGolden],
    actual: &[ExpansionQuery],
) {
    if expected.len() != actual.len() {
        push_gate_failure(
            failures,
            case_id,
            "queries.len",
            expected.len().to_string(),
            actual.len().to_string(),
        );
    }
    for (position, expected) in expected.iter().enumerate() {
        let Some(actual) = actual.get(position) else {
            break;
        };
        let field = |name: &str| format!("queries[{position}].{name}");
        for (name, expected, actual) in [
            ("term", &expected.term, &actual.term),
            ("kind", &expected.kind, &actual.kind.as_str().to_owned()),
            (
                "match_expression",
                &expected.match_expression,
                &actual.match_expression,
            ),
            (
                "ranking_expression",
                &expected.ranking_expression,
                &actual.ranking_expression,
            ),
        ] {
            if expected != actual {
                push_gate_failure(
                    failures,
                    case_id,
                    field(name),
                    display_text(expected),
                    display_text(actual),
                );
            }
        }
        if expected.term_rank != actual.term_rank {
            push_gate_failure(
                failures,
                case_id,
                field("term_rank"),
                expected.term_rank.to_string(),
                actual.term_rank.to_string(),
            );
        }
        if expected.retrieved_count != actual.retrieved_count {
            push_gate_failure(
                failures,
                case_id,
                field("retrieved_count"),
                expected.retrieved_count.to_string(),
                actual.retrieved_count.to_string(),
            );
        }
        if expected.selected.len() != actual.selected.len() {
            push_gate_failure(
                failures,
                case_id,
                field("selected.len"),
                expected.selected.len().to_string(),
                actual.selected.len().to_string(),
            );
        }
        for (index, expected) in expected.selected.iter().enumerate() {
            let Some(actual) = actual.selected.get(index) else {
                break;
            };
            if expected.document_id != actual.document_id {
                push_gate_failure(
                    failures,
                    case_id,
                    field(&format!("selected[{index}].document_id")),
                    expected.document_id.to_string(),
                    actual.document_id.to_string(),
                );
            }
            if expected.bm25_rank != actual.bm25_rank {
                push_gate_failure(
                    failures,
                    case_id,
                    field(&format!("selected[{index}].bm25_rank")),
                    expected.bm25_rank.to_string(),
                    actual.bm25_rank.to_string(),
                );
            }
            compare_bits(
                failures,
                ulp_flips,
                case_id,
                field(&format!("selected[{index}].bm25_score")),
                &expected.bm25_score_bits,
                actual.bm25_score,
            );
        }
    }
}

/// `hits`: the interleaved sequence, every field in order (§1.2).
fn compare_expansion_hits(
    failures: &mut Vec<Failure>,
    ulp_flips: &mut Vec<serde_json::Value>,
    case_id: &str,
    expected: &[ExpansionHitGolden],
    actual: &[sag2_index::ExpansionHit],
) {
    if expected.len() != actual.len() {
        push_gate_failure(
            failures,
            case_id,
            "hits.len",
            expected.len().to_string(),
            actual.len().to_string(),
        );
    }
    for (position, expected) in expected.iter().enumerate() {
        let Some(actual) = actual.get(position) else {
            break;
        };
        let field = |name: &str| format!("hits[{position}].{name}");
        for (name, expected, actual) in [
            ("id", expected.id, actual.id),
            ("document_id", expected.document_id, actual.document_id),
        ] {
            if expected != actual {
                push_gate_failure(
                    failures,
                    case_id,
                    field(name),
                    expected.to_string(),
                    actual.to_string(),
                );
            }
        }
        if expected.expansion_term != actual.expansion_term {
            push_gate_failure(
                failures,
                case_id,
                field("expansion_term"),
                display_text(&expected.expansion_term),
                display_text(&actual.expansion_term),
            );
        }
        if expected.expansion_term_rank != actual.expansion_term_rank {
            push_gate_failure(
                failures,
                case_id,
                field("expansion_term_rank"),
                expected.expansion_term_rank.to_string(),
                actual.expansion_term_rank.to_string(),
            );
        }
        if expected.expansion_bm25_rank != actual.expansion_bm25_rank {
            push_gate_failure(
                failures,
                case_id,
                field("expansion_bm25_rank"),
                expected.expansion_bm25_rank.to_string(),
                actual.expansion_bm25_rank.to_string(),
            );
        }
        compare_bits(
            failures,
            ulp_flips,
            case_id,
            field("score"),
            &expected.score_bits,
            actual.score,
        );
    }
}

/// `eq_f64_bits` (§1.0 layer S); a difference inside `eq_f64` is an `ulp_flip`
/// and still fails the gate.
fn compare_bits(
    failures: &mut Vec<Failure>,
    ulp_flips: &mut Vec<serde_json::Value>,
    case_id: &str,
    path: String,
    expected_bits: &str,
    actual: f64,
) {
    let Some(expected) = bits_to_f64(expected_bits) else {
        push_gate_failure(
            failures,
            case_id,
            path,
            expected_bits.to_owned(),
            "unparseable golden bits".to_owned(),
        );
        return;
    };
    if actual.to_bits() == expected.to_bits() {
        return;
    }
    let delta = actual - expected;
    if failures.len() < MAX_FAILURES {
        failures.push(Failure {
            case_id: case_id.to_owned(),
            path: path.clone(),
            expected: format!("{expected_bits} ({expected:e})"),
            actual: format!("{:016x} ({actual:e})", actual.to_bits()),
            delta: Some(delta),
        });
    }
    if within_tolerance(expected, actual) {
        ulp_flips.push(serde_json::json!({
            "case_id": case_id,
            "path": path,
            "expected": expected_bits,
            "actual": format!("{:016x}", actual.to_bits()),
            "delta": delta,
        }));
    }
}

/// Gate D `freq`: rule 5's short-key table and its tokenizer probes
/// (design §4.1 rule 5, §4.3).
///
/// Three checks, because the table is a decision structure rather than data:
/// set equality over every stored key (no missing, no extra), an explicit
/// anchor-position tally (the boundary case that a one-sided window silently
/// drops), and `tokenize(key)` against the reference for the probe sample.
fn gate_d_freq(
    options: &Options,
    manifest: &Manifest,
    failures: &mut Vec<Failure>,
) -> Result<serde_json::Value, String> {
    verify_manifest_files(options, manifest, &["freq_keys.txt.gz", "freq_probes.jsonl"])?;
    let spec = manifest
        .freq
        .as_ref()
        .ok_or_else(|| "manifest has no freq section".to_owned())?;
    let failures_before = failures.len();
    if sag2_core::max_key_chars() != spec.max_chars {
        return Err(format!(
            "unsupported: this build indexes FREQ keys up to {} code points, golden says {}",
            sag2_core::max_key_chars(),
            spec.max_chars
        ));
    }
    if sag2_core::anchor_every() != spec.anchor_every {
        return Err(format!(
            "unsupported: this build anchors every {} entries, golden says {}",
            sag2_core::anchor_every(),
            spec.anchor_every
        ));
    }

    let keys_path = options.golden.join("freq_keys.txt.gz");
    let compressed = fs::read(&keys_path)
        .map_err(|error| format!("cannot read {}: {error}", keys_path.display()))?;
    let text = String::from_utf8(gunzip(&keys_path, &compressed)?)
        .map_err(|error| format!("{} is not UTF-8: {error}", keys_path.display()))?;
    let golden: Vec<&str> = text.lines().collect();
    if golden.len() != spec.keys {
        return Err(format!(
            "golden inconsistency: freq_keys.txt.gz holds {} keys, manifest says {}",
            golden.len(),
            spec.keys
        ));
    }

    // Set equality over the whole table.
    let entries = sag2_core::short_entries();
    let entries_set: BTreeSet<&str> = entries.iter().copied().collect();
    let golden_set: BTreeSet<&str> = golden.iter().copied().collect();
    let missing: Vec<&&str> = golden_set.difference(&entries_set).collect();
    let extra: Vec<&&str> = entries_set.difference(&golden_set).collect();
    if entries.len() != golden.len() {
        push_gate_failure(
            failures,
            "freq/keys",
            "count",
            golden.len().to_string(),
            entries.len().to_string(),
        );
    }
    for key in missing.iter().take(10) {
        push_gate_failure(
            failures,
            "freq/keys",
            "missing",
            format!("{key:?} is in the golden"),
            "absent from the compiled table".to_owned(),
        );
    }
    for key in extra.iter().take(10) {
        push_gate_failure(
            failures,
            "freq/keys",
            "extra",
            "absent from the golden".to_owned(),
            format!("{key:?} is in the compiled table"),
        );
    }
    if missing.is_empty() && extra.is_empty() {
        // Both sides are in UTF-8 byte order, so equal sets must be equal
        // sequences; the binary search depends on that order.
        for (position, (expected, actual)) in golden.iter().zip(&entries).enumerate() {
            if expected != actual {
                push_gate_failure(
                    failures,
                    "freq/keys",
                    format!("order[{position}]"),
                    format!("{expected:?}"),
                    format!("{actual:?}"),
                );
                break;
            }
        }
    }

    // Anchor coverage: every anchor-position key must be *findable* through
    // the public lookup (the entry is in the blob either way; the binary
    // search is what the anchor boundary breaks), and the tally must not
    // silently become zero on a future rebuild.
    let stride = spec.anchor_every;
    let mut anchors = 0_usize;
    let mut anchor_missing = 0_usize;
    for (position, key) in golden.iter().enumerate() {
        if position % stride != 0 {
            continue;
        }
        anchors += 1;
        if !sag2_core::freq_contains(key) {
            anchor_missing += 1;
            if anchor_missing <= 10 {
                push_gate_failure(
                    failures,
                    "freq/anchors",
                    format!("anchor[{position}]"),
                    format!("{key:?} is findable"),
                    "the lookup reports it absent".to_owned(),
                );
            }
        }
    }
    if anchors != spec.anchor_entries {
        return Err(format!(
            "golden inconsistency: {} anchor positions, manifest says {}",
            anchors, spec.anchor_entries
        ));
    }
    if anchors == 0 {
        push_gate_failure(
            failures,
            "freq/anchors",
            "count",
            "> 0 anchor entries",
            "0".to_owned(),
        );
    }

    // Probes: `tokenize(key)` must equal the reference's token list.
    let probes_path = options.golden.join("freq_probes.jsonl");
    let probes_text = fs::read_to_string(&probes_path)
        .map_err(|error| format!("cannot read {}: {error}", probes_path.display()))?;
    let mut probes: Vec<FreqProbe> = Vec::new();
    for (number, line) in probes_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        probes.push(
            serde_json::from_str(line)
                .map_err(|error| format!("{}:{}: {error}", probes_path.display(), number + 1))?,
        );
    }
    if probes.len() != spec.probes {
        return Err(format!(
            "golden inconsistency: freq_probes.jsonl holds {} probes, manifest says {}",
            probes.len(),
            spec.probes
        ));
    }
    let mut probe_diffs = 0_usize;
    for probe in &probes {
        let actual = sag2_core::tokenize(&probe.key);
        if actual == probe.tokens {
            continue;
        }
        probe_diffs += 1;
        if probe_diffs <= 10 {
            push_gate_failure(
                failures,
                "freq/probes",
                format!("tokenize({:?})", probe.key),
                format!("{:?}", probe.tokens),
                format!("{actual:?}"),
            );
        }
    }

    let passed = failures.len() == failures_before;
    println!(
        "freq: {} keys ({} missing, {} extra), {} anchors ({} missing), {} probes ({} diffs)",
        golden.len(),
        missing.len(),
        extra.len(),
        anchors,
        anchor_missing,
        probes.len(),
        probe_diffs
    );
    Ok(serde_json::json!({
        "keys": golden.len(),
        "missing": missing.len(),
        "extra": extra.len(),
        "anchors": anchors,
        "anchor_missing": anchor_missing,
        "probes": probes.len(),
        "corpus_probes": spec.corpus_probes,
        "probe_diffs": probe_diffs,
        "passed": passed,
    }))
}

fn run_gate_d(options: &Options) -> Result<ExitCode, String> {
    let fixtures = options
        .fixtures
        .clone()
        .unwrap_or_else(|| vec!["tokenizer".to_owned()]);
    for fixture in &fixtures {
        if !matches!(
            fixture.as_str(),
            "tokenizer" | "chunks" | "index" | "freq"
        ) {
            return Err(format!(
                "gate D fixture {fixture} is not implemented; use tokenizer, chunks, index or freq"
            ));
        }
    }
    let manifest = preflight(options)?;
    let mut failures: Vec<Failure> = Vec::new();
    let mut sections = serde_json::Map::new();
    let mut passed = true;

    // `freq` is rule 5 of the tokenizer, so `tokenizer` always runs it; naming
    // `freq` alone is how a table change is checked by itself.
    if fixtures
        .iter()
        .any(|fixture| fixture == "tokenizer" || fixture == "freq")
    {
        let section = gate_d_freq(options, &manifest, &mut failures)?;
        passed &= section["passed"].as_bool().unwrap_or(false);
        sections.insert("freq".to_owned(), section);
    }
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
            "schema_exclusions": [
                "rust_index_metadata (design §3.2 Rust-only provenance table; \
                 D2 compares its key set and values)"
            ],
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

/// `struct.pack('<d', value).hex()`: the golden's score representation.
fn hex_bits(value: f64) -> String {
    value
        .to_le_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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

// ---------------------------------------------------------------------------
// Gate C (search, design §1.3)
// ---------------------------------------------------------------------------

/// The manifest's Gate C section.
#[derive(Deserialize)]
struct SearchSpec {
    cases: usize,
    corpora: usize,
    payloads: usize,
    endpoint: String,
    golden: String,
    corpus_golden: String,
}

/// One `search_corpus.jsonl` line: the corpus and the index parameters a case
/// runs against.
#[derive(Deserialize)]
struct SearchCorpus {
    fixture: String,
    fingerprint: String,
    chunk_chars: usize,
    corpus: Vec<SearchDocument>,
}

#[derive(Deserialize)]
struct SearchDocument {
    title: String,
    text: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    idx: Option<String>,
}

/// The resolved `search_sag2` options a case ran with.
#[derive(Deserialize)]
struct SearchOptionsGolden {
    candidate_k: usize,
    seed_k: usize,
    max_hops: u8,
    expansion_term_limit: usize,
}

/// One `search.jsonl.gz` line.  C-1 compares `payloads`/`payloads_sha256`; C-2
/// compares the result fields, which are kept as untyped JSON because they are
/// compared field by field (`compare_json_exact`) rather than deserialised.
#[derive(Deserialize)]
struct SearchCase {
    case_id: String,
    fixture: String,
    index_workers: usize,
    query: String,
    options: SearchOptionsGolden,
    endpoint: String,
    payloads: Vec<String>,
    payloads_sha256: Vec<String>,
    ranked_ids: Vec<i64>,
    ranked_scores_bits: Vec<String>,
    baseline_ids: Vec<i64>,
    trace: serde_json::Value,
    jev_usage: serde_json::Value,
    algorithm: String,
}

fn read_search_cases(options: &Options, spec: &SearchSpec) -> Result<Vec<SearchCase>, String> {
    let path = options.golden.join(&spec.golden);
    let compressed = fs::read(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let data = gunzip(&path, &compressed)?;
    let text = String::from_utf8(data)
        .map_err(|error| format!("{} is not UTF-8: {error}", path.display()))?;
    let mut cases = Vec::new();
    for line in text.lines() {
        let case: SearchCase =
            serde_json::from_str(line).map_err(|error| format!("{}: {error}", path.display()))?;
        if case.payloads.len() != case.payloads_sha256.len() {
            return Err(format!(
                "{}: case {} has {} payloads but {} digests",
                path.display(),
                case.case_id,
                case.payloads.len(),
                case.payloads_sha256.len()
            ));
        }
        if case.ranked_ids.len() != case.ranked_scores_bits.len() {
            return Err(format!(
                "{}: case {} has {} ranked ids but {} scores",
                path.display(),
                case.case_id,
                case.ranked_ids.len(),
                case.ranked_scores_bits.len()
            ));
        }
        if case.algorithm != "SAG2" {
            return Err(format!(
                "{}: case {} says algorithm {:?}, not SAG2",
                path.display(),
                case.case_id,
                case.algorithm
            ));
        }
        if !case.trace.is_array() || !case.jev_usage.is_object() {
            return Err(format!(
                "{}: case {} has a malformed trace or jev_usage",
                path.display(),
                case.case_id
            ));
        }
        if case.endpoint != spec.endpoint {
            return Err(format!(
                "{}: case {} endpoints {:?} and {:?} disagree",
                path.display(),
                case.case_id,
                case.endpoint,
                spec.endpoint
            ));
        }
        cases.push(case);
    }
    Ok(cases)
}

fn read_search_corpora(options: &Options, spec: &SearchSpec) -> Result<BTreeMap<String, SearchCorpus>, String> {
    let path = options.golden.join(&spec.corpus_golden);
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut corpora = BTreeMap::new();
    for line in text.lines() {
        let corpus: SearchCorpus =
            serde_json::from_str(line).map_err(|error| format!("{}: {error}", path.display()))?;
        corpora.insert(corpus.fixture.clone(), corpus);
    }
    Ok(corpora)
}

/// `sag2.py:1036-1043` as the golden's C-2 shape.
fn search_result_value(result: &sag2::SearchResult) -> serde_json::Value {
    serde_json::json!({
        "ranked_ids": result.candidates.iter().map(|c| c.hit.id).collect::<Vec<_>>(),
        "ranked_scores_bits": result
            .candidates
            .iter()
            .map(|c| hex_bits(c.jev_score))
            .collect::<Vec<_>>(),
        "baseline_ids": result.baseline.iter().map(|c| c.hit.id).collect::<Vec<_>>(),
        "trace": result.trace.iter().map(trace_entry_value).collect::<Vec<_>>(),
        "jev_usage": search_usage_value(&result.usage),
        "algorithm": result.algorithm,
    })
}

fn trace_entry_value(entry: &sag2::TraceEntry) -> serde_json::Value {
    match entry {
        sag2::TraceEntry::Retrieval(trace) => serde_json::json!({
            "stage": trace.stage,
            "hop": trace.hop,
            "count": trace.count,
            "candidate_document_ids": trace.candidate_document_ids,
            "candidate_ids": trace.candidate_ids,
            "terms": trace.terms,
            "term_scores": trace.term_scores.iter().map(term_score_value).collect::<Vec<_>>(),
            "expansion_queries": trace.expansion_queries.iter().map(expansion_query_value).collect::<Vec<_>>(),
        }),
        sag2::TraceEntry::Judge(trace) => serde_json::json!({
            "stage": "jev",
            "hop": trace.hop,
            "count": trace.count,
            "carried_seed_count": trace.carried_seed_count,
            "protected_bm25_count": trace.protected_bm25_count,
            "evidence_ids": trace.evidence_ids,
            "scores": trace
                .scores
                .iter()
                .map(|score| serde_json::json!({"id": score.id, "score_bits": hex_bits(score.score)}))
                .collect::<Vec<_>>(),
            "batches": trace.batches.iter().map(batch_value).collect::<Vec<_>>(),
        }),
    }
}

fn term_score_value(term: &sag2::ExpansionTerm) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    object.insert("term".to_owned(), serde_json::json!(term.term));
    object.insert("score_bits".to_owned(), serde_json::json!(hex_bits(term.score)));
    object.insert(
        "unweighted_score_bits".to_owned(),
        serde_json::json!(hex_bits(term.unweighted_score)),
    );
    object.insert(
        "seed_weight_bits".to_owned(),
        serde_json::json!(hex_bits(term.seed_weight)),
    );
    object.insert(
        "seed_jev_score_bits".to_owned(),
        serde_json::json!(hex_bits(term.seed_jev_score)),
    );
    object.insert(
        "term_frequency".to_owned(),
        serde_json::json!(term.term_frequency),
    );
    object.insert(
        "document_frequency".to_owned(),
        serde_json::json!(term.document_frequency),
    );
    object.insert(
        "query_context_score_bits".to_owned(),
        serde_json::json!(hex_bits(term.query_context_score)),
    );
    object.insert(
        "seed_document_id".to_owned(),
        serde_json::json!(term.seed_document_id),
    );
    object.insert("kind".to_owned(), serde_json::json!(term.kind.as_str()));
    if let Some(name) = &term.name {
        object.insert("name".to_owned(), serde_json::json!(name));
    }
    serde_json::Value::Object(object)
}

fn expansion_query_value(query: &sag2::ExpansionQuery) -> serde_json::Value {
    serde_json::json!({
        "term": query.term,
        "term_rank": query.term_rank,
        "kind": query.kind.as_str(),
        "match_expression": query.match_expression,
        "ranking_expression": query.ranking_expression,
        "retrieved_count": query.retrieved_count,
        "selected": query
            .selected
            .iter()
            .map(|selected| serde_json::json!({
                "document_id": selected.document_id,
                "bm25_rank": selected.bm25_rank,
                "bm25_score_bits": hex_bits(selected.bm25_score),
            }))
            .collect::<Vec<_>>(),
    })
}

fn batch_value(batch: &sag2::BatchDetail) -> serde_json::Value {
    serde_json::json!({
        "candidate_count": batch.candidate_count,
        "estimated_input_tokens": batch.estimated_input_tokens,
        "request_attempts": batch.request_attempts,
        "http_attempts": batch.http_attempts,
        "failed_attempts": json_from_sag2(&sag2_json::Value::Object(batch.failed_attempts.clone())),
        "input_tokens": batch.input_tokens,
        "output_tokens": batch.output_tokens,
        "model": batch.model,
    })
}

fn search_usage_value(usage: &sag2::Usage) -> serde_json::Value {
    let batches: Vec<serde_json::Value> = usage
        .batches
        .iter()
        .map(|batch| {
            let mut value = batch_value(&batch.detail);
            value
                .as_object_mut()
                .expect("batch_value is an object")
                .insert("hop".to_owned(), serde_json::json!(batch.hop));
            value
        })
        .collect();
    serde_json::json!({
        "requests": usage.requests,
        "judgments": usage.judgments,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "batches": batches,
    })
}

fn json_from_sag2(value: &sag2_json::Value) -> serde_json::Value {
    serde_json::from_str(&sag2_json::compact_json(value))
        .expect("sag2-json output is valid JSON")
}

/// The first byte at which two payloads differ (the gate's failure evidence).
fn first_byte_difference(expected: &[u8], actual: &[u8]) -> Option<usize> {
    let common = expected.len().min(actual.len());
    for offset in 0..common {
        if expected[offset] != actual[offset] {
            return Some(offset);
        }
    }
    (expected.len() != actual.len()).then_some(common)
}

/// The JSON pointer of the first semantic difference, as design §1.3 asks for
/// (`/questions/candidate_12`).  Byte comparison remains the gate; this only
/// makes the failure readable.
fn first_json_pointer(expected: &str, actual: &str) -> Option<String> {
    let expected: serde_json::Value = serde_json::from_str(expected).ok()?;
    let actual: serde_json::Value = serde_json::from_str(actual).ok()?;
    pointer_difference("", &expected, &actual)
}

fn pointer_difference(
    prefix: &str,
    expected: &serde_json::Value,
    actual: &serde_json::Value,
) -> Option<String> {
    if expected == actual {
        return None;
    }
    match (expected, actual) {
        (serde_json::Value::Object(expected), serde_json::Value::Object(actual)) => {
            let mut keys: BTreeSet<&str> = expected.keys().map(String::as_str).collect();
            keys.extend(actual.keys().map(String::as_str));
            for key in keys {
                let escaped = key.replace('~', "~0").replace('/', "~1");
                let pointer = format!("{prefix}/{escaped}");
                match (expected.get(key), actual.get(key)) {
                    (Some(left), Some(right)) => {
                        if let Some(found) = pointer_difference(&pointer, left, right) {
                            return Some(found);
                        }
                    }
                    _ => return Some(pointer),
                }
            }
            None
        }
        (serde_json::Value::Array(expected), serde_json::Value::Array(actual)) => {
            for position in 0..expected.len().max(actual.len()) {
                let pointer = format!("{prefix}/{position}");
                match (expected.get(position), actual.get(position)) {
                    (Some(left), Some(right)) => {
                        if let Some(found) = pointer_difference(&pointer, left, right) {
                            return Some(found);
                        }
                    }
                    _ => return Some(pointer),
                }
            }
            None
        }
        _ => Some(prefix.to_owned()),
    }
}

/// C-1: compare the payload count, every payload byte and every journal digest
/// (`sag2.py:49`).  Returns `(payloads, equal bytes, equal digests, diffs)`.
fn compare_payloads(
    case: &SearchCase,
    recorded: &[Vec<u8>],
    failures: &mut Vec<Failure>,
) -> (usize, usize, usize, usize) {
    let prefix = |index: usize| format!("payloads[{index}]");
    if recorded.len() != case.payloads.len() {
        push_gate_failure(
            failures,
            &case.case_id,
            "payloads.len",
            case.payloads.len().to_string(),
            recorded.len().to_string(),
        );
    }
    let mut diffs = usize::from(recorded.len() != case.payloads.len());
    let mut equal = 0_usize;
    let mut digests_equal = 0_usize;
    let compared = recorded.len().min(case.payloads.len());
    for (index, actual) in recorded.iter().enumerate().take(compared) {
        let expected = case.payloads[index].as_bytes();
        if expected == actual {
            equal += 1;
        } else {
            diffs += 1;
            let offset = first_byte_difference(expected, actual).unwrap_or(0);
            let pointer = first_json_pointer(&case.payloads[index], &String::from_utf8_lossy(actual))
                .unwrap_or_else(|| "?".to_owned());
            push_gate_failure(
                failures,
                &case.case_id,
                format!("{} (byte {offset}, {pointer})", prefix(index)),
                display_text(&case.payloads[index]),
                display_text(&String::from_utf8_lossy(actual)),
            );
        }
        if let Some(expected_digest) = case.payloads_sha256.get(index) {
            let mut input = String::from("[");
            input.push_str(&sag2_json::compact_json(&sag2_json::Value::Str(
                case.endpoint.clone(),
            )));
            input.push(',');
            input.push_str(&String::from_utf8_lossy(actual));
            input.push(']');
            let digest = sha256_hex(input.as_bytes());
            if &digest == expected_digest {
                digests_equal += 1;
            } else {
                diffs += 1;
                push_gate_failure(
                    failures,
                    &case.case_id,
                    format!("payloads_sha256[{index}]"),
                    expected_digest.clone(),
                    digest,
                );
            }
        }
    }
    (compared, equal, digests_equal, diffs)
}

/// Gate C `search`: replay `search_sag2` under `OracleJudge` and compare both
/// halves design §1.3 requires - the request payload bytes (C-1) and the result
/// object (C-2).  `elapsed_ms` is the only excluded field and is declared in
/// the report.
fn run_gate_c(options: &Options) -> Result<ExitCode, String> {
    let manifest = preflight(options)?;
    let spec = manifest
        .search
        .as_ref()
        .ok_or_else(|| "manifest has no search section".to_owned())?;
    verify_manifest_files(
        options,
        &manifest,
        &[spec.golden.as_str(), spec.corpus_golden.as_str()],
    )?;

    let corpora = read_search_corpora(options, spec)?;
    if corpora.len() != spec.corpora {
        return Err(format!(
            "golden inconsistency: {} holds {} corpora, manifest says {}",
            spec.corpus_golden,
            corpora.len(),
            spec.corpora
        ));
    }
    let cases = read_search_cases(options, spec)?;
    if cases.len() != spec.cases {
        return Err(format!(
            "golden inconsistency: {} holds {} cases, manifest says {}",
            spec.golden,
            cases.len(),
            spec.cases
        ));
    }
    let payload_total: usize = cases.iter().map(|case| case.payloads.len()).sum();
    if payload_total != spec.payloads {
        return Err(format!(
            "golden inconsistency: {} payloads recorded, manifest says {}",
            payload_total, spec.payloads
        ));
    }
    for case in &cases {
        if !corpora.contains_key(&case.fixture) {
            return Err(format!(
                "golden inconsistency: case {} uses corpus {} with no definition",
                case.case_id, case.fixture
            ));
        }
    }

    let selected: Vec<&SearchCase> = match &options.fixtures {
        None => cases.iter().collect(),
        Some(fixtures) => {
            for fixture in fixtures {
                if !matches!(fixture.as_str(), "search" | "C0" | "C3" | "C4")
                    && !fixture.starts_with("w")
                {
                    return Err(format!(
                        "gate C fixture {fixture} is not implemented; the search section \
                         covers C0, C3 and C4"
                    ));
                }
            }
            cases
                .iter()
                .filter(|case| {
                    fixtures.iter().any(|fixture| {
                        fixture == "search"
                            || fixture == &case.fixture
                            || fixture == &format!("w{}", case.index_workers)
                    })
                })
                .collect()
        }
    };
    let selected: Vec<&SearchCase> = match &options.filter {
        None => selected,
        Some(filter) => selected
            .into_iter()
            .filter(|case| case.case_id.contains(filter.as_str()))
            .collect(),
    };
    if selected.is_empty() {
        return Err(format!(
            "no Gate C cases selected from {}",
            options.golden.join(&spec.golden).display()
        ));
    }

    // `--index-dir` keeps the built indexes for inspection; without it they
    // live in a temporary directory that dies with the run.
    let temporary;
    let directory: PathBuf = match &options.index_dir {
        Some(directory) => {
            fs::create_dir_all(directory)
                .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
            directory.clone()
        }
        None => {
            temporary = tempfile::tempdir()
                .map_err(|error| format!("cannot create the Gate C index directory: {error}"))?;
            temporary.path().to_path_buf()
        }
    };
    let mut indexes: BTreeMap<(String, usize), CorpusIndex> = BTreeMap::new();
    for case in &selected {
        let key = (case.fixture.clone(), case.index_workers);
        if indexes.contains_key(&key) {
            continue;
        }
        let corpus = corpora.get(&case.fixture).ok_or_else(|| {
            format!(
                "golden inconsistency: case {} uses corpus {} with no definition",
                case.case_id, case.fixture
            )
        })?;
        let documents: Vec<sag2_index::Document> = corpus
            .corpus
            .iter()
            .map(|document| sag2_index::Document {
                title: &document.title,
                text: &document.text,
                path: document.path.as_deref(),
                idx: document.idx.as_deref(),
            })
            .collect();
        let path = directory.join(format!("{}-w{}.sqlite", case.fixture, case.index_workers));
        let (index, _) = sag2_index::CorpusIndex::open(
            &path,
            &documents,
            &corpus.fingerprint,
            sag2_index::IndexOptions {
                chunk_chars: corpus.chunk_chars,
                index_workers: case.index_workers,
                source_dir: None,
                rebuild: sag2_index::RebuildPolicy::Always,
            },
        )
        .map_err(|error| {
            format!(
                "cannot build the {} corpus at index_workers={}: {error}",
                case.fixture, case.index_workers
            )
        })?;
        indexes.insert(key, index);
    }
    println!(
        "gate C: {} cases, {} corpora, {} indexes built",
        selected.len(),
        corpora.len(),
        indexes.len()
    );

    let mut failures: Vec<Failure> = Vec::new();
    let mut per_fixture: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut payloads_compared = 0_usize;
    let mut payloads_equal = 0_usize;
    let mut digests_equal = 0_usize;
    let mut c1_passed = 0_usize;
    let mut c2_passed = 0_usize;
    for case in &selected {
        let index = indexes
            .get(&(case.fixture.clone(), case.index_workers))
            .ok_or_else(|| format!("{}: its index was not built", case.case_id))?;
        let judge = sag2::BatchedJudge::new(sag2::OracleJudge::new());
        let options = sag2::SearchOptions {
            candidate_k: case.options.candidate_k,
            seed_k: case.options.seed_k,
            max_hops: case.options.max_hops,
            expansion_term_limit: case.options.expansion_term_limit,
            context: None,
        };
        let result = match sag2::search(index, &judge, &case.query, &options) {
            Ok(result) => result,
            Err(error) => {
                push_gate_failure(
                    &mut failures,
                    &case.case_id,
                    "search",
                    "Ok",
                    format!("{error}"),
                );
                let entry = per_fixture
                    .entry(format!("{} w{}", case.fixture, case.index_workers))
                    .or_insert((0, 0));
                entry.0 += 1;
                continue;
            }
        };
        let recorded = judge.transport().payloads();
        let (compared, equal, digests, c1_diffs) =
            compare_payloads(case, &recorded, &mut failures);
        payloads_compared += compared;
        payloads_equal += equal;
        digests_equal += digests;
        if c1_diffs == 0 {
            c1_passed += 1;
        }
        let expected = serde_json::json!({
            "ranked_ids": case.ranked_ids,
            "ranked_scores_bits": case.ranked_scores_bits,
            "baseline_ids": case.baseline_ids,
            "trace": case.trace,
            "jev_usage": case.jev_usage,
            "algorithm": case.algorithm,
        });
        let actual = search_result_value(&result);
        let mut gate = GateFailures::new(&mut failures);
        compare_json_exact(&case.case_id, "result", &expected, &actual, &mut gate);
        let c2_ok = gate.diffs == 0;
        if c2_ok {
            c2_passed += 1;
        }
        let entry = per_fixture
            .entry(format!("{} w{}", case.fixture, case.index_workers))
            .or_insert((0, 0));
        entry.0 += 1;
        if c1_diffs == 0 && c2_ok {
            entry.1 += 1;
        }
    }

    let passed: usize = per_fixture.values().map(|(_, passed)| passed).sum();
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
    println!(
        "gate C: {} cases, {} passed, {} failed; C-1 {}/{} cases, {}/{} payload bytes, \
         {}/{} digests; C-2 {}/{} cases",
        selected.len(),
        passed,
        selected.len() - passed,
        c1_passed,
        selected.len(),
        payloads_equal,
        payloads_compared,
        digests_equal,
        payloads_compared,
        c2_passed,
        selected.len()
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
            "tiktoken": "o200k_base",
        },
        "tolerances": {"level": "S", "f64": "bits", "ids": "exact"},
        "gates": {"C": {
            "cases": selected.len(),
            "passed": passed,
            "failed": selected.len() - passed,
            "payloads": payloads_compared,
            "payload_bytes_equal": payloads_equal,
            "payload_digests_equal": digests_equal,
            "subchecks": {
                "c1_payloads": {"cases": selected.len(), "passed": c1_passed},
                "c2_result": {"cases": selected.len(), "passed": c2_passed},
            },
            "fixtures": per_fixture.iter().map(|(fixture, (total, passed))| {
                serde_json::json!({"fixture": fixture, "cases": total, "passed": passed})
            }).collect::<Vec<_>>(),
            "index_workers": indexes.keys().map(|(fixture, workers)| {
                serde_json::json!({"fixture": fixture, "index_workers": workers})
            }).collect::<Vec<_>>(),
        }},
        "excluded_fields": ["elapsed_ms"],
        "excluded_checks": ["db_physical_bytes"],
        "failures": failures,
        "ulp_flips": [],
        "unsupported": [],
    });
    write_report(&options.report, &report);

    Ok(if failures_empty(&report) {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
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
