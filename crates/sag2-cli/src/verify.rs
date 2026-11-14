//! `sag2-cli verify --gate A`: replay the golden Gate A cases (design §1.1, §7.3).
//!
//! Comparison uses only the §1.0 tolerance functions: `eq_ids` on the ordered id
//! sequences and `eq_f64_bits` on the scores.  Exit codes follow §1.0:
//! `0` passed, `2` a comparison failed, `3` the tooling/data cannot be used.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use sag2_index::{open_readonly, CorpusIndex, QueryError, QueryOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Generator version this verifier understands.
const GENERATOR_VERSION: &str = "0.1.0";

/// Minimum interpreter the goldens may come from (design §11 divergence D5).
const REQUIRED_PYTHON: (u64, u64) = (3, 12);

/// `eq_f64(a, b) := |a-b| <= 1e-12 * max(1.0, |a|, |b|)` (design §1.0).
const RELATIVE_TOLERANCE: f64 = 1e-12;

/// Failures are buffered so the report shows the full picture (§1.0 rule 1).
const MAX_FAILURES: usize = 200;

#[derive(Deserialize)]
struct Manifest {
    generator_version: String,
    python: String,
    sqlite: String,
    files: Vec<ManifestFile>,
    fixtures: Vec<ManifestFixture>,
}

#[derive(Deserialize)]
struct ManifestFile {
    path: String,
    sha256: String,
    lines: usize,
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
    #[serde(default)]
    error: Option<String>,
}

struct Options {
    gate: String,
    golden: PathBuf,
    cache: PathBuf,
    report: PathBuf,
    filter: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            gate: "A".to_owned(),
            golden: PathBuf::from("golden"),
            cache: std::env::temp_dir().join("sag2-verify-cache"),
            report: PathBuf::from("report.json"),
            filter: None,
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
    path: &'static str,
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
    "usage: sag2-cli verify --gate A --golden golden/ [--index-cache <dir>] \
     [--filter <substring>] [--report report.json]"
}

fn run(options: &Options) -> Result<ExitCode, String> {
    if !options.gate.eq_ignore_ascii_case("A") {
        return Err(format!(
            "gate {} is not implemented in P0; only gate A has goldens",
            options.gate
        ));
    }

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

    for file in &manifest.files {
        let path = options.golden.join(&file.path);
        let bytes = fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        if sha256_hex(&bytes) != file.sha256 {
            return Err(format!("{} does not match the manifest sha256", file.path));
        }
        let lines = bytes.iter().filter(|byte| **byte == b'\n').count();
        if lines != file.lines {
            return Err(format!(
                "{} has {lines} lines, manifest says {}",
                file.path, file.lines
            ));
        }
    }

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
        let bytes =
            fs::read(&source).map_err(|error| format!("cannot read {}: {error}", source.display()))?;
        if sha256_hex(&bytes) != fixture.index_sha256 {
            return Err(format!("{} does not match the manifest sha256", fixture.index));
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
        let hits = index.retrieve(&case.expression, &excluded, case.limit, case.term.as_deref());
        let before = failures.len();
        check(case, hits, &mut failures, &mut ulp_flips);
        let entry = per_fixture.entry(case.fixture.clone()).or_insert((0, 0));
        entry.0 += 1;
        if failures.len() == before {
            entry.1 += 1;
        }
    }

    println!("{:<12} {:>7} {:>7} {:>7}", "fixture", "cases", "passed", "failed");
    for (fixture, (total, passed)) in &per_fixture {
        println!(
            "{:<12} {:>7} {:>7} {:>7}",
            fixture,
            total,
            passed,
            total - passed
        );
    }
    let passed = per_fixture.values().map(|(_, passed)| passed).sum::<usize>();
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

    let sqlite_note = if manifest.sqlite != sag2_index::sqlite_version() {
        format!(
            "golden sqlite {} vs runtime {}: scores still compared bit-exactly \
             (measured identical); ulp_flips guard is active",
            manifest.sqlite,
            sag2_index::sqlite_version()
        )
    } else {
        String::new()
    };
    let report = serde_json::json!({
        "golden": {
            "dir": options.golden.display().to_string(),
            "generated_by": format!("export_golden.py@{}", manifest.generator_version),
            "python": manifest.python,
            "sqlite": manifest.sqlite,
            "runtime_sqlite": sag2_index::sqlite_version(),
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
        "notes": if sqlite_note.is_empty() { vec![] } else { vec![sqlite_note] },
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
            push_failure(failures, case, "retrieve", "results".into(), format!("error: {error}"), None);
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

    let ids: Vec<i64> = hits.iter().map(|hit| hit.id).collect();
    if ids != case.ids {
        push_failure(failures, case, "ids", format!("{:?}", case.ids), format!("{ids:?}"), None);
    }
    let document_ids: Vec<i64> = hits.iter().map(|hit| hit.document_id).collect();
    if document_ids != case.document_ids {
        push_failure(
            failures,
            case,
            "document_ids",
            format!("{:?}", case.document_ids),
            format!("{document_ids:?}"),
            None,
        );
    }
    if hits.len() != case.scores_bits.len() {
        push_failure(
            failures,
            case,
            "length",
            format!("{} scores", case.scores_bits.len()),
            format!("{} hits", hits.len()),
            None,
        );
    }
    for (index, expected_hex) in case.scores_bits.iter().enumerate() {
        if index >= hits.len() {
            break;
        }
        let Some(expected) = bits_to_f64(expected_hex) else {
            push_failure(
                failures,
                case,
                "scores_bits",
                expected_hex.clone(),
                "unparseable golden bits".into(),
                None,
            );
            continue;
        };
        let actual = hits[index].score;
        if actual.to_bits() == expected.to_bits() {
            continue;
        }
        let delta = actual - expected;
        push_failure(
            failures,
            case,
            "scores",
            format!("{expected_hex} ({expected:e})"),
            format!("{:016x} ({actual:e})", actual.to_bits()),
            Some(delta),
        );
        if within_tolerance(expected, actual) {
            ulp_flips.push(serde_json::json!({
                "case_id": case.case_id,
                "position": index,
                "expected": expected_hex,
                "actual": format!("{:016x}", actual.to_bits()),
                "delta": delta,
            }));
        }
    }
}

fn push_failure(
    failures: &mut Vec<Failure>,
    case: &Case,
    path: &'static str,
    expected: String,
    actual: String,
    delta: Option<f64>,
) {
    if failures.len() < MAX_FAILURES {
        failures.push(Failure {
            case_id: case.case_id.clone(),
            path,
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
    let text = fs::read_to_string(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
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
    let text = fs::read_to_string(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut cases = Vec::new();
    for line in text.lines() {
        let case: Case =
            serde_json::from_str(line).map_err(|error| format!("{}: {error}", path.display()))?;
        if case.error.is_none()
            && (case.ids.len() != case.document_ids.len()
                || case.ids.len() != case.scores_bits.len())
        {
            return Err(format!(
                "{}: case {} has inconsistent golden arrays",
                path.display(),
                case.case_id
            ));
        }
        cases.push(case);
    }
    Ok(cases)
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
