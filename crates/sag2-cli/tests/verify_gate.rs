//! Exit-code contract of `sag2-cli verify` (design §1.0 rules 2-3, §7.1).
//!
//! These exercise the tooling gates directly: a gate run outside the certified
//! environment, or against the wrong reference checkout, must compare nothing
//! and exit `3`.  The comparison itself is covered by the `verify` unit tests
//! and by the committed golden replay.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use serde_json::{json, Value};

fn reference(digest: &str) -> Value {
    json!({
        "sag2_py_sha256": digest,
        "sag2_rank_py_sha256": digest,
        "chunking_py_sha256": digest,
        "commit": "0000000000000000000000000000000000000000",
    })
}

fn write_manifest(dir: &Path, sqlite: &str, reference: Option<Value>) {
    let mut manifest = json!({
        "generator_version": "0.8.1",
        "python": "3.12.11",
        "sqlite": sqlite,
        "jieba": "0.42.1",
        "pyuca": "9.0.0",
        "unicode": "15.0.0",
        "files": [],
        "fixtures": [],
    });
    if let Some(reference) = reference {
        manifest["reference"] = reference;
    }
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).expect("serialise manifest"),
    )
    .expect("write manifest");
}

fn run_verify(golden: &Path, report: &Path, reference: Option<&Path>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_sag2-cli"));
    command
        .args(["verify", "--gate", "A", "--golden"])
        .arg(golden)
        .arg("--report")
        .arg(report);
    if let Some(reference) = reference {
        command.arg("--reference").arg(reference);
    }
    command.output().expect("run sag2-cli")
}

fn report_of(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).expect("report")).expect("report json")
}

fn unsupported(report: &Value) -> &str {
    report["unsupported"][0]
        .as_str()
        .expect("one unsupported entry")
}

#[test]
fn sqlite_outside_the_allowlist_exits_three_without_comparing() {
    let temp = tempfile::tempdir().expect("temp dir");
    write_manifest(temp.path(), "3.45.1", Some(reference(&"00".repeat(32))));
    let report = temp.path().join("report.json");

    let output = run_verify(temp.path(), &report, None);

    assert_eq!(
        output.status.code(),
        Some(3),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = report_of(&report);
    assert!(unsupported(&report).contains("allowlist"), "{report}");
    assert!(
        report["gates"]
            .as_object()
            .is_some_and(|gates| gates.is_empty()),
        "no gate may run: {report}"
    );
}

#[test]
fn reference_digest_mismatch_exits_three() {
    let temp = tempfile::tempdir().expect("temp dir");
    let golden = temp.path().join("golden");
    fs::create_dir_all(&golden).expect("golden dir");
    write_manifest(&golden, "3.50.4", Some(reference(&"00".repeat(32))));
    let checkout = temp.path().join("reference");
    fs::create_dir_all(&checkout).expect("reference dir");
    for name in ["sag2.py", "sag2_rank.py", "chunking.py"] {
        fs::write(checkout.join(name), b"not the reference").expect("reference file");
    }
    let report = temp.path().join("report.json");

    let output = run_verify(&golden, &report, Some(&checkout));

    assert_eq!(
        output.status.code(),
        Some(3),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(unsupported(&report_of(&report)).contains("reference"));
}

#[test]
fn missing_reference_checkout_exits_three() {
    let temp = tempfile::tempdir().expect("temp dir");
    let golden = temp.path().join("golden");
    fs::create_dir_all(&golden).expect("golden dir");
    write_manifest(&golden, "3.50.4", Some(reference(&"00".repeat(32))));
    let missing = temp.path().join("absent");
    let report = temp.path().join("report.json");

    let output = run_verify(&golden, &report, Some(&missing));

    assert_eq!(output.status.code(), Some(3));
    assert!(unsupported(&report_of(&report)).contains("reference"));
}

#[test]
fn manifest_without_reference_provenance_is_rejected() {
    let temp = tempfile::tempdir().expect("temp dir");
    write_manifest(temp.path(), "3.50.4", None);
    let report = temp.path().join("report.json");

    let output = run_verify(temp.path(), &report, None);

    assert_eq!(output.status.code(), Some(3));
    assert!(unsupported(&report_of(&report)).contains("reference"));
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// A Gate A golden whose only fixture index must come from `--index-dir`.
fn write_index_dir_golden(root: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let golden = root.join("golden");
    let reference_dir = root.join("reference");
    fs::create_dir_all(&golden).expect("golden dir");
    fs::create_dir_all(&reference_dir).expect("reference dir");
    let source = b"reference placeholder";
    for name in ["sag2.py", "sag2_rank.py", "chunking.py"] {
        fs::write(reference_dir.join(name), source).expect("reference file");
    }
    let digest = sha256_hex(source);
    let signature = "[\"fixture\",2200,5,null,\"desktop-sentences-v1\",\"plain-text-default\"]";
    let case = json!({
        "case_id": "C0/index-dir",
        "fixture": "C0",
        "expression": "\"anchor\"",
        "excluded": [],
        "limit": 1,
        "term": null,
        "ids": [],
        "document_ids": [],
        "scores_bits": [],
        "hits": [],
    });
    let retrieve = format!("{case}\n");
    fs::write(golden.join("retrieve.jsonl"), &retrieve).expect("retrieve golden");
    let dump = format!(
        "{}\n",
        json!({"type": "metadata", "fixture": "C0", "signature": signature})
    );
    fs::write(golden.join("index_dump.jsonl"), &dump).expect("index dump");
    let manifest = json!({
        "generator_version": "0.8.1",
        "python": "3.12.11",
        "sqlite": "3.50.4",
        "jieba": "0.42.1",
        "pyuca": "9.0.0",
        "unicode": "15.0.0",
        "reference": {
            "sag2_py_sha256": digest,
            "sag2_rank_py_sha256": digest,
            "chunking_py_sha256": digest,
            "commit": "0".repeat(40),
        },
        "files": [
            {
                "path": "retrieve.jsonl",
                "sha256": sha256_hex(retrieve.as_bytes()),
                "lines": 1,
            },
            {"path": "index_dump.jsonl", "sha256": sha256_hex(dump.as_bytes()), "lines": 1},
        ],
        "fixtures": [{
            "fixture": "C0",
            "index": "C0.sqlite",
            "index_sha256": "0".repeat(64),
            "chunks": 1,
            "signature": signature,
        }],
    });
    fs::write(
        golden.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).expect("manifest json"),
    )
    .expect("manifest");
    (golden, reference_dir)
}

fn run_verify_index_dir(
    golden: &Path,
    reference_dir: &Path,
    index_dir: &Path,
    report: &Path,
) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sag2-cli"))
        .args(["verify", "--gate", "A", "--golden"])
        .arg(golden)
        .arg("--report")
        .arg(report)
        .arg("--reference")
        .arg(reference_dir)
        .arg("--index-dir")
        .arg(index_dir)
        .output()
        .expect("run sag2-cli")
}

#[test]
fn gate_a_index_dir_never_falls_back_to_the_golden_indexes() {
    let temp = tempfile::tempdir().expect("temp dir");
    let (golden, reference_dir) = write_index_dir_golden(temp.path());
    let index_dir = temp.path().join("indexes");
    fs::create_dir_all(&index_dir).expect("index dir");
    let report = temp.path().join("report.json");

    let output = run_verify_index_dir(&golden, &reference_dir, &index_dir, &report);

    assert_eq!(
        output.status.code(),
        Some(3),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        unsupported(&report_of(&report)).contains("fixture C0 has no index"),
        "--index-dir must not fall back to golden/indexes: {}",
        unsupported(&report_of(&report))
    );
}

#[test]
fn gate_a_index_dir_with_a_corrupted_index_exits_three() {
    let temp = tempfile::tempdir().expect("temp dir");
    let (golden, reference_dir) = write_index_dir_golden(temp.path());
    let index_dir = temp.path().join("indexes");
    fs::create_dir_all(&index_dir).expect("index dir");
    fs::write(index_dir.join("C0.sqlite"), b"not a database").expect("corrupt index");
    let report = temp.path().join("report.json");

    let output = run_verify_index_dir(&golden, &reference_dir, &index_dir, &report);

    assert_eq!(
        output.status.code(),
        Some(3),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        unsupported(&report_of(&report)).contains("cannot open fixture C0"),
        "{}",
        unsupported(&report_of(&report))
    );
}
