//! Gate A evidence on the Python-built fixtures (`golden/indexes/*.sqlite`).
//!
//! These tests read the committed goldens, so they exercise the same files the
//! `sag2-cli verify --gate A` replay walks, without duplicating 221 cases.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::Connection;
use sag2_index::{
    open, open_readonly, IndexOptions, OpenError, QueryOptions, RebuildPolicy, CHUNK_POLICY,
    INDEX_SCHEMA,
};

fn golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden")
}

/// The `metadata.signature` the exporter recorded for `fixture`.
fn signature_of(fixture: &str) -> String {
    let text = fs::read_to_string(golden().join("index_dump.jsonl")).expect("index_dump");
    for line in text.lines() {
        let value: serde_json::Value = serde_json::from_str(line).expect("dump line");
        if value["type"] == "metadata" && value["fixture"] == fixture {
            return value["signature"].as_str().expect("signature").to_owned();
        }
    }
    panic!("no signature for {fixture}");
}

fn fixture_path(fixture: &str) -> PathBuf {
    golden().join("indexes").join(format!("{fixture}.sqlite"))
}

#[test]
fn opens_a_python_built_index_and_loads_its_statistics() {
    let index = open_readonly(
        &fixture_path("C0"),
        &signature_of("C0"),
        QueryOptions::default(),
    )
    .expect("C0 opens");
    assert_eq!(
        index.signature(),
        "[\"fixture\",2200,5,null,\"desktop-sentences-v1\",\"plain-text-default\"]"
    );
    assert_eq!(index.statistics().total, 23);
    assert_eq!(index.statistics().document_count, 23);

    let hits = index
        .retrieve("\"anchor\" OR \"birthplace\"", &BTreeSet::new(), 7, None)
        .expect("retrieve");
    assert_eq!(hits.first().map(|hit| hit.id), Some(1));
    assert_eq!(hits[0].title, "Anchor");
}

#[test]
fn rejects_a_signature_from_another_corpus() {
    let error = open_readonly(
        &fixture_path("C0"),
        &sag2_index::signature("not-fixture", 2200, None),
        QueryOptions::default(),
    )
    .expect_err("must refuse");
    match error {
        OpenError::SignatureMismatch { found, .. } => assert!(found.contains("fixture")),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn workers_one_and_eight_produce_identical_hits() {
    let signature = signature_of("C1-rare140");
    let single = open_readonly(
        &fixture_path("C1-rare140"),
        &signature,
        QueryOptions {
            workers: 1,
            cache_entries: 256,
        },
    )
    .expect("single reader");
    let many = open_readonly(
        &fixture_path("C1-rare140"),
        &signature,
        QueryOptions {
            workers: 8,
            cache_entries: 256,
        },
    )
    .expect("eight readers");

    let rare: Vec<String> = (0..5).map(|index| format!("rare{index}")).collect();
    let common: Vec<String> = (0..4).map(|index| format!("common{index}")).collect();
    let quoted = |terms: &[String]| {
        terms
            .iter()
            .map(|term| format!("\"{term}\""))
            .collect::<Vec<_>>()
            .join(" OR ")
    };
    let expressions = [
        quoted(&rare),
        quoted(&common),
        quoted(&[rare.clone(), common.clone()].concat()),
        "\"rare0\"".to_owned(),
    ];
    let excluded: BTreeSet<i64> = [1, 3].into_iter().collect();

    let mut compared = 0;
    for expression in &expressions {
        for limit in [1, 7, 100] {
            let left = single
                .retrieve(expression, &excluded, limit, None)
                .expect("single");
            let right = many
                .retrieve(expression, &excluded, limit, None)
                .expect("many");
            let shape = |hits: &[sag2_index::Hit]| {
                hits.iter()
                    .map(|hit| (hit.id, hit.document_id, hit.score.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(shape(&left), shape(&right), "{expression} limit {limit}");
            compared += 1;
        }
    }
    assert_eq!(compared, 12);
}

#[test]
fn format_version_mismatch_is_refused_and_never_rebuilds() {
    let temp = tempfile::tempdir().expect("temp dir");
    let tampered = temp.path().join("C0.sqlite");
    fs::copy(fixture_path("C0"), &tampered).expect("copy fixture");

    let db = Connection::open(&tampered).expect("open for tampering");
    db.execute_batch(
        "CREATE TABLE rust_index_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
         INSERT INTO rust_index_metadata VALUES('format_version','2');
         INSERT INTO rust_index_metadata VALUES('chunk_policy_version','desktop-sentences-v1');",
    )
    .expect("tamper");
    drop(db);

    for rebuild in [
        RebuildPolicy::Strict,
        RebuildPolicy::OnCorpusChange,
        RebuildPolicy::Always,
    ] {
        let options = IndexOptions {
            chunk_chars: 2200,
            index_workers: 1,
            source_dir: None,
            rebuild,
        };
        match open(&tampered, &options, "fixture") {
            Err(OpenError::UnsupportedIndexVersion { found, supported }) => {
                assert_eq!((found, supported), (2, 1));
            }
            other => panic!("{rebuild:?} must refuse, got {other:?}"),
        }
    }

    // The refusing open must not have touched the file.
    let db = Connection::open(&tampered).expect("reopen");
    let chunks: i64 = db
        .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
        .expect("chunks");
    assert_eq!(chunks, 23);
    let version: String = db
        .query_row(
            "SELECT value FROM rust_index_metadata WHERE key='format_version'",
            [],
            |row| row.get(0),
        )
        .expect("format_version");
    assert_eq!(version, "2");
}

#[test]
fn foreign_database_is_refused_instead_of_dropped() {
    let temp = tempfile::tempdir().expect("temp dir");
    let foreign = temp.path().join("foreign.sqlite");
    let db = Connection::open(&foreign).expect("create");
    db.execute_batch("CREATE TABLE chunks (id INTEGER PRIMARY KEY, title TEXT NOT NULL);")
        .expect("create chunks");
    drop(db);

    let strict = IndexOptions {
        chunk_chars: 2200,
        index_workers: 1,
        source_dir: None,
        rebuild: RebuildPolicy::Strict,
    };
    assert!(matches!(
        open(&foreign, &strict, "foreign"),
        Err(OpenError::ForeignDatabase)
    ));
    let compatible = IndexOptions {
        rebuild: RebuildPolicy::OnCorpusChange,
        ..strict.clone()
    };
    assert!(matches!(
        open(&foreign, &compatible, "foreign"),
        Err(OpenError::ForeignDatabase)
    ));
    // `Always` may rebuild, which needs the builder P1 owns.
    let always = IndexOptions {
        rebuild: RebuildPolicy::Always,
        ..strict
    };
    assert!(matches!(
        open(&foreign, &always, "foreign"),
        Err(OpenError::BuildUnavailable)
    ));

    let db = Connection::open(&foreign).expect("reopen");
    let chunks: i64 = db
        .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
        .expect("chunks");
    assert_eq!(chunks, 0);
}

#[test]
fn signature_mismatch_follows_the_policy() {
    let temp = tempfile::tempdir().expect("temp dir");
    let copy = temp.path().join("C0.sqlite");
    fs::copy(fixture_path("C0"), &copy).expect("copy fixture");

    // The stored signature is `["fixture",2200,...]`; ask for another corpus.
    let strict = IndexOptions {
        chunk_chars: 2200,
        index_workers: 1,
        source_dir: None,
        rebuild: RebuildPolicy::Strict,
    };
    assert!(matches!(
        open(&copy, &strict, "another-corpus"),
        Err(OpenError::IndexSignatureMismatch {
            field: "fingerprint"
        })
    ));
    let compatible = IndexOptions {
        rebuild: RebuildPolicy::OnCorpusChange,
        ..strict
    };
    assert!(matches!(
        open(&copy, &compatible, "another-corpus"),
        Err(OpenError::BuildUnavailable)
    ));
}

#[test]
fn signature_constant_matches_the_reference_chunk_policy() {
    assert_eq!(INDEX_SCHEMA, 5);
    assert_eq!(CHUNK_POLICY, "desktop-sentences-v1");
}
