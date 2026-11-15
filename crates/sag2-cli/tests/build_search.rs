//! End-to-end `build` + `search` on the real binary.
//!
//! The unit tests cover the argument parsing; this drives the process the way
//! the README shows it, including the exit codes of the failure paths.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sag2-cli"))
        .args(arguments)
        .output()
        .expect("the binary runs")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr is UTF-8")
}

fn write_corpus(directory: &Path) -> String {
    let path = directory.join("corpus.jsonl");
    fs::write(
        &path,
        concat!(
            "{\"title\":\"Straße\",\"text\":\"Die Straße führt zum alten Hafen. Der Hafen ist ruhig.\"}\n",
            "{\"title\":\"Markt\",\"text\":\"Der Markt öffnet am Morgen. Die Straße zum Markt ist voll.\"}\n",
        ),
    )
    .expect("corpus writes");
    path.to_string_lossy().into_owned()
}

#[test]
fn build_then_search_round_trips() {
    let directory = tempfile::tempdir().expect("temp dir");
    let corpus = write_corpus(directory.path());
    let index = directory.path().join("de.sqlite");

    let built = run(&[
        "build",
        "--input",
        &corpus,
        "--output",
        index.to_str().expect("index path"),
        "--profile",
        "multilingual-v1-de",
    ]);
    assert_eq!(built.status.code(), Some(0), "{}", stderr(&built));
    let built_stdout = stdout(&built);
    for expected in [
        "profile:     multilingual-v1-de",
        "documents:   2",
        "action:      Built",
    ] {
        assert!(
            built_stdout.contains(expected),
            "missing {expected:?} in:\n{built_stdout}"
        );
    }

    let found = run(&[
        "search",
        "--index",
        index.to_str().expect("index path"),
        "--query",
        "STRASSE",
    ]);
    assert_eq!(found.status.code(), Some(0), "{}", stderr(&found));
    let found_stdout = stdout(&found);
    for expected in [
        "profile:    multilingual-v1-de",
        "mode:       lexical",
        "expression: \"strasse\"",
        "hits:       2",
        "doc=1",
    ] {
        assert!(
            found_stdout.contains(expected),
            "missing {expected:?} in:\n{found_stdout}"
        );
    }

    let phrase3 = run(&[
        "search",
        "--index",
        index.to_str().expect("index path"),
        "--query",
        "STRASSE",
        "--query-mode",
        "phrase3",
        "--limit",
        "1",
        "--judge",
        "oracle",
    ]);
    assert_eq!(phrase3.status.code(), Some(0), "{}", stderr(&phrase3));
    let phrase3_stdout = stdout(&phrase3);
    assert!(phrase3_stdout.contains("mode:       phrase3"));
    assert!(phrase3_stdout.contains("judge:      oracle"));
}

#[test]
fn refused_open_exits_three_with_the_typed_error() {
    let directory = tempfile::tempdir().expect("temp dir");
    let corpus = write_corpus(directory.path());
    let index = directory.path().join("de.sqlite");
    assert_eq!(
        run(&[
            "build",
            "--input",
            &corpus,
            "--output",
            index.to_str().expect("index path"),
            "--profile",
            "multilingual-v1-de",
        ])
        .status
        .code(),
        Some(0)
    );

    // T1's language-less identifier is never mapped onto a language.
    let connection = rusqlite::Connection::open(&index).expect("open the index");
    connection
        .execute(
            "UPDATE rust_index_metadata SET value='multilingual-v1' WHERE key='analysis'",
            [],
        )
        .expect("tamper the recorded profile");
    drop(connection);
    let legacy = run(&[
        "search",
        "--index",
        index.to_str().expect("index path"),
        "--query",
        "strasse",
    ]);
    assert_eq!(legacy.status.code(), Some(3));
    assert!(
        stderr(&legacy).contains("is not one this build knows"),
        "{}",
        stderr(&legacy)
    );

    let missing = run(&[
        "search",
        "--index",
        directory.path().join("nope.sqlite").to_str().expect("path"),
        "--query",
        "x",
    ]);
    assert_eq!(missing.status.code(), Some(3));
    assert!(
        stderr(&missing).contains("index not found"),
        "{}",
        stderr(&missing)
    );

    let dictionary = run(&[
        "build",
        "--input",
        &corpus,
        "--output",
        directory.path().join("ja.sqlite").to_str().expect("path"),
        "--profile",
        "multilingual-v1-ja",
    ]);
    assert_eq!(dictionary.status.code(), Some(3));
    assert!(
        stderr(&dictionary).contains("needs a dictionary directory"),
        "{}",
        stderr(&dictionary)
    );

    let profile = run(&[
        "build",
        "--input",
        &corpus,
        "--output",
        directory.path().join("x.sqlite").to_str().expect("path"),
        "--profile",
        "multilingual-v1",
    ]);
    assert_eq!(profile.status.code(), Some(3));
    assert!(
        stderr(&profile).contains("unknown analysis profile"),
        "{}",
        stderr(&profile)
    );
}

/// The committed `golden/indexes/C0.sqlite` was written by `sag2.py`, which
/// predates `rust_index_metadata.analysis`; `search` must read it back through
/// the reference pipeline rather than refuse or guess.
#[test]
fn a_python_built_index_reads_back_through_the_recorded_profile() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/indexes/C0.sqlite");
    let found = run(&[
        "search",
        "--index",
        fixture.to_str().expect("fixture path"),
        "--query",
        "anchor birthplace",
        "--limit",
        "3",
    ]);
    assert_eq!(found.status.code(), Some(0), "{}", stderr(&found));
    let found_stdout = stdout(&found);
    for expected in [
        "profile:    sag2-zh-v1",
        "expression: \"anchor\" OR \"birthplace\"",
        "hits:       3",
        "doc=1",
        "Anchor",
    ] {
        assert!(
            found_stdout.contains(expected),
            "missing {expected:?} in:\n{found_stdout}"
        );
    }
}
