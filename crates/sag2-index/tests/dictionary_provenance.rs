//! Dictionary profiles fail closed: a `multilingual-v1-ja`/`-ko` build or open
//! without a dictionary directory is a typed error, and an index whose
//! `analysis_dict` row names another dictionary is provenance, never a rebuild.
//!
//! No real dictionary is needed here: the directory rules are exercised at the
//! resolution boundary, and the provenance test edits the recorded fingerprint
//! directly.  The offline `#[ignore]`d test that loads a real ipadic/ko-dic
//! lives in `sag2-core/tests/lindera_dictionary.rs`.

use rusqlite::Connection;
use sag2_index::{
    open, open_readonly, signature, AnalysisError, CorpusIndex, Document, IndexOptions, Language,
    OpenError, Profile, QueryOptions, RebuildPolicy,
};

fn corpus() -> [Document<'static>; 2] {
    [
        Document {
            title: "Anchor",
            text: "Alice Paul birthplace depicts a bridge.",
            path: None,
            idx: None,
        },
        Document {
            title: "Bridge",
            text: "The target is the bridge near the harbour.",
            path: None,
            idx: None,
        },
    ]
}

fn options(profile: Profile) -> IndexOptions {
    IndexOptions {
        chunk_chars: 2200,
        index_workers: 1,
        source_dir: None,
        profile,
        dictionary_dir: None,
        rebuild: RebuildPolicy::Always,
    }
}

fn japanese() -> Profile {
    Profile::Multilingual {
        language: Language::Ja,
    }
}

#[test]
fn a_dictionary_profile_without_a_directory_cannot_build() {
    for language in [Language::Ja, Language::Ko] {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("index.sqlite");
        let error = CorpusIndex::open(
            &path,
            &corpus(),
            "fixture",
            options(Profile::Multilingual { language }),
        )
        .expect_err("no dictionary directory");
        match error {
            OpenError::Analysis(AnalysisError::DictionaryMissing { language: found }) => {
                assert_eq!(found, language);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(
            !path.exists(),
            "a refused build must not write a degraded index"
        );
    }
}

#[test]
fn a_dictionary_profile_without_a_directory_cannot_open() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("index.sqlite");
    CorpusIndex::open(&path, &corpus(), "fixture", options(Profile::Sag2ZhV1))
        .expect("build the reference index");

    let signature = signature("fixture", 2200, None);
    let read_only = open_readonly(&path, &signature, japanese(), QueryOptions::default())
        .expect_err("no dictionary directory");
    match read_only {
        OpenError::Analysis(AnalysisError::DictionaryMissing { .. }) => {}
        other => panic!("unexpected {other:?}"),
    }

    // The free `open` resolves the profile's dictionary before touching the
    // file, so it reports the same typed error instead of reusing the index.
    let error = match open(&path, &options(japanese()), "fixture") {
        Ok(_) => panic!("a dictionary profile cannot open without a dictionary"),
        Err(error) => error,
    };
    match error {
        OpenError::Analysis(AnalysisError::DictionaryMissing { .. }) => {}
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn a_dictionary_free_index_records_none_and_refuses_another_dictionary() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("index.sqlite");
    CorpusIndex::open(&path, &corpus(), "fixture", options(Profile::Sag2ZhV1))
        .expect("build the reference index");

    {
        let db = Connection::open(&path).expect("open the index file");
        let recorded: String = db
            .query_row(
                "SELECT value FROM rust_index_metadata WHERE key='analysis_dict'",
                [],
                |row| row.get(0),
            )
            .expect("analysis_dict row");
        assert_eq!(recorded, "none");
        // A fingerprint recorded where a dictionary-free profile was expected
        // is a provenance mismatch even with `RebuildPolicy::Always`.
        db.execute(
            "UPDATE rust_index_metadata SET value=?1 WHERE key='analysis_dict'",
            [format!("{}-fake", "0".repeat(59))],
        )
        .expect("rewrite the provenance row");
    }

    let error = CorpusIndex::open(&path, &corpus(), "fixture", options(Profile::Sag2ZhV1))
        .expect_err("a changed dictionary is a provenance mismatch");
    match error {
        OpenError::IncompatibleIndex {
            field,
            index,
            runtime,
        } => {
            assert_eq!(field, "analysis_dict");
            assert_eq!(index, format!("{}-fake", "0".repeat(59)));
            assert_eq!(runtime, "none");
        }
        other => panic!("unexpected {other:?}"),
    }

    // The refusal never rebuilt: the row is still the one that was written.
    let db = Connection::open(&path).expect("reopen the index file");
    let recorded: String = db
        .query_row(
            "SELECT value FROM rust_index_metadata WHERE key='analysis_dict'",
            [],
            |row| row.get(0),
        )
        .expect("analysis_dict row");
    assert_eq!(recorded, format!("{}-fake", "0".repeat(59)));
}

#[cfg(not(feature = "lindera"))]
#[test]
fn a_dictionary_directory_needs_the_lindera_feature() {
    let dictionary = tempfile::tempdir().expect("temp dir");
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("index.sqlite");
    let mut options = options(japanese());
    options.dictionary_dir = Some(dictionary.path().to_owned());
    let error = CorpusIndex::open(&path, &corpus(), "fixture", options)
        .expect_err("this build has no lindera feature");
    match error {
        OpenError::Analysis(AnalysisError::DictionaryUnsupported {
            language: Language::Ja,
        }) => {}
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn a_missing_directory_is_a_typed_dictionary_error() {
    // `DictionaryAnalysis` owns the resolution rules, but the index boundary
    // must not turn a bad directory into a panic.
    let directory = tempfile::tempdir().expect("temp dir");
    let mut options = options(japanese());
    options.dictionary_dir = Some(directory.path().join("absent"));
    let path = directory.path().join("index.sqlite");
    let error = CorpusIndex::open(&path, &corpus(), "fixture", options)
        .expect_err("the dictionary directory does not exist");
    match error {
        OpenError::Analysis(AnalysisError::Dictionary { .. }) => {}
        OpenError::Analysis(AnalysisError::DictionaryUnsupported { .. }) => {}
        other => panic!("unexpected {other:?}"),
    }
}
