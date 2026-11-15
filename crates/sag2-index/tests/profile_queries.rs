//! Query-side tokenisation follows the profile the index was opened with, and
//! that profile normalises before the certified tokenizer.
//!
//! `Sag2ZhV1` stays byte-identical: it never normalises and its expressions are
//! the pinned ones.  A multilingual profile folds case, accents and
//! orthographic variants, so a query reaches the tokens the build stored.

use sag2_core::{match_query, match_query_with, Language, Profile, QueryMode};
use sag2_index::{
    open_readonly, CorpusIndex, Document, IndexOptions, OpenError, QueryOptions, RebuildPolicy,
    Seed,
};

const GERMAN: Profile = Profile::Multilingual {
    language: Language::De,
};

const QUERY: &str = "harbour town guide";

fn corpus() -> [Document<'static>; 2] {
    [
        Document {
            title: "Straße",
            text: "Die Straße ist alt und ruhig.",
            path: None,
            idx: None,
        },
        Document {
            title: "Guide",
            text: "The harbour town guide describes the old street.",
            path: None,
            idx: None,
        },
    ]
}

fn build(profile: Profile) -> (tempfile::TempDir, CorpusIndex) {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("index.sqlite");
    let (index, _) = CorpusIndex::open(
        &path,
        &corpus(),
        "fixture",
        IndexOptions {
            chunk_chars: 2200,
            index_workers: 1,
            source_dir: None,
            profile,
            dictionary_dir: None,
            rebuild: RebuildPolicy::Always,
        },
    )
    .expect("build");
    (directory, index)
}

#[test]
fn queries_use_the_index_profile_pipeline() {
    let (_directory, index) = build(GERMAN);
    assert_eq!(index.profile(), GERMAN);

    for mode in [QueryMode::Lexical, QueryMode::Phrase3] {
        assert_eq!(
            index.match_query_with(mode, QUERY).expect("expression"),
            match_query_with(mode, GERMAN, QUERY),
            "{mode:?}"
        );
    }
    // `match_query` stays the reference expression even on another profile's
    // index, so the default path cannot move.
    assert_eq!(
        index.match_query(QUERY).expect("expression"),
        match_query(QUERY)
    );

    let expression = index
        .match_query_with(QueryMode::Phrase3, QUERY)
        .expect("expression");
    assert!(
        !index
            .retrieve(&expression, &Default::default(), 5, None)
            .expect("retrieve")
            .is_empty(),
        "the index answers the expression its own pipeline built"
    );
}

#[test]
fn a_multilingual_index_matches_case_and_orthography_variants() {
    let (directory, index) = build(GERMAN);

    // The build stored `strasse` for the `ß` title; both spellings reach it.
    for query in ["Straße", "STRASSE", "strasse"] {
        let expression = index.match_query(query).expect("expression");
        assert_eq!(expression, "\"strasse\"", "{query:?}");
        let hits = index
            .retrieve(&expression, &Default::default(), 5, None)
            .expect("retrieve");
        assert!(
            hits.iter().any(|hit| hit.title == "Straße"),
            "{query:?} must reach the `Straße` document"
        );
    }

    // The reference profile does not fold, and cannot read this index.
    let signature = index.signature().to_owned();
    let path = directory.path().join("index.sqlite");
    drop(index);
    let error = open_readonly(
        &path,
        &signature,
        Profile::Sag2ZhV1,
        QueryOptions::default(),
    )
    .expect_err("the reference profile must refuse the index");
    match error {
        OpenError::AnalysisMismatch { index, expected } => {
            assert_eq!(index.as_deref(), Some("multilingual-v1-de"));
            assert_eq!(expected, Profile::Sag2ZhV1);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn a_read_only_open_carries_the_index_profile() {
    let (directory, index) = build(GERMAN);
    let signature = index.signature().to_owned();
    let path = directory.path().join("index.sqlite");
    drop(index);

    let reopened = open_readonly(&path, &signature, GERMAN, QueryOptions::default()).expect("open");
    assert_eq!(reopened.profile(), GERMAN);
    assert_eq!(
        reopened.match_query("STRASSE").expect("expression"),
        "\"strasse\""
    );
}

#[test]
fn expansion_terms_follow_the_index_profile() {
    // Same corpus, two profiles: the result is a function of the index's
    // pipeline, not of a hard-coded reference tokenizer.  The seed and query
    // are chosen without foldable characters, so the two agree here; the
    // assertion fails as soon as a site falls back to `tokenize`.
    let seeds = [Seed {
        document_id: 1,
        jev_score: 1.0,
        content: "The harbour town guide.".to_owned(),
    }];
    let (_reference_directory, reference) = build(Profile::Sag2ZhV1);
    let (_multilingual_directory, multilingual) = build(GERMAN);
    assert_eq!(
        multilingual
            .expansion_terms(QUERY, &seeds, 10)
            .expect("multilingual expansion"),
        reference
            .expansion_terms(QUERY, &seeds, 10)
            .expect("reference expansion")
    );
}
