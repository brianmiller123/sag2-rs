//! Query-side tokenisation follows the profile the index was opened with.
//!
//! `Profile::MultilingualV1` shares the reference tokenizer until T3-T5 land,
//! so these assertions pin the *contract*: each expression must equal the one
//! the profile-specific builder produces.  When the pipelines diverge, a query
//! site still calling `tokenize` would fall back to `Sag2ZhV1` here and fail.

use sag2_core::{match_query, match_query_with, Profile, QueryMode};
use sag2_index::{
    open_readonly, CorpusIndex, Document, IndexOptions, QueryOptions, RebuildPolicy, Seed,
};

const QUERY: &str = "anchor birthplace depicts";

fn corpus() -> [Document<'static>; 2] {
    [
        Document {
            title: "Anchor",
            text: "Anchor birthplace depicts the town.",
            path: None,
            idx: None,
        },
        Document {
            title: "Guide",
            text: "The guide describes a harbour town.",
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
            rebuild: RebuildPolicy::Always,
        },
    )
    .expect("build");
    (directory, index)
}

#[test]
fn queries_use_the_index_profile_pipeline() {
    let (_directory, index) = build(Profile::MultilingualV1);
    assert_eq!(index.profile(), Profile::MultilingualV1);

    for mode in [QueryMode::Lexical, QueryMode::Phrase3] {
        assert_eq!(
            index.match_query_with(mode, QUERY).expect("expression"),
            match_query_with(mode, Profile::MultilingualV1, QUERY),
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
fn a_read_only_open_carries_the_index_profile() {
    let (directory, index) = build(Profile::MultilingualV1);
    let signature = index.signature().to_owned();
    let path = directory.path().join("index.sqlite");
    drop(index);

    let reopened = open_readonly(
        &path,
        &signature,
        Profile::MultilingualV1,
        QueryOptions::default(),
    )
    .expect("open");
    assert_eq!(reopened.profile(), Profile::MultilingualV1);
    assert_eq!(
        reopened
            .match_query_with(QueryMode::Phrase3, QUERY)
            .expect("expression"),
        match_query_with(QueryMode::Phrase3, Profile::MultilingualV1, QUERY)
    );
}

#[test]
fn expansion_terms_follow_the_index_profile() {
    // Same corpus, two profiles: the result is a function of the index's
    // pipeline, not of a hard-coded reference tokenizer.
    let seeds = [Seed {
        document_id: 1,
        jev_score: 1.0,
        content: "Alice Paul was born in Anchor.".to_owned(),
    }];
    let (_reference_directory, reference) = build(Profile::Sag2ZhV1);
    let (_multilingual_directory, multilingual) = build(Profile::MultilingualV1);
    assert_eq!(
        multilingual
            .expansion_terms(QUERY, &seeds, 10)
            .expect("multilingual expansion"),
        reference
            .expansion_terms(QUERY, &seeds, 10)
            .expect("reference expansion")
    );
}
