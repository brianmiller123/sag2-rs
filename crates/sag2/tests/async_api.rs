//! The async wrappers must agree with the synchronous API and must apply
//! backpressure at the index's query-worker count.

#![cfg(all(feature = "async", feature = "tiktoken"))]

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use futures_core::future::Future;
use sag2::{
    AsyncIndex, BatchedJudge, CorpusIndex, Document, IndexOptions, Judge, OracleJudge, Profile,
    RebuildPolicy, SearchOptions,
};

fn corpus() -> [Document<'static>; 5] {
    [
        Document {
            title: "Anchor",
            text: "Anchor birthplace depicts the town.",
            path: None,
            idx: None,
        },
        Document {
            title: "Multiple",
            text: "Multiple relates to Anchor directly.",
            path: None,
            idx: None,
        },
        Document {
            title: "Bridge",
            text: "Bridge connects Multiple to the town.",
            path: None,
            idx: None,
        },
        Document {
            title: "High",
            text: "High value passage about the town.",
            path: None,
            idx: None,
        },
        Document {
            title: "Series",
            text: "Series orders the remaining passages.",
            path: None,
            idx: None,
        },
    ]
}

fn open_index(directory: &Path) -> CorpusIndex {
    let documents = corpus();
    let (index, _) = CorpusIndex::open(
        &directory.join("async.sqlite"),
        &documents,
        "fixture",
        IndexOptions {
            chunk_chars: 2200,
            index_workers: 1,
            source_dir: None,
            profile: Profile::Sag2ZhV1,
            dictionary_dir: None,
            rebuild: RebuildPolicy::Always,
        },
    )
    .expect("build the fixture index");
    index
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("build a current-thread runtime")
}

/// The futures are moved into tokio tasks in real use; assert the bound at
/// compile time.
fn assert_send<F: Future + Send>(future: F) -> F {
    future
}

#[test]
fn async_search_matches_the_synchronous_loop() {
    let directory = tempfile::tempdir().expect("temp dir");
    let index = open_index(directory.path());
    let expression = index
        .match_query("anchor birthplace depicts")
        .expect("query");
    let excluded = BTreeSet::new();
    let sync_hits = index
        .retrieve(&expression, &excluded, 5, None)
        .expect("sync retrieve");
    let sync_result = sag2::search(
        &index,
        &BatchedJudge::new(OracleJudge::new()),
        "anchor birthplace depicts",
        &SearchOptions::default(),
    )
    .expect("sync search");

    let async_index = AsyncIndex::new(open_index(directory.path()));
    assert_eq!(
        async_index.available_permits(),
        async_index.index().query_workers()
    );

    runtime().block_on(async {
        let hits = assert_send(async_index.retrieve(&expression, excluded, 5, None))
            .await
            .expect("async retrieve");
        assert_eq!(hits, sync_hits);

        let judge: Arc<dyn Judge> = Arc::new(BatchedJudge::new(OracleJudge::new()));
        let result = assert_send(async_index.search(
            judge,
            "anchor birthplace depicts",
            SearchOptions::default(),
        ))
        .await
        .expect("async search");
        assert_eq!(result.candidates, sync_result.candidates);
        assert_eq!(result.baseline, sync_result.baseline);
        assert_eq!(result.trace, sync_result.trace);
        assert_eq!(result.usage, sync_result.usage);
        assert_eq!(result.algorithm, sync_result.algorithm);
    });
}

#[test]
fn async_open_reads_an_existing_index() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("async.sqlite");
    let _ = open_index(directory.path());

    runtime().block_on(async {
        let index = AsyncIndex::open(&path, Profile::Sag2ZhV1)
            .await
            .expect("async open");
        let expression = index.index().match_query("anchor").expect("query");
        let hits = assert_send(index.retrieve(&expression, BTreeSet::new(), 3, None))
            .await
            .expect("async retrieve");
        assert!(!hits.is_empty(), "the fixture index should match anchor");
    });
}

#[test]
fn async_expansion_matches_the_synchronous_call() {
    let directory = tempfile::tempdir().expect("temp dir");
    let index = open_index(directory.path());
    let seeds = vec![sag2::Seed {
        document_id: 1,
        jev_score: 1.0,
        content: "Anchor birthplace depicts the town.".to_owned(),
    }];
    let sync_terms = index
        .expansion_terms("anchor birthplace", &seeds, 20)
        .expect("sync expansion");

    let async_index = AsyncIndex::new(open_index(directory.path()));
    let terms = runtime()
        .block_on(assert_send(async_index.expansion(
            "anchor birthplace",
            seeds.clone(),
            20,
        )))
        .expect("async expansion");
    assert_eq!(terms, sync_terms);
}
