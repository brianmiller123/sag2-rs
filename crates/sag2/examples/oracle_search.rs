//! Full SAG2 search with the deterministic `OracleJudge` over the committed
//! `golden/indexes/C0.sqlite` fixture.
//!
//! ```sh
//! cargo run --example oracle_search
//! ```
//!
//! `OracleJudge` is the reference's fixture transport (`tests/test_sag2.py`):
//! it scores by document title, so no network is involved.  Swap it for a
//! `BatchedJudge<T>` once a real `Transport` exists (P3).

use std::path::PathBuf;

use sag2::{search, BatchedJudge, OracleJudge, Profile, SearchOptions, TraceEntry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/indexes/C0.sqlite");
    let index = sag2::open_existing(&path, Profile::Sag2ZhV1)?;

    let judge = BatchedJudge::new(OracleJudge::new());
    let result = search(
        &index,
        &judge,
        "anchor birthplace depicts",
        &SearchOptions::default(),
    )?;

    println!("algorithm: {} ({} ms)", result.algorithm, result.elapsed_ms);
    for (rank, candidate) in result.candidates.iter().take(5).enumerate() {
        println!(
            "rank {rank}: id={} jev_score={:.3} title={:?}",
            candidate.hit.id, candidate.jev_score, candidate.hit.title
        );
    }
    let stages: Vec<&str> = result
        .trace
        .iter()
        .map(|entry| match entry {
            TraceEntry::Retrieval(retrieval) => retrieval.stage,
            TraceEntry::Judge(_) => "jev",
        })
        .collect();
    println!("trace: {stages:?}");
    println!(
        "usage: {} requests, {} judgments, {} input tokens",
        result.usage.requests, result.usage.judgments, result.usage.input_tokens
    );
    if result.candidates.is_empty() {
        return Err("the fixture index returned no candidates".into());
    }
    Ok(())
}
