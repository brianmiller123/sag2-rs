//! BM25-only recall: no judge, no payload builder, no async runtime.
//!
//! ```sh
//! cargo run --example bm25_only
//! ```
//!
//! Reads the committed `golden/indexes/C0.sqlite` fixture, so it needs no
//! network and no Python.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/indexes/C0.sqlite");
    let index = sag2::open_existing(&path)?;

    let expression = index.match_query("anchor birthplace depicts")?;
    let hits = index.retrieve(&expression, &Default::default(), 5, None)?;

    println!("index: {}", index.path().display());
    println!("signature: {}", index.signature());
    println!("expression: {expression}");
    for hit in &hits {
        println!(
            "id={} document={} score={:.6} title={:?}",
            hit.id, hit.document_id, hit.score, hit.title
        );
    }
    if hits.is_empty() {
        return Err("the fixture index returned no hits".into());
    }
    Ok(())
}
