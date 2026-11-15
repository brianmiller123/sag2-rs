//! SAG2 retrieval core as a library: BM25 recall from a SAG2 index, and the
//! SAG2 search loop with its judge seam.
//!
//! This is the facade design §2.2 places between retrieval and the transport:
//! it knows the retrieval API and the judge seam, but not SQLite, tokio or
//! HTTP.  [`search`] takes `&dyn Judge`; [`BatchedJudge<T: Transport>`] owns the
//! payload/budget behaviour and a transport supplies the responses.
//!
//! # Two paths
//!
//! **BM25-only recall** needs no judge and no transport.  This build of the
//! crate (default features off) is enough:
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // An index built by this workspace or by the Python implementation.
//! let index = sag2::open_existing("golden/indexes/C0.sqlite", sag2::Profile::Sag2ZhV1)?;
//! let expression = index.match_query("anchor birthplace depicts")?;
//! let hits = index.retrieve(&expression, &Default::default(), 5, None)?;
//! for hit in hits {
//!     println!("{} {} {:.6}", hit.id, hit.title, hit.score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! **Full SAG2 search** adds a [`Judge`].  [`OracleJudge`] is the deterministic
//! fixture judge the gates replay; a real deployment implements [`Judge`] or
//! [`Transport`] (P3 adds HTTP):
//!
//! ```no_run
//! # #[cfg(feature = "tiktoken")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use sag2::{search, BatchedJudge, OracleJudge, SearchOptions};
//!
//! let index = sag2::open_existing("golden/indexes/C0.sqlite", sag2::Profile::Sag2ZhV1)?;
//! let judge = BatchedJudge::new(OracleJudge::new());
//! let result = search(
//!     &index,
//!     &judge,
//!     "anchor birthplace depicts",
//!     &SearchOptions::default(),
//! )?;
//! for candidate in result.candidates.iter().take(5) {
//!     println!("{} {} {}", candidate.hit.id, candidate.hit.title, candidate.jev_score);
//! }
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "tiktoken"))]
//! # fn main() {}
//! ```
//!
//! # Features
//!
//! | feature | default | enables | without it |
//! |---|---|---|---|
//! | `tiktoken` | on | the payload builder and its 32000/64000-token packing: [`BatchedJudge`], [`Payload`], [`Transport`], [`OracleJudge`], the budget constants | only a custom [`Judge`] can score; retrieval and [`search`] still work |
//! | `async` | off | `AsyncIndex` and `async_index::search`, which run retrieval and the search loop on tokio's blocking pool with a backpressure semaphore | nothing tokio-shaped is compiled or linked |
//!
//! `tiktoken` gates only payload/budget construction.  The accounting types
//! ([`BatchDetail`], [`RoundUsage`], [`UsageBatch`], [`Usage`]) stay available
//! without it because [`Judge::score`] and [`SearchResult`] return them.
//!
//! # Three properties are contractual and gate-covered
//!
//! * **Compensated summation.** [`rank_dpp`] uses CPython 3.12's `sum()` in the
//!   norm, the kernel dot product and the projection (`sag2.py:133/142/163`);
//!   a naive `+=` drifts the residuals and reorders the DPP head (design §9 R5).
//! * **Insertion order.** Every map-like structure in a numeric path is an
//!   ordered vector: term vectors keep the first occurrence of each term, and
//!   `retained`/`carried`/`protected` keep Python's dict-update order.
//! * **Payload bytes.** `compact_json(state)` is computed once and carried as
//!   a string inside the payload (`sag2.py:849`), so double serialisation pins
//!   the float format and key order (design §1.3 C-1).

#![deny(unsafe_code)]
#![deny(missing_docs)]

use std::path::Path;

mod dpp;
mod judge;
mod search;

#[cfg(feature = "async")]
pub mod async_index;

pub use dpp::{rank_dpp, DppError};
#[cfg(feature = "tiktoken")]
pub use judge::{
    Answer, BatchedJudge, Criteria, Instructions, OracleJudge, Payload, Question, Response,
    Transport, BATCH_BUDGET, CONTEXT_BUDGET, CRITERION, EVIDENCE_SCORE_NOTE, HISTORY_NOTE,
    QUESTION_OVERHEAD, STATE_OVERHEAD,
};
pub use judge::{BatchDetail, Judge, JudgeError, RoundUsage, Usage, UsageBatch};
pub use sag2_index::{
    inspect, open_readonly, BuildError, CorpusIndex, Document, ExpansionError, ExpansionHit,
    ExpansionKind, ExpansionQuery, ExpansionTerm, Hit, IndexOptions, OpenAction, OpenError,
    OpenOutcome, Profile, QueryError, QueryOptions, RebuildPolicy, RebuildReason, Seed, Selected,
    SqliteError, Statistics, StatisticsError,
};
/// The facade's ordered JSON object, used where Python has a dict.
pub use sag2_json::Object;
/// The facade's ordered JSON value: the type of [`SearchContext`]'s history
/// fields and of [`BatchDetail::failed_attempts`].  Re-exported so a consumer
/// never depends on `sag2-json` directly.
pub use sag2_json::Value;
pub use search::{
    search, Candidate, JudgeTrace, RetrievalTrace, SearchContext, SearchError, SearchOptions,
    SearchResult, TraceEntry, TraceScore,
};

#[cfg(feature = "async")]
pub use async_index::{AsyncError, AsyncIndex};

/// The judge model the reference client is constructed with
/// (`tests/test_sag2.py:35`: `"jev-latest"`).
pub const DEFAULT_MODEL: &str = "jev-latest";

/// `sag2.py:935-940` default candidate count (`candidate_k`).
pub const DEFAULT_CANDIDATE_K: usize = 100;
/// `sag2.py:935-940` default expansion seed count (`seed_k`).
pub const DEFAULT_SEED_K: usize = 10;
/// `sag2.py:935-940` default number of expansion hops.
pub const DEFAULT_MAX_HOPS: u8 = 2;
/// `sag2.py:935-940` default number of expansion terms per hop.
pub const DEFAULT_EXPANSION_TERM_LIMIT: usize = 20;

/// Open an existing index file read-only, validating it against the signature
/// stored in its `metadata` row and against `profile`.
///
/// This is the entry point for an index this process did not build — the golden
/// fixtures, or an index produced by the Python implementation.  The stored
/// signature is checked, and so is `rust_index_metadata` when the file has it;
/// an index built under another [`Profile`] is refused rather than read with
/// the wrong token stream.  Use [`CorpusIndex::open`] when you hold the corpus
/// and want the build/reuse decision table.
pub fn open_existing(path: impl AsRef<Path>, profile: Profile) -> Result<CorpusIndex, OpenError> {
    let path = path.as_ref();
    let existing = inspect(path)?;
    if !existing.file_exists {
        return Err(OpenError::MissingDatabase(path.to_owned()));
    }
    let signature = existing.signature.ok_or(OpenError::MissingSignature)?;
    open_readonly(path, &signature, profile, QueryOptions::default())
}
