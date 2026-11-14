//! SAG2 orchestration: `rank_dpp`, the judge payload builder and
//! `search_sag2` (design §1.3, §8, §9 R5).
//!
//! This is the facade design §2.2 places between retrieval and the transport:
//! it knows the retrieval API and the judge seam, but not SQLite, tokio or
//! HTTP.  `search` takes `&dyn Judge`; `BatchedJudge<T: Transport>` owns the
//! payload/budget behaviour and a transport supplies the responses.
//!
//! Three properties are contractual and gate-covered:
//!
//! * **Compensated summation.** `rank_dpp` uses CPython 3.12's `sum()` in the
//!   norm, the kernel dot product and the projection (`sag2.py:133/142/163`);
//!   a naive `+=` drifts the residuals and reorders the DPP head (design §9 R5).
//! * **Insertion order.** Every map-like structure in a numeric path is an
//!   ordered vector: term vectors keep the first occurrence of each term, and
//!   `retained`/`carried`/`protected` keep Python's dict-update order.
//! * **Payload bytes.** `compact_json(state)` is computed once and carried as
//!   a string inside the payload (`sag2.py:849`), so double serialisation pins
//!   the float format and key order (design §1.3 C-1).

#![deny(unsafe_code)]

mod dpp;
mod judge;
mod search;

pub use dpp::{rank_dpp, DppError};
pub use judge::{
    Answer, BatchDetail, BatchedJudge, Criteria, Instructions, Judge, JudgeError, OracleJudge,
    Payload, Question, Response, RoundUsage, Transport, Usage, UsageBatch,
};
pub use sag2_index::{CorpusIndex, ExpansionQuery, ExpansionTerm, Hit};
pub use search::{
    search, Candidate, JudgeTrace, RetrievalTrace, SearchContext, SearchError, SearchOptions,
    SearchResult, TraceEntry, TraceScore,
};

/// The judge model the reference client is constructed with
/// (`tests/test_sag2.py:35`: `"jev-latest"`).
pub const DEFAULT_MODEL: &str = "jev-latest";

/// `sag2.py:935-940` defaults.
pub const DEFAULT_CANDIDATE_K: usize = 100;
pub const DEFAULT_SEED_K: usize = 10;
pub const DEFAULT_MAX_HOPS: u8 = 2;
pub const DEFAULT_EXPANSION_TERM_LIMIT: usize = 20;
