//! Read-only access to a SAG2 FTS5 index (design §3, Gate A).
//!
//! This is the only crate in the workspace that depends on `rusqlite`, so the
//! `links = "sqlite3"` constraint and the `Connection` lifetime live in one
//! place (design §2.3).  P0 covers the read path: the corpus signature and the
//! §3.4 open decision table, the cached statistics, and the `retrieve` SQL with
//! its pruning, LRU and epoch semantics.  Building an index needs the tokenizer
//! and the chunker, which P1 adds.

#![deny(unsafe_code)]

mod open;
mod pool;
mod py312;
mod retrieve;
mod signature;
mod statistics;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sag2_json::{compact_json, Value};

pub use open::{
    decide, inspect, open, open_readonly, Decision, ExistingIndex, IndexOptions, OpenAction,
    OpenError, OpenOutcome, RebuildPolicy, RebuildReason, RuntimeVersions, VersionCheck,
    DEFAULT_CHUNK_CHARS, FORMAT_VERSION, PENDING_BUILD_RUNTIME, RUNTIME_VERSIONS,
};
pub use pool::{QueryOptions, QUERY_CACHE_ENTRIES, QUERY_WORKERS};
pub use retrieve::{Hit, QueryError};
pub use signature::{changed_field, signature, CHUNK_POLICY, INDEX_SCHEMA};
pub use statistics::{Statistics, StatisticsError};

impl std::fmt::Debug for CorpusIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CorpusIndex")
            .field("path", &self.path)
            .field("signature", &self.signature)
            .field("total", &self.statistics.total)
            .finish_non_exhaustive()
    }
}

/// The SQLite version compiled into this build (`rusqlite`'s bundled
/// amalgamation).  Gate A records it so a mismatch with the golden is visible.
pub fn sqlite_version() -> &'static str {
    rusqlite::version()
}

/// One opened index: statistics, a read-only connection pool and the signature
/// it was validated against (`sag2.py:174-206` `CorpusIndex`).
pub struct CorpusIndex {
    path: PathBuf,
    signature: String,
    statistics: Statistics,
    pool: pool::QueryPool,
}

impl CorpusIndex {
    pub(crate) fn new(
        path: PathBuf,
        signature: String,
        statistics: Statistics,
        pool: pool::QueryPool,
    ) -> Self {
        CorpusIndex {
            path,
            signature,
            statistics,
            pool,
        }
    }

    /// The file this index reads.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The validated `metadata.signature`.
    pub fn signature(&self) -> &str {
        &self.signature
    }

    /// Corpus statistics loaded from `index_statistics`.
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// `CorpusIndex.retrieve` (`sag2.py:454-464`).
    ///
    /// `excluded` is a set, so it is always ascending and de-duplicated before
    /// it reaches SQLite as a single `json_each(?)` parameter (design §13.5).
    pub fn retrieve(
        &self,
        expression: &str,
        excluded: &BTreeSet<i64>,
        limit: usize,
        term: Option<&str>,
    ) -> Result<Vec<Hit>, QueryError> {
        let excluded_ids: Vec<i64> = excluded.iter().copied().collect();
        let excluded_json = compact_json(&Value::Array(
            excluded_ids.iter().copied().map(Value::Int).collect(),
        ));
        self.pool
            .retrieve(expression, &excluded_ids, &excluded_json, limit, term)
    }
}
