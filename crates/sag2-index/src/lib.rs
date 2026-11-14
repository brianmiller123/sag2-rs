//! Read-only access to a SAG2 FTS5 index (design §3, Gate A).
//!
//! This is the only crate in the workspace that depends on `rusqlite`, so the
//! `links = "sqlite3"` constraint and the `Connection` lifetime live in one
//! place (design §2.3).  P0 covers the read path: the corpus signature and the
//! §3.4 open decision table, the cached statistics, and the `retrieve` SQL with
//! its pruning, LRU and epoch semantics.  P1 adds [`CorpusIndex::open`]'s
//! building branch (design §1.4 D1-D3): the Python schema, the signature, the
//! cached seven-key statistics and the `rust_index_metadata` provenance rows.

#![deny(unsafe_code)]

mod build;
mod expand;
mod open;
mod pool;
mod py312;
mod retrieve;
mod signature;
mod statistics;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use sag2_json::{compact_json, Value};

pub use build::{BuildError, Document};
pub use expand::{
    ExpansionError, ExpansionHit, ExpansionKind, ExpansionQuery, ExpansionTerm, Seed, Selected,
};
pub use open::{
    decide, inspect, open, open_readonly, Decision, ExistingIndex, IndexOptions, OpenAction,
    OpenError, OpenOutcome, RebuildPolicy, RebuildReason, RuntimeVersions, FORMAT_VERSION,
    FORMAT_VERSION_STR, JIEBA_DICT_SHA256, RUNTIME_VERSIONS, SAG_RANK_SHA256, SQLITE_VERSION,
    TOKENIZER_VERSION, UCA_KEYS_SHA256, UCA_VERSION, UNICODE_VERSION,
};
pub use pool::{register_rank, QueryOptions, QUERY_CACHE_ENTRIES, QUERY_WORKERS};
pub use retrieve::{Hit, QueryError, BASE_SQL, SAG_STATS_SQL};
pub use sag2_core::DEFAULT_CHUNK_CHARS;
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
    /// `self.frequencies`, the mutable frequency memo (`sag2.py:551-583`).
    frequencies: Mutex<BTreeMap<String, i64>>,
    /// `self.name_pattern`, compiled on first use from the corpus aliases.
    name_pattern: OnceLock<Option<expand::NamePattern>>,
}

impl CorpusIndex {
    pub(crate) fn new(
        path: PathBuf,
        signature: String,
        statistics: Statistics,
        pool: pool::QueryPool,
    ) -> Self {
        let frequencies = statistics.frequencies.clone();
        CorpusIndex {
            path,
            signature,
            statistics,
            pool,
            frequencies: Mutex::new(frequencies),
            name_pattern: OnceLock::new(),
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
