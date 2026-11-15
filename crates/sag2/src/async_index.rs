//! Async wrappers (feature `async`): retrieval and the SAG2 search loop on
//! tokio's blocking pool, with a semaphore for backpressure.
//!
//! Every entry point moves the work into [`tokio::task::spawn_blocking`], so the
//! caller's executor is never blocked: the whole SAG2 loop — BM25 retrieval,
//! expansion and the synchronous judge calls — runs on a blocking thread.  The
//! returned futures are ordinary `Send` futures; nothing here needs an I/O
//! driver or a reactor.
//!
//! # Backpressure
//!
//! [`AsyncIndex`] holds a [`tokio::sync::Semaphore`] with one permit per query
//! worker ([`CorpusIndex::query_workers`], i.e. the pool's read-only
//! connections).  Each call takes a permit before touching the pool and holds
//! it until the blocking task finishes.  More in-flight calls than workers only
//! queue on the pool's mutexes, so the semaphore is the natural concurrency
//! cap: it turns overload into async waiting instead of thread-pool growth.
//!
//! # Why `Judge` stays synchronous
//!
//! [`Judge::score`] is synchronous by contract, and it must stay that way: the
//! search loop calls it between retrieval steps, holds no async state, and the
//! crate must not force every consumer onto one executor.  A judge that needs
//! to await (an HTTP provider, say) wraps itself: inside a multi-thread tokio
//! runtime it can use `tokio::task::block_in_place` plus
//! [`tokio::runtime::Handle::block_on`], or it can drive its own runtime.  That
//! keeps the async decision in the layer that actually performs I/O, and it is
//! why this module can run the whole loop on one blocking thread.
//!
//! ```no_run
//! # #[cfg(feature = "tiktoken")]
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use std::sync::Arc;
//! use sag2::{AsyncIndex, BatchedJudge, Judge, OracleJudge, Profile, SearchOptions};
//!
//! let index = AsyncIndex::open("golden/indexes/C0.sqlite", Profile::Sag2ZhV1).await?;
//! let judge: Arc<dyn Judge> = Arc::new(BatchedJudge::new(OracleJudge::new()));
//! let result = index
//!     .search(judge, "anchor birthplace depicts", SearchOptions::default())
//!     .await?;
//! println!("{} candidates", result.candidates.len());
//! # Ok(())
//! # }
//! # fn main() {}
//! ```

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::{
    open_existing, search as search_sync, CorpusIndex, ExpansionError, ExpansionTerm, Hit, Judge,
    OpenError, Profile, QueryError, SearchError, SearchOptions, SearchResult, Seed,
};

/// A failure of the async wrappers.
#[derive(Debug)]
pub enum AsyncError {
    /// Opening the index file failed.
    Open(OpenError),
    /// The retrieval call failed.
    Query(QueryError),
    /// The expansion call failed.
    Expansion(ExpansionError),
    /// The SAG2 search loop failed.
    Search(SearchError),
    /// The blocking task panicked, or the runtime is shutting down.
    Cancelled,
}

impl std::fmt::Display for AsyncError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsyncError::Open(error) => write!(formatter, "{error}"),
            AsyncError::Query(error) => write!(formatter, "{error}"),
            AsyncError::Expansion(error) => write!(formatter, "{error}"),
            AsyncError::Search(error) => write!(formatter, "{error}"),
            AsyncError::Cancelled => write!(formatter, "the blocking SAG2 task was cancelled"),
        }
    }
}

impl std::error::Error for AsyncError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AsyncError::Open(error) => Some(error),
            AsyncError::Query(error) => Some(error),
            AsyncError::Expansion(error) => Some(error),
            AsyncError::Search(error) => Some(error),
            AsyncError::Cancelled => None,
        }
    }
}

impl From<OpenError> for AsyncError {
    fn from(error: OpenError) -> Self {
        AsyncError::Open(error)
    }
}

impl From<QueryError> for AsyncError {
    fn from(error: QueryError) -> Self {
        AsyncError::Query(error)
    }
}

impl From<ExpansionError> for AsyncError {
    fn from(error: ExpansionError) -> Self {
        AsyncError::Expansion(error)
    }
}

impl From<SearchError> for AsyncError {
    fn from(error: SearchError) -> Self {
        AsyncError::Search(error)
    }
}

/// An opened [`CorpusIndex`] shared across tasks, with bounded concurrency.
///
/// Cloning is not supported; share it behind an `Arc` instead.  Every method
/// takes a permit from the internal semaphore (one per query worker), so using
/// one `AsyncIndex` from many tasks applies backpressure instead of piling up
/// blocking-pool threads.
pub struct AsyncIndex {
    inner: Arc<CorpusIndex>,
    permits: Arc<Semaphore>,
}

impl AsyncIndex {
    /// Open an existing index file on the blocking pool.
    ///
    /// The file is validated against the signature stored in its `metadata`
    /// row and against `profile`, exactly like [`crate::open_existing`].  The
    /// permit count becomes the index's query worker count.
    pub async fn open(path: impl AsRef<Path>, profile: Profile) -> Result<Self, AsyncError> {
        let path: PathBuf = path.as_ref().to_owned();
        let index = tokio::task::spawn_blocking(move || open_existing(path, profile))
            .await
            .map_err(|_| AsyncError::Cancelled)??;
        Ok(AsyncIndex::new(index))
    }

    /// Wrap an already-open index; the permit count is its query workers.
    pub fn new(index: CorpusIndex) -> Self {
        let permits = Semaphore::new(index.query_workers());
        AsyncIndex {
            inner: Arc::new(index),
            permits: Arc::new(permits),
        }
    }

    /// The wrapped index.
    pub fn index(&self) -> &Arc<CorpusIndex> {
        &self.inner
    }

    /// Permits free right now; equal to the pool size while idle.
    pub fn available_permits(&self) -> usize {
        self.permits.available_permits()
    }

    async fn permit(&self) -> Result<tokio::sync::OwnedSemaphorePermit, AsyncError> {
        self.permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| AsyncError::Cancelled)
    }

    /// [`CorpusIndex::retrieve`] on the blocking pool.
    pub async fn retrieve(
        &self,
        expression: &str,
        excluded: BTreeSet<i64>,
        limit: usize,
        term: Option<String>,
    ) -> Result<Vec<Hit>, AsyncError> {
        let permit = self.permit().await?;
        let index = Arc::clone(&self.inner);
        let expression = expression.to_owned();
        Ok(tokio::task::spawn_blocking(move || {
            let _permit = permit;
            index.retrieve(&expression, &excluded, limit, term.as_deref())
        })
        .await
        .map_err(|_| AsyncError::Cancelled)??)
    }

    /// [`CorpusIndex::expansion_terms`] on the blocking pool.
    pub async fn expansion(
        &self,
        query: &str,
        seeds: Vec<Seed>,
        limit: usize,
    ) -> Result<Vec<ExpansionTerm>, AsyncError> {
        let permit = self.permit().await?;
        let index = Arc::clone(&self.inner);
        let query = query.to_owned();
        Ok(tokio::task::spawn_blocking(move || {
            let _permit = permit;
            index.expansion_terms(&query, &seeds, limit)
        })
        .await
        .map_err(|_| AsyncError::Cancelled)??)
    }

    /// [`crate::search`] under this index's backpressure limit.
    pub async fn search(
        &self,
        judge: Arc<dyn Judge>,
        query: impl Into<String> + Send,
        options: SearchOptions,
    ) -> Result<SearchResult, AsyncError> {
        let permit = self.permit().await?;
        let index = Arc::clone(&self.inner);
        let query = query.into();
        Ok(tokio::task::spawn_blocking(move || {
            let _permit = permit;
            search_sync(&index, judge.as_ref(), &query, &options)
        })
        .await
        .map_err(|_| AsyncError::Cancelled)??)
    }
}

/// Run the SAG2 search loop on the blocking pool.
///
/// The [`CorpusIndex`] and the owned query/options are moved into one
/// [`tokio::task::spawn_blocking`] task, so the caller's executor is never
/// blocked.  This free function has no backpressure of its own: use
/// [`AsyncIndex::search`] when many tasks share one index.
pub async fn search(
    index: Arc<CorpusIndex>,
    judge: Arc<dyn Judge>,
    query: String,
    options: SearchOptions,
) -> Result<SearchResult, AsyncError> {
    Ok(
        tokio::task::spawn_blocking(move || search_sync(&index, judge.as_ref(), &query, &options))
            .await
            .map_err(|_| AsyncError::Cancelled)??,
    )
}
