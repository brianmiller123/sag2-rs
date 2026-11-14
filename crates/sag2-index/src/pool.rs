//! Read-only connection pool with the reference LRU and epoch semantics
//! (`sag2.py:466-549`, `sag2.py:551-565`).
//!
//! Python keeps one reader connection per worker thread, each with its own
//! `OrderedDict` cache and its own `(PRAGMA data_version, total_changes)` epoch.
//! The Rust pool keeps the same shape: `workers` independent read-only
//! connections, each with its own LRU, dispatched round-robin.  Because the
//! cache only memoises a pure function of the query, `workers = 1` and
//! `workers = 8` must return identical hits (tested in `tests/gate_a.rs`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use rusqlite::{Connection, OpenFlags, OptionalExtension};

use crate::open::OpenError;
use crate::py312;
use crate::retrieve::{
    decode_idfs, materialize, pruned_sql, quote_term, quoted_phrases, run_query, top_idf_indices,
    Hit, QueryError, BASE_SQL, SAG_STATS_SQL,
};

/// `sag2.py:35` `QUERY_WORKERS`.
pub const QUERY_WORKERS: usize = 8;

/// `sag2.py:36` `QUERY_CACHE_ENTRIES`.
pub const QUERY_CACHE_ENTRIES: usize = 256;

/// Read-path parallelism and cache size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryOptions {
    /// Number of read-only connections; clamped to at least one.
    pub workers: usize,
    /// `QUERY_CACHE_ENTRIES` per connection.
    pub cache_entries: usize,
}

impl Default for QueryOptions {
    fn default() -> Self {
        QueryOptions {
            workers: QUERY_WORKERS,
            cache_entries: QUERY_CACHE_ENTRIES,
        }
    }
}

/// `(expression, sorted excluded ids, limit, term)` (`sag2.py:499`).
type CacheKey = (String, Vec<i64>, usize, Option<String>);

struct QueryCache {
    capacity: usize,
    entries: Vec<(CacheKey, Vec<(i64, f64)>)>,
}

impl QueryCache {
    fn new(capacity: usize) -> Self {
        QueryCache {
            capacity,
            entries: Vec::new(),
        }
    }

    fn take(&mut self, key: &CacheKey) -> Option<Vec<(i64, f64)>> {
        let position = self
            .entries
            .iter()
            .position(|(candidate, _)| candidate == key)?;
        Some(self.entries.remove(position).1)
    }

    fn put(&mut self, key: CacheKey, value: Vec<(i64, f64)>) {
        if self.capacity == 0 {
            return;
        }
        self.entries.retain(|(existing, _)| existing != &key);
        self.entries.push((key, value));
        while self.entries.len() > self.capacity {
            self.entries.remove(0);
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

struct Reader {
    db: Connection,
    epoch: Option<(i64, u64)>,
    cache: QueryCache,
    rank_stats: HashMap<String, Vec<u8>>,
}

impl Reader {
    fn open(path: &Path, cache_entries: usize) -> Result<Self, OpenError> {
        let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        // `_initialize_queries` (`sag2.py:424-425`) for every connection.
        db.execute_batch("PRAGMA mmap_size=2147483648")?;
        register_rank(&db)?;
        Ok(Reader {
            db,
            epoch: None,
            cache: QueryCache::new(cache_entries),
            rank_stats: HashMap::new(),
        })
    }

    fn data_version(&self) -> Result<i64, rusqlite::Error> {
        self.db
            .query_row("PRAGMA data_version", [], |row| row.get(0))
    }

    fn retrieve(
        &mut self,
        expression: &str,
        excluded: &[i64],
        excluded_json: &str,
        limit: usize,
        term: Option<&str>,
    ) -> Result<Vec<Hit>, QueryError> {
        // `sag2.py:454-464`: pin one read snapshot for the statistics lookup,
        // the ranking query and the `chunks` materialisation.
        let own_transaction = self.db.is_autocommit();
        if own_transaction {
            self.db.execute_batch("BEGIN")?;
        }
        let result = self.retrieve_snapshot(expression, excluded, excluded_json, limit, term);
        if own_transaction && !self.db.is_autocommit() {
            self.db.execute_batch("ROLLBACK")?;
        }
        result
    }

    fn retrieve_snapshot(
        &mut self,
        expression: &str,
        excluded: &[i64],
        excluded_json: &str,
        limit: usize,
        term: Option<&str>,
    ) -> Result<Vec<Hit>, QueryError> {
        // `data_version` pins the read snapshot and detects writes by other
        // connections (`sag2.py:470-476`).
        let epoch = (self.data_version()?, self.db.total_changes());
        if self.epoch != Some(epoch) {
            self.cache.clear();
            if self.epoch.is_some() {
                self.rank_stats.clear();
            }
            self.epoch = Some(epoch);
        }

        let key: CacheKey = (
            expression.to_owned(),
            excluded.to_vec(),
            limit,
            term.map(str::to_owned),
        );

        let ranked = if let Some(cached) = self.cache.take(&key) {
            self.cache.put(key, cached.clone());
            cached
        } else {
            let ranked = self.rank(expression, excluded_json, limit, term)?;
            self.cache.put(key, ranked.clone());
            ranked
        };

        materialize(&self.db, &ranked)
    }

    fn rank(
        &mut self,
        expression: &str,
        excluded_json: &str,
        limit: usize,
        term: Option<&str>,
    ) -> Result<Vec<(i64, f64)>, QueryError> {
        if !self.rank_stats.contains_key(expression) {
            let blob: Option<Option<Vec<u8>>> = self
                .db
                .query_row(SAG_STATS_SQL, [expression], |row| row.get(0))
                .optional()?;
            self.rank_stats
                .insert(expression.to_owned(), blob.flatten().unwrap_or_default());
        }
        let stats = self
            .rank_stats
            .get(expression)
            .cloned()
            .unwrap_or_default();

        let query = match term {
            Some(term) => format!("({expression}) AND {}", quote_term(term)),
            None => expression.to_owned(),
        };

        let phrases = quoted_phrases(expression);
        let idfs = decode_idfs(&stats)?;
        if term.is_none() && limit >= 100 && phrases.len() >= 8 && idfs.len() == phrases.len() {
            let selected = top_idf_indices(&idfs);
            let subset = selected
                .iter()
                .map(|&index| format!("\"{}\"", phrases[index]))
                .collect::<Vec<_>>()
                .join(" OR ");
            let possible = run_query(
                &self.db,
                &pruned_sql(),
                &stats,
                &query,
                excluded_json,
                limit,
                Some(&subset),
            )?;
            // `sum()` here is the compensated Python 3.12 sum (`sag2.py:514`).
            let bound = 2.2
                * py312::sum(
                    idfs.iter()
                        .enumerate()
                        .filter(|(index, _)| !selected.contains(index))
                        .map(|(_, idf)| *idf),
                );
            if possible.len() == limit
                && possible
                    .last()
                    .is_some_and(|(_, score)| -*score > bound)
            {
                return Ok(possible);
            }
        }
        Ok(run_query(
            &self.db,
            BASE_SQL,
            &stats,
            &query,
            excluded_json,
            limit,
            None,
        )?)
    }
}

/// `load_rank_extension` (`sag2.py:424-425`) for one connection.
#[allow(unsafe_code)]
fn register_rank(db: &Connection) -> Result<(), OpenError> {
    // SAFETY: the handle is valid for as long as `db` is alive, and
    // `sag_register_rank` only prepares one statement on that connection.
    let handle = unsafe { db.handle() } as *mut sag2_rank_sys::sqlite3;
    // SAFETY: same live handle as above; not used after `db` is dropped.
    unsafe { sag2_rank_sys::register(handle) }
        .map_err(|error| OpenError::RankExtension(error.code()))
}

/// A fixed-size pool of read-only readers.
pub(crate) struct QueryPool {
    readers: Vec<Mutex<Reader>>,
    next: AtomicUsize,
}

impl QueryPool {
    pub(crate) fn open(path: &Path, options: QueryOptions) -> Result<Self, OpenError> {
        let workers = options.workers.max(1);
        let mut readers = Vec::with_capacity(workers);
        for _ in 0..workers {
            readers.push(Mutex::new(Reader::open(path, options.cache_entries)?));
        }
        Ok(QueryPool {
            readers,
            next: AtomicUsize::new(0),
        })
    }

    pub(crate) fn retrieve(
        &self,
        expression: &str,
        excluded: &[i64],
        excluded_json: &str,
        limit: usize,
        term: Option<&str>,
    ) -> Result<Vec<Hit>, QueryError> {
        let slot = self.next.fetch_add(1, Ordering::Relaxed) % self.readers.len();
        let mut reader = self.readers[slot]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        reader.retrieve(expression, excluded, excluded_json, limit, term)
    }
}
