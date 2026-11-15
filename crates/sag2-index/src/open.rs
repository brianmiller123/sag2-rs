//! Index open: the `metadata.signature` check and the decision table
//! (`sag2.py:197-320`, design §3.4).
//!
//! The table separates two failure classes that must never be conflated:
//!
//! * **corpus signature changed** — this index was not built for this corpus,
//!   so rebuilding is allowed (and always reported through [`OpenOutcome`]);
//! * **provenance or version mismatch** — this index came from another
//!   implementation or format, so it is rejected and never rebuilt, because a
//!   rebuild would hide the inconsistency.
//!
//! The free [`open`] entry point has no corpus, so a decision that would build
//! returns [`OpenError::BuildUnavailable`]; [`CorpusIndex::open`] carries the
//! corpus and performs the build (§3.4).
//!
//! The analysis profile follows the same fail-closed rule: the profile a caller
//! reads with must match `rust_index_metadata.analysis`, because reading an
//! index with another tokenizer would mix two token streams.  A row without the
//! key predates profiles and is the reference pipeline.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags, OptionalExtension};

use sag2_core::{Profile, DEFAULT_CHUNK_CHARS};

use crate::build::{self, BuildError, Document};
use crate::pool::{QueryOptions, QueryPool};
use crate::signature::{changed_field, signature, CHUNK_POLICY};
use crate::statistics::{self, StatisticsError};
use crate::CorpusIndex;

/// `rust_index_metadata.format_version` this build reads and writes.
pub const FORMAT_VERSION: i64 = 1;

/// [`FORMAT_VERSION`] in the string form the provenance table stores.
pub const FORMAT_VERSION_STR: &str = "1";

/// Design §3.3 example: every feature that changes the token stream is in the
/// version, `cjk` being the compatibility switch (design §2.4).
pub const TOKENIZER_VERSION: &str = "jieba-0.42.1/re_skip-1/nfkc-15.0.0/lower-15.0.0/cjk-on";

/// `data/dict.txt`, the certified jieba 0.42.1 dictionary
/// (`crates/sag2-core/data/PROVENANCE.md`).
pub const JIEBA_DICT_SHA256: &str =
    "7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8";

/// DUCET 9.0.0 (`allkeys-9.0.0.txt` shipped with pyuca 1.2); P2 vendors the
/// same file for the collation.
pub const UCA_KEYS_SHA256: &str =
    "0633f4520c99f249b0c53aa1442cd2521702041fb00a32df944fec13c9da3ed5";

/// `crates/sag2-rank-sys/vendor/sag_rank.c` (design §9 R15).
pub const SAG_RANK_SHA256: &str =
    "a87ab48d4c6fecd57f9d9b4b86109a20917451766942a88d1a040f593025726b";

/// `unicodedata.unidata_version` of the reference interpreter.
pub const UNICODE_VERSION: &str = "15.0.0";

/// `pyuca.Collator.UCA_VERSION`.
pub const UCA_VERSION: &str = "9.0.0";

/// The SQLite version this build's amalgamation reports.
pub const SQLITE_VERSION: &str = "3.50.2";

/// The `rust_index_metadata` values the running build certifies (design §3.3).
#[derive(Debug, Clone, Copy)]
pub struct RuntimeVersions {
    /// Tokenizer identity.
    pub tokenizer_version: &'static str,
    /// Embedded jieba dictionary digest.
    pub jieba_dict_sha256: &'static str,
    /// `unicodedata.unidata_version` of the reference interpreter.
    pub unicode_version: &'static str,
    /// `pyuca` UCA version.
    pub uca_version: &'static str,
    /// Digest of the UCA key tables.
    pub uca_keys_sha256: &'static str,
    /// SQLite version this build links (the vendored amalgamation).
    pub sqlite_version: &'static str,
    /// Digest of the vendored `sag_rank.c`.
    pub sag_rank_sha256: &'static str,
    /// `sag2-index`'s own version, the troubleshooting anchor.
    pub sag2_rs_version: &'static str,
    /// `CHUNK_POLICY`.
    pub chunk_policy_version: &'static str,
}

/// The contract this build certifies (design §3.3).
pub const RUNTIME_VERSIONS: RuntimeVersions = RuntimeVersions {
    tokenizer_version: TOKENIZER_VERSION,
    jieba_dict_sha256: JIEBA_DICT_SHA256,
    unicode_version: UNICODE_VERSION,
    uca_version: UCA_VERSION,
    uca_keys_sha256: UCA_KEYS_SHA256,
    sqlite_version: SQLITE_VERSION,
    sag_rank_sha256: SAG_RANK_SHA256,
    sag2_rs_version: env!("CARGO_PKG_VERSION"),
    chunk_policy_version: CHUNK_POLICY,
};

/// How hard an open may try to make the index match the corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildPolicy {
    /// Never rebuild; a signature mismatch is an error (library default).
    Strict,
    /// Rebuild when the corpus signature changed (CLI/binding default, Python
    /// compatible).
    OnCorpusChange,
    /// Rebuild unless the index is unsupported or incompatible.
    Always,
}

/// Options for [`open`].
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Chunk target in characters ([`sag2_core::DEFAULT_CHUNK_CHARS`]).
    pub chunk_chars: usize,
    /// `index_workers` (`sag2.py:196`): documents are chunked and tokenised in
    /// this many threads.  The inserted rows are byte-identical for every
    /// value (design §2.5), so this is a throughput knob, not a format one.
    pub index_workers: usize,
    /// Build the contentless `source_dir` layout against this directory.
    pub source_dir: Option<PathBuf>,
    /// The analysis profile the index is built and read with.
    ///
    /// Persisted as `rust_index_metadata.analysis`; opening an index under a
    /// different profile is refused rather than silently re-tokenised.
    pub profile: Profile,
    /// How hard the open may try to make the index match the corpus.
    pub rebuild: RebuildPolicy,
}

impl Default for IndexOptions {
    fn default() -> Self {
        IndexOptions {
            chunk_chars: DEFAULT_CHUNK_CHARS,
            index_workers: 1,
            source_dir: None,
            profile: Profile::Sag2ZhV1,
            rebuild: RebuildPolicy::Strict,
        }
    }
}

/// What the open did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAction {
    /// The existing index was used as-is.
    Reused,
    /// No index was present and the corpus was built.
    Built,
    /// The index was rebuilt from the corpus.
    Rebuilt,
}

/// Why a rebuild happened; always surfaced, never silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildReason {
    /// No index file was present.
    MissingIndex,
    /// The corpus signature changed.
    SignatureChanged {
        /// The signature component that differs (`changed_field`'s name).
        field: &'static str,
    },
    /// The file has index tables but no usable provenance row.
    CompiledIn,
}

/// The observable outcome of [`open`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenOutcome {
    /// What happened to the file.
    pub action: OpenAction,
    /// Why it was rebuilt, when it was.
    pub rebuilt: Option<RebuildReason>,
}

impl OpenOutcome {
    fn reused() -> Self {
        OpenOutcome {
            action: OpenAction::Reused,
            rebuilt: None,
        }
    }
}

/// A refused open.
#[derive(Debug)]
pub enum OpenError {
    /// The index file does not exist.
    MissingDatabase(PathBuf),
    /// The database has no `metadata` row, so its provenance is unknown.
    MissingSignature,
    /// The stored signature is not the one this corpus produces.
    SignatureMismatch {
        /// The signature the corpus produces.
        expected: String,
        /// The signature stored in the file.
        found: String,
    },
    /// `metadata.signature` is for another corpus.
    IndexSignatureMismatch {
        /// The signature component that differs.
        field: &'static str,
    },
    /// `chunks`/`chunk_fts` exist without a `metadata` row (design §3.4, divergence D1).
    ForeignDatabase,
    /// `format_version` is not [`FORMAT_VERSION`]; never rebuilt.
    UnsupportedIndexVersion {
        /// The version found in the file.
        found: i64,
        /// The version this build reads and writes.
        supported: i64,
    },
    /// `rust_index_metadata` exists without the key.
    MissingVersionKey(&'static str),
    /// A version key does not match the running build; never rebuilt.
    IncompatibleIndex {
        /// The provenance key that differs.
        field: &'static str,
        /// The value stored in the index.
        index: String,
        /// The value this build certifies.
        runtime: String,
    },
    /// `rust_index_metadata.analysis` names another analysis profile; never
    /// rebuilt, because rebuilding under the requested profile would hide the
    /// mismatch.
    AnalysisMismatch {
        /// The recorded profile; `None` for an index that predates the key
        /// (implicitly [`Profile::Sag2ZhV1`], the reference pipeline).
        index: Option<String>,
        /// The profile the caller asked for.
        expected: Profile,
    },
    /// The `chunks` schema is the `source_dir` variant, which P0 does not read.
    SourceDirUnsupported,
    /// The caller supplied no corpus, so a build decision is refused.
    BuildUnavailable,
    /// Building from the corpus failed.
    Build(BuildError),
    /// `sag_register_rank` failed.
    RankExtension(i32),
    /// The cached statistics are unusable.
    Statistics(StatisticsError),
    /// SQLite itself failed.
    Sqlite(crate::SqliteError),
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::MissingDatabase(path) => {
                write!(formatter, "index not found: {}", path.display())
            }
            OpenError::MissingSignature => {
                write!(
                    formatter,
                    "database has no metadata row; refusing to guess its provenance"
                )
            }
            OpenError::SignatureMismatch { expected, found } => write!(
                formatter,
                "index signature mismatch: expected {expected}, found {found}"
            ),
            OpenError::IndexSignatureMismatch { field } => {
                write!(
                    formatter,
                    "index was built for a different corpus ({field})"
                )
            }
            OpenError::ForeignDatabase => write!(
                formatter,
                "database has chunks tables but no metadata row; refusing to drop its data"
            ),
            OpenError::UnsupportedIndexVersion { found, supported } => write!(
                formatter,
                "unsupported index format_version {found} (this build reads {supported})"
            ),
            OpenError::MissingVersionKey(key) => {
                write!(formatter, "rust_index_metadata is missing {key}")
            }
            OpenError::IncompatibleIndex {
                field,
                index,
                runtime,
            } => write!(
                formatter,
                "index {field} {index} is incompatible with runtime {runtime}"
            ),
            OpenError::AnalysisMismatch { index, expected } => match index {
                Some(found) => write!(
                    formatter,
                    "index analysis profile {found} cannot be read as {expected}"
                ),
                None => write!(
                    formatter,
                    "index predates analysis profiles (implicitly {}) and cannot be read as {expected}",
                    Profile::Sag2ZhV1
                ),
            },
            OpenError::SourceDirUnsupported => write!(
                formatter,
                "reading source_dir indexes needs the source tree; P0 reads materialised chunks"
            ),
            OpenError::BuildUnavailable => write!(
                formatter,
                "index building needs a corpus; use CorpusIndex::open with the corpus rows"
            ),
            OpenError::Build(error) => write!(formatter, "cannot build the index: {error}"),
            OpenError::RankExtension(code) => {
                write!(
                    formatter,
                    "sag_register_rank failed with SQLite code {code}"
                )
            }
            OpenError::Statistics(error) => write!(formatter, "{error}"),
            OpenError::Sqlite(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for OpenError {}

impl From<rusqlite::Error> for OpenError {
    fn from(error: rusqlite::Error) -> Self {
        OpenError::Sqlite(error.into())
    }
}

impl From<StatisticsError> for OpenError {
    fn from(error: StatisticsError) -> Self {
        OpenError::Statistics(error)
    }
}

impl From<BuildError> for OpenError {
    fn from(error: BuildError) -> Self {
        OpenError::Build(error)
    }
}

/// What is already on disk, as far as the decision table cares.
#[derive(Debug, Clone, Default)]
pub struct ExistingIndex {
    /// Whether the path exists at all.
    pub file_exists: bool,
    /// Any of `chunks`, `chunk_fts`, `source_files` exists.
    pub has_index_tables: bool,
    /// The `metadata` table exists and holds a row.
    pub has_metadata_row: bool,
    /// `metadata.signature`, when there is a row.
    pub signature: Option<String>,
    /// Present when the index was written by this port.
    pub rust_metadata: Option<BTreeMap<String, String>>,
}

/// The outcome of the decision table (design §3.4).
#[derive(Debug)]
pub enum Decision {
    /// No usable index: build one.
    Build,
    /// Reuse the existing index.
    Reuse,
    /// Rebuild the existing index for the given reason.
    Rebuild(RebuildReason),
    /// Refuse to touch the index.
    Reject(OpenError),
}

/// Apply the §3.4 decision table.
///
/// `profile` is the analysis profile the caller reads with.  An index built
/// under another profile is rejected under every [`RebuildPolicy`]: a rebuild
/// would silently replace the token stream the file was built for.
pub fn decide(
    policy: RebuildPolicy,
    existing: &ExistingIndex,
    expected_signature: &str,
    runtime: &RuntimeVersions,
    profile: Profile,
) -> Decision {
    if !existing.file_exists {
        return Decision::Build;
    }
    if !existing.has_metadata_row {
        return if existing.has_index_tables {
            match policy {
                RebuildPolicy::Always => Decision::Rebuild(RebuildReason::CompiledIn),
                RebuildPolicy::Strict | RebuildPolicy::OnCorpusChange => {
                    Decision::Reject(OpenError::ForeignDatabase)
                }
            }
        } else {
            Decision::Build
        };
    }

    let stored = existing.signature.as_deref().unwrap_or_default();
    if stored != expected_signature {
        let field = changed_field(expected_signature, stored);
        return match policy {
            RebuildPolicy::Strict => Decision::Reject(OpenError::IndexSignatureMismatch { field }),
            RebuildPolicy::OnCorpusChange | RebuildPolicy::Always => {
                Decision::Rebuild(RebuildReason::SignatureChanged { field })
            }
        };
    }

    match &existing.rust_metadata {
        None => match check_analysis(None, profile) {
            Err(error) => Decision::Reject(error),
            // Python-built index: readable, provenance is `Python`.
            Ok(()) => match policy {
                RebuildPolicy::Always => Decision::Rebuild(RebuildReason::CompiledIn),
                RebuildPolicy::Strict | RebuildPolicy::OnCorpusChange => Decision::Reuse,
            },
        },
        Some(metadata) => match check_versions(metadata, runtime, profile) {
            Err(error) => Decision::Reject(error),
            Ok(()) => match policy {
                RebuildPolicy::Always => Decision::Rebuild(RebuildReason::CompiledIn),
                RebuildPolicy::Strict | RebuildPolicy::OnCorpusChange => Decision::Reuse,
            },
        },
    }
}

/// Compare the recorded `analysis` value with the requested profile.
///
/// A row without the key predates profiles; it was tokenised by the reference
/// pipeline, so it reads as [`Profile::Sag2ZhV1`] and nothing else.
fn check_analysis(stored: Option<&str>, profile: Profile) -> Result<(), OpenError> {
    match stored {
        Some(value) if value == profile.as_str() => Ok(()),
        Some(value) => Err(OpenError::AnalysisMismatch {
            index: Some(value.to_owned()),
            expected: profile,
        }),
        None if profile == Profile::Sag2ZhV1 => Ok(()),
        None => Err(OpenError::AnalysisMismatch {
            index: None,
            expected: profile,
        }),
    }
}

fn check_versions(
    metadata: &BTreeMap<String, String>,
    runtime: &RuntimeVersions,
    profile: Profile,
) -> Result<(), OpenError> {
    let format = metadata
        .get("format_version")
        .ok_or(OpenError::MissingVersionKey("format_version"))?;
    let found = format
        .parse::<i64>()
        .map_err(|_| OpenError::UnsupportedIndexVersion {
            found: -1,
            supported: FORMAT_VERSION,
        })?;
    if found != FORMAT_VERSION {
        return Err(OpenError::UnsupportedIndexVersion {
            found,
            supported: FORMAT_VERSION,
        });
    }
    // Design §3.3: the v1 key set is fixed.  Design §3.2 rule 4 only makes
    // *unknown* keys ignorable; a required key that is absent is a rejection.
    let checks: [(&'static str, &'static str); 9] = [
        ("tokenizer_version", runtime.tokenizer_version),
        ("chunk_policy_version", runtime.chunk_policy_version),
        ("jieba_dict_sha256", runtime.jieba_dict_sha256),
        ("unicode_version", runtime.unicode_version),
        ("uca_version", runtime.uca_version),
        ("uca_keys_sha256", runtime.uca_keys_sha256),
        ("sqlite_version", runtime.sqlite_version),
        ("sag_rank_sha256", runtime.sag_rank_sha256),
        ("sag2_rs_version", runtime.sag2_rs_version),
    ];
    for (field, expected) in checks {
        let stored = metadata
            .get(field)
            .ok_or(OpenError::MissingVersionKey(field))?;
        if stored == expected {
            continue;
        }
        return Err(OpenError::IncompatibleIndex {
            field,
            index: stored.clone(),
            runtime: expected.to_owned(),
        });
    }
    check_analysis(metadata.get("analysis").map(String::as_str), profile)
}

/// Read the open-relevant facts about `path` without writing anything.
pub fn inspect(path: &Path) -> Result<ExistingIndex, OpenError> {
    if !path.exists() {
        return Ok(ExistingIndex::default());
    }
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let tables: BTreeSet<String> = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table'")?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<_, _>>()?;

    let has_index_tables = ["chunks", "chunk_fts", "source_files"]
        .iter()
        .any(|name| tables.contains(*name));
    let has_metadata_row = if tables.contains("metadata") {
        db.query_row("SELECT EXISTS(SELECT 1 FROM metadata)", [], |row| {
            row.get::<_, bool>(0)
        })?
    } else {
        false
    };
    let signature: Option<String> = if has_metadata_row {
        db.query_row("SELECT signature FROM metadata LIMIT 1", [], |row| {
            row.get(0)
        })
        .optional()?
    } else {
        None
    };
    let rust_metadata = if tables.contains("rust_index_metadata") {
        let mut entries = BTreeMap::new();
        let mut statement = db.prepare("SELECT key, value FROM rust_index_metadata")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            entries.insert(row.get::<_, String>(0)?, row.get::<_, String>(1)?);
        }
        Some(entries)
    } else {
        None
    };
    Ok(ExistingIndex {
        file_exists: true,
        has_index_tables,
        has_metadata_row,
        signature,
        rust_metadata,
    })
}

/// Open an existing index, applying the §3.4 decision table.
///
/// A decision to build is reported as [`OpenError::BuildUnavailable`]: this
/// entry point has no corpus, and half-building an index is worse than saying
/// so.  [`CorpusIndex::open`] is the builder.
pub fn open(
    path: &Path,
    options: &IndexOptions,
    fingerprint: &str,
) -> Result<(CorpusIndex, OpenOutcome), OpenError> {
    let expected = signature(
        fingerprint,
        options.chunk_chars,
        options.source_dir.as_deref(),
    );
    let existing = inspect(path)?;
    match decide(
        options.rebuild,
        &existing,
        &expected,
        &RUNTIME_VERSIONS,
        options.profile,
    ) {
        Decision::Reject(error) => Err(error),
        Decision::Build | Decision::Rebuild(_) => Err(OpenError::BuildUnavailable),
        Decision::Reuse => Ok((
            open_readonly(path, &expected, options.profile, QueryOptions::default())?,
            OpenOutcome::reused(),
        )),
    }
}

impl CorpusIndex {
    /// Build or reuse the index for `corpus` (`sag2.py:197-311`, design §2.5).
    ///
    /// With `options.source_dir` the index is written (contentless FTS plus
    /// `source_files`), but the P0 reader still refuses the
    /// `start_char`/`end_char` layout with [`OpenError::SourceDirUnsupported`].
    pub fn open(
        path: &Path,
        corpus: &[Document<'_>],
        fingerprint: &str,
        options: IndexOptions,
    ) -> Result<(Self, OpenOutcome), OpenError> {
        let expected = signature(
            fingerprint,
            options.chunk_chars,
            options.source_dir.as_deref(),
        );
        let existing = inspect(path)?;
        let outcome = match decide(
            options.rebuild,
            &existing,
            &expected,
            &RUNTIME_VERSIONS,
            options.profile,
        ) {
            Decision::Reject(error) => return Err(error),
            Decision::Reuse => OpenOutcome::reused(),
            Decision::Build => {
                build::build(path, corpus, fingerprint, &options, &RUNTIME_VERSIONS)?;
                OpenOutcome {
                    action: OpenAction::Built,
                    rebuilt: None,
                }
            }
            Decision::Rebuild(reason) => {
                build::build(path, corpus, fingerprint, &options, &RUNTIME_VERSIONS)?;
                OpenOutcome {
                    action: OpenAction::Rebuilt,
                    rebuilt: Some(reason),
                }
            }
        };
        Ok((
            open_readonly(path, &expected, options.profile, QueryOptions::default())?,
            outcome,
        ))
    }
}

/// Open `path` read-only, verifying that it was built for `expected_signature`
/// and `profile`.
pub fn open_readonly(
    path: &Path,
    expected_signature: &str,
    profile: Profile,
    options: QueryOptions,
) -> Result<CorpusIndex, OpenError> {
    open_readonly_with(
        path,
        expected_signature,
        profile,
        options,
        &RUNTIME_VERSIONS,
    )
}

fn open_readonly_with(
    path: &Path,
    expected_signature: &str,
    profile: Profile,
    options: QueryOptions,
    runtime: &RuntimeVersions,
) -> Result<CorpusIndex, OpenError> {
    let existing = inspect(path)?;
    if !existing.file_exists {
        return Err(OpenError::MissingDatabase(path.to_owned()));
    }
    let stored = existing.signature.ok_or(OpenError::MissingSignature)?;
    if stored != expected_signature {
        return Err(OpenError::SignatureMismatch {
            expected: expected_signature.to_owned(),
            found: stored,
        });
    }
    match &existing.rust_metadata {
        Some(metadata) => check_versions(metadata, runtime, profile)?,
        None => check_analysis(None, profile)?,
    }

    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    if !has_materialised_chunks(&db)? {
        return Err(OpenError::SourceDirUnsupported);
    }
    let statistics = statistics::load(&db, &stored)?;
    drop(db);

    Ok(CorpusIndex::new(
        path.to_owned(),
        stored,
        statistics,
        QueryPool::open(path, options)?,
    ))
}

/// `chunks` holds `content`/`full_doc` rather than `start_char`/`end_char`.
fn has_materialised_chunks(db: &Connection) -> Result<bool, rusqlite::Error> {
    let mut statement = db.prepare("PRAGMA table_info(chunks)")?;
    let mut rows = statement.query([])?;
    let mut materialised = false;
    while let Some(row) = rows.next()? {
        if row.get::<_, String>(1)? == "content" {
            materialised = true;
        }
    }
    Ok(materialised)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        decide, Decision, ExistingIndex, OpenError, Profile, RebuildPolicy, RebuildReason,
        RuntimeVersions, FORMAT_VERSION, RUNTIME_VERSIONS,
    };

    const EXPECTED: &str =
        "[\"corpus\",2200,5,null,\"desktop-sentences-v1\",\"plain-text-default\"]";
    const OTHER: &str = "[\"other\",2200,5,null,\"desktop-sentences-v1\",\"plain-text-default\"]";

    const RUNTIME: RuntimeVersions = RuntimeVersions {
        tokenizer_version: "tokera-1",
        jieba_dict_sha256: "dict-1",
        unicode_version: "15.0.0",
        uca_version: "9.0.0",
        uca_keys_sha256: "uca-1",
        sqlite_version: "3.50.2",
        sag_rank_sha256: "rank-1",
        sag2_rs_version: "0.1.0",
        chunk_policy_version: "desktop-sentences-v1",
    };

    /// The §3.3 v1 key set in the order `check_versions` requires it.
    const V1_KEYS: [&str; 9] = [
        "tokenizer_version",
        "chunk_policy_version",
        "jieba_dict_sha256",
        "unicode_version",
        "uca_version",
        "uca_keys_sha256",
        "sqlite_version",
        "sag_rank_sha256",
        "sag2_rs_version",
    ];

    fn existing(signature: Option<&str>, rust: Option<BTreeMap<String, String>>) -> ExistingIndex {
        ExistingIndex {
            file_exists: true,
            has_index_tables: true,
            has_metadata_row: signature.is_some(),
            signature: signature.map(str::to_owned),
            rust_metadata: rust,
        }
    }

    fn rust_metadata(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    }

    fn current_metadata() -> BTreeMap<String, String> {
        rust_metadata(&[
            ("format_version", "1"),
            ("analysis", "sag2-zh-v1"),
            ("tokenizer_version", "tokera-1"),
            ("chunk_policy_version", "desktop-sentences-v1"),
            ("jieba_dict_sha256", "dict-1"),
            ("unicode_version", "15.0.0"),
            ("uca_version", "9.0.0"),
            ("uca_keys_sha256", "uca-1"),
            ("sqlite_version", "3.50.2"),
            ("sag_rank_sha256", "rank-1"),
            ("sag2_rs_version", "0.1.0"),
        ])
    }

    fn rejection(decision: Decision) -> OpenError {
        match decision {
            Decision::Reject(error) => error,
            other => panic!("expected a rejection, got {other:?}"),
        }
    }

    fn is_rebuild(decision: Decision) -> Option<RebuildReason> {
        match decision {
            Decision::Rebuild(reason) => Some(reason),
            other => panic!("expected a rebuild, got {other:?}"),
        }
    }

    fn is_reuse(decision: Decision) {
        assert!(matches!(decision, Decision::Reuse), "got {decision:?}");
    }

    #[test]
    fn missing_file_builds_for_every_policy() {
        let missing = ExistingIndex::default();
        for policy in [
            RebuildPolicy::Strict,
            RebuildPolicy::OnCorpusChange,
            RebuildPolicy::Always,
        ] {
            assert!(matches!(
                decide(policy, &missing, EXPECTED, &RUNTIME, Profile::Sag2ZhV1),
                Decision::Build
            ));
        }
    }

    #[test]
    fn empty_database_builds() {
        let empty = ExistingIndex {
            file_exists: true,
            ..ExistingIndex::default()
        };
        assert!(matches!(
            decide(
                RebuildPolicy::Strict,
                &empty,
                EXPECTED,
                &RUNTIME,
                Profile::Sag2ZhV1
            ),
            Decision::Build
        ));
    }

    #[test]
    fn foreign_database_is_rejected_unless_always() {
        let foreign = ExistingIndex {
            file_exists: true,
            has_index_tables: true,
            has_metadata_row: false,
            signature: None,
            rust_metadata: None,
        };
        assert!(matches!(
            rejection(decide(
                RebuildPolicy::Strict,
                &foreign,
                EXPECTED,
                &RUNTIME,
                Profile::Sag2ZhV1
            )),
            OpenError::ForeignDatabase
        ));
        assert!(matches!(
            rejection(decide(
                RebuildPolicy::OnCorpusChange,
                &foreign,
                EXPECTED,
                &RUNTIME,
                Profile::Sag2ZhV1,
            )),
            OpenError::ForeignDatabase
        ));
        assert_eq!(
            is_rebuild(decide(
                RebuildPolicy::Always,
                &foreign,
                EXPECTED,
                &RUNTIME,
                Profile::Sag2ZhV1
            )),
            Some(RebuildReason::CompiledIn)
        );
    }

    #[test]
    fn signature_change_rebuilds_or_rejects_by_policy() {
        let changed = existing(Some(OTHER), None);
        match rejection(decide(
            RebuildPolicy::Strict,
            &changed,
            EXPECTED,
            &RUNTIME,
            Profile::Sag2ZhV1,
        )) {
            OpenError::IndexSignatureMismatch { field } => assert_eq!(field, "fingerprint"),
            other => panic!("unexpected {other}"),
        }
        assert_eq!(
            is_rebuild(decide(
                RebuildPolicy::OnCorpusChange,
                &changed,
                EXPECTED,
                &RUNTIME,
                Profile::Sag2ZhV1,
            )),
            Some(RebuildReason::SignatureChanged {
                field: "fingerprint"
            })
        );
        assert_eq!(
            is_rebuild(decide(
                RebuildPolicy::Always,
                &changed,
                EXPECTED,
                &RUNTIME,
                Profile::Sag2ZhV1
            )),
            Some(RebuildReason::SignatureChanged {
                field: "fingerprint"
            })
        );
    }

    #[test]
    fn python_built_index_is_reused_and_always_rebuilds() {
        let python = existing(Some(EXPECTED), None);
        is_reuse(decide(
            RebuildPolicy::Strict,
            &python,
            EXPECTED,
            &RUNTIME,
            Profile::Sag2ZhV1,
        ));
        is_reuse(decide(
            RebuildPolicy::OnCorpusChange,
            &python,
            EXPECTED,
            &RUNTIME,
            Profile::Sag2ZhV1,
        ));
        assert_eq!(
            is_rebuild(decide(
                RebuildPolicy::Always,
                &python,
                EXPECTED,
                &RUNTIME,
                Profile::Sag2ZhV1
            )),
            Some(RebuildReason::CompiledIn)
        );
    }

    #[test]
    fn rust_built_index_is_reused_when_versions_match() {
        let rust = existing(Some(EXPECTED), Some(current_metadata()));
        is_reuse(decide(
            RebuildPolicy::Strict,
            &rust,
            EXPECTED,
            &RUNTIME,
            Profile::Sag2ZhV1,
        ));
        is_reuse(decide(
            RebuildPolicy::OnCorpusChange,
            &rust,
            EXPECTED,
            &RUNTIME,
            Profile::Sag2ZhV1,
        ));
    }

    #[test]
    fn unknown_format_version_is_rejected_by_every_policy() {
        for version in ["2", "0", "not-a-number"] {
            let rust = existing(
                Some(EXPECTED),
                Some(rust_metadata(&[("format_version", version)])),
            );
            for policy in [
                RebuildPolicy::Strict,
                RebuildPolicy::OnCorpusChange,
                RebuildPolicy::Always,
            ] {
                match rejection(decide(policy, &rust, EXPECTED, &RUNTIME, Profile::Sag2ZhV1)) {
                    OpenError::UnsupportedIndexVersion { supported, .. } => {
                        assert_eq!(supported, FORMAT_VERSION)
                    }
                    other => panic!("unexpected {other}"),
                }
            }
        }
    }

    #[test]
    fn missing_required_version_keys_are_rejected() {
        for key in std::iter::once("format_version").chain(V1_KEYS) {
            let mut metadata = current_metadata();
            metadata.remove(key);
            let rust = existing(Some(EXPECTED), Some(metadata));
            for policy in [
                RebuildPolicy::Strict,
                RebuildPolicy::OnCorpusChange,
                RebuildPolicy::Always,
            ] {
                match rejection(decide(policy, &rust, EXPECTED, &RUNTIME, Profile::Sag2ZhV1)) {
                    OpenError::MissingVersionKey(field) => assert_eq!(field, key),
                    other => panic!("{key}: unexpected {other}"),
                }
            }
        }
    }

    #[test]
    fn certified_sqlite_version_matches_the_bundled_amalgamation() {
        assert_eq!(RUNTIME_VERSIONS.sqlite_version, crate::sqlite_version());
    }

    #[test]
    fn incompatible_version_is_rejected_by_every_policy() {
        for (key, value) in [
            ("tokenizer_version", "tokera-2"),
            ("chunk_policy_version", "desktop-sentences-v2"),
            ("jieba_dict_sha256", "dict-2"),
            ("unicode_version", "16.0.0"),
            ("uca_version", "10.0.0"),
            ("uca_keys_sha256", "uca-2"),
            ("sqlite_version", "3.50.4"),
            ("sag_rank_sha256", "rank-2"),
            ("sag2_rs_version", "0.2.0"),
        ] {
            let mut metadata = current_metadata();
            metadata.insert(key.to_owned(), value.to_owned());
            let rust = existing(Some(EXPECTED), Some(metadata));
            for policy in [
                RebuildPolicy::Strict,
                RebuildPolicy::OnCorpusChange,
                RebuildPolicy::Always,
            ] {
                match rejection(decide(policy, &rust, EXPECTED, &RUNTIME, Profile::Sag2ZhV1)) {
                    OpenError::IncompatibleIndex { field, .. } => assert_eq!(field, key),
                    other => panic!("unexpected {other}"),
                }
            }
        }
    }

    #[test]
    fn another_analysis_profile_is_rejected_by_every_policy() {
        let rust = existing(Some(EXPECTED), Some(current_metadata()));
        for policy in [
            RebuildPolicy::Strict,
            RebuildPolicy::OnCorpusChange,
            RebuildPolicy::Always,
        ] {
            match rejection(decide(
                policy,
                &rust,
                EXPECTED,
                &RUNTIME,
                Profile::MultilingualV1,
            )) {
                OpenError::AnalysisMismatch { index, expected } => {
                    assert_eq!(index.as_deref(), Some("sag2-zh-v1"));
                    assert_eq!(expected, Profile::MultilingualV1);
                }
                other => panic!("{policy:?} must refuse, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_multilingual_index_is_reused_only_under_its_own_profile() {
        let mut metadata = current_metadata();
        metadata.insert("analysis".to_owned(), "multilingual-v1".to_owned());
        let rust = existing(Some(EXPECTED), Some(metadata));
        is_reuse(decide(
            RebuildPolicy::Strict,
            &rust,
            EXPECTED,
            &RUNTIME,
            Profile::MultilingualV1,
        ));
        assert!(matches!(
            rejection(decide(
                RebuildPolicy::Strict,
                &rust,
                EXPECTED,
                &RUNTIME,
                Profile::Sag2ZhV1
            )),
            OpenError::AnalysisMismatch { .. }
        ));
    }

    #[test]
    fn a_row_without_analysis_is_the_reference_profile() {
        let mut metadata = current_metadata();
        metadata.remove("analysis");
        let rust = existing(Some(EXPECTED), Some(metadata));
        is_reuse(decide(
            RebuildPolicy::Strict,
            &rust,
            EXPECTED,
            &RUNTIME,
            Profile::Sag2ZhV1,
        ));
        match rejection(decide(
            RebuildPolicy::Strict,
            &rust,
            EXPECTED,
            &RUNTIME,
            Profile::MultilingualV1,
        )) {
            OpenError::AnalysisMismatch { index, expected } => {
                assert_eq!(index, None);
                assert_eq!(expected, Profile::MultilingualV1);
            }
            other => panic!("unexpected {other}"),
        }
    }

    #[test]
    fn a_python_index_is_never_read_as_multilingual() {
        let python = existing(Some(EXPECTED), None);
        is_reuse(decide(
            RebuildPolicy::Strict,
            &python,
            EXPECTED,
            &RUNTIME,
            Profile::Sag2ZhV1,
        ));
        assert!(matches!(
            rejection(decide(
                RebuildPolicy::OnCorpusChange,
                &python,
                EXPECTED,
                &RUNTIME,
                Profile::MultilingualV1
            )),
            OpenError::AnalysisMismatch { index: None, .. }
        ));
    }

    #[test]
    fn an_unknown_analysis_value_is_rejected() {
        let mut metadata = current_metadata();
        metadata.insert("analysis".to_owned(), "sag2-zh-v0".to_owned());
        let rust = existing(Some(EXPECTED), Some(metadata));
        for profile in [Profile::Sag2ZhV1, Profile::MultilingualV1] {
            match rejection(decide(
                RebuildPolicy::Always,
                &rust,
                EXPECTED,
                &RUNTIME,
                profile,
            )) {
                OpenError::AnalysisMismatch { index, .. } => {
                    assert_eq!(index.as_deref(), Some("sag2-zh-v0"))
                }
                other => panic!("unexpected {other}"),
            }
        }
    }
}
