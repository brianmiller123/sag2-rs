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
//! Building an index needs the tokenizer and the chunker, which P1 owns.  So a
//! decision that would build returns [`OpenError::BuildUnavailable`] instead of
//! silently doing nothing; the decision itself is complete and unit-tested.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags, OptionalExtension};

use crate::pool::{QueryOptions, QueryPool};
use crate::signature::{changed_field, signature, CHUNK_POLICY};
use crate::statistics::{self, StatisticsError};
use crate::CorpusIndex;

/// `sag2.py`/`chunking.py` default chunk size (`chunking.py:6`).
pub const DEFAULT_CHUNK_CHARS: usize = 2200;

/// `rust_index_metadata.format_version` this build reads and writes.
pub const FORMAT_VERSION: i64 = 1;

/// How this build certifies one `rust_index_metadata` value (design §3.3).
///
/// A field is either certified or explicitly owned by a later plan.  The
/// `PendingBuild` variant is visible in the type on purpose: forgetting to
/// certify a field is a compile-time edit, and an index that carries the key
/// is rejected rather than silently accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionCheck {
    /// This build certifies the exact value an index must carry.
    Certified(&'static str),
    /// The value belongs to a component a later plan writes; until then any
    /// index that carries the key is rejected.
    PendingBuild,
}

impl VersionCheck {
    /// The certified value, if this build has one.
    pub fn certified(self) -> Option<&'static str> {
        match self {
            VersionCheck::Certified(value) => Some(value),
            VersionCheck::PendingBuild => None,
        }
    }
}

/// The `rust_index_metadata` keys whose values must match the running build.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeVersions {
    /// Tokenizer identity; P1 owns the tokenizer.
    pub tokenizer_version: VersionCheck,
    /// Embedded jieba dictionary digest; P1 owns the tokenizer.
    pub jieba_dict_sha256: VersionCheck,
    /// `unicodedata.unidata_version` of the reference interpreter.
    pub unicode_version: VersionCheck,
    /// `pyuca` UCA version.
    pub uca_version: VersionCheck,
    /// Digest of the UCA key tables; P2 owns the collation.
    pub uca_keys_sha256: VersionCheck,
    /// SQLite version this build links (the vendored amalgamation).
    pub sqlite_version: VersionCheck,
    /// Digest of the vendored `sag_rank.c`; P1 publishes the build digest.
    pub sag_rank_sha256: VersionCheck,
    /// `sag2-index`'s own version, the troubleshooting anchor.
    pub sag2_rs_version: VersionCheck,
    /// `CHUNK_POLICY`.
    pub chunk_policy_version: VersionCheck,
}

/// Message recorded when an index carries a key this build cannot certify yet.
pub const PENDING_BUILD_RUNTIME: &str = "<pending P1 build>";

/// The contract this build certifies (design §3.3).
pub const RUNTIME_VERSIONS: RuntimeVersions = RuntimeVersions {
    // P1 owns the tokenizer, its dictionary, the build digest and the UCA
    // tables; the values are declared here so the gap is in the type, not in a
    // silently skipped check.
    tokenizer_version: VersionCheck::PendingBuild,
    jieba_dict_sha256: VersionCheck::PendingBuild,
    unicode_version: VersionCheck::Certified("15.0.0"),
    uca_version: VersionCheck::Certified("9.0.0"),
    uca_keys_sha256: VersionCheck::PendingBuild,
    sqlite_version: VersionCheck::Certified("3.50.2"),
    sag_rank_sha256: VersionCheck::PendingBuild,
    sag2_rs_version: VersionCheck::Certified(env!("CARGO_PKG_VERSION")),
    chunk_policy_version: VersionCheck::Certified(CHUNK_POLICY),
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
    pub chunk_chars: usize,
    pub source_dir: Option<PathBuf>,
    pub rebuild: RebuildPolicy,
}

impl Default for IndexOptions {
    fn default() -> Self {
        IndexOptions {
            chunk_chars: DEFAULT_CHUNK_CHARS,
            source_dir: None,
            rebuild: RebuildPolicy::Strict,
        }
    }
}

/// What the open did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAction {
    Reused,
    Built,
    Rebuilt,
}

/// Why a rebuild happened; always surfaced, never silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildReason {
    MissingIndex,
    SignatureChanged { field: &'static str },
    CompiledIn,
}

/// The observable outcome of [`open`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenOutcome {
    pub action: OpenAction,
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
    SignatureMismatch { expected: String, found: String },
    /// `metadata.signature` is for another corpus.
    IndexSignatureMismatch { field: &'static str },
    /// `chunks`/`chunk_fts` exist without a `metadata` row (design §3.4, divergence D1).
    ForeignDatabase,
    /// `format_version` is not [`FORMAT_VERSION`]; never rebuilt.
    UnsupportedIndexVersion { found: i64, supported: i64 },
    /// `rust_index_metadata` exists without the key.
    MissingVersionKey(&'static str),
    /// A version key does not match the running build; never rebuilt.
    IncompatibleIndex {
        field: &'static str,
        index: String,
        runtime: String,
    },
    /// The `chunks` schema is the `source_dir` variant, which P0 does not read.
    SourceDirUnsupported,
    /// Building is P1 scope; P0 refuses instead of writing a partial index.
    BuildUnavailable,
    /// `sag_register_rank` failed.
    RankExtension(i32),
    /// The cached statistics are unusable.
    Statistics(StatisticsError),
    /// SQLite itself failed.
    Sqlite(rusqlite::Error),
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::MissingDatabase(path) => write!(formatter, "index not found: {}", path.display()),
            OpenError::MissingSignature => {
                write!(formatter, "database has no metadata row; refusing to guess its provenance")
            }
            OpenError::SignatureMismatch { expected, found } => write!(
                formatter,
                "index signature mismatch: expected {expected}, found {found}"
            ),
            OpenError::IndexSignatureMismatch { field } => {
                write!(formatter, "index was built for a different corpus ({field})")
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
            OpenError::SourceDirUnsupported => write!(
                formatter,
                "reading source_dir indexes needs the source tree; P0 reads materialised chunks"
            ),
            OpenError::BuildUnavailable => write!(
                formatter,
                "index building needs the tokenizer and chunker (P1); P0 only reads existing indexes"
            ),
            OpenError::RankExtension(code) => {
                write!(formatter, "sag_register_rank failed with SQLite code {code}")
            }
            OpenError::Statistics(error) => write!(formatter, "{error}"),
            OpenError::Sqlite(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for OpenError {}

impl From<rusqlite::Error> for OpenError {
    fn from(error: rusqlite::Error) -> Self {
        OpenError::Sqlite(error)
    }
}

impl From<StatisticsError> for OpenError {
    fn from(error: StatisticsError) -> Self {
        OpenError::Statistics(error)
    }
}

/// What is already on disk, as far as the decision table cares.
#[derive(Debug, Clone, Default)]
pub struct ExistingIndex {
    pub file_exists: bool,
    /// Any of `chunks`, `chunk_fts`, `source_files` exists.
    pub has_index_tables: bool,
    /// The `metadata` table exists and holds a row.
    pub has_metadata_row: bool,
    pub signature: Option<String>,
    /// Present when the index was written by this port.
    pub rust_metadata: Option<BTreeMap<String, String>>,
}

/// The outcome of the decision table (design §3.4).
#[derive(Debug)]
pub enum Decision {
    Build,
    Reuse,
    Rebuild(RebuildReason),
    Reject(OpenError),
}

/// Apply the §3.4 decision table.
pub fn decide(
    policy: RebuildPolicy,
    existing: &ExistingIndex,
    expected_signature: &str,
    runtime: &RuntimeVersions,
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
        None => match policy {
            // Python-built index: readable, provenance is `Python`.
            RebuildPolicy::Always => Decision::Rebuild(RebuildReason::CompiledIn),
            RebuildPolicy::Strict | RebuildPolicy::OnCorpusChange => Decision::Reuse,
        },
        Some(metadata) => match check_versions(metadata, runtime) {
            Err(error) => Decision::Reject(error),
            Ok(()) => match policy {
                RebuildPolicy::Always => Decision::Rebuild(RebuildReason::CompiledIn),
                RebuildPolicy::Strict | RebuildPolicy::OnCorpusChange => Decision::Reuse,
            },
        },
    }
}

fn check_versions(
    metadata: &BTreeMap<String, String>,
    runtime: &RuntimeVersions,
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
    let checks: [(&'static str, VersionCheck); 9] = [
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
    for (field, check) in checks {
        let stored = metadata
            .get(field)
            .ok_or(OpenError::MissingVersionKey(field))?;
        let runtime_value = match check {
            VersionCheck::Certified(expected) if stored == expected => continue,
            VersionCheck::Certified(expected) => expected,
            // An index that carries a value this build cannot certify is
            // refused: accepting it would launder an unknown provenance.
            VersionCheck::PendingBuild => PENDING_BUILD_RUNTIME,
        };
        return Err(OpenError::IncompatibleIndex {
            field,
            index: stored.clone(),
            runtime: runtime_value.to_owned(),
        });
    }
    Ok(())
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
/// A decision to build is reported as [`OpenError::BuildUnavailable`]: building
/// needs the tokenizer and chunker (P1), and P0 must not half-build an index.
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
    match decide(options.rebuild, &existing, &expected, &RUNTIME_VERSIONS) {
        Decision::Reject(error) => Err(error),
        Decision::Build | Decision::Rebuild(_) => Err(OpenError::BuildUnavailable),
        Decision::Reuse => Ok((
            open_readonly(path, &expected, QueryOptions::default())?,
            OpenOutcome::reused(),
        )),
    }
}

/// Open `path` read-only, verifying that it was built for `expected_signature`.
pub fn open_readonly(
    path: &Path,
    expected_signature: &str,
    options: QueryOptions,
) -> Result<CorpusIndex, OpenError> {
    open_readonly_with(path, expected_signature, options, &RUNTIME_VERSIONS)
}

fn open_readonly_with(
    path: &Path,
    expected_signature: &str,
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
    if let Some(metadata) = &existing.rust_metadata {
        check_versions(metadata, runtime)?;
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
        decide, Decision, ExistingIndex, OpenError, RebuildPolicy, RebuildReason, RuntimeVersions,
        VersionCheck, FORMAT_VERSION, PENDING_BUILD_RUNTIME, RUNTIME_VERSIONS,
    };

    const EXPECTED: &str =
        "[\"corpus\",2200,5,null,\"desktop-sentences-v1\",\"plain-text-default\"]";
    const OTHER: &str = "[\"other\",2200,5,null,\"desktop-sentences-v1\",\"plain-text-default\"]";

    const RUNTIME: RuntimeVersions = RuntimeVersions {
        tokenizer_version: VersionCheck::Certified("tokera-1"),
        jieba_dict_sha256: VersionCheck::Certified("dict-1"),
        unicode_version: VersionCheck::Certified("15.0.0"),
        uca_version: VersionCheck::Certified("9.0.0"),
        uca_keys_sha256: VersionCheck::Certified("uca-1"),
        sqlite_version: VersionCheck::Certified("3.50.2"),
        sag_rank_sha256: VersionCheck::Certified("rank-1"),
        sag2_rs_version: VersionCheck::Certified("0.1.0"),
        chunk_policy_version: VersionCheck::Certified("desktop-sentences-v1"),
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
                decide(policy, &missing, EXPECTED, &RUNTIME),
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
            decide(RebuildPolicy::Strict, &empty, EXPECTED, &RUNTIME),
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
            rejection(decide(RebuildPolicy::Strict, &foreign, EXPECTED, &RUNTIME)),
            OpenError::ForeignDatabase
        ));
        assert!(matches!(
            rejection(decide(
                RebuildPolicy::OnCorpusChange,
                &foreign,
                EXPECTED,
                &RUNTIME
            )),
            OpenError::ForeignDatabase
        ));
        assert_eq!(
            is_rebuild(decide(RebuildPolicy::Always, &foreign, EXPECTED, &RUNTIME)),
            Some(RebuildReason::CompiledIn)
        );
    }

    #[test]
    fn signature_change_rebuilds_or_rejects_by_policy() {
        let changed = existing(Some(OTHER), None);
        match rejection(decide(RebuildPolicy::Strict, &changed, EXPECTED, &RUNTIME)) {
            OpenError::IndexSignatureMismatch { field } => assert_eq!(field, "fingerprint"),
            other => panic!("unexpected {other}"),
        }
        assert_eq!(
            is_rebuild(decide(
                RebuildPolicy::OnCorpusChange,
                &changed,
                EXPECTED,
                &RUNTIME
            )),
            Some(RebuildReason::SignatureChanged {
                field: "fingerprint"
            })
        );
        assert_eq!(
            is_rebuild(decide(RebuildPolicy::Always, &changed, EXPECTED, &RUNTIME)),
            Some(RebuildReason::SignatureChanged {
                field: "fingerprint"
            })
        );
    }

    #[test]
    fn python_built_index_is_reused_and_always_rebuilds() {
        let python = existing(Some(EXPECTED), None);
        is_reuse(decide(RebuildPolicy::Strict, &python, EXPECTED, &RUNTIME));
        is_reuse(decide(
            RebuildPolicy::OnCorpusChange,
            &python,
            EXPECTED,
            &RUNTIME,
        ));
        assert_eq!(
            is_rebuild(decide(RebuildPolicy::Always, &python, EXPECTED, &RUNTIME)),
            Some(RebuildReason::CompiledIn)
        );
    }

    #[test]
    fn rust_built_index_is_reused_when_versions_match() {
        let rust = existing(Some(EXPECTED), Some(current_metadata()));
        is_reuse(decide(RebuildPolicy::Strict, &rust, EXPECTED, &RUNTIME));
        is_reuse(decide(
            RebuildPolicy::OnCorpusChange,
            &rust,
            EXPECTED,
            &RUNTIME,
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
                match rejection(decide(policy, &rust, EXPECTED, &RUNTIME)) {
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
                match rejection(decide(policy, &rust, EXPECTED, &RUNTIME)) {
                    OpenError::MissingVersionKey(field) => assert_eq!(field, key),
                    other => panic!("{key}: unexpected {other}"),
                }
            }
        }
    }

    #[test]
    fn pending_build_versions_reject_a_rust_built_index() {
        let rust = existing(Some(EXPECTED), Some(current_metadata()));
        for policy in [
            RebuildPolicy::Strict,
            RebuildPolicy::OnCorpusChange,
            RebuildPolicy::Always,
        ] {
            match rejection(decide(policy, &rust, EXPECTED, &RUNTIME_VERSIONS)) {
                OpenError::IncompatibleIndex { field, runtime, .. } => {
                    assert_eq!(field, "tokenizer_version");
                    assert_eq!(runtime, PENDING_BUILD_RUNTIME);
                }
                other => panic!("unexpected {other}"),
            }
        }
    }

    /// The build must certify everything except the P1/P2-owned values; this
    /// list has to reach zero before P1's Gate D.
    #[test]
    fn pending_build_fields_are_exactly_the_pipeline_owned_ones() {
        let fields = [
            ("tokenizer_version", RUNTIME_VERSIONS.tokenizer_version),
            (
                "chunk_policy_version",
                RUNTIME_VERSIONS.chunk_policy_version,
            ),
            ("jieba_dict_sha256", RUNTIME_VERSIONS.jieba_dict_sha256),
            ("unicode_version", RUNTIME_VERSIONS.unicode_version),
            ("uca_version", RUNTIME_VERSIONS.uca_version),
            ("uca_keys_sha256", RUNTIME_VERSIONS.uca_keys_sha256),
            ("sqlite_version", RUNTIME_VERSIONS.sqlite_version),
            ("sag_rank_sha256", RUNTIME_VERSIONS.sag_rank_sha256),
            ("sag2_rs_version", RUNTIME_VERSIONS.sag2_rs_version),
        ];
        let pending: Vec<&str> = fields
            .iter()
            .filter(|(_, check)| check.certified().is_none())
            .map(|(field, _)| *field)
            .collect();
        assert_eq!(
            pending,
            [
                "tokenizer_version",
                "jieba_dict_sha256",
                "uca_keys_sha256",
                "sag_rank_sha256"
            ]
        );
    }

    #[test]
    fn certified_sqlite_version_matches_the_bundled_amalgamation() {
        assert_eq!(
            RUNTIME_VERSIONS.sqlite_version.certified(),
            Some(crate::sqlite_version())
        );
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
                match rejection(decide(policy, &rust, EXPECTED, &RUNTIME)) {
                    OpenError::IncompatibleIndex { field, .. } => assert_eq!(field, key),
                    other => panic!("unexpected {other}"),
                }
            }
        }
    }
}
