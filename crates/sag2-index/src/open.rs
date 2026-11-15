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
//!
//! The dictionary a Japanese profile segments with is provenance too:
//! `rust_index_metadata.analysis_dict` records the fingerprint of the directory
//! the build loaded, and opening the index with another dictionary is an
//! [`OpenError::IncompatibleIndex`] whatever the rebuild policy says.
//! A dictionary-free build writes no such row (an absent row reads as
//! [`ANALYSIS_DICT_NONE`]), so a default index keeps exactly the ten design
//! §3.3 keys.  A profile that needs a dictionary and gets none is
//! [`OpenError::Analysis`] wrapping [`AnalysisError::DictionaryMissing`].
//!
//! The multilingual pipeline is versioned for the same reason.
//! `rust_index_metadata.analysis_version` records [`MULTILINGUAL_VERSION`] for
//! every non-default profile: the nine design version keys are identical across
//! multilingual pipeline revisions, so they cannot tell two token streams
//! apart.  An absent or different value is a stale index (one from before the
//! key existed is stale by definition); a rebuild policy replaces it, and a
//! read-only open reports [`OpenError::AnalysisVersion`].

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rusqlite::{Connection, OpenFlags, OptionalExtension};

use sag2_core::{
    analysis::REMOVED_MULTILINGUAL_V1, AnalysisError, DictionaryAnalysis, Language, Profile,
    DEFAULT_CHUNK_CHARS,
};

use crate::build::{self, BuildError, Document};
use crate::pool::{QueryOptions, QueryPool};
use crate::signature::{changed_field, signature, CHUNK_POLICY};
use crate::statistics::{self, StatisticsError};
use crate::CorpusIndex;

/// `rust_index_metadata.format_version` this build reads and writes.
pub const FORMAT_VERSION: i64 = 1;

/// [`FORMAT_VERSION`] in the string form the provenance table stores.
pub const FORMAT_VERSION_STR: &str = "1";

/// The `rust_index_metadata` key that records the analysis profile.
pub const ANALYSIS_KEY: &str = "analysis";

/// The `rust_index_metadata` key that records the dictionary provenance.
pub const ANALYSIS_DICT_KEY: &str = "analysis_dict";

/// The `rust_index_metadata` key that records the multilingual pipeline
/// revision a non-default profile's tokens came from.
pub const ANALYSIS_VERSION_KEY: &str = "analysis_version";

/// The multilingual pipeline revision this build writes.
///
/// One constant for every `multilingual-v1-<lang>` profile: the value names the
/// pipeline (normaliser, word policy, language folds), not the language, so a
/// profile id keeps naming exactly one token stream.  Bump it whenever the
/// multilingual tokenizer or normaliser changes; an index recorded under another
/// value (or without the key, which every pre-batch index is) is stale by
/// definition and is rebuilt, never read with a tokenizer that did not produce
/// its tokens.  The reference profile records no such row.
pub const MULTILINGUAL_VERSION: &str = "ml-1";

/// The [`ANALYSIS_DICT_KEY`] value of a dictionary-free profile.
///
/// It is deliberately not a hash, so it can never collide with a
/// [`DictionaryAnalysis::fingerprint`] (64 lowercase hex digits).  No build
/// writes it - an absent row *is* this value, which keeps a default index at the
/// ten design §3.3 keys - but a hand-edited row equal to it still reads as a
/// dictionary-free provenance.
pub const ANALYSIS_DICT_NONE: &str = "none";

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
    /// The lindera dictionary directory a Japanese `profile` needs.
    ///
    /// The directory holds a compiled ipadic (`metadata.json`, `char_def.bin`,
    /// `unk.bin`, `matrix.mtx`, `dict.trie`, `dict.valsidx`, `dict.vals`,
    /// `dict.wordsidx`, `dict.words`).  It is loaded once per open and its
    /// fingerprint is recorded as `rust_index_metadata.analysis_dict`.  A
    /// Japanese profile without it fails closed
    /// ([`AnalysisError::DictionaryMissing`]); every other profile has no
    /// dictionary to read and refuses one
    /// ([`AnalysisError::DictionaryUnused`]).
    ///
    /// Requires the `lindera` feature; without it such a profile is
    /// [`AnalysisError::DictionaryUnsupported`].
    pub dictionary_dir: Option<PathBuf>,
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
            dictionary_dir: None,
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
    /// The index records another multilingual pipeline revision, or none: its
    /// tokens came from a tokenizer this build no longer has.
    AnalysisVersionChanged,
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
    /// A multilingual index does not record the pipeline revision that produced
    /// its tokens, or records another one.
    ///
    /// The nine design version keys are identical across pipeline revisions, so
    /// they cannot tell two token streams apart.  An index from before the key
    /// existed is stale by definition; under a rebuild policy it is rebuilt,
    /// and a read-only open reports this error.
    AnalysisVersion {
        /// The recorded revision; `None` when the key is absent.
        index: Option<String>,
        /// The revision this build writes.
        runtime: &'static str,
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
    /// `rust_index_metadata.analysis` names a profile this build cannot parse,
    /// so the file cannot be opened as what it records.
    UnknownAnalysisProfile {
        /// The stored value.
        value: String,
    },
    /// Building from the corpus failed.
    Build(BuildError),
    /// The analysis pipeline could not be resolved: a dictionary profile has no
    /// dictionary directory, this build has no `lindera` feature, or the
    /// dictionary is unreadable.
    Analysis(AnalysisError),
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
            OpenError::AnalysisVersion { index, runtime } => match index {
                Some(found) => write!(
                    formatter,
                    "index analysis_version {found} is incompatible with runtime {runtime}"
                ),
                None => write!(
                    formatter,
                    "index has no analysis_version: a multilingual index built before \
                     the pipeline was versioned is stale by definition (this build \
                     writes {runtime}); rebuild it"
                ),
            },
            OpenError::AnalysisMismatch { index, expected } => match index {
                Some(found) if found == REMOVED_MULTILINGUAL_V1 => write!(
                    formatter,
                    "index analysis profile {found} is the removed language-less profile; \
                     rebuild the index with a language profile (for example {})",
                    Profile::Multilingual { language: Language::Ja }
                ),
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
            OpenError::UnknownAnalysisProfile { value } => write!(
                formatter,
                "index analysis profile {value:?} is not one this build knows; \
                 rebuild the index with sag2-zh-v1 or multilingual-v1-<lang>"
            ),
            OpenError::BuildUnavailable => write!(
                formatter,
                "index building needs a corpus; use CorpusIndex::open with the corpus rows"
            ),
            OpenError::Build(error) => write!(formatter, "cannot build the index: {error}"),
            OpenError::Analysis(error) => write!(formatter, "{error}"),
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

impl From<AnalysisError> for OpenError {
    fn from(error: AnalysisError) -> Self {
        OpenError::Analysis(error)
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
/// under another profile is rejected under every [`RebuildPolicy`] - before the
/// signature is even considered - because a rebuild would silently replace the
/// token stream the file was built for.  A multilingual profile additionally
/// requires the recorded [`ANALYSIS_VERSION_KEY`] to name this build's
/// [`MULTILINGUAL_VERSION`]; a stale or unrecorded pipeline revision is a
/// rebuild under [`RebuildPolicy::OnCorpusChange`] and
/// [`RebuildPolicy::Always`], and a rejection under [`RebuildPolicy::Strict`].
///
/// This is the dictionary-free form: a profile that does not segment with a
/// dictionary has no fingerprint, so [`decide_with_dictionary`] is the entry
/// point that also compares `rust_index_metadata.analysis_dict`.
pub fn decide(
    policy: RebuildPolicy,
    existing: &ExistingIndex,
    expected_signature: &str,
    runtime: &RuntimeVersions,
    profile: Profile,
) -> Decision {
    decide_with_dictionary(policy, existing, expected_signature, runtime, profile, None)
}

/// [`decide`] with the dictionary fingerprint the caller resolved.
///
/// `dictionary` is the [`DictionaryAnalysis::fingerprint`] of the directory the
/// profile was resolved with, or `None` for a dictionary-free profile.  An index
/// recorded with another dictionary is refused under every [`RebuildPolicy`],
/// exactly like a profile mismatch: rebuilding would replace the token stream
/// the index was built for.
pub fn decide_with_dictionary(
    policy: RebuildPolicy,
    existing: &ExistingIndex,
    expected_signature: &str,
    runtime: &RuntimeVersions,
    profile: Profile,
    dictionary: Option<&str>,
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

    // The analysis profile and its dictionary are checked before the corpus
    // signature: an index built by another pipeline is refused whatever the
    // signature says, because a rebuild would replace the token stream it was
    // built for.
    let analysis = existing
        .rust_metadata
        .as_ref()
        .and_then(|metadata| metadata.get("analysis").map(String::as_str));
    if let Err(error) = check_analysis(analysis, profile) {
        return Decision::Reject(error);
    }
    // A multilingual index must record the pipeline that produced its tokens.
    // An index from before the key existed, or under another revision, is stale
    // by definition: a rebuild policy replaces it, `Strict` reports why.
    if let Err(error) = check_analysis_version(existing.rust_metadata.as_ref(), profile) {
        return match policy {
            RebuildPolicy::Strict => Decision::Reject(error),
            RebuildPolicy::OnCorpusChange | RebuildPolicy::Always => {
                Decision::Rebuild(RebuildReason::AnalysisVersionChanged)
            }
        };
    }
    if let Err(error) = check_dictionary(existing.rust_metadata.as_ref(), dictionary) {
        return Decision::Reject(error);
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
        Some(metadata) => match check_versions(metadata, runtime, profile, dictionary) {
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

/// Require the recorded multilingual pipeline revision.
///
/// The reference profile records no `analysis_version` row and is unaffected:
/// its token stream is the one the nine design version keys already certify.
/// A multilingual profile must record the revision that produced its tokens,
/// because those same nine keys are identical across pipeline revisions and
/// cannot tell one stream from another.  An absent row belongs to an index from
/// before the pipeline was versioned, which is stale by definition.
fn check_analysis_version(
    metadata: Option<&BTreeMap<String, String>>,
    profile: Profile,
) -> Result<(), OpenError> {
    if profile == Profile::Sag2ZhV1 {
        return Ok(());
    }
    let stored =
        metadata.and_then(|metadata| metadata.get(ANALYSIS_VERSION_KEY).map(String::as_str));
    match stored {
        Some(value) if value == MULTILINGUAL_VERSION => Ok(()),
        Some(value) => Err(OpenError::AnalysisVersion {
            index: Some(value.to_owned()),
            runtime: MULTILINGUAL_VERSION,
        }),
        None => Err(OpenError::AnalysisVersion {
            index: None,
            runtime: MULTILINGUAL_VERSION,
        }),
    }
}

/// Compare the recorded dictionary fingerprint with the resolved one.
///
/// `dictionary` is the fingerprint of the directory the profile was resolved
/// with, or `None` for a dictionary-free profile; the recorded value is
/// [`ANALYSIS_DICT_NONE`] for the same case.  A row without the key predates
/// this provenance and is `"none"`, so every index built before the dictionary
/// profiles still opens.  Any other difference is
/// [`OpenError::IncompatibleIndex`] - a provenance mismatch, never a rebuild.
fn check_dictionary(
    metadata: Option<&BTreeMap<String, String>>,
    dictionary: Option<&str>,
) -> Result<(), OpenError> {
    let expected = dictionary.unwrap_or(ANALYSIS_DICT_NONE);
    let stored = metadata.and_then(|metadata| metadata.get(ANALYSIS_DICT_KEY).map(String::as_str));
    match stored {
        Some(value) if value == expected => Ok(()),
        None if expected == ANALYSIS_DICT_NONE => Ok(()),
        Some(value) => Err(OpenError::IncompatibleIndex {
            field: ANALYSIS_DICT_KEY,
            index: value.to_owned(),
            runtime: expected.to_owned(),
        }),
        None => Err(OpenError::IncompatibleIndex {
            field: ANALYSIS_DICT_KEY,
            index: ANALYSIS_DICT_NONE.to_owned(),
            runtime: expected.to_owned(),
        }),
    }
}

fn check_versions(
    metadata: &BTreeMap<String, String>,
    runtime: &RuntimeVersions,
    profile: Profile,
    dictionary: Option<&str>,
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
    check_analysis(metadata.get("analysis").map(String::as_str), profile)?;
    check_analysis_version(Some(metadata), profile)?;
    check_dictionary(Some(metadata), dictionary)
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
    let dictionary =
        DictionaryAnalysis::resolve(options.profile, options.dictionary_dir.as_deref())?
            .map(Arc::new);
    let expected = signature(
        fingerprint,
        options.chunk_chars,
        options.source_dir.as_deref(),
    );
    let existing = inspect(path)?;
    match decide_with_dictionary(
        options.rebuild,
        &existing,
        &expected,
        &RUNTIME_VERSIONS,
        options.profile,
        dictionary.as_deref().map(DictionaryAnalysis::fingerprint),
    ) {
        Decision::Reject(error) => Err(error),
        Decision::Build | Decision::Rebuild(_) => Err(OpenError::BuildUnavailable),
        Decision::Reuse => Ok((
            open_readonly_with(
                path,
                &expected,
                options.profile,
                dictionary,
                QueryOptions::default(),
                &RUNTIME_VERSIONS,
            )?,
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
        let dictionary =
            DictionaryAnalysis::resolve(options.profile, options.dictionary_dir.as_deref())?
                .map(Arc::new);
        let expected = signature(
            fingerprint,
            options.chunk_chars,
            options.source_dir.as_deref(),
        );
        let existing = inspect(path)?;
        let outcome = match decide_with_dictionary(
            options.rebuild,
            &existing,
            &expected,
            &RUNTIME_VERSIONS,
            options.profile,
            dictionary.as_deref().map(DictionaryAnalysis::fingerprint),
        ) {
            Decision::Reject(error) => return Err(error),
            Decision::Reuse => OpenOutcome::reused(),
            Decision::Build => {
                build::build(
                    path,
                    corpus,
                    fingerprint,
                    &options,
                    &RUNTIME_VERSIONS,
                    dictionary.as_deref(),
                )?;
                OpenOutcome {
                    action: OpenAction::Built,
                    rebuilt: None,
                }
            }
            Decision::Rebuild(reason) => {
                build::build(
                    path,
                    corpus,
                    fingerprint,
                    &options,
                    &RUNTIME_VERSIONS,
                    dictionary.as_deref(),
                )?;
                OpenOutcome {
                    action: OpenAction::Rebuilt,
                    rebuilt: Some(reason),
                }
            }
        };
        Ok((
            open_readonly_with(
                path,
                &expected,
                options.profile,
                dictionary,
                QueryOptions::default(),
                &RUNTIME_VERSIONS,
            )?,
            outcome,
        ))
    }
}

/// Open `path` read-only, verifying that it was built for `expected_signature`
/// and `profile`.
///
/// This entry point resolves no dictionary, so a Japanese `profile` fails with
/// [`OpenError::Analysis`] wrapping [`AnalysisError::DictionaryMissing`].  Use
/// [`open`] with `IndexOptions::dictionary_dir` when the index needs a
/// dictionary.
pub fn open_readonly(
    path: &Path,
    expected_signature: &str,
    profile: Profile,
    options: QueryOptions,
) -> Result<CorpusIndex, OpenError> {
    let dictionary = DictionaryAnalysis::resolve(profile, None)?.map(Arc::new);
    open_readonly_with(
        path,
        expected_signature,
        profile,
        dictionary,
        options,
        &RUNTIME_VERSIONS,
    )
}

/// Open an existing index as the profile and dictionary it records.
///
/// This is the entry point for a caller that does not know how the file was
/// built: the profile comes from `rust_index_metadata.analysis` (a row without
/// the key is [`Profile::Sag2ZhV1`], and an unparseable one is
/// [`OpenError::UnknownAnalysisProfile`]), and `dictionary_dir` resolves the
/// runtime dictionary a Japanese index records.  The recorded
/// provenance is still validated: the signature, the version keys, the profile
/// and the dictionary fingerprint all have to agree.
pub fn open_recorded(
    path: &Path,
    dictionary_dir: Option<&Path>,
    options: QueryOptions,
) -> Result<CorpusIndex, OpenError> {
    let existing = inspect(path)?;
    if !existing.file_exists {
        return Err(OpenError::MissingDatabase(path.to_owned()));
    }
    let signature = existing
        .signature
        .clone()
        .ok_or(OpenError::MissingSignature)?;
    let profile = recorded_profile(&existing)?;
    let dictionary = DictionaryAnalysis::resolve(profile, dictionary_dir)?.map(Arc::new);
    open_readonly_with(
        path,
        &signature,
        profile,
        dictionary,
        options,
        &RUNTIME_VERSIONS,
    )
}

/// The profile [`ExistingIndex`] records, without opening the file.
///
/// A file with no `analysis` row was built by the reference pipeline (or by
/// this port before profiles existed), so it reads as [`Profile::Sag2ZhV1`].
pub fn recorded_profile(existing: &ExistingIndex) -> Result<Profile, OpenError> {
    let stored = existing
        .rust_metadata
        .as_ref()
        .and_then(|metadata| metadata.get(ANALYSIS_KEY));
    match stored {
        None => Ok(Profile::Sag2ZhV1),
        Some(value) => value
            .parse::<Profile>()
            .map_err(|_| OpenError::UnknownAnalysisProfile {
                value: value.clone(),
            }),
    }
}

fn open_readonly_with(
    path: &Path,
    expected_signature: &str,
    profile: Profile,
    dictionary: Option<Arc<DictionaryAnalysis>>,
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
    let dictionary_fingerprint = dictionary.as_deref().map(DictionaryAnalysis::fingerprint);
    match &existing.rust_metadata {
        Some(metadata) => check_versions(metadata, runtime, profile, dictionary_fingerprint)?,
        None => {
            check_analysis(None, profile)?;
            check_dictionary(None, dictionary_fingerprint)?;
        }
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
        profile,
        dictionary,
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
        decide, decide_with_dictionary, Decision, ExistingIndex, Language, OpenError, Profile,
        RebuildPolicy, RebuildReason, RuntimeVersions, ANALYSIS_VERSION_KEY, FORMAT_VERSION,
        MULTILINGUAL_VERSION, REMOVED_MULTILINGUAL_V1, RUNTIME_VERSIONS,
    };

    const MULTILINGUAL: Profile = Profile::Multilingual {
        language: Language::Ja,
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

    /// The metadata of a current multilingual index: the ten design keys plus
    /// `analysis` and `analysis_version`.
    fn multilingual_metadata() -> BTreeMap<String, String> {
        let mut metadata = current_metadata();
        metadata.insert("analysis".to_owned(), MULTILINGUAL.as_str().to_owned());
        metadata.insert(
            ANALYSIS_VERSION_KEY.to_owned(),
            MULTILINGUAL_VERSION.to_owned(),
        );
        metadata
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
            match rejection(decide(policy, &rust, EXPECTED, &RUNTIME, MULTILINGUAL)) {
                OpenError::AnalysisMismatch { index, expected } => {
                    assert_eq!(index.as_deref(), Some("sag2-zh-v1"));
                    assert_eq!(expected, MULTILINGUAL);
                }
                other => panic!("{policy:?} must refuse, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_multilingual_index_is_reused_only_under_its_own_profile() {
        let rust = existing(Some(EXPECTED), Some(multilingual_metadata()));
        is_reuse(decide(
            RebuildPolicy::Strict,
            &rust,
            EXPECTED,
            &RUNTIME,
            MULTILINGUAL,
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
    fn a_multilingual_index_without_a_pipeline_version_is_stale() {
        // Every index built before the multilingual pipeline was versioned
        // records `analysis` but no `analysis_version`; its tokens came from a
        // tokenizer this build no longer has.
        let mut metadata = current_metadata();
        metadata.insert("analysis".to_owned(), MULTILINGUAL.as_str().to_owned());
        let rust = existing(Some(EXPECTED), Some(metadata));

        let error = rejection(decide(
            RebuildPolicy::Strict,
            &rust,
            EXPECTED,
            &RUNTIME,
            MULTILINGUAL,
        ));
        match &error {
            OpenError::AnalysisVersion { index, runtime } => {
                assert_eq!(*index, None);
                assert_eq!(*runtime, MULTILINGUAL_VERSION);
            }
            other => panic!("unexpected {other}"),
        }
        let message = error.to_string();
        assert!(message.contains("stale by definition"), "{message}");
        assert!(message.contains("rebuild it"), "{message}");

        for policy in [RebuildPolicy::OnCorpusChange, RebuildPolicy::Always] {
            assert_eq!(
                is_rebuild(decide(policy, &rust, EXPECTED, &RUNTIME, MULTILINGUAL)),
                Some(RebuildReason::AnalysisVersionChanged),
                "{policy:?}"
            );
        }
    }

    #[test]
    fn a_bumped_multilingual_pipeline_version_is_stale() {
        // Simulates the next pipeline revision: the recorded value is not this
        // build's constant.
        let mut metadata = multilingual_metadata();
        metadata.insert(ANALYSIS_VERSION_KEY.to_owned(), "ml-0".to_owned());
        let rust = existing(Some(EXPECTED), Some(metadata));

        match rejection(decide(
            RebuildPolicy::Strict,
            &rust,
            EXPECTED,
            &RUNTIME,
            MULTILINGUAL,
        )) {
            OpenError::AnalysisVersion { index, runtime } => {
                assert_eq!(index.as_deref(), Some("ml-0"));
                assert_eq!(runtime, MULTILINGUAL_VERSION);
            }
            other => panic!("unexpected {other}"),
        }
        assert_eq!(
            is_rebuild(decide(
                RebuildPolicy::OnCorpusChange,
                &rust,
                EXPECTED,
                &RUNTIME,
                MULTILINGUAL
            )),
            Some(RebuildReason::AnalysisVersionChanged)
        );
    }

    #[test]
    fn the_reference_profile_ignores_the_pipeline_version() {
        // The default key set is the ten design keys exactly: no
        // `analysis_version` row, and its absence must not stale an index.
        let rust = existing(Some(EXPECTED), Some(current_metadata()));
        is_reuse(decide(
            RebuildPolicy::Strict,
            &rust,
            EXPECTED,
            &RUNTIME,
            Profile::Sag2ZhV1,
        ));
        // A stray row is an unknown key (design §3.2 rule 4), not a stale one.
        let mut stray = current_metadata();
        stray.insert(ANALYSIS_VERSION_KEY.to_owned(), "ml-0".to_owned());
        is_reuse(decide(
            RebuildPolicy::Strict,
            &existing(Some(EXPECTED), Some(stray)),
            EXPECTED,
            &RUNTIME,
            Profile::Sag2ZhV1,
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
            MULTILINGUAL,
        )) {
            OpenError::AnalysisMismatch { index, expected } => {
                assert_eq!(index, None);
                assert_eq!(expected, MULTILINGUAL);
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
                MULTILINGUAL
            )),
            OpenError::AnalysisMismatch { index: None, .. }
        ));
    }

    #[test]
    fn the_removed_language_less_profile_is_never_mapped() {
        let mut metadata = current_metadata();
        metadata.insert("analysis".to_owned(), REMOVED_MULTILINGUAL_V1.to_owned());
        let rust = existing(Some(EXPECTED), Some(metadata));
        for profile in [Profile::Sag2ZhV1, MULTILINGUAL] {
            for policy in [
                RebuildPolicy::Strict,
                RebuildPolicy::OnCorpusChange,
                RebuildPolicy::Always,
            ] {
                let error = rejection(decide(policy, &rust, EXPECTED, &RUNTIME, profile));
                match &error {
                    OpenError::AnalysisMismatch { index, .. } => {
                        assert_eq!(index.as_deref(), Some(REMOVED_MULTILINGUAL_V1))
                    }
                    other => panic!("unexpected {other}"),
                }
                let message = error.to_string();
                assert!(
                    message.contains("rebuild the index with a language profile"),
                    "{message}"
                );
            }
        }
    }

    #[test]
    fn an_analysis_mismatch_wins_over_a_signature_change() {
        // Both the signature and the profile disagree; the profile check runs
        // first, so no policy may treat this as a rebuildable index.
        let changed = existing(Some(OTHER), Some(current_metadata()));
        for policy in [
            RebuildPolicy::Strict,
            RebuildPolicy::OnCorpusChange,
            RebuildPolicy::Always,
        ] {
            match rejection(decide(policy, &changed, EXPECTED, &RUNTIME, MULTILINGUAL)) {
                OpenError::AnalysisMismatch { index, expected } => {
                    assert_eq!(index.as_deref(), Some(Profile::Sag2ZhV1.as_str()));
                    assert_eq!(expected, MULTILINGUAL);
                }
                other => panic!("{policy:?} must refuse by profile, got {other:?}"),
            }
        }

        // Matching profiles keep the signature rules exactly as they were.
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
                Profile::Sag2ZhV1
            )),
            Some(RebuildReason::SignatureChanged {
                field: "fingerprint"
            })
        );
    }

    #[test]
    fn an_unknown_analysis_value_is_rejected() {
        let mut metadata = current_metadata();
        metadata.insert("analysis".to_owned(), "sag2-zh-v0".to_owned());
        let rust = existing(Some(EXPECTED), Some(metadata));
        for profile in [Profile::Sag2ZhV1, MULTILINGUAL] {
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

    /// A `multilingual-v1-ja` index with `analysis_dict` set to `dictionary`.
    fn dictionary_index(dictionary: &str) -> ExistingIndex {
        let mut metadata = current_metadata();
        metadata.insert(
            "analysis".to_owned(),
            Profile::Multilingual {
                language: Language::Ja,
            }
            .as_str()
            .to_owned(),
        );
        metadata.insert(
            ANALYSIS_VERSION_KEY.to_owned(),
            MULTILINGUAL_VERSION.to_owned(),
        );
        metadata.insert(
            crate::open::ANALYSIS_DICT_KEY.to_owned(),
            dictionary.to_owned(),
        );
        existing(Some(EXPECTED), Some(metadata))
    }

    #[test]
    fn the_same_dictionary_is_reused() {
        const FINGERPRINT: &str =
            "0000000000000000000000000000000000000000000000000000000000000001";
        let index = dictionary_index(FINGERPRINT);
        for policy in [RebuildPolicy::Strict, RebuildPolicy::OnCorpusChange] {
            is_reuse(decide_with_dictionary(
                policy,
                &index,
                EXPECTED,
                &RUNTIME,
                MULTILINGUAL,
                Some(FINGERPRINT),
            ));
        }
    }

    #[test]
    fn a_different_dictionary_is_rejected_under_every_policy() {
        const STORED: &str = "0000000000000000000000000000000000000000000000000000000000000001";
        const RESOLVED: &str = "0000000000000000000000000000000000000000000000000000000000000002";
        let index = dictionary_index(STORED);
        for policy in [
            RebuildPolicy::Strict,
            RebuildPolicy::OnCorpusChange,
            // Even an explicit rebuild must not replace the token stream the
            // index was built for.
            RebuildPolicy::Always,
        ] {
            match rejection(decide_with_dictionary(
                policy,
                &index,
                EXPECTED,
                &RUNTIME,
                MULTILINGUAL,
                Some(RESOLVED),
            )) {
                OpenError::IncompatibleIndex {
                    field,
                    index,
                    runtime,
                } => {
                    assert_eq!(field, crate::open::ANALYSIS_DICT_KEY);
                    assert_eq!(index, STORED);
                    assert_eq!(runtime, RESOLVED);
                }
                other => panic!("{policy:?} must refuse by dictionary, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_none_row_never_matches_a_dictionary() {
        // The stored value is a fingerprint, the caller resolved none: refuse.
        let mut reference = current_metadata();
        reference.insert(
            crate::open::ANALYSIS_DICT_KEY.to_owned(),
            "00000000000000000000000000000000000000000000000000000000000000ff".to_owned(),
        );
        let index = existing(Some(EXPECTED), Some(reference));
        assert!(matches!(
            rejection(decide_with_dictionary(
                RebuildPolicy::Strict,
                &index,
                EXPECTED,
                &RUNTIME,
                Profile::Sag2ZhV1,
                None,
            )),
            OpenError::IncompatibleIndex {
                field: crate::open::ANALYSIS_DICT_KEY,
                ..
            }
        ));

        // The stored value is `none`, the caller resolved a fingerprint: refuse.
        let ja = {
            let mut metadata = current_metadata();
            metadata.insert("analysis".to_owned(), MULTILINGUAL.as_str().to_owned());
            metadata.insert(
                ANALYSIS_VERSION_KEY.to_owned(),
                MULTILINGUAL_VERSION.to_owned(),
            );
            metadata.insert(
                crate::open::ANALYSIS_DICT_KEY.to_owned(),
                crate::open::ANALYSIS_DICT_NONE.to_owned(),
            );
            existing(Some(EXPECTED), Some(metadata))
        };
        assert!(matches!(
            rejection(decide_with_dictionary(
                RebuildPolicy::Always,
                &ja,
                EXPECTED,
                &RUNTIME,
                MULTILINGUAL,
                Some("00000000000000000000000000000000000000000000000000000000000000aa"),
            )),
            OpenError::IncompatibleIndex {
                field: crate::open::ANALYSIS_DICT_KEY,
                ..
            }
        ));
        // The sentinel is not a hash, so `none` can never equal a fingerprint.
        assert_ne!(crate::open::ANALYSIS_DICT_NONE.len(), 64);
    }

    #[test]
    fn a_legacy_row_without_the_key_is_none() {
        // Every index built before the dictionary profiles has no
        // `analysis_dict` row; that is `none`, not a fingerprint.
        let rust = existing(Some(EXPECTED), Some(current_metadata()));
        is_reuse(decide_with_dictionary(
            RebuildPolicy::Strict,
            &rust,
            EXPECTED,
            &RUNTIME,
            Profile::Sag2ZhV1,
            None,
        ));
        // A dictionary profile cannot read it: the missing row is `none`.
        let ja = existing(Some(EXPECTED), Some(multilingual_metadata()));
        assert!(matches!(
            rejection(decide_with_dictionary(
                RebuildPolicy::Always,
                &ja,
                EXPECTED,
                &RUNTIME,
                MULTILINGUAL,
                Some("00000000000000000000000000000000000000000000000000000000000000aa"),
            )),
            OpenError::IncompatibleIndex {
                field: crate::open::ANALYSIS_DICT_KEY,
                ..
            }
        ));
    }
}
