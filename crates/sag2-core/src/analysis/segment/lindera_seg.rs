//! Japanese and Korean segmentation with `lindera` (feature `lindera`).
//!
//! Both languages need a morphological dictionary that is far too large to
//! ship in this repository (ipadic ≈ 10.5 MB, ko-dic ≈ 19.9 MB of compiled
//! artifacts), so the dictionary is a *runtime* directory: the caller passes it
//! through the index options and the crate loads it once per opened index with
//! `lindera`'s filesystem loader.  Nothing is downloaded at build time and the
//! default build compiles none of this.
//!
//! [`LinderaSegmenter::fingerprint`] is the provenance anchor: it hashes the
//! dictionary contents (never the absolute path), the fingerprint is stored in
//! the index's `rust_index_metadata.analysis_dict` row, and opening the index
//! with a different dictionary is refused instead of silently re-tokenising
//! (see [`crate::analysis::DictionaryAnalysis::resolve`]).

use std::borrow::Cow;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use lindera::dictionary::load_fs_dictionary_with_options;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use sha2::{Digest, Sha256};

use crate::analysis::AnalysisError;

/// `metadata.json` names the dictionary kind and format version; every other
/// file in a lindera directory is a headerless compiled array.
const METADATA: &str = "metadata.json";

/// A `lindera` segmenter over one loaded dictionary directory.
///
/// `Mode::Normal` is `lindera`'s default lattice; no user dictionary and no
/// post-processing filters are applied, so the output is exactly the
/// dictionary's Viterbi best path.  The dictionary is read eagerly
/// (`use_mmap = false`), so the loaded segmenter holds no file handles and a
/// later change to the directory cannot alter an opened index's token stream.
pub struct LinderaSegmenter {
    segmenter: Segmenter,
    dir: PathBuf,
    fingerprint: String,
}

impl LinderaSegmenter {
    /// Load the dictionary compiled into `dir`.
    ///
    /// A missing directory, a missing component or a malformed dictionary all
    /// come back as [`AnalysisError::Dictionary`]; nothing aborts the process.
    ///
    /// The fingerprint is computed here, from the same directory the loader
    /// reads, so the value an index records always describes the lattice that
    /// was actually loaded (C7).  `open` is the only place that produces a
    /// segmenter, so there is no path that records a fingerprint from one scan
    /// and segments with another.
    pub fn open(dir: &Path) -> Result<Self, AnalysisError> {
        if !dir.is_dir() {
            return Err(AnalysisError::dictionary(
                dir,
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "the dictionary directory does not exist",
                ),
            ));
        }
        let fingerprint = Self::fingerprint(dir)?;
        let dictionary = load_fs_dictionary_with_options(dir, false)
            .map_err(|error| AnalysisError::dictionary(dir, error))?;
        Ok(LinderaSegmenter {
            segmenter: Segmenter::new(Mode::Normal, dictionary, None),
            dir: dir.to_owned(),
            fingerprint,
        })
    }

    /// The fingerprint of the directory this segmenter loaded.
    pub fn dictionary_fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Segment `text` into the dictionary's tokens.
    ///
    /// The tokens are the token surfaces in input order; whitespace is not a
    /// token (`keep_whitespace` stays off).  The profile pipeline then applies
    /// the same `HAS_CONTENT`/stopword filter as every other profile.
    pub fn segment(&self, text: &str) -> Result<Vec<String>, AnalysisError> {
        let tokens = self
            .segmenter
            .segment(Cow::Borrowed(text))
            .map_err(|error| AnalysisError::dictionary(&self.dir, error))?;
        Ok(tokens
            .into_iter()
            .map(|token| token.surface.into_owned())
            .filter(|surface| !surface.is_empty())
            .collect())
    }

    /// A stable identifier of the dictionary contents in `dir`.
    ///
    /// `sha256` over the raw `metadata.json` bytes followed by every file in
    /// the directory in name order, each contributing its relative file name
    /// and the `sha256` of its contents.  Absolute paths are not part of the
    /// hash, so the same dictionary copied elsewhere fingerprints identically.
    ///
    /// A missing or unreadable directory or file is
    /// [`AnalysisError::Dictionary`]; a directory without `metadata.json` is
    /// the same error, because `lindera` cannot load it either.
    ///
    /// Symlinks are followed, because `lindera`'s loader reads the same paths:
    /// a dictionary assembled out of symlinks must fingerprint like the files
    /// it points at (C7).  `fs::metadata` follows and `DirEntry::metadata`
    /// does not, so the scan uses the former.
    pub fn fingerprint(dir: &Path) -> Result<String, AnalysisError> {
        let metadata = fs::read(dir.join(METADATA))
            .map_err(|error| AnalysisError::dictionary(dir.join(METADATA), error))?;
        let mut names: Vec<String> = fs::read_dir(dir)
            .map_err(|error| AnalysisError::dictionary(dir, error))?
            .map(|entry| {
                let entry = entry.map_err(|error| AnalysisError::dictionary(dir, error))?;
                let path = entry.path();
                let metadata =
                    fs::metadata(&path).map_err(|error| AnalysisError::dictionary(path, error))?;
                if metadata.is_file() {
                    // A lindera dictionary is flat; a subdirectory (a user
                    // dictionary, say) is not part of the system dictionary.
                    Ok(Some(entry.file_name().to_string_lossy().into_owned()))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<Option<String>>, AnalysisError>>()?
            .into_iter()
            .flatten()
            .collect();
        names.sort_unstable();

        let mut hasher = Sha256::new();
        hasher.update((metadata.len() as u64).to_le_bytes());
        hasher.update(&metadata);
        for name in names {
            hasher.update((name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
            if name == METADATA {
                hasher.update(sha256_hex(&metadata).as_bytes());
            } else {
                let path = dir.join(&name);
                let bytes =
                    fs::read(&path).map_err(|error| AnalysisError::dictionary(path, error))?;
                hasher.update(sha256_hex(&bytes).as_bytes());
            }
        }
        Ok(hex(hasher.finalize().as_slice()))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes).as_slice())
}

fn hex(digest: &[u8]) -> String {
    let mut text = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut text, "{byte:02x}").expect("writing to a String cannot fail");
    }
    text
}

impl std::fmt::Debug for LinderaSegmenter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LinderaSegmenter")
            .field("dir", &self.dir)
            .field("fingerprint", &self.fingerprint)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::LinderaSegmenter;

    fn write(directory: &std::path::Path, name: &str, contents: &[u8]) {
        fs::write(directory.join(name), contents).expect("write fixture");
    }

    #[test]
    fn open_reports_a_missing_directory_without_panicking() {
        let directory = tempfile::tempdir().expect("temp dir");
        let missing = directory.path().join("absent");
        let error = LinderaSegmenter::open(&missing).expect_err("a missing dictionary");
        assert!(matches!(
            error,
            crate::analysis::AnalysisError::Dictionary { .. }
        ));
    }

    #[test]
    fn open_reports_a_directory_without_a_dictionary() {
        let directory = tempfile::tempdir().expect("temp dir");
        let error = LinderaSegmenter::open(directory.path()).expect_err("an empty directory");
        assert!(matches!(
            error,
            crate::analysis::AnalysisError::Dictionary { .. }
        ));
    }

    #[test]
    fn fingerprint_is_stable_and_path_independent() {
        let first = tempfile::tempdir().expect("temp dir");
        let second = tempfile::tempdir().expect("temp dir");
        for directory in [first.path(), second.path()] {
            write(directory, "metadata.json", br#"{"name":"test"}"#);
            write(directory, "dict.vals", b"values");
        }
        let left = LinderaSegmenter::fingerprint(first.path()).expect("fingerprint");
        let right = LinderaSegmenter::fingerprint(second.path()).expect("fingerprint");
        assert_eq!(left, right);
        assert_eq!(left.len(), 64);
        assert_ne!(left, "none");
    }

    #[test]
    fn fingerprint_changes_with_the_contents() {
        let directory = tempfile::tempdir().expect("temp dir");
        write(directory.path(), "metadata.json", br#"{"name":"test"}"#);
        write(directory.path(), "dict.vals", b"values");
        let before = LinderaSegmenter::fingerprint(directory.path()).expect("fingerprint");
        write(directory.path(), "dict.vals", b"other values");
        let after = LinderaSegmenter::fingerprint(directory.path()).expect("fingerprint");
        assert_ne!(before, after);

        write(directory.path(), "dict.vals", b"values");
        write(directory.path(), "dict.trie", b"trie");
        let added = LinderaSegmenter::fingerprint(directory.path()).expect("fingerprint");
        assert_ne!(before, added);
    }

    #[cfg(unix)]
    #[test]
    fn fingerprint_follows_symlinks() {
        // A dictionary assembled from symlinks is the same dictionary: the
        // scan must hash the file a symlink points at instead of skipping it.
        use std::os::unix::fs::symlink;

        let real = tempfile::tempdir().expect("temp dir");
        write(real.path(), "metadata.json", br#"{"name":"test"}"#);
        write(real.path(), "dict.vals", b"values");

        let linked = tempfile::tempdir().expect("temp dir");
        fs::copy(
            real.path().join("metadata.json"),
            linked.path().join("metadata.json"),
        )
        .expect("copy metadata");
        symlink(
            real.path().join("dict.vals"),
            linked.path().join("dict.vals"),
        )
        .expect("symlink");

        assert_eq!(
            LinderaSegmenter::fingerprint(real.path()).expect("fingerprint"),
            LinderaSegmenter::fingerprint(linked.path()).expect("fingerprint"),
        );
    }

    #[test]
    fn fingerprint_reports_a_missing_metadata_file() {
        let directory = tempfile::tempdir().expect("temp dir");
        let error =
            LinderaSegmenter::fingerprint(directory.path()).expect_err("a directory without data");
        assert!(matches!(
            error,
            crate::analysis::AnalysisError::Dictionary { .. }
        ));
    }
}
