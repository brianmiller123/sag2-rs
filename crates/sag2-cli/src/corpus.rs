//! Corpus input for `sag2-cli build`: a JSONL file or a directory of UTF-8
//! text files.
//!
//! Both forms produce the same rows the index build takes
//! ([`sag2::Document`]) plus the corpus fingerprint its signature records:
//!
//! * **JSONL** - one object per line, `{"title": ..., "text": ..., "path"?,
//!   "idx"?}` (the Gate C corpus shape, minus the case fields).  The
//!   fingerprint is the `sha256` of the file, like the reference's
//!   `corpus_fingerprint`.
//! * **directory** - every regular file below it, recursively, sorted by
//!   relative path.  The title is the file stem, `path` is the relative path,
//!   and the fingerprint is the `sha256` of `relpath\0sha256(bytes)\n` over the
//!   sorted list, so the same tree always fingerprints the same way.

use std::fs;
use std::path::{Path, PathBuf};

use sag2::Document;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// One corpus row, owned so the command can lend [`Document`]s from a slice.
pub struct Row {
    /// The document title.
    title: String,
    /// The document text.
    text: String,
    /// The relative path, when the row has one.
    path: Option<String>,
    /// The raw `idx` value, when the row has one.
    idx: Option<String>,
}

/// A loaded corpus and the fingerprint its index records.
pub struct Corpus {
    /// The value passed to `CorpusIndex::open` as `fingerprint`.
    pub fingerprint: String,
    /// The rows, in build order.
    pub rows: Vec<Row>,
}

#[derive(Deserialize)]
struct JsonRow {
    title: String,
    text: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    idx: Option<String>,
}

impl Corpus {
    /// Load `input`, deciding between the JSONL and the directory form.
    pub fn load(input: &Path) -> Result<Self, String> {
        if input.is_dir() {
            Self::from_directory(input)
        } else if input.is_file() {
            Self::from_jsonl(input)
        } else {
            Err(format!("corpus input {} does not exist", input.display()))
        }
    }

    /// Borrow the rows as the documents the index build takes.
    pub fn documents(&self) -> Vec<Document<'_>> {
        self.rows
            .iter()
            .map(|row| Document {
                title: &row.title,
                text: &row.text,
                path: row.path.as_deref(),
                idx: row.idx.as_deref(),
            })
            .collect()
    }

    fn from_jsonl(input: &Path) -> Result<Self, String> {
        let bytes =
            fs::read(input).map_err(|error| format!("cannot read {}: {error}", input.display()))?;
        let fingerprint = sha256_hex(&bytes);
        let text = String::from_utf8(bytes)
            .map_err(|_| format!("{} is not valid UTF-8", input.display()))?;
        let mut rows = Vec::new();
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let row: JsonRow = serde_json::from_str(line)
                .map_err(|error| format!("{}:{}: {error}", input.display(), index + 1))?;
            rows.push(Row {
                title: row.title,
                text: row.text,
                path: row.path,
                idx: row.idx,
            });
        }
        if rows.is_empty() {
            return Err(format!(
                "{} holds no corpus rows (one JSON object per line)",
                input.display()
            ));
        }
        Ok(Corpus { fingerprint, rows })
    }

    fn from_directory(input: &Path) -> Result<Self, String> {
        let mut files = Vec::new();
        collect_files(input, &mut files)?;
        files.sort();
        if files.is_empty() {
            return Err(format!("{} holds no corpus files", input.display()));
        }
        let mut digest = Sha256::new();
        let mut rows = Vec::new();
        for file in files {
            let bytes = fs::read(&file)
                .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
            let relative = file
                .strip_prefix(input)
                .unwrap_or(&file)
                .to_string_lossy()
                .into_owned();
            let text = String::from_utf8(bytes.clone())
                .map_err(|_| format!("{} is not valid UTF-8", file.display()))?;
            digest.update(relative.as_bytes());
            digest.update([0]);
            digest.update(sha256_hex(&bytes).as_bytes());
            digest.update(b"\n");
            let title = file
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .filter(|stem| !stem.is_empty())
                .unwrap_or_else(|| relative.clone());
            rows.push(Row {
                title,
                text,
                path: Some(relative),
                idx: None,
            });
        }
        Ok(Corpus {
            fingerprint: hex(&digest.finalize()),
            rows,
        })
    }
}

/// Every regular file below `directory`, recursively (symlinks are not
/// followed: `Path::is_dir` is false for them, so they are taken as files).
fn collect_files(directory: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let mut entries: Vec<PathBuf> = fs::read_dir(directory)
        .map_err(|error| format!("cannot list {}: {error}", directory.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<_, _>>()
        .map_err(|error| format!("cannot list {}: {error}", directory.display()))?;
    entries.sort();
    for entry in entries {
        if entry.is_dir() {
            collect_files(&entry, files)?;
        } else {
            files.push(entry);
        }
    }
    Ok(())
}

/// `sha256` of `bytes`, lowercase hex.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    hex(&digest.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
