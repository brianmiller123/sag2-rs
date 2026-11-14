//! The `metadata.signature` string (`sag2.py:211-220`, design §3.2).
//!
//! The signature is the first row of the `metadata` table and the only thing
//! that tells a reader whether the database was built for this corpus.  It is
//! `compact_json([fingerprint, chunk_chars, INDEX_SCHEMA, source_dir_or_null,
//! CHUNK_POLICY, *(["plain-text-default"] if not source_dir)])` — byte for byte
//! what [`sag2_json::compact_json`] produces, which is why the type lives here
//! rather than in `sag2-core`.

use std::path::{Path, PathBuf};

use sag2_json::{compact_json, Value};

/// `sag2.py:34` `INDEX_SCHEMA`.
pub const INDEX_SCHEMA: i64 = 5;

/// `chunking.py:7` `CHUNK_POLICY`.
pub const CHUNK_POLICY: &str = "desktop-sentences-v1";

/// Field names in signature order, used to name a signature mismatch.
const FIELDS: [&str; 6] = [
    "fingerprint",
    "chunk_chars",
    "index_schema",
    "source_dir",
    "chunk_policy",
    "plain_text_default",
];

/// Build the corpus signature for `fingerprint`, `chunk_chars` and `source_dir`.
///
/// `sag2.py:203` stores `source_dir.resolve()` and signs *that*, so a relative
/// `--source-dir` must not produce a foreign signature: the path is made
/// absolute (and symlink-resolved when the directory exists) before it is
/// embedded.  `Path::resolve` in `pathlib` never fails on a missing tail, so
/// the lexical fallback mirrors it.
pub fn signature(fingerprint: &str, chunk_chars: usize, source_dir: Option<&Path>) -> String {
    let mut fields = vec![
        Value::Str(fingerprint.to_owned()),
        Value::Int(chunk_chars as i64),
        Value::Int(INDEX_SCHEMA),
        match source_dir {
            Some(path) => Value::Str(resolve(path).to_string_lossy().into_owned()),
            None => Value::Null,
        },
        Value::Str(CHUNK_POLICY.to_owned()),
    ];
    if source_dir.is_none() {
        fields.push(Value::Str("plain-text-default".to_owned()));
    }
    compact_json(&Value::Array(fields))
}

/// `Path.resolve()` (`pathlib`): absolute, symlinks resolved where they exist.
fn resolve(path: &Path) -> PathBuf {
    if let Ok(resolved) = std::fs::canonicalize(path) {
        return resolved;
    }
    if path.is_absolute() {
        return path.to_owned();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(path),
        Err(_) => path.to_owned(),
    }
}

/// Name the first signature field that differs between `expected` and `stored`.
///
/// Returns `"signature"` when either side is not a JSON array or when only the
/// length differs, so the caller always has a field name to report.
pub fn changed_field(expected: &str, stored: &str) -> &'static str {
    let (Ok(expected), Ok(stored)) = (
        serde_json::from_str::<Vec<serde_json::Value>>(expected),
        serde_json::from_str::<Vec<serde_json::Value>>(stored),
    ) else {
        return "signature";
    };
    expected
        .iter()
        .zip(stored.iter())
        .position(|(left, right)| left != right)
        .and_then(|index| FIELDS.get(index).copied())
        .unwrap_or("signature")
}

#[cfg(test)]
mod tests {
    use super::{changed_field, signature, CHUNK_POLICY};

    #[test]
    fn matches_the_reference_signature_bytes() {
        // The exact string `sag2.py` wrote into golden/indexes/C0.sqlite.
        assert_eq!(
            signature("fixture", 2200, None),
            "[\"fixture\",2200,5,null,\"desktop-sentences-v1\",\"plain-text-default\"]"
        );
        assert!(signature("fix", 10000, None).contains(CHUNK_POLICY));
    }

    #[test]
    fn source_dir_drops_the_plain_text_marker() {
        let string = signature("fix", 2200, Some(std::path::Path::new("/tmp/corpus")));
        assert_eq!(
            string,
            "[\"fix\",2200,5,\"/tmp/corpus\",\"desktop-sentences-v1\"]"
        );
    }

    #[test]
    fn relative_source_dir_is_signed_as_an_absolute_path() {
        let directory = tempfile::tempdir().expect("temp dir");
        let absolute = std::fs::canonicalize(directory.path()).expect("canonicalize");
        let signed = signature("fix", 2200, Some(directory.path()));
        let expected = format!(
            "[\"fix\",2200,5,\"{}\",\"desktop-sentences-v1\"]",
            absolute.to_string_lossy()
        );
        assert_eq!(signed, expected);
        assert_eq!(signature("fix", 2200, Some(&absolute)), expected);
    }

    #[test]
    fn names_the_first_differing_field() {
        let left = signature("a", 2200, None);
        assert_eq!(changed_field(&left, &signature("b", 2200, None)), "fingerprint");
        assert_eq!(changed_field(&left, &signature("a", 10000, None)), "chunk_chars");
        assert_eq!(
            changed_field(
                &left,
                &signature("a", 2200, Some(std::path::Path::new("/tmp/x")))
            ),
            "source_dir"
        );
        assert_eq!(changed_field(&left, "not-json"), "signature");
    }
}
