//! Thai word segmentation: `nlpo3`'s dictionary maximal matching (`newmm`).
//!
//! Thai writes without spaces, so the shared §4 pipeline can only emit one
//! token per code point (no dictionary entry covers Thai).  [`segment`] runs
//! the PyThaiNLP algorithm instead: dict-based maximal matching constrained by
//! Thai character clusters, over the CC0 word list vendored in
//! `data/thai/words_th.txt`.  The crate ships the algorithm but no dictionary,
//! which is why the list is part of the build (`include_str!`), keeping the
//! index build offline and version-stable.
//!
//! The ASCII model and the final filters are *not* here: they are shared with
//! every other profile and live in [`crate::analysis::tokenize_with`].

use std::sync::LazyLock;

use nlpo3::tokenizer::newmm::NewmmTokenizer;
use nlpo3::tokenizer::tokenizer_trait::Tokenizer;

use crate::analysis::{AnalysisError, Language};

/// `data/thai/words_th.txt`: the PyThaiNLP `words_th` corpus (CC0-1.0), byte
/// for byte; see `data/PROVENANCE.md`.
static WORDS: &str = include_str!("../../../data/thai/words_th.txt");

/// The tokenizer over [`WORDS`], built once per process.
///
/// One word per line; a few entries are multi-word expressions that keep their
/// inner spaces (`newmm` matches them as one unit).
static TOKENIZER: LazyLock<NewmmTokenizer> =
    LazyLock::new(|| NewmmTokenizer::from_word_list(WORDS.lines().map(str::to_owned).collect()));

/// Segment `text` into Thai words.
///
/// Tokens are the tokenizer's slices of `text`; whitespace-only slices are
/// dropped, so an empty or blank input yields no token at all.
///
/// Safe mode is PyThaiNLP's default: for an input of 140 character clusters or
/// more, the scan window moves before a run the dictionary cannot cover, so a
/// long unspaced paragraph never becomes one over-long token.  `parallel` is
/// off: the token stream does not depend on a worker pool, so an index built
/// twice gets the same tokens.
///
/// `newmm` reports the input it cannot cover as an error; that error is
/// [`AnalysisError::Segmentation`] here, never an abort and never a
/// character-level fallback.
pub fn segment(text: &str) -> Result<Vec<String>, AnalysisError> {
    let tokens =
        TOKENIZER
            .segment(text, true, false)
            .map_err(|error| AnalysisError::Segmentation {
                language: Language::Th,
                source: error.into(),
            })?;
    Ok(tokens
        .into_iter()
        .filter(|token| !token.trim().is_empty())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{segment, WORDS};
    use crate::analysis::AnalysisError;
    use sha2::Digest;

    #[test]
    fn the_vendored_word_list_is_the_pinned_upstream_file() {
        // C6: the provenance table documents the digest; this makes a silent
        // edit of the word list a test failure instead of a note.
        const SHA256: &str = "12bf4d04e8d59a8b235eb9e7ac59b18e5f4d56320f25e75e9c9f8f565543b481";
        let bytes = include_bytes!("../../../data/thai/words_th.txt");
        assert_eq!(bytes.len(), 1_520_294, "bytes, see data/PROVENANCE.md");
        assert_eq!(
            WORDS.lines().count(),
            62_109,
            "lines, see data/PROVENANCE.md"
        );
        let digest = sha2::Sha256::digest(bytes);
        let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(hex, SHA256, "the vendored Thai word list changed");
    }

    #[test]
    fn blank_input_yields_no_token() {
        assert!(segment("").expect("tokens").is_empty());
        assert!(segment(" \t\n ").expect("tokens").is_empty());
    }

    #[test]
    fn a_thai_word_is_one_token() {
        assert_eq!(segment("ห้องสมุด").expect("tokens"), ["ห้องสมุด"]);
    }

    #[test]
    fn an_unspaced_sentence_splits_into_words() {
        // Not one long token and not one token per code point.
        let words = segment("ภาษาไทยไม่มีช่องว่างระหว่างคำ").expect("tokens");
        assert!(words.len() > 1, "{words:?}");
        assert_eq!(words.concat(), "ภาษาไทยไม่มีช่องว่างระหว่างคำ");
    }

    #[test]
    fn a_segmentation_failure_is_a_typed_error_never_a_panic() {
        // `newmm` reports the input it cannot cover as `Err`; every one of
        // these inputs is user text, so none of them may abort, and none may
        // silently degrade to code points either (C3).
        for text in [
            "\u{0E31}",
            "\u{0E31}\u{0E31}",
            "\u{0E48}",
            "\u{0E33}",
            "\u{0E4D}\u{0E32}",
            "abc\u{0301}",
            "\u{200B}",
            "🙂",
            "\u{0E01}\u{0E31}\u{0E31}",
            "\u{0301}\u{0301}",
        ] {
            match segment(text) {
                Ok(tokens) => assert!(
                    !tokens.is_empty() || text.trim().is_empty(),
                    "{text:?} produced no token"
                ),
                Err(error) => assert!(
                    matches!(error, AnalysisError::Segmentation { .. }),
                    "{text:?} produced {error:?}"
                ),
            }
        }
    }
}
