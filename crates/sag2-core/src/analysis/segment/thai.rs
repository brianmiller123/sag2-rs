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
pub fn segment(text: &str) -> Vec<String> {
    TOKENIZER
        .segment(text, true, false)
        .expect("newmm reaches every character cluster: the graph is built from the same prefixes")
        .into_iter()
        .filter(|token| !token.trim().is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::segment;

    #[test]
    fn blank_input_yields_no_token() {
        assert!(segment("").is_empty());
        assert!(segment(" \t\n ").is_empty());
    }

    #[test]
    fn a_thai_word_is_one_token() {
        assert_eq!(segment("ห้องสมุด"), ["ห้องสมุด"]);
    }

    #[test]
    fn an_unspaced_sentence_splits_into_words() {
        // Not one long token and not one token per code point.
        let words = segment("ภาษาไทยไม่มีช่องว่างระหว่างคำ");
        assert!(words.len() > 1, "{words:?}");
        assert_eq!(words.concat(), "ภาษาไทยไม่มีช่องว่างระหว่างคำ");
    }
}
