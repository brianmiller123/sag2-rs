//! The Japanese segmenter against a real lindera dictionary.
//!
//! `lindera` dictionaries are ~10-20 MB of compiled artifacts, so they are not
//! in this repository and this test is `#[ignore]`d by default.  Point
//! `SAG2_DICT_DIR` at a compiled ipadic directory and run:
//!
//! ```sh
//! SAG2_DICT_DIR=/path/to/ipadic \
//!   cargo test --offline -p sag2-core --features lindera --test lindera_dictionary -- --ignored
//! ```
//!
//! The directory must hold the files lindera's filesystem loader reads:
//! `metadata.json`, `char_def.bin`, `unk.bin`, `matrix.mtx`, `dict.trie`,
//! `dict.valsidx`, `dict.vals`, `dict.wordsidx`, `dict.words`.

#![cfg(feature = "lindera")]

use std::path::PathBuf;

use sag2_core::analysis::segment::lindera_seg::LinderaSegmenter;
use sag2_core::{tokenize_with_segmenter, DictionaryAnalysis, Language, Profile};

const SAMPLE: &str = "東京都に住んでいます。";

fn japanese() -> Profile {
    Profile::Multilingual {
        language: Language::Ja,
    }
}

fn dictionary_dir() -> PathBuf {
    std::env::var_os("SAG2_DICT_DIR")
        .map(PathBuf::from)
        .expect("set SAG2_DICT_DIR to a compiled lindera dictionary directory")
}

#[test]
#[ignore = "needs a real ipadic directory in SAG2_DICT_DIR"]
fn a_real_dictionary_segments_into_words() {
    let segmenter = LinderaSegmenter::open(&dictionary_dir()).expect("load the dictionary");
    let tokens = segmenter.segment(SAMPLE).expect("segment");

    assert!(tokens.len() > 1, "{tokens:?}");
    assert!(tokens.iter().any(|token| token.chars().count() > 1));
    // The tokens are the input's slices: nothing is dropped or duplicated.
    assert_eq!(tokens.concat(), SAMPLE);
    // The fingerprint is a stable identifier of the loaded dictionary.
    let fingerprint = LinderaSegmenter::fingerprint(&dictionary_dir()).expect("fingerprint");
    assert_eq!(fingerprint.len(), 64);
    assert_ne!(fingerprint, "none");
}

#[test]
#[ignore = "needs a real ipadic directory in SAG2_DICT_DIR"]
fn a_real_dictionary_drives_the_profile_pipeline() {
    let profile = japanese();
    let dictionary = DictionaryAnalysis::resolve(profile, Some(&dictionary_dir()))
        .expect("resolve the dictionary")
        .expect("the Japanese profile resolves one");
    let tokens = tokenize_with_segmenter(profile, SAMPLE, Some(&dictionary))
        .expect("tokenise with the dictionary");

    assert!(tokens.len() > 1, "{tokens:?}");
    // The resolved fingerprint is what the index records.
    assert_eq!(dictionary.fingerprint().len(), 64);
    // Without the dictionary the same profile is a typed error: the dictionary
    // path is the only Japanese pipeline, so a regression in the wiring cannot
    // hide behind a character-level fallback.
    let without = tokenize_with_segmenter(profile, SAMPLE, None);
    assert!(without.is_err(), "{without:?}");
}
