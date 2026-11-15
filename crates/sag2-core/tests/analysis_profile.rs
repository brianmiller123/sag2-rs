//! The analysis-profile seam: [`tokenize`] is the `Sag2ZhV1` profile, and the
//! profiles stay byte-identical until the multilingual segmenters land (T3-T5).
//!
//! The pinned vectors below are the current certified output; Gate D's
//! `golden/tokenizer.jsonl` replay stays the authority, and these only guard
//! the plumbing from moving the default stream.

use sag2_core::{tokenize, tokenize_with, Profile};

/// `(text, tokens)` — the `tokenize` output this build certifies.
const PINNED: [(&str, &[&str]); 6] = [
    ("Alice Paul birthplace", &["alice", "paul", "birthplace"]),
    (
        "中华人民共和国成立了",
        &[
            "中华",
            "华人",
            "人民",
            "共和",
            "共和国",
            "中华人民共和国",
            "成立",
        ],
    ),
    (
        "Café naïve — Ærøskøbing",
        &[
            "caf", "é", "na", "ï", "ve", "æ", "r", "ø", "sk", "ø", "bing",
        ],
    ),
    (
        "hello.world_name-42",
        &["hello", "world", "name", "hello.world_name-42"],
    ),
    (
        "The quick brown fox jumps over the lazy dog.",
        &["quick", "brown", "fox", "jumps", "over", "lazy", "dog"],
    ),
    (
        "2024-11-15T09:30:00Z",
        &["15t09", "2024-11-15t09", "30", "00z"],
    ),
];

#[test]
fn tokenize_is_the_sag2_zh_profile() {
    for (text, expected) in PINNED {
        assert_eq!(tokenize(text), expected, "{text:?}");
        assert_eq!(tokenize_with(Profile::Sag2ZhV1, text), expected, "{text:?}");
    }
}

#[test]
fn multilingual_v1_is_identical_until_the_segmenters_land() {
    for (text, expected) in PINNED {
        assert_eq!(
            tokenize_with(Profile::MultilingualV1, text),
            expected,
            "{text:?}"
        );
    }
}

#[test]
fn profile_identifiers_are_the_metadata_values() {
    assert_eq!(Profile::Sag2ZhV1.as_str(), "sag2-zh-v1");
    assert_eq!(Profile::MultilingualV1.as_str(), "multilingual-v1");
    assert_eq!(
        "multilingual-v1".parse::<Profile>(),
        Ok(Profile::MultilingualV1)
    );
    assert_eq!("sag2-zh-v1".parse::<Profile>(), Ok(Profile::Sag2ZhV1));
    assert!("sag2-zh-v0".parse::<Profile>().is_err());
}
