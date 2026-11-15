//! The analysis-profile seam: [`tokenize`] is the `Sag2ZhV1` profile, and every
//! multilingual profile normalises before the same certified pipeline.
//!
//! The pinned vectors are the current certified `Sag2ZhV1` output; Gate D's
//! `golden/tokenizer.jsonl` replay stays the authority, and these guard the
//! default path from moving.

use sag2_core::{tokenize, tokenize_with, Language, Profile};

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
fn a_multilingual_profile_normalises_before_the_certified_pipeline() {
    let german = Profile::Multilingual {
        language: Language::De,
    };
    // Foldable text reaches different tokens through the multilingual profile:
    // `ß` becomes `ss` before the tokenizer runs.
    assert_eq!(tokenize_with(german, "Straße"), ["strasse"]);
    assert_ne!(tokenize("Straße"), tokenize_with(german, "Straße"));

    // Text the fold does not touch keeps the certified tokens.
    for text in [
        "Alice Paul birthplace",
        "hello.world_name-42",
        "The quick brown fox jumps over the lazy dog.",
        "2024-11-15T09:30:00Z",
        "中华人民共和国成立了",
    ] {
        assert_eq!(tokenize_with(german, text), tokenize(text), "{text:?}");
    }
}

#[test]
fn profile_identifiers_are_the_metadata_values() {
    assert_eq!(Profile::Sag2ZhV1.as_str(), "sag2-zh-v1");
    let german = Profile::Multilingual {
        language: Language::De,
    };
    assert_eq!(german.as_str(), "multilingual-v1-de");
    assert_eq!("multilingual-v1-de".parse::<Profile>(), Ok(german));
    assert_eq!("sag2-zh-v1".parse::<Profile>(), Ok(Profile::Sag2ZhV1));
    assert!("sag2-zh-v0".parse::<Profile>().is_err());
    // T1's language-less identifier is gone and is not mapped onto anything.
    assert!("multilingual-v1".parse::<Profile>().is_err());
}
