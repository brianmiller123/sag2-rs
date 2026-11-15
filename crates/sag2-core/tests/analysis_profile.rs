//! The analysis-profile seam: [`tokenize`] is the `Sag2ZhV1` profile, and every
//! multilingual profile normalises before the same certified pipeline.
//!
//! The pinned vectors are the current certified `Sag2ZhV1` output; Gate D's
//! `golden/tokenizer.jsonl` replay stays the authority, and these guard the
//! default path from moving.

use sag2_core::analysis::normalize::normalize;
use sag2_core::{tokenize, tokenize_with, Language, Profile};

/// [`tokenize_with`] for a profile that never needs a dictionary; the tests
/// below all use one, so the typed error is a test failure here.
fn tokens(profile: Profile, text: &str) -> Vec<String> {
    tokenize_with(profile, text).expect("a dictionary-free profile always tokenises")
}

/// `(text, tokens)` — the `tokenize` output this build certifies.
const PINNED: [(&str, &[&str]); 7] = [
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
    // The L1 word-character policy is a multilingual-profile policy: the
    // reference pipeline still shreds the accented word and still keeps the
    // CJK run whole.
    ("bibliothèque 東京", &["biblioth", "è", "que", "東京"]),
];

#[test]
fn tokenize_is_the_sag2_zh_profile() {
    for (text, expected) in PINNED {
        assert_eq!(tokenize(text), expected, "{text:?}");
        assert_eq!(tokens(Profile::Sag2ZhV1, text), expected, "{text:?}");
    }
}

#[test]
fn a_multilingual_profile_normalises_before_the_certified_pipeline() {
    let german = Profile::Multilingual {
        language: Language::De,
    };
    // Foldable text reaches different tokens through the multilingual profile:
    // `ß` becomes `ss` before the tokenizer runs.
    assert_eq!(tokens(german, "Straße"), ["strasse"]);
    assert_ne!(tokenize("Straße"), tokens(german, "Straße"));

    // Text the fold does not touch keeps the certified tokens wherever the two
    // classifiers agree: word characters separated by spaces, and Han runs.
    // (The `[._-]` compounds of `PINNED` do not: the multilingual word set ends
    // a run at those characters, see the dedicated test below.)
    for text in [
        "Alice Paul birthplace",
        "The quick brown fox jumps over the lazy dog.",
        "中华人民共和国成立了",
        "東京の図書館は古い。",
    ] {
        assert_eq!(tokens(german, text), tokenize(text), "{text:?}");
    }
}

/// Assert the L1 shape of one multilingual word run: every surviving code point
/// of the run comes first, in text order, and the whole run is the last token.
///
/// The run is compared against the profile's own spelling of the word
/// ([`normalize`]): Greek drops the tonos, Arabic the harakat.  [`keep`] drops
/// the code points that are not letters or numbers (combining marks like the
/// Devanagari anusvara), so the code-point prefix is a subsequence of the run
/// rather than the run itself.
fn assert_code_points_then_the_run(language: Language, word: &str) {
    let profile = Profile::Multilingual { language };
    let whole = normalize(language, word);
    let stream = tokens(profile, word);
    let (run, prefix) = stream
        .split_last()
        .unwrap_or_else(|| panic!("{profile}: {word:?} produced no token"));
    assert_eq!(run, &whole, "{profile}: the whole run is the last token");
    let mut characters = whole.chars();
    for token in prefix {
        let mut token_characters = token.chars();
        let character = token_characters.next().expect("a token is not empty");
        assert_eq!(
            token_characters.next(),
            None,
            "{profile}: {token:?} is not one code point"
        );
        assert!(
            characters.any(|expected| expected == character),
            "{profile}: {token:?} is out of order in {whole:?}"
        );
    }
}

#[test]
fn a_multilingual_profile_keeps_a_word_together_in_every_script() {
    // L1: the word characters of a multilingual profile are the Unicode letters
    // and digits, not the ASCII + `U+4E00-U+9FD5` set the reference classifier
    // pins, so an accented, Cyrillic, Greek, Arabic, Devanagari or Hangul word
    // is one run instead of one token per code point.
    for (language, word) in [
        (Language::Fr, "bibliothèque"),
        (Language::Ko, "도서관에서"),
        (Language::Ru, "библиотека"),
        (Language::El, "βιβλιοθήκη"),
        (Language::Ar, "المكتبة"),
        (Language::Hi, "बंदरगाह"),
        (Language::Pl, "pociągi"),
    ] {
        let profile = Profile::Multilingual { language };
        assert_code_points_then_the_run(language, word);
        // The code points are the recall path (§5.1): they are still there.
        assert!(
            tokens(profile, word).len() > 1,
            "{profile}: {word:?} keeps its code points"
        );
    }
    // Spot-check the order and the members for the assignment's example: the
    // run comes after its code points, never before.
    let french = Profile::Multilingual {
        language: Language::Fr,
    };
    assert_eq!(
        tokens(french, "bibliothèque"),
        [
            "b",
            "i",
            "b",
            "l",
            "i",
            "o",
            "t",
            "h",
            "è",
            "q",
            "u",
            "e",
            "bibliothèque"
        ]
    );
}

#[test]
fn a_single_code_point_run_is_not_duplicated() {
    // A one-code-point run has nothing to add: the code point is the token.
    for (language, text) in [
        (Language::Ko, "역"),
        (Language::Fr, "é"),
        (Language::Ru, "я"),
    ] {
        let profile = Profile::Multilingual { language };
        assert_eq!(tokens(profile, text), [text], "{profile}");
    }
}

#[test]
fn an_ascii_word_run_stays_one_token() {
    // The reference pipeline already reaches the word level for ASCII, so a
    // multilingual profile emits no code-point stream for it: those tokens
    // would only spend the query expression budget on characters.
    for (language, text) in [
        (Language::Fr, "gare"),
        (Language::De, "strasse"),
        (Language::Other, "harbour"),
    ] {
        let profile = Profile::Multilingual { language };
        assert_eq!(tokens(profile, text), [text], "{profile}");
    }
}

#[test]
fn a_stacked_mark_ends_a_word_run() {
    // A word character is a letter or mark with `combining_class == 0` (L1):
    // the Devanagari vowel signs and anusvara (ccc 0) stay inside the run, the
    // nukta (ccc 7) and the virama (ccc 9) end it.
    let hindi = Profile::Multilingual {
        language: Language::Hi,
    };
    assert!(tokens(hindi, "बंदरगाह")
        .iter()
        .any(|token| token == "बंदरगाह"));
    let nukta = tokens(hindi, "बाज़ार");
    assert!(
        !nukta.iter().any(|token| token == "बाज़ार"),
        "the nukta ends the run: {nukta:?}"
    );
    assert!(nukta.iter().any(|token| token == "बाज"), "{nukta:?}");
    let virama = tokens(hindi, "स्कूल");
    assert!(
        !virama.iter().any(|token| token == "स्कूल"),
        "the virama ends the run: {virama:?}"
    );
}

#[test]
fn a_multilingual_profile_leaves_han_runs_to_jieba() {
    // Chinese has no language slot of its own: `Sag2ZhV1` is the only Chinese
    // classifier, and a Han run inside any multilingual profile is cut by the
    // same jieba path, with the same code-point tokens outside it.
    for language in [Language::Other, Language::Ko, Language::Fr] {
        let profile = Profile::Multilingual { language };
        for text in [
            "中华人民共和国成立了。",
            "图书馆在公园旁边，天气很好。",
            "東京の図書館は古い。",
        ] {
            assert_eq!(tokens(profile, text), tokenize(text), "{profile}: {text:?}");
        }
    }
}

#[test]
fn a_multilingual_profile_ends_a_word_at_the_ascii_interface_characters() {
    // The multilingual classifier is the script-based word set (L1), not the
    // reference's `[._-]` interface set, so a compound keeps only its parts.
    let german = Profile::Multilingual {
        language: Language::De,
    };
    assert_eq!(
        tokens(german, "hello.world_name-42"),
        ["hello", "world", "name", "42"]
    );
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
