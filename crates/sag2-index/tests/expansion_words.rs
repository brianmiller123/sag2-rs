//! T6: expansion candidates are words for a word-level profile.
//!
//! `expansion_terms` used to drop a candidate when *every* token of it occurred
//! somewhere in the query - the certified `sag2.py` rule, which is a dead end
//! for character-level languages (one character is enough to drop a whole word)
//! and wrong for multi-token name phrases.  A [`Profile::Multilingual`] profile
//! now compares the candidate's token run instead: a one-token term is skipped
//! when that word occurs anywhere in the query, a multi-token term only when the
//! whole run occurs contiguously.  [`Profile::Sag2ZhV1`] keeps the certified
//! rule byte for byte.
//!
//! Thai is the word-level profile that needs no external data; Arabic is the
//! character-level control.  The Japanese dictionary test is `#[ignore]`d
//! because it needs a real ipadic in `SAG2_DICT_DIR`.

use std::collections::BTreeSet;

use sag2_index::{
    CorpusIndex, Document, ExpansionKind, IndexOptions, Language, Profile, RebuildPolicy, Seed,
};

fn document<'a>(title: &'a str, text: &'a str) -> Document<'a> {
    Document {
        title,
        text,
        path: None,
        idx: None,
    }
}

fn open(profile: Profile, corpus: &[Document<'_>]) -> (tempfile::TempDir, CorpusIndex) {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("index.sqlite");
    let (index, _) = CorpusIndex::open(
        &path,
        corpus,
        "fixture",
        IndexOptions {
            chunk_chars: 2200,
            index_workers: 1,
            source_dir: None,
            profile,
            dictionary_dir: None,
            rebuild: RebuildPolicy::Always,
        },
    )
    .expect("build the fixture index");
    (directory, index)
}

fn thai() -> Profile {
    Profile::Multilingual {
        language: Language::Th,
    }
}

fn seed(content: &str) -> Seed {
    Seed {
        document_id: 1,
        jev_score: 0.01,
        content: content.to_owned(),
    }
}

fn terms(index: &CorpusIndex, query: &str, seed_text: &str) -> Vec<(String, ExpansionKind)> {
    index
        .expansion_terms(query, &[seed(seed_text)], 20)
        .expect("expansion")
        .into_iter()
        .map(|term| (term.term, term.kind))
        .collect()
}

#[test]
fn thai_candidates_are_words_and_the_query_word_is_skipped() {
    let corpus = [
        document("คู่มือ", "ห้องสมุดและหนังสือ"),
        document("รายวัน", "หนังสือพิมพ์รายวัน"),
    ];
    let (_directory, index) = open(thai(), &corpus);
    let found = terms(&index, "ห้องสมุด", "ห้องสมุดมีหนังสือ");

    assert!(
        found.iter().any(|(term, _)| term == "หนังสือ"),
        "the seed word must survive: {found:?}"
    );
    assert!(
        !found.iter().any(|(term, _)| term == "ห้องสมุด"),
        "the query word must be skipped: {found:?}"
    );
    // `มี` occurs in the seed but in no document, so its frequency is zero.
    assert!(
        !found.iter().any(|(term, _)| term == "มี"),
        "a term absent from the corpus never matches: {found:?}"
    );
    // One whole word per candidate: no 2/3-token window of the seed sentence
    // and no single-character noise.
    for (term, _) in &found {
        assert!(!term.contains(' '), "window candidate {term:?}");
        assert!(
            term.chars().count() > 1,
            "single-character candidate {term:?}"
        );
    }
}

#[test]
fn thai_a_different_word_sharing_the_query_characters_survives() {
    let corpus = [document("ข่าว", "หนังสือพิมพ์รายวัน")];
    let (_directory, index) = open(thai(), &corpus);
    let found = terms(&index, "หนังสือ", "หนังสือพิมพ์รายวัน");

    // `หนังสือพิมพ์` starts with the query word's characters but is another
    // word, so the word-level rule must keep it.
    assert!(
        found.iter().any(|(term, _)| term == "หนังสือพิมพ์"),
        "{found:?}"
    );
    assert!(!found.iter().any(|(term, _)| term == "หนังสือ"), "{found:?}");
}

#[test]
fn a_name_phrase_is_skipped_only_when_it_is_a_contiguous_query_run() {
    let corpus = [
        document("Anchor", "Alice Paul ห้องสมุดและหนังสือ"),
        document("Guide", "หนังสือพิมพ์รายวัน"),
    ];
    let (_directory, index) = open(thai(), &corpus);

    // Both words are in the query, but not as one run: the phrase survives.
    let scattered = terms(&index, "paul alice", "Alice Paul");
    assert!(
        scattered
            .iter()
            .any(|(term, kind)| term == "alice paul" && *kind == ExpansionKind::NamePhrase),
        "a scattered token set must not drop the phrase: {scattered:?}"
    );

    // The same phrase as one contiguous query run is already in the query.
    let contiguous = terms(&index, "alice paul", "Alice Paul");
    assert!(
        !contiguous.iter().any(|(term, _)| term == "alice paul"),
        "the query phrase must be skipped: {contiguous:?}"
    );
}

#[test]
fn a_character_level_profile_keeps_only_characters_absent_from_the_query() {
    let corpus = [document("كتاب", "كتاب علم")];
    let (_directory, index) = open(
        Profile::Multilingual {
            language: Language::Ar,
        },
        &corpus,
    );
    let found = terms(&index, "علم", "كتاب علم");

    // Arabic tokenises to one code point per token: the three query characters
    // are skipped, the four characters of the other word are kept.
    let kept: BTreeSet<&str> = found.iter().map(|(term, _)| term.as_str()).collect();
    assert_eq!(
        kept,
        ["ا", "ب", "ت", "ك"].into_iter().collect(),
        "{found:?}"
    );
    assert!(found.iter().all(|(term, _)| term.chars().count() == 1));
}

#[test]
fn the_reference_profile_keeps_its_certified_candidate_sequence() {
    let corpus = [
        document("Anchor", "Alice Paul birthplace depicts a bridge."),
        document("Bridge", "The target is the bridge near the harbour."),
    ];
    let (_directory, index) = open(Profile::Sag2ZhV1, &corpus);
    let found = index
        .expansion_terms(
            "anchor birthplace depicts",
            &[seed("Alice Paul birthplace depicts a bridge.")],
            20,
        )
        .expect("expansion");

    // The output of `0528f9b` on this input, pinned: terms, kinds, score bits
    // and document frequencies.  The reference filter is unchanged, so this is
    // the Gate B contract at unit scale.
    let observed: Vec<(&str, ExpansionKind, u64, i64)> = found
        .iter()
        .map(|term| {
            (
                term.term.as_str(),
                term.kind,
                term.score.to_bits(),
                term.document_frequency,
            )
        })
        .collect();
    assert_eq!(
        observed,
        [
            (
                "alice paul",
                ExpansionKind::NamePhrase,
                0x3ffa_7700_7f3e_7fbf,
                1
            ),
            ("bridge", ExpansionKind::Word, 0x3fdb_d83c_0451_e84b, 2),
        ]
    );
}

#[cfg(feature = "lindera")]
#[test]
#[ignore = "needs a real ipadic directory in SAG2_DICT_DIR"]
fn japanese_candidates_are_dictionary_words() {
    let dictionary_dir: std::path::PathBuf = std::env::var_os("SAG2_DICT_DIR")
        .map(std::path::PathBuf::from)
        .expect("set SAG2_DICT_DIR to a compiled ipadic directory");
    let profile = Profile::Multilingual {
        language: Language::Ja,
    };
    let corpus = [
        document("東京", "東京都に住んでいます。図書館で本を読みます。"),
        document("図書館", "図書館には本がたくさんあります。"),
    ];
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("index.sqlite");
    let (index, _) = CorpusIndex::open(
        &path,
        &corpus,
        "fixture",
        IndexOptions {
            chunk_chars: 2200,
            index_workers: 1,
            source_dir: None,
            profile,
            dictionary_dir: Some(dictionary_dir),
            rebuild: RebuildPolicy::Always,
        },
    )
    .expect("build the dictionary index");

    let found = terms(&index, "東京都", "東京都に住んでいます。");
    assert!(
        !found.iter().any(|(term, _)| term == "東京都"),
        "the query word must be skipped: {found:?}"
    );
    assert!(
        found.iter().any(|(term, _)| term.chars().count() > 1),
        "the dictionary must yield word candidates: {found:?}"
    );
    for (term, _) in &found {
        assert!(!term.contains(' '), "window candidate {term:?}");
    }
}
