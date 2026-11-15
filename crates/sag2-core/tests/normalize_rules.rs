//! The multilingual normalisation tables: `NFKC -> fold_script -> lower`.
//!
//! Each affected language pins at least twenty `(input, expected)` pairs and
//! each unaffected language pins five, so a rule change is visible as a diff
//! rather than a recall number somewhere else.  `fold_script` is checked
//! separately where the distinction matters: it is the step between NFKC and
//! lower, and the tables below give its output after `lower`.

use sag2_core::analysis::normalize::{fold_script, normalize};
use sag2_core::{nfd, tokenize, tokenize_with, Language, Profile};

/// [`tokenize_with`] for a profile that never needs a dictionary; the tests
/// below all use one, so the typed error is a test failure here.
fn tokens(profile: Profile, text: &str) -> Vec<String> {
    tokenize_with(profile, text).expect("a dictionary-free profile always tokenises")
}

/// German exercises the shared Latin fold in both cases plus NFKC, and the
/// DIN 5007-2 umlaut expansion that makes `Müller` and `Mueller` one term.
const GERMAN: [(&str, &str); 30] = [
    ("Straße", "strasse"),
    ("STRASSE", "strasse"),
    ("ẞ", "ss"),
    ("Straßenbahn", "strassenbahn"),
    ("Ærøskøbing", "aeroskobing"),
    ("œ", "oe"),
    ("œuvre", "oeuvre"),
    ("ŒUVRE", "oeuvre"),
    ("Øre", "ore"),
    ("Łódź", "lódź"),
    ("Đà Nẵng", "dà nẵng"),
    ("Maður", "madur"),
    ("Þingvellir", "thingvellir"),
    ("ıi", "ii"),
    ("ſ", "s"),
    ("Müller", "mueller"),
    ("MÜLLER", "mueller"),
    ("Mueller", "mueller"),
    ("für", "fuer"),
    ("über", "ueber"),
    ("Öl", "oel"),
    ("Ärger", "aerger"),
    ("Grüße", "gruesse"),
    ("Fußgänger", "fussgaenger"),
    ("GROSS", "gross"),
    ("Café", "café"),
    ("L'œuvre", "l'oeuvre"),
    ("ﬁ", "fi"),
    ("①", "1"),
    ("die Straße", "die strasse"),
];

/// Greek: NFD, drop every combining mark (tonos and dialytika), NFKC, lower.
const GREEK: [(&str, &str); 22] = [
    ("Αθήνα", "αθηνα"),
    ("ΑΘΉΝΑ", "αθηνα"),
    ("αθηνα", "αθηνα"),
    ("Ελλάδα", "ελλαδα"),
    ("ΕΛΛΆΔΑ", "ελλαδα"),
    ("τόνος", "τονος"),
    ("άέήίόύώ", "αεηιουω"),
    ("ΆΈΉΊΌΎΏ", "αεηιουω"),
    ("ϊ", "ι"),
    ("ϋ", "υ"),
    ("ΐ", "ι"),
    ("ΰ", "υ"),
    ("ΣΟΦΊΑ", "σοφια"),
    // Final sigma is the `lower` step's business, not the fold's.
    ("ΟΔΟΣ", "οδο\u{3C2}"),
    ("Παρίσι", "παρισι"),
    ("Θεσσαλονίκη", "θεσσαλονικη"),
    ("Αριστοτέλης", "αριστοτελη\u{3C2}"),
    ("Ελληνικά", "ελληνικα"),
    ("naïve", "naive"),
    ("Straße", "strasse"),
    ("Æ", "ae"),
    ("Αθήνα και Θεσσαλονίκη", "αθηνα και θεσσαλονικη"),
];

/// Vietnamese: the tone marks carry lexical identity (`ma`, `má`, `mà`, `mả`,
/// `mã`, `mạ` are six words), so only the shared Latin fold applies; the
/// diacritic-free spelling is the index layer's business (FTS5 `unicode61`).
const VIETNAMESE: [(&str, &str); 22] = [
    ("Việt Nam", "việt nam"),
    ("VIỆT NAM", "việt nam"),
    ("việt nam", "việt nam"),
    ("Hà Nội", "hà nội"),
    ("Đà Nẵng", "dà nẵng"),
    ("Thành phố Hồ Chí Minh", "thành phố hồ chí minh"),
    ("Tôi yêu tiếng Việt", "tôi yêu tiếng việt"),
    ("ế", "ế"),
    ("ô", "ô"),
    ("ơ", "ơ"),
    ("ư", "ư"),
    ("ă", "ă"),
    // `đ` is the shared Latin fold's rule (`đ→d`), not a mark.
    ("đ", "d"),
    ("ÁÀẢÃẠ", "áàảãạ"),
    ("ấầẩẫậ", "ấầẩẫậ"),
    ("ắằẳẵặ", "ắằẳẵặ"),
    ("ếềểễệ", "ếềểễệ"),
    ("ốồổỗộ", "ốồổỗộ"),
    ("ớờởỡợ", "ớờởỡợ"),
    ("ứừửữự", "ứừửữự"),
    ("ýỳỷỹỵ", "ýỳỷỹỵ"),
    ("ƠƯ", "ơư"),
];

/// Arabic: hamza carriers, alef maksura, tatweel; `ة` is kept.
const ARABIC: [(&str, &str); 22] = [
    ("أحمد", "احمد"),
    ("إسلام", "اسلام"),
    ("آمن", "امن"),
    ("ٱلله", "الله"),
    ("مصطفى", "مصطفي"),
    ("مؤمن", "مومن"),
    ("شيئ", "شيي"),
    ("محمّد", "محمد"),
    ("مُحَمَّد", "محمد"),
    ("كـــتاب", "كتاب"),
    ("مدرسة", "مدرسة"),
    ("سيارة", "سيارة"),
    ("الفتاة", "الفتاة"),
    ("قُرْآن", "قران"),
    ("عَلَى", "علي"),
    ("إِنَّ", "ان"),
    ("حَمَاس", "حماس"),
    ("بِسْمِ", "بسم"),
    ("اللَّه", "الله"),
    ("ﻻ", "لا"),
    ("مرحباً", "مرحبا"),
    ("رَؤُوف", "رووف"),
];

const RUSSIAN: [(&str, &str); 5] = [
    ("Москва", "москва"),
    ("Санкт-Петербург", "санкт-петербург"),
    ("Россия", "россия"),
    ("Привет, мир!", "привет, мир!"),
    ("ЁЖИК", "ёжик"),
];

const JAPANESE: [(&str, &str); 5] = [
    ("東京", "東京"),
    ("こんにちは", "こんにちは"),
    ("カタカナ", "カタカナ"),
    ("日本語のテキスト", "日本語のテキスト"),
    ("ｶﾀｶﾅ", "カタカナ"),
];

/// Chinese has no language slot of its own: it is the reference profile's own
/// language, so it is pinned through the no-rule profile and the reference one.
const CHINESE: [(&str, &str); 5] = [
    ("北京", "北京"),
    ("中文分词", "中文分词"),
    ("ＡＢＣ", "abc"),
    ("２０２４年", "2024年"),
    ("中华人民共和国", "中华人民共和国"),
];

fn assert_table(language: Language, table: &[(&str, &str)]) {
    for (input, expected) in table {
        assert_eq!(
            normalize(language, input),
            *expected,
            "{}: {input:?}",
            language.as_str()
        );
    }
}

#[test]
fn german_folds_the_shared_latin_letters() {
    assert_table(Language::De, &GERMAN);
}

#[test]
fn german_expands_the_umlauts() {
    // The realistic German ASCII spelling is transliteration, not deaccenting:
    // `Mueller` must reach `Müller`.  FTS5 `unicode61` removes diacritics
    // (`müller` indexes as `muller`) but cannot bridge `muller` and `mueller`,
    // so this is the fold the German profile needs.
    for (input, expected) in [
        ("Müller", "mueller"),
        ("MÜLLER", "mueller"),
        ("mueller", "mueller"),
        ("für", "fuer"),
        ("Straße", "strasse"),
        ("Strasse", "strasse"),
    ] {
        assert_eq!(normalize(Language::De, input), expected, "{input:?}");
    }
    // The umlaut expansion is the only German rule; French-style stripping is
    // not part of it, and the shared Latin fold still runs first.
    assert_eq!(fold_script(Language::De, "Müller"), "Mueller");
    assert_eq!(fold_script(Language::De, "Café"), "Café");
}

#[test]
fn german_emits_the_code_points_and_both_ascii_spellings() {
    // `fold_script` turns `Müller` into the transliteration `mueller`, which
    // FTS5 `unicode61` cannot map back to the deaccented `muller`.  The word
    // run therefore keeps its code points and carries both spellings, in the
    // order: code points, transliteration, deaccented spelling.  A query typed
    // either way reaches the tokens the build stored.
    let german = Profile::Multilingual {
        language: Language::De,
    };
    assert_eq!(
        tokens(german, "Müller"),
        ["m", "ü", "l", "l", "e", "r", "mueller", "muller"]
    );
    assert_eq!(
        tokens(german, "Bäckerei"),
        [
            "b",
            "ä",
            "c",
            "k",
            "e",
            "r",
            "e",
            "i",
            "baeckerei",
            "backerei"
        ]
    );
    // `ß` is the shared Latin fold's business: `strasse` is both the
    // transliteration and the deaccented spelling, so it is emitted once.  The
    // code-point prefix still goes through the reference filters, which drop
    // the stopword `a`.
    assert_eq!(
        tokens(german, "Straße"),
        ["s", "t", "r", "ß", "e", "strasse"]
    );
}

#[test]
fn the_german_variants_are_canonical_invariant() {
    // The gate reads `lower(nfkc(text))`, not the raw input, so the NFD
    // spelling of the word emits the same stream as its NFC form: an ASCII
    // `muller` query reaches an NFD corpus the same way it reaches an NFC one.
    let german = Profile::Multilingual {
        language: Language::De,
    };
    assert_eq!(tokens(german, "Mu\u{0308}ller"), tokens(german, "Müller"));
    assert_eq!(
        tokens(german, "Mu\u{0308}ller"),
        ["m", "ü", "l", "l", "e", "r", "mueller", "muller"]
    );
    // `lower` turns `ẞ` into `ß`, which is in the gate's set too.
    assert_eq!(tokens(german, "ẞ"), ["ß", "ss"]);
}

#[test]
fn a_non_composing_mark_does_not_desync_the_german_variants() {
    // `deaccent` deletes combining marks, so a second walk over a deaccented
    // text could merge two word runs (`x` + U+0308 + `Bäckerei` is two runs
    // before the strip and one after) and shift every later word's variants.
    // The spelling is derived from the paired accented run instead, so the
    // German word after the mark keeps both spellings.
    let german = Profile::Multilingual {
        language: Language::De,
    };
    let stream = tokens(german, "Müller q\u{0308}Bäckerei");
    for expected in ["mueller", "muller", "baeckerei", "backerei"] {
        assert!(
            stream.iter().any(|token| token == expected),
            "{expected:?} missing from {stream:?}"
        );
    }
}

#[test]
fn only_german_gets_the_second_ascii_spelling() {
    // The extra spelling is a German rule, read from the pre-fold text: the
    // same text under another language keeps its L1 stream.
    for language in [Language::Fr, Language::Other, Language::Pl] {
        let profile = Profile::Multilingual { language };
        let stream = tokens(profile, "Müller");
        assert!(
            !stream
                .iter()
                .any(|token| token == "mueller" || token == "muller"),
            "{profile}: {stream:?}"
        );
    }
}

#[test]
fn the_german_fold_is_idempotent() {
    for text in [
        "Müller",
        "MÜLLER",
        "mueller",
        "Straße",
        "für",
        "Fußgänger",
        "schön",
    ] {
        let once = normalize(Language::De, text);
        assert_eq!(normalize(Language::De, &once), once, "{text:?}");
    }
}

#[test]
fn the_default_profile_still_shreds_the_accented_words() {
    // `Sag2ZhV1` never enters `normalize`: the German fold is a multilingual
    // profile rule, so the certified tokens for these words do not move.
    assert_eq!(
        tokenize("Müller bibliothèque"),
        ["m", "ü", "ller", "biblioth", "è", "que"]
    );
    assert_eq!(
        tokens(Profile::Sag2ZhV1, "Müller bibliothèque"),
        tokenize("Müller bibliothèque")
    );
}

#[test]
fn greek_drops_tonos_and_dialytika() {
    assert_table(Language::El, &GREEK);
    // The marks are gone from the fold, before `lower` runs.
    assert_eq!(fold_script(Language::El, "Αθήνα"), "Αθηνα");
    // Greek keeps its strip: `ά` alone still reaches `α`.
    assert_eq!(normalize(Language::El, "ά"), "α");
}

#[test]
fn vietnamese_keeps_the_tone_marks() {
    assert_table(Language::Vi, &VIETNAMESE);
    assert_eq!(fold_script(Language::Vi, "Việt"), "Việt");
    // Lexical identity: NFKC and lower apply, the tone mark stays.
    assert_eq!(normalize(Language::Vi, "trường"), "trường");
    assert_ne!(normalize(Language::Vi, "ma"), normalize(Language::Vi, "má"));
    assert_ne!(normalize(Language::Vi, "má"), normalize(Language::Vi, "mà"));
}

#[test]
fn vietnamese_is_canonical_invariant() {
    // Keeping the marks means the composed and decomposed spellings have to
    // reach the same token stream on their own: NFKC composes the NFD input, so
    // no mark-dropping step is left to hide a difference.
    let profile = Profile::Multilingual {
        language: Language::Vi,
    };
    for text in ["trường", "Hà Nội", "Đà Nẵng", "Việt Nam"] {
        let decomposed = nfd(text);
        assert_ne!(decomposed, text, "{text:?} has a decomposed spelling");
        assert_eq!(
            normalize(Language::Vi, &decomposed),
            normalize(Language::Vi, text),
            "{text:?}"
        );
        assert_eq!(
            tokens(profile, &decomposed),
            tokens(profile, text),
            "{text:?}"
        );
    }
}

#[test]
fn arabic_folds_the_hamza_carriers_and_drops_the_tatweel() {
    assert_table(Language::Ar, &ARABIC);
    assert_eq!(fold_script(Language::Ar, "مُحَمَّد"), "محمد");
    // The fold is the only step besides NFKC: `ة` and the alef are untouched.
    assert_eq!(fold_script(Language::Ar, "مدرسة"), "مدرسة");
}

#[test]
fn unaffected_languages_keep_their_text() {
    assert_table(Language::Ru, &RUSSIAN);
    assert_table(Language::Ja, &JAPANESE);
    // `Other` is the no-language-specific-rule profile, which is what CJK has.
    assert_table(Language::Other, &CHINESE);
    // And the reference profile never enters normalisation at all.
    for (text, _) in &CHINESE {
        assert_eq!(tokens(Profile::Sag2ZhV1, text), tokenize(text));
    }
}

#[test]
fn the_fold_leaves_unaffected_languages_alone() {
    for (text, _) in RUSSIAN.iter().chain(JAPANESE.iter()).chain(CHINESE.iter()) {
        for language in [Language::Ru, Language::Ja, Language::Other] {
            assert_eq!(fold_script(language, text), *text, "{text:?}");
        }
    }
}

#[test]
fn the_reference_profile_never_normalises() {
    // `Sag2ZhV1` keeps the certified tokens even where a German profile folds.
    assert_ne!(
        tokens(
            Profile::Multilingual {
                language: Language::De
            },
            "Straße"
        ),
        tokenize("Straße")
    );
    assert_eq!(tokenize("Straße"), tokens(Profile::Sag2ZhV1, "Straße"));
}
