//! The multilingual normalisation tables: `NFKC -> fold_script -> lower`.
//!
//! Each affected language pins at least twenty `(input, expected)` pairs and
//! each unaffected language pins five, so a rule change is visible as a diff
//! rather than a recall number somewhere else.  `fold_script` is checked
//! separately where the distinction matters: it is the step between NFKC and
//! lower, and the tables below give its output after `lower`.

use sag2_core::analysis::normalize::{fold_script, normalize};
use sag2_core::{tokenize, tokenize_with, Language, Profile};

/// [`tokenize_with`] for a profile that never needs a dictionary; the tests
/// below all use one, so the typed error is a test failure here.
fn tokens(profile: Profile, text: &str) -> Vec<String> {
    tokenize_with(profile, text).expect("a dictionary-free profile always tokenises")
}

/// German exercises the shared Latin fold in both cases plus NFKC.
const GERMAN: [(&str, &str); 22] = [
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
    ("Müller", "müller"),
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

/// Vietnamese: the tone (and vowel-quality) marks are combining marks in NFD.
const VIETNAMESE: [(&str, &str); 22] = [
    ("Việt Nam", "viet nam"),
    ("VIỆT NAM", "viet nam"),
    ("việt nam", "viet nam"),
    ("Hà Nội", "ha noi"),
    ("Đà Nẵng", "da nang"),
    ("Thành phố Hồ Chí Minh", "thanh pho ho chi minh"),
    ("Tôi yêu tiếng Việt", "toi yeu tieng viet"),
    ("ế", "e"),
    ("ô", "o"),
    ("ơ", "o"),
    ("ư", "u"),
    ("ă", "a"),
    ("đ", "d"),
    ("ÁÀẢÃẠ", "aaaaa"),
    ("ấầẩẫậ", "aaaaa"),
    ("ắằẳẵặ", "aaaaa"),
    ("ếềểễệ", "eeeee"),
    ("ốồổỗộ", "ooooo"),
    ("ớờởỡợ", "ooooo"),
    ("ứừửữự", "uuuuu"),
    ("ýỳỷỹỵ", "yyyyy"),
    ("ƠƯ", "ou"),
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
fn greek_drops_tonos_and_dialytika() {
    assert_table(Language::El, &GREEK);
    // The marks are gone from the fold, before `lower` runs.
    assert_eq!(fold_script(Language::El, "Αθήνα"), "Αθηνα");
}

#[test]
fn vietnamese_drops_the_tone_marks() {
    assert_table(Language::Vi, &VIETNAMESE);
    assert_eq!(fold_script(Language::Vi, "Việt"), "Viet");
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
