//! `multilingual-v1-th`: Thai words come from `nlpo3`'s newmm over the vendored
//! PyThaiNLP word list, while `tokenize` (the certified `sag2-zh-v1` profile)
//! must keep producing the same tokens it produced before the segmenter
//! existed.
//!
//! The pinned `THAI` vectors are the real output of this build; the `DEFAULT`
//! vectors were captured on `a0c1146`, before the Thai profile had a segmenter,
//! and are the regression net for the default path (Gate D's
//! `golden/tokenizer.jsonl` replay stays the authority for the corpus).

use sag2_core::{tokenize, tokenize_with, Language, Profile};

/// [`tokenize_with`] for a profile that never needs a dictionary; the tests
/// below all use one, so the typed error is a test failure here.
fn tokens(profile: Profile, text: &str) -> Vec<String> {
    tokenize_with(profile, text).expect("a dictionary-free profile always tokenises")
}

/// `multilingual-v1-th`.
fn thai() -> Profile {
    Profile::Multilingual {
        language: Language::Th,
    }
}

/// `(text, tokens)` — the Thai profile output this build certifies.
///
/// Maximal matching keeps corpus compounds whole (`ห้องสมุดประชาชน`,
/// `กรุงเทพมหานคร`, `ปัญญาประดิษฐ์`), and the reference `[._-]` split and
/// `FREQ` 2-/3-codepoint expansion deliberately do not run: a Thai word is what
/// the segmenter produced.  `hello.world_name-42` shows the ASCII consequence:
/// newmm cuts at the punctuation and every ASCII stretch then goes through the
/// certified ASCII model, so the reference profile's compound token has no Thai
/// counterpart.
const THAI: [(&str, &[&str]); 17] = [
    ("ห้องสมุด", &["ห\u{e49}องสม\u{e38}ด"]),
    ("ประชาชน", &["ประชาชน"]),
    ("กรุงเทพมหานคร", &["กร\u{e38}งเทพมหานคร"]),
    ("ภาษาไทย", &["ภาษาไทย"]),
    (
        "ห้องสมุดประชาชนกรุงเทพมหานครภาษาไทย",
        &[
            "ห\u{e49}องสม\u{e38}ดประชาชน",
            "กร\u{e38}งเทพมหานคร",
            "ภาษาไทย",
        ],
    ),
    ("การค้นหาข้อมูล", &["การ", "ค\u{e49}นหา", "ข\u{e49}อม\u{e39}ล"]),
    (
        "ฉันรักประเทศไทย",
        &["ฉ\u{e31}น", "ร\u{e31}ก", "ประเทศ", "ไทย"],
    ),
    ("ราคาสินค้าลดลง", &["ราคา", "ส\u{e34}นค\u{e49}า", "ลดลง"]),
    (
        "มหาวิทยาลัยเกษตรศาสตร์",
        &["มหาว\u{e34}ทยาล\u{e31}ยเกษตรศาสตร\u{e4c}"],
    ),
    ("ระบบฐานข้อมูล", &["ระบบ", "ฐานข\u{e49}อม\u{e39}ล"]),
    ("เศรษฐกิจพอเพียง", &["เศรษฐก\u{e34}จพอเพ\u{e35}ยง"]),
    (
        "เทคโนโลยีปัญญาประดิษฐ์",
        &["เทคโนโลย\u{e35}", "ป\u{e31}ญญาประด\u{e34}ษฐ\u{e4c}"],
    ),
    (
        "สวัสดีครับ ยินดีต้อนรับสู่กรุงเทพฯ",
        &[
            "สว\u{e31}สด\u{e35}",
            "คร\u{e31}บ",
            "ย\u{e34}นด\u{e35}ต\u{e49}อนร\u{e31}บ",
            "ส\u{e39}\u{e48}",
            "กร\u{e38}งเทพฯ",
        ],
    ),
    (
        "นักศึกษาสามารถยืมหนังสือและใช้คอมพิวเตอร์ได้ฟรี",
        &[
            "น\u{e31}กศ\u{e36}กษา",
            "สามารถ",
            "ย\u{e37}ม",
            "หน\u{e31}งส\u{e37}อ",
            "และ",
            "ใช\u{e49}",
            "คอมพ\u{e34}วเตอร\u{e4c}",
            "ได\u{e49}",
            "ฟร\u{e35}",
        ],
    ),
    ("hello.world_name-42", &["hello", "world", "name", "42"]),
    ("Alice Paul birthplace", &["alice", "paul", "birthplace"]),
    ("", &[]),
];

/// `(text, tokens)` — `tokenize` on the same inputs, captured on `a0c1146`.
const DEFAULT: [(&str, &[&str]); 19] = [
    ("ห้องสมุด", &["ห", "อ", "ง", "ส", "ม", "ด"]),
    ("ประชาชน", &["ป", "ร", "ะ", "ช", "า", "ช", "น"]),
    (
        "กรุงเทพมหานคร",
        &["ก", "ร", "ง", "เ", "ท", "พ", "ม", "ห", "า", "น", "ค", "ร"],
    ),
    ("ภาษาไทย", &["ภ", "า", "ษ", "า", "ไ", "ท", "ย"]),
    (
        "ห้องสมุดประชาชนกรุงเทพมหานครภาษาไทย",
        &[
            "ห", "อ", "ง", "ส", "ม", "ด", "ป", "ร", "ะ", "ช", "า", "ช", "น", "ก", "ร", "ง", "เ",
            "ท", "พ", "ม", "ห", "า", "น", "ค", "ร", "ภ", "า", "ษ", "า", "ไ", "ท", "ย",
        ],
    ),
    (
        "การค้นหาข้อมูล",
        &["ก", "า", "ร", "ค", "น", "ห", "า", "ข", "อ", "ม", "ล"],
    ),
    (
        "ฉันรักประเทศไทย",
        &[
            "ฉ", "น", "ร", "ก", "ป", "ร", "ะ", "เ", "ท", "ศ", "ไ", "ท", "ย",
        ],
    ),
    (
        "ราคาสินค้าลดลง",
        &["ร", "า", "ค", "า", "ส", "น", "ค", "า", "ล", "ด", "ล", "ง"],
    ),
    (
        "มหาวิทยาลัยเกษตรศาสตร์",
        &[
            "ม", "ห", "า", "ว", "ท", "ย", "า", "ล", "ย", "เ", "ก", "ษ", "ต", "ร", "ศ", "า", "ส",
            "ต", "ร",
        ],
    ),
    (
        "ระบบฐานข้อมูล",
        &["ร", "ะ", "บ", "บ", "ฐ", "า", "น", "ข", "อ", "ม", "ล"],
    ),
    (
        "เศรษฐกิจพอเพียง",
        &[
            "เ", "ศ", "ร", "ษ", "ฐ", "ก", "จ", "พ", "อ", "เ", "พ", "ย", "ง",
        ],
    ),
    (
        "เทคโนโลยีปัญญาประดิษฐ์",
        &[
            "เ", "ท", "ค", "โ", "น", "โ", "ล", "ย", "ป", "ญ", "ญ", "า", "ป", "ร", "ะ", "ด", "ษ",
            "ฐ",
        ],
    ),
    (
        "สวัสดีครับ ยินดีต้อนรับสู่กรุงเทพฯ",
        &[
            "ส", "ว", "ส", "ด", "ค", "ร", "บ", "ย", "น", "ด", "ต", "อ", "น", "ร", "บ", "ส", "ก",
            "ร", "ง", "เ", "ท", "พ", "ฯ",
        ],
    ),
    (
        "นักศึกษาสามารถยืมหนังสือและใช้คอมพิวเตอร์ได้ฟรี",
        &[
            "น", "ก", "ศ", "ก", "ษ", "า", "ส", "า", "ม", "า", "ร", "ถ", "ย", "ม", "ห", "น", "ง",
            "ส", "อ", "แ", "ล", "ะ", "ใ", "ช", "ค", "อ", "ม", "พ", "ว", "เ", "ต", "อ", "ร", "ไ",
            "ด", "ฟ", "ร",
        ],
    ),
    (
        "ราคา iPhone 15 Pro คือ 29,900 บาท",
        &[
            "ร", "า", "ค", "า", "iphone", "15", "pro", "ค", "อ", "29", "900", "บ", "า", "ท",
        ],
    ),
    (
        "hello.world_name-42",
        &["hello", "world", "name", "hello.world_name-42"],
    ),
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
    ("", &[]),
];

#[test]
fn thai_words_are_pinned() {
    for (text, expected) in THAI {
        assert_eq!(tokens(thai(), text), expected, "{text:?}");
    }
}

#[test]
fn the_default_profile_is_unchanged_for_those_inputs() {
    for (text, expected) in DEFAULT {
        assert_eq!(tokenize(text), expected, "{text:?}");
        assert_eq!(tokens(Profile::Sag2ZhV1, text), expected, "{text:?}");
    }
}

#[test]
fn mixed_thai_latin_and_digits_keep_the_ascii_model() {
    // `NFKC` + lower run before segmentation (`iPhone` -> `iphone`); the Latin
    // and digit stretches then take the certified ASCII cut, which splits
    // `29,900` at the non-alphanumeric byte exactly as the reference does.
    assert_eq!(
        tokens(thai(), "ราคา iPhone 15 Pro คือ 29,900 บาท"),
        [
            "ราคา",
            "iphone",
            "15",
            "pro",
            "ค\u{e37}อ",
            "29",
            "900",
            "บาท"
        ]
    );
}

#[test]
fn the_reference_filters_apply_to_thai_tokens() {
    // `the` and `is` are reference stopwords and `.` has no content: the final
    // filter is the one every profile shares.
    assert_eq!(tokens(thai(), "the ราคา is ราคา."), ["ราคา", "ราคา"]);
    // Whitespace between Thai words is dropped, not tokenized.
    assert_eq!(tokens(thai(), " \tห้องสมุด\n"), ["ห\u{e49}องสม\u{e38}ด"]);
}

/// The content words of a five-sentence sample, by hand: `(sentence, words)`.
const CONTENT_WORDS: [(&str, &[&str]); 5] = [
    ("การค้นหาข้อมูล", &["การ", "ค\u{e49}นหา", "ข\u{e49}อม\u{e39}ล"]),
    (
        "ฉันรักประเทศไทย",
        &["ฉ\u{e31}น", "ร\u{e31}ก", "ประเทศ", "ไทย"],
    ),
    ("ราคาสินค้าลดลง", &["ราคา", "ส\u{e34}นค\u{e49}า", "ลดลง"]),
    ("ระบบฐานข้อมูล", &["ระบบ", "ฐานข\u{e49}อม\u{e39}ล"]),
    (
        "นักศึกษาสามารถยืมหนังสือและใช้คอมพิวเตอร์ได้ฟรี",
        &[
            "น\u{e31}กศ\u{e36}กษา",
            "สามารถ",
            "ย\u{e37}ม",
            "หน\u{e31}งส\u{e37}อ",
            "และ",
            "ใช\u{e49}",
            "คอมพ\u{e34}วเตอร\u{e4c}",
            "ได\u{e49}",
            "ฟร\u{e35}",
        ],
    ),
];

#[test]
fn content_words_are_whole_tokens() {
    let mut matched = 0;
    let mut total = 0;
    for (text, words) in CONTENT_WORDS {
        let tokens = tokens(thai(), text);
        assert_eq!(tokens.concat(), text, "no code point is dropped: {text:?}");
        for word in words {
            total += 1;
            if tokens.iter().any(|token| token == word) {
                matched += 1;
            }
            // The default profile has no Thai dictionary entry, so it emits one
            // token per code point and never a whole word.
            assert!(
                !tokenize(text).iter().any(|token| token == word),
                "{word:?} is not a default-profile token"
            );
        }
    }
    assert_eq!(
        matched, total,
        "{matched}/{total} content words are whole tokens"
    );
    assert_eq!(total, 21);
}
