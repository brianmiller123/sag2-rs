//! Profile normalisation: the steps a multilingual profile runs before the §4
//! tokenizer.
//!
//! The order is fixed and [`normalize`] is its only entry point:
//!
//! 1. `NFKC`,
//! 2. [`fold_script`] - the shared Latin fold plus the language rules,
//! 3. lower (the Final_Sigma-aware table the tokenizer already certifies).
//!
//! The fold is deliberately lossy: it trades orthographic detail for recall, so
//! `Straße`, `STRASSE` and `strasse` all reach one term.  Index building and
//! every query-side call go through the same function, which is why a stored
//! token and a query token can never be normalised by different rules.
//!
//! [`crate::Profile::Sag2ZhV1`] never enters this module.

use crate::unicode::{combining_class, lower, nfd, nfkc};

use super::Language;

/// `NFKC -> fold_script -> lower` for `language`.
pub fn normalize(language: Language, text: &str) -> String {
    lower(&fold_script(language, &nfkc(text)))
}

/// The script fold of [`normalize`]: the shared Latin replacements plus the
/// language-specific rules.
///
/// | scope | rule |
/// |---|---|
/// | every language | `ß→ss æ→ae œ→oe ø→o ł→l đ→d ð→d þ→th ı→i ſ→s` (both cases) |
/// | [`Language::El`] | `NFD`, drop every combining mark, `NFKC` |
/// | [`Language::De`] | `ä→ae ö→oe ü→ue` (both cases) |
/// | [`Language::Ar`] | `أإآٱ→ا`, `ى→ي`, `ؤ→و`, `ئ→ي`; delete `U+0640` and `U+064B–U+0652`; `ة` is kept |
/// | [`Language::Tr`] | `İ→i` and delete `U+0307`, so `İstanbul`, `ISTANBUL` and `istanbul` reach one term |
/// | [`Language::Mt`] | `ħ→h` (both cases) |
/// | [`Language::Bg`] | `ѝ→и` (both cases) |
/// | [`Language::Uk`] | `U+0027` and `U+2019` → `U+02BC`, so `м'ясо`, `м’ясо` and `мʼясо` are one word |
/// | other languages | no further rule |
///
/// The [`Language::Tr`], [`Language::Mt`], [`Language::Bg`] and [`Language::Uk`]
/// rules are the ones `unicode61` cannot supply.  It removes
/// combining diacritics (`ř`, `ő`, `ș` and `ġ` all fold to their base letter),
/// so the Latin-script profiles need no rule of their own, but it has no entry
/// for `ħ` (a letter with stroke, not a decomposed mark) and no Cyrillic table
/// at all, and it cannot bridge the two punctuation spellings of the Ukrainian
/// apostrophe to the letter `U+02BC`.  Turkish is not a fold at all: `lower(İ)`
/// produces `i` + `U+0307`, and that mark's ccc 230 ends a word run, so the
/// word was stored as two tokens.
///
/// Dropping *every* combining mark is what the Greek profile asks for: NFD
/// cannot tell a tonos from a circumflex, so `ά` and `α` collapse together.
/// Vietnamese does not: its tone marks carry lexical identity (`ma`, `má`,
/// `mà`, `mả`, `mã`, `mạ` are six words), and stripping them here would merge
/// those six into one before the index ever sees them.  That distinction lives
/// in the token stream, not in the FTS index: `chunk_fts` uses SQLite's
/// `unicode61` with `remove_diacritics=1`, which folds the marks on both sides
/// of a match, so tone-exact queries are not available and the recall comes
/// from word-token distinctiveness rather than from preserved index terms.
pub fn fold_script(language: Language, text: &str) -> String {
    let folded = fold_latin(text);
    match language {
        Language::El => nfkc(&strip_marks(&nfd(&folded))),
        Language::De => fold_german(&folded),
        Language::Ar => fold_arabic(&folded),
        Language::Tr => fold_turkish(&folded),
        Language::Mt => fold_maltese(&folded),
        Language::Bg => fold_bulgarian(&folded),
        Language::Uk => fold_ukrainian(&folded),
        _ => folded,
    }
}

/// The shared Latin fold, including the uppercase spellings so the result does
/// not depend on the later `lower` step.
fn fold_latin(text: &str) -> String {
    let mut folded = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            'ß' | 'ẞ' => folded.push_str("ss"),
            'æ' | 'Æ' => folded.push_str("ae"),
            'œ' | 'Œ' => folded.push_str("oe"),
            'ø' | 'Ø' => folded.push('o'),
            'ł' | 'Ł' => folded.push('l'),
            'đ' | 'Đ' => folded.push('d'),
            'ð' | 'Ð' => folded.push('d'),
            'þ' | 'Þ' => folded.push_str("th"),
            'ı' => folded.push('i'),
            'ſ' => folded.push('s'),
            other => folded.push(other),
        }
    }
    folded
}

/// The German umlaut expansion (DIN 5007-2 style), including the uppercase
/// spellings: `Müller` and the transliteration `Mueller` reach one term.
///
/// This is not a diacritic strip.  FTS5 `unicode61` already removes the umlaut
/// (`müller` indexes as `muller`), but it cannot bridge `muller` and `mueller`;
/// the transliteration is the spelling a German user types on an ASCII
/// keyboard, so the fold has to produce it.  `ß→ss` is the shared Latin fold's
/// rule and runs first.
fn fold_german(text: &str) -> String {
    let mut folded = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            'ä' | 'Ä' => folded.push_str("ae"),
            'ö' | 'Ö' => folded.push_str("oe"),
            'ü' | 'Ü' => folded.push_str("ue"),
            other => folded.push(other),
        }
    }
    folded
}

/// The diacritic-free spelling FTS5 `unicode61` indexes for `text`: the shared
/// Latin fold, `NFD`, the combining marks dropped, lower.
///
/// The German profile folds `ä/ö/ü` into `ae/oe/ue`, so the token it stores is
/// the transliteration and FTS can no longer derive the deaccented spelling
/// from it (`müller` indexes as `muller`, `mueller` does not).  The multilingual
/// tokenizer emits this spelling next to the transliteration for exactly that
/// reason; every other profile has no use for it.
pub(crate) fn deaccent(text: &str) -> String {
    lower(&strip_marks(&nfd(&fold_latin(&nfkc(text)))))
}

/// Drop every code point with a non-zero canonical combining class.
fn strip_marks(text: &str) -> String {
    text.chars()
        .filter(|character| combining_class(*character as u32) == 0)
        .collect()
}

/// The Arabic orthography rules: hamza carriers, alef maksura and tatweel.
fn fold_arabic(text: &str) -> String {
    let mut folded = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            // `أ` `إ` `آ` `ٱ` -> `ا`.
            '\u{0623}' | '\u{0625}' | '\u{0622}' | '\u{0671}' => folded.push('\u{0627}'),
            // `ى` and `ئ` -> `ي`.
            '\u{0649}' | '\u{0626}' => folded.push('\u{064A}'),
            // `ؤ` -> `و`.
            '\u{0624}' => folded.push('\u{0648}'),
            // Tatweel and the harakat (U+064B..U+0652) are deleted; `ة` stays.
            '\u{0640}' => {}
            '\u{064B}'..='\u{0652}' => {}
            other => folded.push(other),
        }
    }
    folded
}

/// The Turkish dotted and dotless i: `İ→i` and `U+0307` deleted, so
/// `İstanbul`, `ISTANBUL` and `istanbul` all reach `istanbul`.
///
/// This runs inside the fold, before `lower`, because `İ` is one precomposed
/// capital at that point: `lower(İ)` is `i` + U+0307, and that mark's ccc 230
/// makes [`crate::tokenize`]'s word set end the run, so the word used to be
/// stored as `i` and `stanbul`.  Mapping the capital first means the mark is
/// never produced; the filter is left to remove a mark that arrived
/// decomposed.  `NFKC` composes `I` + U+0307 back into `İ`, so a capital dot
/// cannot reach this function on its own: every other U+0307 here is a mark the
/// input spelled separately from its base letter - the lowercase residue of
/// `İ`, or a letter such as `q̇` - and dropping it is the same lossy trade the
/// shared fold makes for `ø` and `đ`.  `ı→i` is the shared Latin fold's rule
/// and has already run.
fn fold_turkish(text: &str) -> String {
    let mut folded = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            'İ' => folded.push('i'),
            '\u{0307}' => {}
            other => folded.push(other),
        }
    }
    folded
}

/// The Maltese `ħ`: `ħ/Ħ→h`.
///
/// U+0127 is a letter with a stroke, not a base letter plus a combining mark,
/// so FTS5 `unicode61` has no entry for it and an ASCII `h` query never reaches
/// a corpus word spelled `ħal`.  `ġ`, `ċ` and `ż` need no rule: they decompose
/// to a base letter plus U+0307, which `remove_diacritics=1` already folds.
fn fold_maltese(text: &str) -> String {
    let mut folded = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            'ħ' | 'Ħ' => folded.push('h'),
            other => folded.push(other),
        }
    }
    folded
}

/// The Bulgarian `ѝ`: `ѝ/Ѝ→и`.
///
/// `unicode61` has no Cyrillic diacritic table, so `ѝ` (the object-form clitic
/// of `и`) and `и` are two unrelated terms today.  Both cases fold to the
/// lowercase letter the later `lower` would produce anyway.
fn fold_bulgarian(text: &str) -> String {
    let mut folded = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            'ѝ' | 'Ѝ' => folded.push('и'),
            other => folded.push(other),
        }
    }
    folded
}

/// The Ukrainian apostrophe: `U+0027` and `U+2019` → `U+02BC`.
///
/// The apostrophe belongs to the word (`м'ясо`, `м’ясо` and `мʼясо` are the
/// same word), but only U+02BC is a letter (`Lm`) and therefore one of
/// [`crate::tokenize`]'s word characters; the two punctuation spellings end
/// the run and leave the word as fragments.  The rule is Ukrainian-only: in
/// Italian and French the apostrophe is a word *boundary* (`l'amore` is an
/// article and a noun), so folding it there would glue two words into one term.
fn fold_ukrainian(text: &str) -> String {
    let mut folded = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\u{0027}' | '\u{2019}' => folded.push('\u{02BC}'),
            other => folded.push(other),
        }
    }
    folded
}

#[cfg(test)]
mod tests {
    use super::{fold_script, normalize};
    use crate::analysis::Language;

    #[test]
    fn nfkc_runs_before_the_fold() {
        // The `œ` ligature is itself the fold target; `ﬁ` is NFKC's business.
        assert_eq!(normalize(Language::De, "ﬁ"), "fi");
        assert_eq!(normalize(Language::De, "①"), "1");
    }

    #[test]
    fn the_latin_fold_applies_to_every_language() {
        for language in [Language::De, Language::Ru, Language::Ja, Language::Other] {
            // `fold_script` runs before `lower`, so the case is still there.
            assert_eq!(
                fold_script(language, "Ærøskøbing Straße œuvre"),
                "aeroskobing Strasse oeuvre"
            );
        }
    }

    #[test]
    fn empty_text_stays_empty() {
        assert_eq!(normalize(Language::El, ""), "");
        assert_eq!(fold_script(Language::Ar, ""), "");
    }
}
