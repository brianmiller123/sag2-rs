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
/// | other languages | no further rule |
///
/// Dropping *every* combining mark is what the Greek profile asks for: NFD
/// cannot tell a tonos from a circumflex, so `ά` and `α` collapse together.
/// Vietnamese does not: its tone marks are lexical identity (`ma`, `má`, `mà`,
/// `mả`, `mã`, `mạ` are six words), and the index layer already removes the
/// diacritics (FTS5 `unicode61`), so stripping them here would only destroy the
/// distinctiveness of the stored word tokens.
pub fn fold_script(language: Language, text: &str) -> String {
    let folded = fold_latin(text);
    match language {
        Language::El => nfkc(&strip_marks(&nfd(&folded))),
        Language::De => fold_german(&folded),
        Language::Ar => fold_arabic(&folded),
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
