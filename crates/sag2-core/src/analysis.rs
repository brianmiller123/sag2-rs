//! Analysis profiles: the switch that decides how text becomes tokens.
//!
//! [`Profile::Sag2ZhV1`] is the certified reference pipeline of design §4 and
//! the only profile that reproduces the Python token stream byte for byte.
//! [`Profile::Multilingual`] is the word-level pipeline for one language: it
//! runs [`normalize::normalize`] before the same §4 tokenizer, so accent, case
//! and orthographic variants of a word collapse onto one term.  A language that
//! writes without spaces ([`Language::Th`]) replaces the §4 word boundaries
//! with its own segmenter (see [`segment`]) and shares the rest: the certified
//! ASCII model and the final filters.  The identifier a profile persists in an
//! index (`rust_index_metadata.analysis`) comes from [`Profile::as_str`].

use std::fmt;
use std::str::FromStr;

pub mod normalize;
pub mod segment;

use crate::ascii_cut;
use crate::tokenize::{keep, tokenize_default};

/// The language a multilingual profile is built for.
///
/// The language decides the profile's normalisation rules (see [`normalize`]);
/// the tokenizer itself stays the shared §4 pipeline until T4/T5 add the
/// language segmenters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    /// Japanese.
    Ja,
    /// Korean.
    Ko,
    /// Thai.
    Th,
    /// Arabic.
    Ar,
    /// Russian.
    Ru,
    /// Greek.
    El,
    /// Vietnamese.
    Vi,
    /// German.
    De,
    /// French.
    Fr,
    /// Polish.
    Pl,
    /// Hindi.
    Hi,
    /// Hebrew.
    He,
    /// Any other language: the shared fold and no language-specific rule.
    Other,
}

impl Language {
    /// The short code a multilingual [`Profile`] identifier embeds.
    pub const fn as_str(self) -> &'static str {
        match self {
            Language::Ja => "ja",
            Language::Ko => "ko",
            Language::Th => "th",
            Language::Ar => "ar",
            Language::Ru => "ru",
            Language::El => "el",
            Language::Vi => "vi",
            Language::De => "de",
            Language::Fr => "fr",
            Language::Pl => "pl",
            Language::Hi => "hi",
            Language::He => "he",
            Language::Other => "other",
        }
    }

    /// The inverse of [`Language::as_str`].
    fn from_code(code: &str) -> Option<Self> {
        match code {
            "ja" => Some(Language::Ja),
            "ko" => Some(Language::Ko),
            "th" => Some(Language::Th),
            "ar" => Some(Language::Ar),
            "ru" => Some(Language::Ru),
            "el" => Some(Language::El),
            "vi" => Some(Language::Vi),
            "de" => Some(Language::De),
            "fr" => Some(Language::Fr),
            "pl" => Some(Language::Pl),
            "hi" => Some(Language::Hi),
            "he" => Some(Language::He),
            "other" => Some(Language::Other),
            _ => None,
        }
    }
}

/// The language-less multilingual profile T1 persisted.
///
/// T3 replaced it with [`Profile::Multilingual`], whose identifier always names
/// a language.  An index that stores this value can only be rebuilt, never
/// mapped onto a language.
pub const REMOVED_MULTILINGUAL_V1: &str = "multilingual-v1";

/// Which analysis pipeline turns text into the tokens an index stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// The certified reference pipeline: `NFKC` + lower + jieba 0.42.1 with
    /// the `[._-]`/`FREQ`/substring post-rules and the stopword filter
    /// (design §4.1, §4.2).
    Sag2ZhV1,
    /// The word-level pipeline for `language`: normalisation first, then the
    /// §4 tokenizer.
    Multilingual {
        /// The language whose normalisation rules apply.
        language: Language,
    },
}

impl Profile {
    /// The stable identifier `rust_index_metadata.analysis` stores.
    pub const fn as_str(self) -> &'static str {
        match self {
            Profile::Sag2ZhV1 => "sag2-zh-v1",
            Profile::Multilingual { language } => match language {
                Language::Ja => "multilingual-v1-ja",
                Language::Ko => "multilingual-v1-ko",
                Language::Th => "multilingual-v1-th",
                Language::Ar => "multilingual-v1-ar",
                Language::Ru => "multilingual-v1-ru",
                Language::El => "multilingual-v1-el",
                Language::Vi => "multilingual-v1-vi",
                Language::De => "multilingual-v1-de",
                Language::Fr => "multilingual-v1-fr",
                Language::Pl => "multilingual-v1-pl",
                Language::Hi => "multilingual-v1-hi",
                Language::He => "multilingual-v1-he",
                Language::Other => "multilingual-v1-other",
            },
        }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// An `analysis` value this build does not know.
///
/// A profile this build cannot name is a provenance mismatch, never a default:
/// reading it as [`Profile::Sag2ZhV1`] would silently mix token streams.  The
/// value is preserved verbatim so a caller can report it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownProfile(String);

impl UnknownProfile {
    /// The value that failed to parse.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for UnknownProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown analysis profile {:?}", self.0)
    }
}

impl std::error::Error for UnknownProfile {}

impl FromStr for Profile {
    type Err = UnknownProfile;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text == Profile::Sag2ZhV1.as_str() {
            return Ok(Profile::Sag2ZhV1);
        }
        let language = text
            .strip_prefix("multilingual-v1-")
            .and_then(Language::from_code);
        match language {
            Some(language) => Ok(Profile::Multilingual { language }),
            None => Err(UnknownProfile(text.to_owned())),
        }
    }
}

/// Tokenize `text` with `profile`.
///
/// [`Profile::Sag2ZhV1`] is the certified pipeline and what [`crate::tokenize()`]
/// always uses; it does not enter [`normalize`] at all, so its token stream and
/// allocations are the ones the goldens pin.  [`Profile::Multilingual`] runs
/// [`normalize::normalize`] for its language and then the same §4 tokenizer -
/// except for [`Language::Th`], where [`thai_tokens`] supplies the word
/// boundaries and keeps the certified ASCII model and the final filters.
pub fn tokenize_with(profile: Profile, text: &str) -> Vec<String> {
    match profile {
        Profile::Sag2ZhV1 => tokenize_default(text),
        Profile::Multilingual { language } => {
            let normalised = normalize::normalize(language, text);
            match language {
                Language::Th => thai_tokens(&normalised),
                _ => tokenize_default(&normalised),
            }
        }
    }
}

/// The Thai pipeline: `normalize` -> `newmm` -> ASCII runs through the
/// certified ASCII model -> the reference `HAS_CONTENT`/stopword filter.
///
/// `newmm` hands back one token per ASCII run, but its rules for that run are
/// not the ones §4.2 certifies (no `[._-]` compounds, no patched `finalseg`).
/// Passing the run to [`ascii_cut::cut_run`] keeps one ASCII tokenizer for every
/// profile.  The reference `[._-]` split and the 2-/3-codepoint `FREQ`
/// expansion do not apply: a Thai word is exactly what the segmenter produced,
/// so `newmm`'s boundaries are the only word boundaries.
fn thai_tokens(normalised: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for word in segment::thai::segment(normalised) {
        if word.is_ascii() {
            ascii_cut::cut_run(&word, &mut tokens);
        } else {
            tokens.push(word);
        }
    }
    tokens.retain(|token| keep(token));
    tokens
}

#[cfg(test)]
mod tests {
    use super::{tokenize_with, Language, Profile, REMOVED_MULTILINGUAL_V1};
    use std::str::FromStr;

    const LANGUAGES: [Language; 13] = [
        Language::Ja,
        Language::Ko,
        Language::Th,
        Language::Ar,
        Language::Ru,
        Language::El,
        Language::Vi,
        Language::De,
        Language::Fr,
        Language::Pl,
        Language::Hi,
        Language::He,
        Language::Other,
    ];

    #[test]
    fn identifiers_round_trip_through_from_str() {
        assert_eq!(
            Profile::from_str(Profile::Sag2ZhV1.as_str()),
            Ok(Profile::Sag2ZhV1)
        );
        for language in LANGUAGES {
            let profile = Profile::Multilingual { language };
            assert_eq!(Profile::from_str(profile.as_str()), Ok(profile));
            assert_eq!(profile.to_string(), profile.as_str());
            // The identifier is the prefix plus the language code: the two
            // tables cannot drift apart.
            assert_eq!(
                profile.as_str(),
                format!("multilingual-v1-{}", language.as_str())
            );
        }
    }

    #[test]
    fn unknown_identifiers_keep_their_value() {
        for value in ["sag2-zh-v2", "multilingual-v1", "multilingual-v1-xx", ""] {
            let error = Profile::from_str(value).expect_err("an unknown identifier");
            assert_eq!(error.as_str(), value);
        }
        // The removed language-less profile is never mapped onto a language.
        assert_eq!(REMOVED_MULTILINGUAL_V1, "multilingual-v1");
    }

    #[test]
    fn the_reference_profile_does_not_normalize() {
        for text in ["Straße", "Αθήνα", "مُحَمَّد", "Việt"] {
            assert_eq!(
                tokenize_with(Profile::Sag2ZhV1, text),
                crate::tokenize(text),
                "{text:?}"
            );
        }
    }

    #[test]
    fn a_multilingual_profile_normalizes_before_the_certified_pipeline() {
        let german = Profile::Multilingual {
            language: Language::De,
        };
        assert_eq!(tokenize_with(german, "Straße"), ["strasse"]);
        assert_ne!(tokenize_with(german, "Straße"), crate::tokenize("Straße"));
    }
}
