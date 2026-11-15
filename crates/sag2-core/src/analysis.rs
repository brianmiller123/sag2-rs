//! Analysis profiles: the switch that decides how text becomes tokens.
//!
//! [`Profile::Sag2ZhV1`] is the certified reference pipeline of design §4 and
//! the only profile that reproduces the Python token stream byte for byte.
//! [`Profile::Multilingual`] is the word-level pipeline for one language: it
//! runs [`normalize::normalize`] before the same §4 post-rules, with the
//! multilingual word character set (`tokenize::is_word_character`) instead of
//! `re_han_default`'s ASCII + Han set, so accent, case and orthographic variants
//! of a word collapse onto one term and a word in any script stays one token.
//! A German word run also carries both ASCII spellings of an umlaut word
//! (`mueller` and `muller`), because the fold's transliteration and FTS5's
//! diacritic removal disagree — see `tokenize::tokenize_multilingual`.  A
//! language that writes without spaces ([`Language::Th`]) replaces the §4 word
//! boundaries with its own segmenter (see [`segment`]) and shares the rest: the
//! certified ASCII model and the final filters.  Japanese needs a runtime
//! dictionary ([`Language::Ja`]); the resolved segmenter is a
//! [`segment::DictionaryAnalysis`] and a profile that needs one without it is a
//! typed error, never a degraded token stream.  Korean needs none: its eojeol
//! are space-delimited, so the word-run pipeline is the whole story and a
//! dictionary-backed Korean profile would be a *separate* profile identifier,
//! never a mode of `multilingual-v1-ko`.  The identifier a profile persists in
//! an index (`rust_index_metadata.analysis`) comes from [`Profile::as_str`].

use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

pub mod normalize;
pub mod segment;

pub use segment::DictionaryAnalysis;

use crate::ascii_cut;
use crate::tokenize::{keep, tokenize_default, tokenize_multilingual};

/// The language a multilingual profile is built for.
///
/// The language decides the profile's normalisation rules (see [`normalize`])
/// and, for [`Language::Th`], the segmenter; every other language shares the
/// multilingual word pipeline (`tokenize::is_word_character`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    /// Japanese.
    Ja,
    /// Korean: word runs, no morphological analysis (the eojeol are
    /// space-delimited, so there is nothing to segment).
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
    /// Italian.
    It,
    /// Spanish.
    Es,
    /// Romanian.
    Ro,
    /// Dutch.
    Nl,
    /// Swedish.
    Sv,
    /// Danish.
    Da,
    /// Norwegian Bokmål.
    Nb,
    /// Portuguese.
    Pt,
    /// Czech.
    Cs,
    /// Serbian, written in both Cyrillic and Latin.
    Sr,
    /// Estonian.
    Et,
    /// Hungarian.
    Hu,
    /// Finnish.
    Fi,
    /// Turkish.
    Tr,
    /// Maltese.
    Mt,
    /// Bulgarian.
    Bg,
    /// Ukrainian.
    Uk,
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
            Language::It => "it",
            Language::Es => "es",
            Language::Ro => "ro",
            Language::Nl => "nl",
            Language::Sv => "sv",
            Language::Da => "da",
            Language::Nb => "nb",
            Language::Pt => "pt",
            Language::Cs => "cs",
            Language::Sr => "sr",
            Language::Et => "et",
            Language::Hu => "hu",
            Language::Fi => "fi",
            Language::Tr => "tr",
            Language::Mt => "mt",
            Language::Bg => "bg",
            Language::Uk => "uk",
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
            "it" => Some(Language::It),
            "es" => Some(Language::Es),
            "ro" => Some(Language::Ro),
            "nl" => Some(Language::Nl),
            "sv" => Some(Language::Sv),
            "da" => Some(Language::Da),
            "nb" => Some(Language::Nb),
            "pt" => Some(Language::Pt),
            "cs" => Some(Language::Cs),
            "sr" => Some(Language::Sr),
            "et" => Some(Language::Et),
            "hu" => Some(Language::Hu),
            "fi" => Some(Language::Fi),
            "tr" => Some(Language::Tr),
            "mt" => Some(Language::Mt),
            "bg" => Some(Language::Bg),
            "uk" => Some(Language::Uk),
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
    /// multilingual word tokenizer (word runs dual-emitted, Han through jieba).
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
                Language::It => "multilingual-v1-it",
                Language::Es => "multilingual-v1-es",
                Language::Ro => "multilingual-v1-ro",
                Language::Nl => "multilingual-v1-nl",
                Language::Sv => "multilingual-v1-sv",
                Language::Da => "multilingual-v1-da",
                Language::Nb => "multilingual-v1-nb",
                Language::Pt => "multilingual-v1-pt",
                Language::Cs => "multilingual-v1-cs",
                Language::Sr => "multilingual-v1-sr",
                Language::Et => "multilingual-v1-et",
                Language::Hu => "multilingual-v1-hu",
                Language::Fi => "multilingual-v1-fi",
                Language::Tr => "multilingual-v1-tr",
                Language::Mt => "multilingual-v1-mt",
                Language::Bg => "multilingual-v1-bg",
                Language::Uk => "multilingual-v1-uk",
                Language::Other => "multilingual-v1-other",
            },
        }
    }

    /// The language whose profile segments with a dictionary directory.
    ///
    /// `Some` for [`Language::Ja`] only: Japanese cannot tokenise without one,
    /// so index build and open refuse it when the caller supplies no directory
    /// (see [`segment::DictionaryAnalysis`]).  Korean is dictionary-free: the
    /// word-run pipeline already produces word-level tokens, and a
    /// dictionary-backed Korean profile would be a separate profile identifier
    /// (`multilingual-v1-ko-<segmenter>`), never a mode of `multilingual-v1-ko`,
    /// because one identifier must name exactly one token stream.
    pub fn dictionary_language(self) -> Option<Language> {
        match self {
            Profile::Sag2ZhV1 => None,
            Profile::Multilingual {
                language: Language::Ja,
            } => Some(Language::Ja),
            Profile::Multilingual { .. } => None,
        }
    }
}

/// A resolved analysis pipeline could not be used.
///
/// Every variant is a hard failure: a profile that needs a dictionary never
/// falls back to the character pipeline, because that would silently build or
/// read an index with a different token stream than the one it records.
#[derive(Debug)]
pub enum AnalysisError {
    /// A dictionary directory is missing, unreadable or malformed.
    Dictionary {
        /// The directory or file that failed.
        path: PathBuf,
        /// The underlying failure (an I/O error or `lindera`'s load error).
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The profile segments with a dictionary and the caller supplied no
    /// directory.
    DictionaryMissing {
        /// The language whose profile needs one.
        language: Language,
    },
    /// The caller supplied a dictionary directory to a profile that does not
    /// tokenise with one.
    ///
    /// Ignoring it silently is how a caller comes to believe a Korean index was
    /// built with morphological analysis it never had: an index records the
    /// pipeline that produced its tokens, so a directory a profile cannot use is
    /// a caller mistake, not a harmless extra argument.
    DictionaryUnused {
        /// The profile the directory was passed for.
        profile: Profile,
    },
    /// The profile segments with a dictionary but this build has no `lindera`
    /// feature, so it cannot load one.
    DictionaryUnsupported {
        /// The language whose profile needs one.
        language: Language,
    },
    /// A segmenter failed on the text itself.
    ///
    /// Thai `newmm` returns an error instead of a token stream for input it
    /// cannot cover; the caller gets that error and never a character-level
    /// fallback, because a fallback would silently change the token stream the
    /// index records.
    Segmentation {
        /// The language whose segmenter failed.
        language: Language,
        /// The segmenter's own error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl AnalysisError {
    #[cfg(feature = "lindera")]
    pub(crate) fn dictionary(
        path: impl Into<PathBuf>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        AnalysisError::Dictionary {
            path: path.into(),
            source: Box::new(source),
        }
    }
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalysisError::Dictionary { path, source } => {
                write!(
                    formatter,
                    "cannot load dictionary {}: {source}",
                    path.display()
                )
            }
            AnalysisError::DictionaryMissing { language } => write!(
                formatter,
                "profile {} needs a dictionary directory; pass IndexOptions::dictionary_dir",
                Profile::Multilingual {
                    language: *language
                }
            ),
            AnalysisError::DictionaryUnsupported { language } => write!(
                formatter,
                "profile {} needs the `lindera` feature, which this build does not have",
                Profile::Multilingual {
                    language: *language
                }
            ),
            AnalysisError::DictionaryUnused { profile } => {
                write!(formatter, "profile {profile} does not use a dictionary")
            }
            AnalysisError::Segmentation { language, source } => write!(
                formatter,
                "cannot segment {} input: {source}",
                Profile::Multilingual {
                    language: *language
                }
            ),
        }
    }
}

impl std::error::Error for AnalysisError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AnalysisError::Dictionary { source, .. } => Some(source.as_ref()),
            AnalysisError::Segmentation { source, .. } => Some(source.as_ref()),
            AnalysisError::DictionaryMissing { .. }
            | AnalysisError::DictionaryUnused { .. }
            | AnalysisError::DictionaryUnsupported { .. } => None,
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

/// Tokenize `text` with `profile`, without a dictionary.
///
/// [`Profile::Sag2ZhV1`] is the certified pipeline and what [`crate::tokenize()`]
/// always uses; it does not enter [`normalize`] at all, so its token stream and
/// allocations are the ones the goldens pin.  [`Profile::Multilingual`] runs
/// [`normalize::normalize`] for its language and then the multilingual word
/// tokenizer - except for [`Language::Th`], where [`thai_tokens`] supplies the
/// word boundaries and keeps the certified ASCII model and the final filters.
///
/// This is the dictionary-free entry point.  [`Language::Ja`] cannot tokenise
/// without a dictionary, so it returns [`AnalysisError::DictionaryMissing`] here
/// rather than a character-level stream that would not match the index: its
/// callers resolve [`segment::DictionaryAnalysis`] first and use
/// [`tokenize_with_segmenter`].  Korean needs no dictionary.
pub fn tokenize_with(profile: Profile, text: &str) -> Result<Vec<String>, AnalysisError> {
    tokenize_with_segmenter(profile, text, None)
}

/// Tokenize `text` with `profile`, using `dictionary` when the profile needs one.
///
/// The returned tokens always come from one pipeline: the dictionary profiles
/// segment the normalised text with the supplied [`segment::DictionaryAnalysis`]
/// and apply the same `HAS_CONTENT`/stopword filter as every other profile.  The
/// failures are the dictionary itself (missing, unreadable or malformed) and the
/// segmenter's own error; a profile that needs a dictionary never produces
/// character tokens.
pub fn tokenize_with_segmenter(
    profile: Profile,
    text: &str,
    dictionary: Option<&segment::DictionaryAnalysis>,
) -> Result<Vec<String>, AnalysisError> {
    match profile {
        Profile::Sag2ZhV1 => Ok(tokenize_default(text)),
        Profile::Multilingual { language } => {
            let normalised = normalize::normalize(language, text);
            if let Some(missing) = profile.dictionary_language() {
                let dictionary =
                    dictionary.ok_or(AnalysisError::DictionaryMissing { language: missing })?;
                let mut tokens = dictionary.segment(&normalised)?;
                tokens.retain(|token| keep(token));
                return Ok(tokens);
            }
            match language {
                Language::Th => thai_tokens(&normalised),
                _ => Ok(tokenize_multilingual(language, &normalised, text)),
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
fn thai_tokens(normalised: &str) -> Result<Vec<String>, AnalysisError> {
    let mut tokens = Vec::new();
    for word in segment::thai::segment(normalised)? {
        if word.is_ascii() {
            ascii_cut::cut_run(&word, &mut tokens);
        } else {
            tokens.push(word);
        }
    }
    tokens.retain(|token| keep(token));
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::{
        tokenize_with, tokenize_with_segmenter, AnalysisError, DictionaryAnalysis, Language,
        Profile, REMOVED_MULTILINGUAL_V1,
    };

    /// [`tokenize_with`] for a profile that never needs a dictionary.
    fn tokens(profile: Profile, text: &str) -> Vec<String> {
        tokenize_with(profile, text).expect("a dictionary-free profile always tokenizes")
    }
    use std::str::FromStr;

    const LANGUAGES: [Language; 30] = [
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
        Language::It,
        Language::Es,
        Language::Ro,
        Language::Nl,
        Language::Sv,
        Language::Da,
        Language::Nb,
        Language::Pt,
        Language::Cs,
        Language::Sr,
        Language::Et,
        Language::Hu,
        Language::Fi,
        Language::Tr,
        Language::Mt,
        Language::Bg,
        Language::Uk,
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
                tokens(Profile::Sag2ZhV1, text),
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
        assert_eq!(
            tokens(german, "Straße"),
            ["s", "t", "r", "ß", "e", "strasse"]
        );
        assert_ne!(tokens(german, "Straße"), crate::tokenize("Straße"));
    }

    #[test]
    fn only_the_japanese_profile_needs_a_dictionary() {
        assert_eq!(Profile::Sag2ZhV1.dictionary_language(), None);
        for language in LANGUAGES {
            let profile = Profile::Multilingual { language };
            let expected = (language == Language::Ja).then_some(Language::Ja);
            assert_eq!(profile.dictionary_language(), expected, "{language:?}");
        }
    }

    #[test]
    fn a_dictionary_profile_without_a_directory_is_a_typed_error() {
        let error = DictionaryAnalysis::resolve(
            Profile::Multilingual {
                language: Language::Ja,
            },
            None,
        )
        .expect_err("no dictionary directory");
        match error {
            AnalysisError::DictionaryMissing {
                language: Language::Ja,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_thai_segmentation_failure_surfaces_through_tokenize_with() {
        // D1: `newmm` never fails on real text, so the error arm is driven
        // through the test-only fault injection seam.  If `segment`'s `Err`
        // were swallowed (`Ok(Vec::new())`, `unwrap_or_default`), this test
        // would see `Ok` and fail.
        use crate::analysis::segment::thai::{ForcedSegmentationFailure, FORCE_SEGMENT_ERROR};
        use std::error::Error as _;

        let thai = Profile::Multilingual {
            language: Language::Th,
        };
        FORCE_SEGMENT_ERROR.with(|flag| flag.set(true));
        let failed = tokenize_with(thai, "ห้องสมุด");
        FORCE_SEGMENT_ERROR.with(|flag| flag.set(false));

        let error = failed.expect_err("the injected segmenter failure reaches the caller");
        match &error {
            AnalysisError::Segmentation { language, source } => {
                assert_eq!(*language, Language::Th);
                assert!(
                    source.downcast_ref::<ForcedSegmentationFailure>().is_some(),
                    "the segmenter's own error is the source"
                );
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(
            error.source().is_some(),
            "`AnalysisError::Segmentation` must expose its source"
        );
        // The flag is thread-local and cleared, so the pipeline is healthy.
        assert_eq!(
            tokens(thai, "ห้องสมุด"),
            ["ห้องสมุด"],
            "the injected failure is not sticky"
        );
    }

    #[test]
    fn a_dictionary_profile_never_degrades_to_characters() {
        let profile = Profile::Multilingual {
            language: Language::Ja,
        };
        match tokenize_with(profile, "東京") {
            Err(AnalysisError::DictionaryMissing {
                language: Language::Ja,
            }) => {}
            other => panic!("Japanese must refuse without a dictionary, got {other:?}"),
        }
        match crate::match_query_with(crate::QueryMode::Lexical, profile, "東京") {
            Err(AnalysisError::DictionaryMissing {
                language: Language::Ja,
            }) => {}
            other => panic!("Japanese must not build an expression, got {other:?}"),
        }
        // Korean needs no dictionary, and its word-level stream is not the
        // character fallback the old wrapper produced.
        let korean = Profile::Multilingual {
            language: Language::Ko,
        };
        let tokens = tokens(korean, "도서관에서");
        assert_eq!(tokens.last().map(String::as_str), Some("도서관에서"));
        assert!(tokens.len() > 1, "{tokens:?}");
    }

    #[test]
    fn a_dictionary_free_profile_refuses_a_dictionary() {
        let directory = tempfile::tempdir().expect("temp dir");
        for profile in [
            Profile::Sag2ZhV1,
            Profile::Multilingual {
                language: Language::De,
            },
            Profile::Multilingual {
                language: Language::Ko,
            },
            Profile::Multilingual {
                language: Language::Th,
            },
        ] {
            assert!(
                DictionaryAnalysis::resolve(profile, None)
                    .expect("resolve without a directory")
                    .is_none(),
                "{profile}"
            );
            // A directory this profile cannot read is a caller mistake: it must
            // never be silently dropped, or a Korean build looks morphological.
            match DictionaryAnalysis::resolve(profile, Some(directory.path())) {
                Err(AnalysisError::DictionaryUnused { profile: found }) => {
                    assert_eq!(found, profile)
                }
                other => panic!("{profile}: unexpected {other:?}"),
            }
        }
    }

    #[test]
    fn an_unused_dictionary_names_the_profile_that_ignores_it() {
        use std::error::Error as _;

        let directory = tempfile::tempdir().expect("temp dir");
        let profile = Profile::Multilingual {
            language: Language::Ko,
        };
        let error = DictionaryAnalysis::resolve(profile, Some(directory.path()))
            .expect_err("Korean has no dictionary pipeline");
        assert_eq!(
            error.to_string(),
            "profile multilingual-v1-ko does not use a dictionary"
        );
        assert!(error.source().is_none());
    }

    #[cfg(not(feature = "lindera"))]
    #[test]
    fn a_dictionary_profile_needs_the_lindera_feature() {
        let directory = tempfile::tempdir().expect("temp dir");
        let error = DictionaryAnalysis::resolve(
            Profile::Multilingual {
                language: Language::Ja,
            },
            Some(directory.path()),
        )
        .expect_err("a build without the feature cannot load one");
        match error {
            AnalysisError::DictionaryUnsupported {
                language: Language::Ja,
            } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn the_segmenter_aware_entry_point_keeps_the_reference_stream() {
        for text in ["Where was Alston born?", "中华人民共和国成立了"] {
            assert_eq!(
                tokenize_with_segmenter(Profile::Sag2ZhV1, text, None).expect("no dictionary"),
                tokens(Profile::Sag2ZhV1, text)
            );
        }
    }
}
