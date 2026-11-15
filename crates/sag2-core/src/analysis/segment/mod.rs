//! Language segmenters for the multilingual profiles.
//!
//! A profile whose language writes without spaces ([`thai`]) replaces the
//! shared §4 jieba pipeline with its own word boundaries;
//! [`crate::analysis::tokenize_with`] keeps the certified ASCII model and the
//! final `HAS_CONTENT`/stopword filter around it, so every profile ends in the
//! same filters.
//!
//! [`Language::Ja`] and [`Language::Ko`] additionally need a morphological
//! dictionary.  That dictionary is a runtime directory, so their segmenter is
//! resolved once per index into a [`DictionaryAnalysis`] and the index fails
//! closed when it is absent: [`crate::analysis::AnalysisError::DictionaryMissing`]
//! on build and on open, never a silently degraded token stream.

use std::path::Path;

use crate::analysis::{AnalysisError, Language, Profile};

#[cfg(feature = "lindera")]
pub mod lindera_seg;
pub mod thai;

/// The dictionary a Japanese or Korean profile tokenises with.
///
/// Built by [`DictionaryAnalysis::resolve`] from the caller's dictionary
/// directory and kept for the lifetime of the index, so the build, name repair,
/// query construction and `expansion_terms` all tokenise through the same
/// loaded dictionary.  Its [`fingerprint`](Self::fingerprint) is the value the
/// index records in `rust_index_metadata.analysis_dict`.
///
/// Without the `lindera` feature the type still exists so the index compiles in
/// both feature states, but [`DictionaryAnalysis::resolve`] can never produce
/// one: a Japanese or Korean profile reports
/// [`AnalysisError::DictionaryUnsupported`].
pub struct DictionaryAnalysis {
    language: Language,
    fingerprint: String,
    #[cfg(feature = "lindera")]
    segmenter: lindera_seg::LinderaSegmenter,
}

impl DictionaryAnalysis {
    /// Resolve `profile` against `dictionary_dir`.
    ///
    /// A profile that does not segment with a dictionary resolves to `None`
    /// whatever the directory says.  A Japanese or Korean profile without a
    /// directory is [`AnalysisError::DictionaryMissing`], and one this build
    /// cannot serve (no `lindera` feature, missing or malformed directory) is a
    /// typed error - the caller must never fall back to the character pipeline.
    pub fn resolve(
        profile: Profile,
        dictionary_dir: Option<&Path>,
    ) -> Result<Option<Self>, AnalysisError> {
        let Some(language) = profile.dictionary_language() else {
            return Ok(None);
        };
        let dir = dictionary_dir.ok_or(AnalysisError::DictionaryMissing { language })?;
        #[cfg(feature = "lindera")]
        {
            // One `open` produces both the segmenter and its fingerprint, so
            // the recorded value cannot come from a different scan than the
            // lattice the index tokenises with (C7).
            let segmenter = lindera_seg::LinderaSegmenter::open(dir)?;
            Ok(Some(DictionaryAnalysis {
                language,
                fingerprint: segmenter.dictionary_fingerprint().to_owned(),
                segmenter,
            }))
        }
        #[cfg(not(feature = "lindera"))]
        {
            let _ = dir;
            Err(AnalysisError::DictionaryUnsupported { language })
        }
    }

    /// The language this dictionary segments.
    pub fn language(&self) -> Language {
        self.language
    }

    /// The `sha256` identifier of the dictionary contents
    /// (`lindera_seg::LinderaSegmenter::fingerprint`).
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Segment `text` with the loaded dictionary.
    pub fn segment(&self, text: &str) -> Result<Vec<String>, AnalysisError> {
        #[cfg(feature = "lindera")]
        {
            self.segmenter.segment(text)
        }
        #[cfg(not(feature = "lindera"))]
        {
            let _ = text;
            Err(AnalysisError::DictionaryUnsupported {
                language: self.language,
            })
        }
    }
}

impl std::fmt::Debug for DictionaryAnalysis {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DictionaryAnalysis")
            .field("language", &self.language)
            .field("fingerprint", &self.fingerprint)
            .finish_non_exhaustive()
    }
}
