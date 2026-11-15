//! Analysis profiles: the switch that decides how text becomes tokens.
//!
//! [`Profile::Sag2ZhV1`] is the certified reference pipeline of design §4 and
//! the only profile that reproduces the Python token stream byte for byte.
//! [`Profile::MultilingualV1`] is the profile the multilingual work lands on:
//! T1 plumbs it through core, index and facade, T3-T5 add its normalisation
//! and segmenters.  The identifier each profile persists in an index
//! (`rust_index_metadata.analysis`) comes from [`Profile::as_str`].

use std::fmt;
use std::str::FromStr;

use crate::tokenize::tokenize_default;

/// Which analysis pipeline turns text into the tokens an index stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// The certified reference pipeline: `NFKC` + lower + jieba 0.42.1 with
    /// the `[._-]`/`FREQ`/substring post-rules and the stopword filter
    /// (design §4.1, §4.2).
    Sag2ZhV1,
    /// The word-level pipeline for the multilingual corpora.
    MultilingualV1,
}

impl Profile {
    /// The stable identifier `rust_index_metadata.analysis` stores.
    pub const fn as_str(self) -> &'static str {
        match self {
            Profile::Sag2ZhV1 => "sag2-zh-v1",
            Profile::MultilingualV1 => "multilingual-v1",
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
/// reading it as [`Profile::Sag2ZhV1`] would silently mix token streams.
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
        match text {
            "sag2-zh-v1" => Ok(Profile::Sag2ZhV1),
            "multilingual-v1" => Ok(Profile::MultilingualV1),
            other => Err(UnknownProfile(other.to_owned())),
        }
    }
}

/// Tokenize `text` with `profile`.
///
/// [`Profile::Sag2ZhV1`] is the certified pipeline and what [`crate::tokenize`]
/// always uses.  [`Profile::MultilingualV1`] shares it today: the normalisation
/// and segmenters for it land in T3-T5, and until they do this arm must stay
/// byte-identical so the profile plumbing can be checked without moving the
/// default token stream.  The two arms are written out separately because they
/// diverge at that point.
pub fn tokenize_with(profile: Profile, text: &str) -> Vec<String> {
    match profile {
        Profile::Sag2ZhV1 => tokenize_default(text),
        Profile::MultilingualV1 => tokenize_default(text),
    }
}

#[cfg(test)]
mod tests {
    use super::{tokenize_with, Profile};
    use std::str::FromStr;

    #[test]
    fn identifiers_round_trip_through_from_str() {
        for profile in [Profile::Sag2ZhV1, Profile::MultilingualV1] {
            assert_eq!(Profile::from_str(profile.as_str()), Ok(profile));
            assert_eq!(profile.to_string(), profile.as_str());
        }
        let error = Profile::from_str("sag2-zh-v2").expect_err("an unknown identifier");
        assert_eq!(error.as_str(), "sag2-zh-v2");
    }

    #[test]
    fn the_two_profiles_tokenize_identically_until_t3() {
        for text in [
            "",
            "Alice Paul birthplace",
            "中华人民共和国成立了",
            "Café naïve — Ærøskøbing",
            "hello.world_name-42",
        ] {
            assert_eq!(
                tokenize_with(Profile::MultilingualV1, text),
                tokenize_with(Profile::Sag2ZhV1, text),
                "{text:?}"
            );
        }
    }
}
