//! Shared command-line plumbing for `build` and `search`.

use std::path::PathBuf;

use sag2_core::Profile;

/// The value of the flag at `args[index]`, advancing `index` past it.
pub fn value<'a>(
    args: &'a [String],
    index: &mut usize,
    flag: &str,
    usage: &str,
) -> Result<&'a str, String> {
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} needs a value\n{usage}"))
}

/// The dictionary directory for `profile`: `--dict-dir`, else `SAG2_DICT_DIR`
/// when the profile reads a dictionary.
///
/// An explicit `--dict-dir` is passed through for every profile: it is a direct
/// instruction about the build in hand, so a profile that cannot read it still
/// reports [`sag2_core::AnalysisError::DictionaryUnused`] instead of building
/// without it.
///
/// `SAG2_DICT_DIR` is ambient state.  It outlives the build that exported it,
/// so it must not turn an unrelated `sag2-zh-v1` build into a failure: the
/// variable is consulted only for a profile that segments with a dictionary and
/// silently dropped for every other profile.  An unset variable and no flag is
/// `None`, which is not an error on its own.
pub fn dictionary_dir(profile: Profile, explicit: Option<PathBuf>) -> Option<PathBuf> {
    if explicit.is_some() {
        return explicit;
    }
    // Only a profile that segments with a dictionary reads the ambient
    // variable; every other profile drops it.
    profile.dictionary_language()?;
    std::env::var_os("SAG2_DICT_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}
