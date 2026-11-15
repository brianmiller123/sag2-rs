//! Shared command-line plumbing for `build` and `search`.

use std::path::PathBuf;

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

/// The dictionary directory: `--dict-dir`, else `SAG2_DICT_DIR`.
///
/// Only the Japanese profile resolves one; every other profile reports
/// `AnalysisError::DictionaryUnused` rather than ignoring the value, so a caller
/// cannot come away believing it loaded a dictionary that was never read.  An
/// unset variable and no flag is `None`, which is not an error on its own.
pub fn dictionary_dir(explicit: Option<PathBuf>) -> Option<PathBuf> {
    explicit.or_else(|| {
        std::env::var_os("SAG2_DICT_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    })
}
