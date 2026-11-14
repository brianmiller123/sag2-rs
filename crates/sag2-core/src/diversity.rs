//! `sag2.py:122` `diversity_tokens` - the term sequence DPP's vectors use.
//!
//! The reference is `re.findall(r"[^\W_]+", text.casefold())`: every maximal
//! run of word characters that is not `_`, after case folding.  CPython's
//! `\w` is `str.isalnum() || '_'`, so `[^\W_]` is exactly `\p{L} \p{N}` for
//! every code point assigned in Unicode 15.0 - the tables Gate D certifies.
//!
//! The order is the reference's order (left to right over the folded string),
//! which `rank_dpp` turns into the insertion order of each candidate's term
//! vector (design §9 R5: the compensated `sum()` runs over it).

use crate::{casefold, is_letter, is_number};

pub fn diversity_tokens(text: &str) -> Vec<String> {
    let folded = casefold(text);
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in folded.chars() {
        let scalar = character as u32;
        if is_letter(scalar) || is_number(scalar) {
            current.push(character);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::diversity_tokens;

    #[test]
    fn are_alnum_runs_after_casefold() {
        assert_eq!(
            diversity_tokens("A.H.G. half-ounce_日本"),
            ["a", "h", "g", "half", "ounce", "日本"]
        );
        assert_eq!(diversity_tokens("  "), Vec::<String>::new());
        assert_eq!(diversity_tokens("ß"), ["ss"]);
    }
}
