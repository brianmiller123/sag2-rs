//! `quote_term` and `match_query` (`sag2.py:95-97`) - the FTS5 expression
//! builders every retrieval path shares.
//!
//! `match_query` is `" OR ".join(quote_term(term) for term in
//! list(dict.fromkeys(tokenize(query)))[:80])`: deduplication keeps the first
//! occurrence of a token, and the 80-term cap applies *after* deduplication.

use std::collections::HashSet;

use crate::tokenize;

/// `'"' + term.replace('"', '""') + '"'`.
pub fn quote_term(term: &str) -> String {
    let mut quoted = String::with_capacity(term.len() + 2);
    quoted.push('"');
    quoted.push_str(&term.replace('"', "\"\""));
    quoted.push('"');
    quoted
}

/// `match_query`: the de-duplicated `OR` of quoted tokens (`sag2.py:69-89`).
pub fn match_query(query: &str) -> String {
    let mut seen = HashSet::new();
    let mut terms = Vec::new();
    for term in tokenize(query) {
        if seen.insert(term.clone()) {
            terms.push(term);
        }
    }
    terms.truncate(80);
    terms
        .iter()
        .map(|term| quote_term(term))
        .collect::<Vec<_>>()
        .join(" OR ")
}

#[cfg(test)]
mod tests {
    use super::{match_query, quote_term};

    #[test]
    fn quotes_by_doubling_embedded_quotes() {
        assert_eq!(quote_term("alice paul"), "\"alice paul\"");
        assert_eq!(quote_term("\"quoted\""), "\"\"\"quoted\"\"\"");
    }

    #[test]
    fn deduplicates_query_tokens_before_the_cap() {
        // `tokenize` lowercases and drops stopwords, so "the" disappears and
        // the repeated word keeps one occurrence.
        assert_eq!(match_query("Anchor anchor"), "\"anchor\"");
        assert_eq!(
            match_query("Where was Alston born?"),
            "\"alston\" OR \"born\""
        );
    }
}
