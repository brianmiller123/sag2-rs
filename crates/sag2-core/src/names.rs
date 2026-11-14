//! `NAMES` and `SENTENCES` from `sag2.py:52-55`, without lookaround.
//!
//! The reference patterns are
//!
//! ```text
//! NAME_WORD = (?:\p{Lu}\.|\p{Lu}[\p{L}\p{M}]*(?:[-’']\p{Lu}[\p{L}\p{M}]*)*)
//! NAMES     = (?<![\p{L}\p{N}]) NAME_WORD (?:[ \t]+ NAME_WORD)+ (?![\p{L}\p{N}])
//! SENTENCES = (?<!\b[A-Z])[.!?]\s+ | [。！？]+\s* | \n+
//! ```
//!
//! Both lookarounds are one character wide, so design §6.2 replaces them with
//! explicit checks after the match.  The scanners below are the pattern text
//! written out: each character class is one of the generated §4.5 tables
//! (`is_uppercase` is `\p{Lu}`, `is_letter` is `\p{L}`, `is_mark` is `\p{M}`),
//! so the classifiers are the reference's own Unicode data rather than a
//! crate's.
//!
//! **Backtracking is part of the contract.**  `(?:[ \t]+ NAME_WORD)+` is greedy
//! and the trailing `(?!…)` can fail, so Python retries shorter parses:
//! `Alice Paul Bob2` matches `Alice Paul`, `A. H. G. Wells2` matches
//! `A. H. G.` and `Alice Paul-Bob2` matches `Alice Paul`.  A match-then-filter
//! implementation would reject all three, so [`names`] enumerates the end
//! positions in the engine's own backtracking order (last word first, longest
//! first) and takes the first one whose right neighbour is not `\p{L}`/`\p{N}`.

use std::ops::Range;

use crate::unicode;

/// `NAME_WORD`'s `[-’']`, the characters that join two capitalised parts.
fn is_joiner(character: char) -> bool {
    matches!(character, '-' | '’' | '\'')
}

fn is_letter_or_number(character: char) -> bool {
    let cp = character as u32;
    unicode::is_letter(cp) || unicode::is_number(cp)
}

/// `\p{L}` or `\p{M}` - the body of a `NAME_WORD`.
fn is_name_body(character: char) -> bool {
    let cp = character as u32;
    unicode::is_letter(cp) || unicode::is_mark(cp)
}

/// One `NAME_WORD`'s possible end positions, longest first.
///
/// `start` must be a `\p{Lu}`.  The first entry is the greedy parse's end; the
/// rest are the ends the engine reaches by shrinking the trailing
/// `[\p{L}\p{M}]*` and by dropping whole `[-’']\p{Lu}…` groups, which is the
/// order Python's backtracking tries them in.  The `\p{Lu}\.` alternative is
/// tried before the long one, so its end (one past the period) comes first.
fn word_ends(characters: &[char], start: usize) -> Vec<usize> {
    let mut ascending = Vec::new();
    let mut index = start + 1;
    ascending.push(index);
    while index < characters.len() && is_name_body(characters[index]) {
        index += 1;
        ascending.push(index);
    }
    while index + 1 < characters.len()
        && is_joiner(characters[index])
        && unicode::is_uppercase(characters[index + 1] as u32)
    {
        index += 2;
        ascending.push(index);
        while index < characters.len() && is_name_body(characters[index]) {
            index += 1;
            ascending.push(index);
        }
    }
    let mut ends: Vec<usize> = ascending.into_iter().rev().collect();
    if characters.get(start + 1) == Some(&'.') {
        ends.insert(0, start + 2);
    }
    ends
}

/// Match `NAME_WORD (?:[ \t]+ NAME_WORD)+` at `start`, returning the end index
/// the reference regex would report for its first success.
fn match_chain(characters: &[char], start: usize) -> Option<usize> {
    if !unicode::is_uppercase(characters[start] as u32) {
        return None;
    }
    // `ends[word][0]` is that word's greedy end; the chain above it is greedy
    // too, so the next word is parsed at the greedy end plus its separator.
    let mut words = vec![word_ends(characters, start)];
    let mut index = words[0][0];
    loop {
        let mut next = index;
        while next < characters.len() && matches!(characters[next], ' ' | '\t') {
            next += 1;
        }
        if next == index || next >= characters.len() {
            break;
        }
        if !unicode::is_uppercase(characters[next] as u32) {
            break;
        }
        let ends = word_ends(characters, next);
        index = ends[0];
        words.push(ends);
    }
    if words.len() < 2 {
        return None;
    }
    // Backtracking order: the last word's ends, then the previous word's, …
    // A shorter word always ends before (and therefore ranks after) a longer
    // chain, so this is a single descending sequence of end positions.
    for ends in words.iter().skip(1).rev() {
        for &end in ends {
            if end == characters.len() || !is_letter_or_number(characters[end]) {
                return Some(end);
            }
        }
    }
    None
}

/// `NAMES.finditer` (`sag2.py:52-54`), as byte ranges into `text`.
pub fn names(text: &str) -> Vec<Range<usize>> {
    let characters: Vec<char> = text.chars().collect();
    let mut offsets: Vec<usize> = text.char_indices().map(|(offset, _)| offset).collect();
    offsets.push(text.len());
    let mut matches = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        let left_ok = start == 0 || !is_letter_or_number(characters[start - 1]);
        if left_ok {
            if let Some(end) = match_chain(&characters, start) {
                matches.push(offsets[start]..offsets[end]);
                start = end;
                continue;
            }
        }
        start += 1;
    }
    matches
}

/// `(?<!\b[A-Z])` at `index`: the period is preceded by a lone capital.
fn ends_initial_abbreviation(characters: &[char], index: usize) -> bool {
    if index == 0 {
        return false;
    }
    let previous = characters[index - 1];
    if !previous.is_ascii_uppercase() {
        return false;
    }
    // `\b` before an ASCII capital means the character before it is not a word
    // character (or it is the first character).
    index == 1 || !unicode::is_word(characters[index - 2] as u32)
}

/// `SENTENCES.split` (`sag2.py:55`), including the empty pieces `re.split`
/// yields at the edges.
pub fn sentences(text: &str) -> Vec<&str> {
    let characters: Vec<char> = text.chars().collect();
    let mut offsets: Vec<usize> = text.char_indices().map(|(offset, _)| offset).collect();
    offsets.push(text.len());
    let mut pieces = Vec::new();
    let mut cursor = 0;
    let mut index = 0;
    while index < characters.len() {
        let mut end = None;
        let character = characters[index];
        if matches!(character, '.' | '!' | '?') && !ends_initial_abbreviation(&characters, index) {
            let mut next = index + 1;
            if next < characters.len() && unicode::is_space(characters[next]) {
                while next < characters.len() && unicode::is_space(characters[next]) {
                    next += 1;
                }
                end = Some(next);
            }
        }
        if end.is_none() && matches!(character, '。' | '！' | '？') {
            let mut next = index + 1;
            while next < characters.len() && matches!(characters[next], '。' | '！' | '？') {
                next += 1;
            }
            while next < characters.len() && unicode::is_space(characters[next]) {
                next += 1;
            }
            end = Some(next);
        }
        if end.is_none() && character == '\n' {
            let mut next = index + 1;
            while next < characters.len() && characters[next] == '\n' {
                next += 1;
            }
            end = Some(next);
        }
        match end {
            Some(end) => {
                pieces.push(&text[offsets[cursor]..offsets[index]]);
                cursor = end;
                index = end;
            }
            None => index += 1,
        }
    }
    pieces.push(&text[offsets[cursor]..]);
    pieces
}

#[cfg(test)]
mod tests {
    use super::{names, sentences};

    fn matched(text: &str) -> Vec<String> {
        names(text)
            .into_iter()
            .map(|range| text[range].to_owned())
            .collect()
    }

    #[test]
    fn names_vectors_from_design_6_3() {
        assert_eq!(matched("Alice Paul visited"), vec!["Alice Paul"]);
        assert!(matched("xAlice Paul").is_empty());
        assert!(matched("Alice Paul2").is_empty());
        assert_eq!(matched("A. H. G. Wells"), vec!["A. H. G. Wells"]);
        assert_eq!(matched("Alice-Paul Smith"), vec!["Alice-Paul Smith"]);
        assert_eq!(matched("Alice\tPaul"), vec!["Alice\tPaul"]);
    }

    #[test]
    fn names_backtrack_to_a_shorter_chain() {
        // CPython: `Alice Paul Bob2` -> `Alice Paul`.
        assert_eq!(matched("Alice Paul Bob2"), vec!["Alice Paul"]);
        // CPython: `A. H. G. Wells2` -> `A. H. G.`.
        assert_eq!(matched("A. H. G. Wells2"), vec!["A. H. G."]);
        // CPython: `Alice Paul-Bob2` -> `Alice Paul`.
        assert_eq!(matched("Alice Paul-Bob2"), vec!["Alice Paul"]);
    }

    #[test]
    fn names_match_the_reference_initial_forms() {
        // The `\p{Lu}\.` alternative inside a chain, and the two-word minimum.
        assert_eq!(matched("J. Smith went."), vec!["J. Smith"]);
        assert!(matched("AB. Smith went.").is_empty());
    }

    #[test]
    fn names_need_two_words_and_both_boundaries() {
        assert!(matched("Alice").is_empty());
        assert!(matched("2Alice Paul").is_empty());
        assert_eq!(matched("(Alice Paul)"), vec!["Alice Paul"]);
        assert_eq!(
            matched("Alice Paul, Bob Smith"),
            vec!["Alice Paul", "Bob Smith"]
        );
    }

    #[test]
    fn sentences_vectors_from_design_6_3() {
        // `re.split` drops the separator: `AB. Smith` -> `AB`, `Smith`.
        assert_eq!(sentences("J. Smith went."), vec!["J. Smith went."]);
        assert_eq!(sentences("AB. Smith went."), vec!["AB", "Smith went."]);
        assert_eq!(sentences("3.5 is pi"), vec!["3.5 is pi"]);
        assert_eq!(sentences("好。 那好！"), vec!["好", "那好", ""]);
        assert_eq!(sentences("line1\nline2"), vec!["line1", "line2"]);
    }

    #[test]
    fn sentences_split_at_newlines_and_cjk_runs() {
        assert_eq!(sentences("a\n\nb"), vec!["a", "b"]);
        assert_eq!(sentences("a！！ b"), vec!["a", "b"]);
        assert_eq!(sentences(""), vec![""]);
    }
}
