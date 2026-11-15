//! `quote_term`, `match_query` and the query modes that build FTS5
//! expressions (`sag2.py:95-97`).
//!
//! `match_query` is `" OR ".join(quote_term(term) for term in
//! list(dict.fromkeys(tokenize(query)))[:80])`: deduplication keeps the first
//! occurrence of a token, and the 80-term cap applies *after* deduplication.
//!
//! [`QueryMode::Phrase3`] is query-side only: it builds word phrases and their
//! three-token windows from the same token stream, so no index, schema or
//! stored token changes for it.
//!
//! Both modes tokenise the query with a [`Profile`].  `match_query` stays on
//! [`Profile::Sag2ZhV1`] - the certified expression - while
//! [`match_query_with`] takes the profile of the index being queried, so a
//! query can never be tokenised by a different pipeline than the one that
//! produced the stored tokens.

use std::collections::HashSet;

use crate::{tokenize_with, Profile};

/// The cap both modes apply to the number of expression terms (`[:80]`,
/// `sag2.py:95-97`).
const EXPRESSION_TERM_BUDGET: usize = 80;

/// The slots [`QueryMode::Phrase3`] keeps for the unigram fallback, so phrase
/// terms cannot crowd the character `OR` out of the expression.
const UNIGRAM_RESERVE: usize = 8;

/// How [`match_query_with`] turns a query into an FTS5 `MATCH` expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum QueryMode {
    /// The certified expression: the de-duplicated `OR` of quoted unigrams
    /// (`sag2.py:69-97`).  This is the default and the only mode the gates
    /// replay.
    #[default]
    Lexical,
    /// Exact word phrases and their three-token windows, followed by the
    /// unigram `OR` as the fallback.
    ///
    /// Intended for the word-level analysis profiles: in the multilingual
    /// measurements the reference unigrams degrade to single characters for
    /// Russian, Arabic, Thai and Hindi, while exact and 3-gram phrases recover
    /// word order.  Query-side only - see [`match_query_with`] for the
    /// construction and budget policy.
    Phrase3,
}

/// `'"' + term.replace('"', '""') + '"'`.
pub fn quote_term(term: &str) -> String {
    let mut quoted = String::with_capacity(term.len() + 2);
    quoted.push('"');
    quoted.push_str(&term.replace('"', "\"\""));
    quoted.push('"');
    quoted
}

/// `match_query`: the de-duplicated `OR` of quoted tokens (`sag2.py:69-89`),
/// tokenised by the reference [`Profile::Sag2ZhV1`] pipeline.
pub fn match_query(query: &str) -> String {
    lexical_expression(Profile::Sag2ZhV1, query)
}

/// The de-duplicated `OR` of quoted tokens for `profile`, capped at the
/// expression budget.
fn lexical_expression(profile: Profile, query: &str) -> String {
    let mut terms = deduped_tokens(profile, query);
    terms.truncate(EXPRESSION_TERM_BUDGET);
    join_terms(&terms)
}

/// The expression for `query` built by `mode` and tokenised by `profile`.
///
/// [`QueryMode::Lexical`] with [`Profile::Sag2ZhV1`] is exactly
/// [`match_query`]: byte-identical, so the default path cannot move.  Callers
/// pass the profile of the index they are querying, so the query stream and the
/// stored stream always come from the same pipeline.
///
/// # `Phrase3` construction
///
/// The query is split on whitespace and every chunk is tokenised with
/// `profile`, keeping token order (no early deduplication).  A
/// one-token chunk contributes its quoted term; a longer chunk contributes the
/// whole chunk as one quoted phrase and then every three-token window of it.
///
/// A query with no whitespace (CJK, Thai) has no chunk boundary to respect, so
/// it contributes the three-token windows of the whole token stream, plus the
/// whole stream as one phrase when the stream is at most three tokens.
///
/// The [`match_query`] unigrams (built with the same `profile`) are always
/// appended as the fallback: exact phrases alone lose Korean recall (90.5% to
/// 49.4% R@10 in the plan's measurements), where the character unigrams act as
/// an accidental stemmer for the agglutinative particles.  Terms already emitted as phrases are not
/// repeated.
///
/// # Budget
///
/// Both modes emit at most 80 expression terms.  The `Phrase3` budget is spent
/// in priority order - whole-chunk phrases, then three-token phrases, then
/// unigrams - but eight slots are held back for the unigram fallback whenever
/// the query has that many distinct unigrams, so truncation can never drop the
/// whole fallback.
pub fn match_query_with(mode: QueryMode, profile: Profile, query: &str) -> String {
    match mode {
        QueryMode::Lexical => lexical_expression(profile, query),
        QueryMode::Phrase3 => match_query_phrase3(profile, query),
    }
}

/// `tokenize(query)` with the first occurrence of each token kept
/// (`list(dict.fromkeys(...))`).
fn deduped_tokens(profile: Profile, query: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut terms = Vec::new();
    for term in tokenize_with(profile, query) {
        if seen.insert(term.clone()) {
            terms.push(term);
        }
    }
    terms
}

fn join_terms(terms: &[String]) -> String {
    terms
        .iter()
        .map(|term| quote_term(term))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// [`QueryMode::Phrase3`]: phrases first, then the unigram fallback.
fn match_query_phrase3(profile: Profile, query: &str) -> String {
    let unigrams = deduped_tokens(profile, query);
    let mut terms = phrase_terms(profile, query);
    let reserve = UNIGRAM_RESERVE.min(unigrams.len());
    terms.truncate(EXPRESSION_TERM_BUDGET - reserve);
    let mut emitted: HashSet<String> = terms.iter().cloned().collect();
    for unigram in unigrams {
        if terms.len() >= EXPRESSION_TERM_BUDGET {
            break;
        }
        if emitted.insert(unigram.clone()) {
            terms.push(unigram);
        }
    }
    join_terms(&terms)
}

/// The phrase terms of `query` in priority order: every whole-chunk phrase,
/// then every three-token window.  Duplicates keep their first position.
fn phrase_terms(profile: Profile, query: &str) -> Vec<String> {
    let mut phrases = Vec::new();
    let mut seen = HashSet::new();
    if query.split_whitespace().count() <= 1 {
        let tokens = tokenize_with(profile, query);
        if tokens.len() <= 3 {
            push_phrase(&mut phrases, &mut seen, &tokens);
        }
        for window in tokens.windows(3) {
            push_phrase(&mut phrases, &mut seen, window);
        }
        return phrases;
    }
    let chunks: Vec<Vec<String>> = query
        .split_whitespace()
        .map(|chunk| tokenize_with(profile, chunk))
        .collect();
    for tokens in &chunks {
        push_phrase(&mut phrases, &mut seen, tokens);
    }
    for tokens in &chunks {
        for window in tokens.windows(3) {
            push_phrase(&mut phrases, &mut seen, window);
        }
    }
    phrases
}

fn push_phrase(phrases: &mut Vec<String>, seen: &mut HashSet<String>, tokens: &[String]) {
    if tokens.is_empty() {
        return;
    }
    let phrase = tokens.join(" ");
    if seen.insert(phrase.clone()) {
        phrases.push(phrase);
    }
}

#[cfg(test)]
mod tests {
    use super::{match_query, match_query_with, quote_term, QueryMode};
    use crate::Profile;

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

    #[test]
    fn lexical_mode_is_the_pinned_match_query() {
        // The strings this module already pinned must come through
        // `match_query_with(Lexical, ..)` byte for byte.
        assert_eq!(
            match_query_with(QueryMode::Lexical, Profile::Sag2ZhV1, "Anchor anchor"),
            match_query("Anchor anchor")
        );
        assert_eq!(
            match_query_with(
                QueryMode::Lexical,
                Profile::Sag2ZhV1,
                "Where was Alston born?"
            ),
            match_query("Where was Alston born?")
        );
    }

    #[test]
    fn phrase3_english_queries_pin_their_expressions() {
        // One-token chunks are quoted terms; the unigram fallback repeats
        // nothing already emitted.
        assert_eq!(
            match_query_with(QueryMode::Phrase3, Profile::Sag2ZhV1, "Anchor anchor"),
            "\"anchor\""
        );
        assert_eq!(
            match_query_with(
                QueryMode::Phrase3,
                Profile::Sag2ZhV1,
                "Where was Alston born?"
            ),
            "\"alston\" OR \"born\""
        );
        assert_eq!(
            match_query_with(
                QueryMode::Phrase3,
                Profile::Sag2ZhV1,
                "anchor birthplace depicts"
            ),
            "\"anchor\" OR \"birthplace\" OR \"depicts\""
        );
    }

    #[test]
    fn phrase3_chinese_queries_pin_their_expressions() {
        // No whitespace: three-token windows over the whole stream and then
        // the unigrams; the seven-token stream is too long to be one phrase.
        assert_eq!(
            match_query_with(
                QueryMode::Phrase3,
                Profile::Sag2ZhV1,
                "中华人民共和国成立了"
            ),
            concat!(
                "\"中华 华人 人民\" OR \"华人 人民 共和\" OR ",
                "\"人民 共和 共和国\" OR \"共和 共和国 中华人民共和国\" OR ",
                "\"共和国 中华人民共和国 成立\" OR \"中华\" OR \"华人\" OR ",
                "\"人民\" OR \"共和\" OR \"共和国\" OR \"中华人民共和国\" OR \"成立\""
            )
        );
        assert_eq!(
            match_query_with(QueryMode::Phrase3, Profile::Sag2ZhV1, "北京大学在哪里"),
            concat!(
                "\"北京 大学 北京大学\" OR \"大学 北京大学 哪里\" OR ",
                "\"北京\" OR \"大学\" OR \"北京大学\" OR \"哪里\""
            )
        );
    }

    #[test]
    fn phrase3_russian_queries_pin_their_expressions() {
        // Each tokenised word is a run of single characters, so the whole
        // chunk phrase and its windows carry the word order the unigrams lose.
        assert_eq!(
            match_query_with(
                QueryMode::Phrase3,
                Profile::Sag2ZhV1,
                "кто родился в Москве"
            ),
            concat!(
                "\"к т о\" OR \"р о д и л с я\" OR \"в\" OR \"м о с к в е\" OR ",
                "\"р о д\" OR \"о д и\" OR \"д и л\" OR \"и л с\" OR \"л с я\" OR ",
                "\"м о с\" OR \"о с к\" OR \"с к в\" OR \"к в е\" OR \"к\" OR ",
                "\"т\" OR \"о\" OR \"р\" OR \"д\" OR \"и\" OR \"л\" OR \"с\" OR ",
                "\"я\" OR \"м\" OR \"е\""
            )
        );
        assert_eq!(
            match_query_with(
                QueryMode::Phrase3,
                Profile::Sag2ZhV1,
                "памятник Александру Пушкину"
            ),
            concat!(
                "\"п а м я т н и к\" OR \"а л е к с а н д р у\" OR ",
                "\"п у ш к и н у\" OR \"п а м\" OR \"а м я\" OR \"м я т\" OR ",
                "\"я т н\" OR \"т н и\" OR \"н и к\" OR \"а л е\" OR \"л е к\" OR ",
                "\"е к с\" OR \"к с а\" OR \"с а н\" OR \"а н д\" OR \"н д р\" OR ",
                "\"д р у\" OR \"п у ш\" OR \"у ш к\" OR \"ш к и\" OR \"к и н\" OR ",
                "\"и н у\" OR \"п\" OR \"а\" OR \"м\" OR \"я\" OR \"т\" OR \"н\" OR ",
                "\"и\" OR \"к\" OR \"л\" OR \"е\" OR \"с\" OR \"д\" OR \"р\" OR ",
                "\"у\" OR \"ш\""
            )
        );
    }

    #[test]
    fn phrase3_thai_queries_pin_their_expressions() {
        // Thai has no whitespace: every window is a phrase over characters the
        // tokenizer surfaces one by one.
        assert_eq!(
            match_query_with(
                QueryMode::Phrase3,
                Profile::Sag2ZhV1,
                "กรุงเทพมหานครเป็นเมืองหลวง"
            ),
            concat!(
                "\"ก ร ง\" OR \"ร ง เ\" OR \"ง เ ท\" OR \"เ ท พ\" OR \"ท พ ม\" OR ",
                "\"พ ม ห\" OR \"ม ห า\" OR \"ห า น\" OR \"า น ค\" OR \"น ค ร\" OR ",
                "\"ค ร เ\" OR \"ร เ ป\" OR \"เ ป น\" OR \"ป น เ\" OR \"น เ ม\" OR ",
                "\"เ ม อ\" OR \"ม อ ง\" OR \"อ ง ห\" OR \"ง ห ล\" OR \"ห ล ว\" OR ",
                "\"ล ว ง\" OR \"ก\" OR \"ร\" OR \"ง\" OR \"เ\" OR \"ท\" OR \"พ\" OR ",
                "\"ม\" OR \"ห\" OR \"า\" OR \"น\" OR \"ค\" OR \"ป\" OR \"อ\" OR ",
                "\"ล\" OR \"ว\""
            )
        );
    }

    #[test]
    fn the_phrase3_budget_reserves_eight_unigrams() {
        // A collision-free letter pool: Cyrillic, Greek, Thai and Arabic
        // letters, each of which the tokenizer surfaces as one token, then the
        // same run reversed (no 3-gram of distinct letters can occur twice).
        // Every three-character chunk is a distinct multi-token phrase, so 80
        // candidate phrases exceed the 72 slots the reserve leaves.
        let letters = concat!(
            "абвгдежзийклмнопрстуфхцчшщъыэюя",
            "αβγδεζηθικλμνξοπρστυφχψω",
            "กขคงจฉชซญดตถทนบปผพฟภมยรลวศสหอฮเแโใไำ",
            "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
        );
        let pool: String = letters.chars().chain(letters.chars().rev()).collect();
        let chunks: Vec<String> = pool
            .chars()
            .collect::<Vec<_>>()
            .chunks(3)
            .map(|chunk| chunk.iter().collect())
            .collect();
        let query = chunks.join(" ");
        let expression = match_query_with(QueryMode::Phrase3, Profile::Sag2ZhV1, &query);
        let terms: Vec<&str> = expression.split(" OR ").collect();
        let phrase = |chunk: &str| {
            quote_term(
                &chunk
                    .chars()
                    .map(|character| character.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )
        };
        assert_eq!(terms.len(), 80);
        // 80 - 8: the phrase group is truncated at the reserve boundary.
        assert_eq!(terms[71], phrase(&chunks[71]));
        assert!(terms[..72].iter().all(|term| term.contains(' ')));
        // The reserve keeps the unigram fallback: the first eight distinct
        // query tokens, in query order.
        assert_eq!(
            &terms[72..],
            ["\"а\"", "\"б\"", "\"в\"", "\"г\"", "\"д\"", "\"е\"", "\"ж\"", "\"з\"",]
        );
        assert!(!expression.contains(&phrase(&chunks[72])));
    }
}
