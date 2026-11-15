//! Gate B: `expansion_terms` and `retrieve_expansion` (`sag2.py:585-754`).
//!
//! Both functions are pure functions of the corpus statistics, the FTS index
//! and their arguments; the only state they share with the reference is the
//! `frequency` memo (`self.frequencies` in Python), which is what
//! [`CorpusIndex::frequency`] keeps.
//!
//! The order-sensitive parts are load-bearing (design §1.2/§13):
//!
//! * `ranked` is insertion-ordered: the first seed that produces a term fixes
//!   its position and a later higher score replaces the value **in place**, so
//!   the final stable `sorted` keeps insertion order for exact ties.
//! * `context` is a naive left fold (`sag2.py:626-635`), not a compensated
//!   `sum()`: the summation order of `dict.fromkeys(tokenize(text))` matters.
//! * `hits` interleave round-robin by rank over the term lists, not term by
//!   term.
//!
//! R5 platform assumption: the `ln` values here (`sag2.py:629/643/652`) and in
//! `rank_dpp` are Rust `f64::ln` against CPython's `math.log`.  Gate B compares
//! the resulting bits, but only for the arguments its fixtures reach; outside
//! that sample equality rests on the platform libm rounding these operations
//! identically to the reference build.

use sag2_core::{names, quote_term, tokenize_with, Profile, QueryMode};
use sag2_json::{compact_json, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::pool::QUERY_WORKERS;
use crate::retrieve::QueryError;
use crate::CorpusIndex;

/// `sag2.py:590,612` `" ".join(tokenize(...))`, with the index's profile.
fn tokenized_phrase(profile: Profile, text: &str) -> String {
    tokenize_with(profile, text).join(" ")
}

/// One seed sentence as the reference builds it: the raw text plus the words
/// `expansion_terms` counts (`sag2.py:608-617`).
struct Sentence {
    text: String,
    words: Vec<String>,
}

/// `sag2.py:585-663`'s seed input (`seed["document_id"]`, `seed["jev_score"]`,
/// `seed["content"]`).
#[derive(Debug, Clone, PartialEq)]
pub struct Seed {
    /// The seed chunk's document id.
    pub document_id: i64,
    /// The judge score the seed was ranked by.
    pub jev_score: f64,
    /// The seed chunk's content.
    pub content: String,
}

/// `"name_phrase"` when the term came from a `NAMES` match (`sag2.py:649`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpansionKind {
    /// A term produced by a `NAMES` phrase match.
    NamePhrase,
    /// A term produced by a seed sentence's word.
    Word,
}

impl ExpansionKind {
    /// The wire name (`"name_phrase"` / `"word"`).
    pub fn as_str(self) -> &'static str {
        match self {
            ExpansionKind::NamePhrase => "name_phrase",
            ExpansionKind::Word => "word",
        }
    }
}

/// One entry of `expansion_terms`' ordered result.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpansionTerm {
    /// The expansion term.
    pub term: String,
    /// The weighted score (`sag2.py:659`).
    pub score: f64,
    /// The unweighted score before seed weighting.
    pub unweighted_score: f64,
    /// The seed-derived weight.
    pub seed_weight: f64,
    /// The seed's judge score.
    pub seed_jev_score: f64,
    /// `self.frequency(term)`.
    pub term_frequency: i64,
    /// The number of documents the term matches.
    pub document_frequency: i64,
    /// The query-context multiplier for the term.
    pub query_context_score: f64,
    /// The document the term was expanded from.
    pub seed_document_id: i64,
    /// Whether the term came from a name phrase or a word.
    pub kind: ExpansionKind,
    /// The matched `NAMES` text verbatim, for `name_phrase` terms only.
    pub name: Option<String>,
}

/// One `selected[]` entry of `retrieve_expansion`'s per-term record.
#[derive(Debug, Clone, PartialEq)]
pub struct Selected {
    /// The selected chunk's document id.
    pub document_id: i64,
    /// Its rank in that term's BM25 retrieval.
    pub bm25_rank: usize,
    /// Its BM25 score.
    pub bm25_score: f64,
}

/// One `queries[]` entry (`sag2.py:723-733`).
#[derive(Debug, Clone, PartialEq)]
pub struct ExpansionQuery {
    /// The expansion term this query was built for.
    pub term: String,
    /// The term's rank in `expansion_terms`.
    pub term_rank: usize,
    /// Whether the term came from a name phrase or a word.
    pub kind: ExpansionKind,
    /// The FTS5 `MATCH` expression.
    pub match_expression: String,
    /// The ranking expression the query ran with.
    pub ranking_expression: String,
    /// How many rows the term retrieved.
    pub retrieved_count: usize,
    /// The rows the reference selected, in order.
    pub selected: Vec<Selected>,
}

/// One `hits[]` entry: the retrieved row plus the §1.2 attribution fields.
///
/// Python merges the chunk row and the attribution into one dict
/// (`sag2.py:741-746`); `hit` is that row and the three fields after it are the
/// attribution Gate B compares.  `search_sag2` needs the full row (the judge
/// payload reads `title`/`heading`/`content`).
#[derive(Debug, Clone, PartialEq)]
pub struct ExpansionHit {
    /// The chunk id.
    pub id: i64,
    /// The chunk's document id.
    pub document_id: i64,
    /// The hit's score in the per-term retrieval.
    pub score: f64,
    /// The expansion term that produced it.
    pub expansion_term: String,
    /// The term's rank in `expansion_terms`.
    pub expansion_term_rank: usize,
    /// The hit's rank in the term's BM25 retrieval.
    pub expansion_bm25_rank: usize,
    /// The full retrieved chunk row.
    pub hit: crate::retrieve::Hit,
}

/// A failure of the expansion path.
#[derive(Debug)]
pub enum ExpansionError {
    /// A retrieval call failed.
    Query(QueryError),
    /// `self.aliases[source.casefold()]` missed (`sag2.py:411` raises
    /// `KeyError`): the pattern matched text whose casefold is not a key.
    UnknownAlias(String),
}

impl std::fmt::Display for ExpansionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpansionError::Query(error) => write!(formatter, "{error}"),
            ExpansionError::UnknownAlias(source) => write!(
                formatter,
                "alias pattern matched {source:?} but its casefold is not an alias key"
            ),
        }
    }
}

impl std::error::Error for ExpansionError {}

impl From<QueryError> for ExpansionError {
    fn from(error: QueryError) -> Self {
        ExpansionError::Query(error)
    }
}

fn is_letter_or_number(character: char) -> bool {
    let cp = character as u32;
    sag2_core::is_letter(cp) || sag2_core::is_number(cp)
}

/// `sag2.py:314-326`'s `name_pattern`, matched without a regex engine.
///
/// `(?<![\p{L}\p{N}])(?:alias1|alias2|…)(?![\p{L}\p{N}])`, `IGNORECASE`, with
/// the alternatives sorted by `(-len(value), value)`.  Python's alternation is
/// ordered, so the sort is what makes the longest alias win.
pub(crate) struct NamePattern {
    /// Each alias as its per-character `str.casefold()` sequence.
    aliases: Vec<Vec<String>>,
}

impl NamePattern {
    /// `None` when the corpus has no aliases (`self.name_pattern is None`).
    pub(crate) fn compile(aliases: &BTreeMap<String, String>) -> Option<Self> {
        if aliases.is_empty() {
            return None;
        }
        let mut keys: Vec<&String> = aliases.keys().collect();
        keys.sort_by(|left, right| {
            right
                .chars()
                .count()
                .cmp(&left.chars().count())
                .then_with(|| left.cmp(right))
        });
        Some(NamePattern {
            aliases: keys
                .into_iter()
                .map(|key| {
                    key.chars()
                        .map(|character| sag2_core::casefold(&character.to_string()))
                        .collect()
                })
                .collect(),
        })
    }

    /// The length in characters of the match starting at `start`, if any.
    ///
    /// The left lookbehind is checked once per position (it sits outside the
    /// group); the right lookahead is checked per alternative, because a failed
    /// alternative lets Python try the next one at the same position.
    fn match_at(&self, characters: &[char], start: usize) -> Option<usize> {
        if start > 0 && is_letter_or_number(characters[start - 1]) {
            return None;
        }
        for alias in &self.aliases {
            if start + alias.len() > characters.len() {
                continue;
            }
            let mut matched = true;
            for (offset, folded) in alias.iter().enumerate() {
                let character = characters[start + offset];
                if sag2_core::casefold(&character.to_string()) != *folded {
                    matched = false;
                    break;
                }
            }
            if !matched {
                continue;
            }
            let end = start + alias.len();
            if end == characters.len() || !is_letter_or_number(characters[end]) {
                return Some(alias.len());
            }
        }
        None
    }
}

impl CorpusIndex {
    pub(crate) fn name_pattern(&self) -> Option<&NamePattern> {
        self.name_pattern
            .get_or_init(|| NamePattern::compile(&self.statistics.aliases))
            .as_ref()
    }

    /// `self.frequency(term)` (`sag2.py:582-591`), memoised in the statistics
    /// map the reference mutates in place.
    pub fn frequency(&self, term: &str) -> Result<i64, QueryError> {
        if let Some(count) = self
            .frequencies
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(term)
        {
            return Ok(*count);
        }
        let count = self.pool.frequency(term)?;
        self.frequencies
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(term.to_owned(), count);
        Ok(count)
    }

    /// `self.has_match(expression, excluded)` (`sag2.py:444-452`).
    pub fn has_match(
        &self,
        expression: &str,
        excluded: &BTreeSet<i64>,
    ) -> Result<bool, QueryError> {
        let excluded_json = compact_json(&Value::Array(
            excluded.iter().copied().map(Value::Int).collect(),
        ));
        self.pool.has_match(expression, &excluded_json)
    }

    /// `self.repair_names(text, excluded)` (`sag2.py:403-421`).
    ///
    /// The three reference rules are kept verbatim: an alias is looked up by
    /// `str.casefold()`; a match whose tokenised form is present in the corpus
    /// is returned unchanged; and the target only replaces the source when the
    /// target's own tokenised form is present.
    pub fn repair_names(
        &self,
        text: &str,
        excluded: &BTreeSet<i64>,
    ) -> Result<String, ExpansionError> {
        let Some(pattern) = self.name_pattern() else {
            return Ok(text.to_owned());
        };
        let characters: Vec<char> = text.chars().collect();
        let mut offsets: Vec<usize> = text.char_indices().map(|(offset, _)| offset).collect();
        offsets.push(text.len());
        let mut repaired = String::with_capacity(text.len());
        let mut cursor = 0;
        let mut index = 0;
        while index < characters.len() {
            let Some(length) = pattern.match_at(&characters, index) else {
                index += 1;
                continue;
            };
            let end = index + length;
            let source = &text[offsets[index]..offsets[end]];
            repaired.push_str(&text[cursor..offsets[index]]);
            repaired.push_str(&self.replacement(source, excluded)?);
            cursor = offsets[end];
            index = end;
        }
        repaired.push_str(&text[cursor..]);
        Ok(repaired)
    }

    /// The `replace` callback of `sag2.py:405-416`.
    fn replacement(
        &self,
        source: &str,
        excluded: &BTreeSet<i64>,
    ) -> Result<String, ExpansionError> {
        if self.has_match(
            &quote_term(&tokenized_phrase(self.profile(), source)),
            excluded,
        )? {
            return Ok(source.to_owned());
        }
        let target = self
            .statistics
            .aliases
            .get(&sag2_core::casefold(source))
            .ok_or_else(|| ExpansionError::UnknownAlias(source.to_owned()))?;
        if self.has_match(
            &quote_term(&tokenized_phrase(self.profile(), target)),
            excluded,
        )? {
            Ok(target.clone())
        } else {
            Ok(source.to_owned())
        }
    }

    /// `sag2.py:420-421`: `match_query(self.repair_names(query, set()))`.
    pub fn match_query(&self, query: &str) -> Result<String, ExpansionError> {
        self.match_query_with(QueryMode::Lexical, query)
    }

    /// [`match_query`](Self::match_query) with an explicit [`QueryMode`].
    ///
    /// The query is tokenised with the profile this index was opened with
    /// ([`CorpusIndex::profile`]), so the query stream and the stored stream
    /// always come from the same pipeline.
    pub fn match_query_with(&self, mode: QueryMode, query: &str) -> Result<String, ExpansionError> {
        let repaired = self.repair_names(query, &BTreeSet::new())?;
        Ok(sag2_core::match_query_with(mode, self.profile(), &repaired))
    }

    /// `sag2.py:585-663` `expansion_terms`.
    pub fn expansion_terms(
        &self,
        query: &str,
        seeds: &[Seed],
        limit: usize,
    ) -> Result<Vec<ExpansionTerm>, ExpansionError> {
        let sentences: Vec<Vec<&str>> = seeds
            .iter()
            .map(|seed| names::sentences(&seed.content))
            .collect();

        // `unknown`: every word and name phrase of every seed sentence that the
        // cached statistics do not already know (`sag2.py:586-596`).
        let mut unknown: BTreeSet<String> = BTreeSet::new();
        for texts in &sentences {
            for text in texts {
                let text: &str = text;
                for word in tokenize_with(self.profile(), text) {
                    unknown.insert(word);
                }
                for range in names::names(text) {
                    unknown.insert(tokenized_phrase(self.profile(), &text[range]));
                }
            }
        }
        {
            let frequencies = self
                .frequencies
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            unknown.retain(|term| !frequencies.contains_key(term));
        }
        if unknown.len() > 1 {
            let terms: Vec<String> = unknown.into_iter().collect();
            let workers = terms.len().min(QUERY_WORKERS);
            let mut results: Vec<(String, i64)> = Vec::new();
            for worker in 0..workers {
                let mut position = worker;
                while position < terms.len() {
                    let term = terms[position].clone();
                    let count = self.pool.frequency(&term)?;
                    results.push((term, count));
                    position += workers;
                }
            }
            let mut frequencies = self
                .frequencies
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for (term, count) in results {
                frequencies.insert(term, count);
            }
        }

        let original: BTreeSet<String> = tokenize_with(self.profile(), query).into_iter().collect();
        // `max([0, *(seed["jev_score"] ...)])`.
        let max_score = seeds
            .iter()
            .map(|seed| seed.jev_score)
            .fold(0.0_f64, f64::max);

        // `ranked`: insertion-ordered, replaced in place by a higher score.
        let mut ranked: Vec<ExpansionTerm> = Vec::new();
        let mut positions: BTreeMap<String, usize> = BTreeMap::new();
        for (seed, texts) in seeds.iter().zip(&sentences) {
            let mut parsed: Vec<Sentence> = Vec::with_capacity(texts.len());
            let mut names_by_term: BTreeMap<String, String> = BTreeMap::new();
            for text in texts {
                let text: &str = text;
                let mut ungrouped = String::new();
                let mut end = 0;
                let mut sentence_names = Vec::new();
                for range in names::names(text) {
                    let term = tokenized_phrase(self.profile(), &text[range.clone()]);
                    names_by_term
                        .entry(term.clone())
                        .or_insert_with(|| text[range.clone()].to_owned());
                    sentence_names.push(term);
                    ungrouped.push_str(&text[end..range.start]);
                    ungrouped.push(' ');
                    end = range.end;
                }
                let mut words =
                    tokenize_with(self.profile(), &format!("{ungrouped}{}", &text[end..]));
                words.extend(sentence_names);
                parsed.push(Sentence {
                    text: text.to_owned(),
                    words,
                });
            }

            // `Counter` over the sentences, in first-appearance order.
            let mut counts: Vec<(String, i64)> = Vec::new();
            let mut count_positions: BTreeMap<String, usize> = BTreeMap::new();
            for sentence in &parsed {
                for word in &sentence.words {
                    match count_positions.get(word) {
                        Some(&position) => counts[position].1 += 1,
                        None => {
                            count_positions.insert(word.clone(), counts.len());
                            counts.push((word.clone(), 1));
                        }
                    }
                }
            }

            // `context_scores`: max of the naive per-sentence context.
            let mut context_scores: BTreeMap<String, f64> = BTreeMap::new();
            for sentence in &parsed {
                let mut context = 0.0_f64;
                let mut visited: BTreeSet<String> = BTreeSet::new();
                for term in tokenize_with(self.profile(), &sentence.text) {
                    if !visited.insert(term.clone()) {
                        continue;
                    }
                    if !original.contains(&term) {
                        continue;
                    }
                    let frequency = self.frequency(&term)?;
                    let ratio = (self.statistics.total - frequency) as f64 + 0.5;
                    context += (1.0 + ratio / (frequency as f64 + 0.5)).ln();
                }
                for word in &sentence.words {
                    let score = context_scores.entry(word.clone()).or_insert(0.0);
                    if context > *score {
                        *score = context;
                    }
                }
            }

            let weight = 0.5
                + 0.5
                    * if max_score > 0.0 {
                        seed.jev_score / max_score
                    } else {
                        0.0
                    };
            for (term, term_frequency) in &counts {
                let document_frequency = self.frequency(term)?;
                if tokenize_with(self.profile(), term)
                    .iter()
                    .all(|word| original.contains(word))
                    || document_frequency == 0
                {
                    continue;
                }
                let context = *context_scores
                    .get(term)
                    .expect("every counted word has a context score");
                let top = (self.statistics.total - document_frequency) as f64 + 0.5;
                let idf = (1.0 + top / (document_frequency as f64 + 0.5)).ln();
                let unweighted = (1.0 + (*term_frequency as f64).ln()) * idf * (1.0 + context);
                let score = unweighted * weight;
                let record = ExpansionTerm {
                    term: term.clone(),
                    score,
                    unweighted_score: unweighted,
                    seed_weight: weight,
                    seed_jev_score: seed.jev_score,
                    term_frequency: *term_frequency,
                    document_frequency,
                    query_context_score: context,
                    seed_document_id: seed.document_id,
                    kind: if names_by_term.contains_key(term) {
                        ExpansionKind::NamePhrase
                    } else {
                        ExpansionKind::Word
                    },
                    name: names_by_term.get(term).cloned(),
                };
                match positions.get(term) {
                    Some(&position) => {
                        if score > ranked[position].score {
                            ranked[position] = record;
                        }
                    }
                    None => {
                        positions.insert(term.clone(), ranked.len());
                        ranked.push(record);
                    }
                }
            }
        }

        // `sorted(..., key=(-score, COLLATOR.sort_key(term)))`, stable.
        let mut decorated: Vec<(ExpansionTerm, Vec<u32>)> = ranked
            .into_iter()
            .map(|term| {
                let key = sag2_core::sort_key(&term.term);
                (term, key)
            })
            .collect();
        decorated.sort_by(|(left, left_key), (right, right_key)| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left_key.cmp(right_key))
        });
        decorated.truncate(limit);
        Ok(decorated.into_iter().map(|(term, _)| term).collect())
    }

    /// `sag2.py:665-754` `retrieve_expansion`.
    pub fn retrieve_expansion(
        &self,
        expression: &str,
        terms: &[ExpansionTerm],
        excluded: &BTreeSet<i64>,
        limit: usize,
    ) -> Result<(Vec<ExpansionHit>, Vec<ExpansionQuery>), ExpansionError> {
        // `sag2.py:666-690`: repair and title grounding, on copies.
        let mut resolved: Vec<ExpansionTerm> = Vec::with_capacity(terms.len());
        for item in terms {
            let mut term = item.clone();
            let original = item.name.clone().unwrap_or_else(|| item.term.clone());
            if !self.has_match(&quote_term(&term.term), excluded)? {
                let repaired = self.repair_names(&original, excluded)?;
                if repaired != original {
                    term.term = tokenized_phrase(self.profile(), &repaired);
                }
            }
            if term.kind == ExpansionKind::NamePhrase
                && !self.has_match(&quote_term(&term.term), excluded)?
            {
                let words: Vec<String> = term.term.split_whitespace().map(str::to_owned).collect();
                let mut matches: Vec<Vec<String>> = Vec::new();
                for start in 0..words.len() {
                    for end in (start + 2)..=words.len() {
                        let candidate = words[start..end].to_vec();
                        if self.statistics.title_phrases.contains(&candidate) {
                            matches.push(candidate);
                        }
                    }
                }
                if let Some(grounded) = matches.into_iter().min_by(|left, right| {
                    right.len().cmp(&left.len()).then_with(|| left.cmp(right))
                }) {
                    let grounded = grounded.join(" ");
                    if self.has_match(&quote_term(&grounded), excluded)? {
                        term.term = grounded;
                    }
                }
            }
            resolved.push(term);
        }

        let ranking = format!(
            "{} OR ({expression})",
            resolved
                .iter()
                .map(|term| quote_term(&term.term))
                .collect::<Vec<_>>()
                .join(" OR ")
        );
        // `sag2.py:699-710`: one list per term, in term order.
        let mut lists: Vec<Vec<crate::retrieve::Hit>> = Vec::with_capacity(resolved.len());
        for term in &resolved {
            lists.push(self.retrieve(&ranking, excluded, limit, Some(&term.term))?);
        }
        let mut queries: Vec<ExpansionQuery> = resolved
            .iter()
            .enumerate()
            .map(|(position, term)| ExpansionQuery {
                term: term.term.clone(),
                term_rank: position + 1,
                kind: term.kind,
                match_expression: quote_term(&term.term),
                ranking_expression: ranking.clone(),
                retrieved_count: lists[position].len(),
                selected: Vec::new(),
            })
            .collect();

        // `sag2.py:734-753`: round-robin by rank, then by term.
        let mut hits: Vec<ExpansionHit> = Vec::new();
        let mut seen: BTreeSet<i64> = BTreeSet::new();
        let mut done = false;
        for rank in 0..limit {
            for (position, rows) in lists.iter().enumerate() {
                if done {
                    break;
                }
                let Some(hit) = rows.get(rank) else {
                    continue;
                };
                if !seen.insert(hit.id) {
                    continue;
                }
                hits.push(ExpansionHit {
                    id: hit.id,
                    document_id: hit.document_id,
                    score: hit.score,
                    expansion_term: resolved[position].term.clone(),
                    expansion_term_rank: position + 1,
                    expansion_bm25_rank: rank + 1,
                    hit: hit.clone(),
                });
                queries[position].selected.push(Selected {
                    document_id: hit.document_id,
                    bm25_rank: rank + 1,
                    bm25_score: hit.score,
                });
                if hits.len() == limit {
                    done = true;
                }
            }
            if done {
                break;
            }
        }
        Ok((hits, queries))
    }
}
