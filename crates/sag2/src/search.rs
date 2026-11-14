//! `search_sag2` (`sag2.py:930-1043`).
//!
//! The loop is ported verbatim; every ordering decision is load-bearing:
//!
//! * `ranked()` sorts by `(-jev_score, -fusion_score, id)`; `fusion_score` is
//!   `1 / (61 + i)` over the retrieval order.
//! * `protected` (hop 0's first five) and `carried` (the previous frontier)
//!   keep Python's dict-update order, which decides the pool order and thus
//!   the `candidate_{id}` insertion order inside a payload.
//! * The hop-2 pool splits into surviving history and at most 20 incoming
//!   candidates, and its evidence is the previous top ten with
//!   `previous_jev_score` attached - the one branch that reaches
//!   `evidence_score_note` and the `Multiple`/`Bridge` oracle scores.
//! * The final head is `rank_dpp(final[:10])` followed by `final[10:]`.
//!
//! `max_hops = 0` still scores once (`sag2.py:1007`); see design §8.4.

use std::collections::BTreeSet;
use std::time::Instant;

use sag2_index::{CorpusIndex, ExpansionQuery, ExpansionTerm, QueryError, Seed};

use crate::judge::{BatchDetail, Judge, JudgeError, RoundUsage};
use crate::{rank_dpp, DppError, Usage, UsageBatch};

/// One candidate: the retrieved chunk plus what the search loop adds.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub hit: sag2_index::Hit,
    pub hop: u32,
    pub fusion_score: f64,
    pub jev_score: f64,
    pub jev_model: Option<String>,
    /// Set only on hop 2's evidence (`sag2.py:1011`).
    pub previous_jev_score: Option<f64>,
}

/// `sag2.py:935-940`.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchOptions {
    pub candidate_k: usize,
    pub seed_k: usize,
    pub max_hops: u8,
    pub expansion_term_limit: usize,
    /// `context` (`sag2.py:939`): `Some` turns on the history branches of the
    /// judge state.
    pub context: Option<SearchContext>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            candidate_k: crate::DEFAULT_CANDIDATE_K,
            seed_k: crate::DEFAULT_SEED_K,
            max_hops: crate::DEFAULT_MAX_HOPS,
            expansion_term_limit: crate::DEFAULT_EXPANSION_TERM_LIMIT,
            context: None,
        }
    }
}

/// The multi-turn search context (`sag2.py:831-846`).
#[derive(Debug, Clone, PartialEq)]
pub struct SearchContext {
    pub original_question: String,
    /// `context["evidence"]`, merged with the current round's evidence.
    pub evidence: Vec<Candidate>,
    /// `context["search_history"]`, carried verbatim.
    pub search_history: Vec<sag2_json::Value>,
    /// `context["previous_decisions"]`, carried verbatim.
    pub previous_decisions: sag2_json::Value,
}

/// One `bm25`/`expansion` trace record (`sag2.py:969-980`).
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalTrace {
    /// `"bm25"` for hop 0, `"expansion"` afterwards.
    pub stage: &'static str,
    pub hop: u32,
    pub count: usize,
    pub candidate_document_ids: Vec<i64>,
    pub candidate_ids: Vec<i64>,
    pub terms: Vec<String>,
    pub term_scores: Vec<ExpansionTerm>,
    pub expansion_queries: Vec<ExpansionQuery>,
}

/// One `jev` trace record (`sag2.py:1014-1031`).
#[derive(Debug, Clone, PartialEq)]
pub struct JudgeTrace {
    pub hop: u32,
    pub count: usize,
    pub carried_seed_count: usize,
    pub protected_bm25_count: usize,
    pub evidence_ids: Vec<i64>,
    /// `ranked()` at the moment the round finished.
    pub scores: Vec<TraceScore>,
    /// The round's batches, without the hop tag.
    pub batches: Vec<BatchDetail>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceScore {
    pub id: i64,
    pub score: f64,
}

/// A trace record, in emission order.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceEntry {
    Retrieval(RetrievalTrace),
    Judge(JudgeTrace),
}

/// `search_sag2`'s return value (`sag2.py:1036-1043`).
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Already DPP-reordered for the first ten.
    pub candidates: Vec<Candidate>,
    pub baseline: Vec<Candidate>,
    pub trace: Vec<TraceEntry>,
    pub usage: Usage,
    pub elapsed_ms: u64,
    pub algorithm: &'static str,
}

/// A failure of the search loop.
#[derive(Debug)]
pub enum SearchError {
    /// `sag2.py:941-942`.
    Hops(u8),
    Query(QueryError),
    Expansion(sag2_index::ExpansionError),
    Judge(JudgeError),
    Dpp(DppError),
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::Hops(hops) => {
                write!(formatter, "SAG2 supports at most two expansion rounds (got {hops})")
            }
            SearchError::Query(error) => write!(formatter, "{error}"),
            SearchError::Expansion(error) => write!(formatter, "{error}"),
            SearchError::Judge(error) => write!(formatter, "{error}"),
            SearchError::Dpp(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for SearchError {}

impl From<QueryError> for SearchError {
    fn from(error: QueryError) -> Self {
        SearchError::Query(error)
    }
}

impl From<sag2_index::ExpansionError> for SearchError {
    fn from(error: sag2_index::ExpansionError) -> Self {
        SearchError::Expansion(error)
    }
}

impl From<JudgeError> for SearchError {
    fn from(error: JudgeError) -> Self {
        SearchError::Judge(error)
    }
}

impl From<DppError> for SearchError {
    fn from(error: DppError) -> Self {
        SearchError::Dpp(error)
    }
}

/// A Python dict keyed by chunk id: insertion order plus replace-in-place
/// (a key keeps its first position when its value is replaced).
struct IdMap {
    entries: Vec<(i64, Candidate)>,
}

impl IdMap {
    fn new() -> Self {
        IdMap {
            entries: Vec::new(),
        }
    }

    fn from_candidates(candidates: impl IntoIterator<Item = Candidate>) -> Self {
        let mut map = IdMap::new();
        map.extend(candidates);
        map
    }

    fn insert(&mut self, candidate: Candidate) {
        match self
            .entries
            .iter_mut()
            .find(|(id, _)| *id == candidate.hit.id)
        {
            Some((_, existing)) => *existing = candidate,
            None => self.entries.push((candidate.hit.id, candidate)),
        }
    }

    fn extend(&mut self, candidates: impl IntoIterator<Item = Candidate>) {
        for candidate in candidates {
            self.insert(candidate);
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn contains(&self, id: i64) -> bool {
        self.entries.iter().any(|(existing, _)| *existing == id)
    }

    fn values(&self) -> impl Iterator<Item = &Candidate> {
        self.entries.iter().map(|(_, candidate)| candidate)
    }
}

/// `sorted(retained.values(), key=lambda c: (-c["jev_score"],
/// -c["fusion_score"], c["id"]))`.  `slice::sort_by` is stable, like
/// Python's `list.sort`.
fn ranked(retained: &IdMap) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = retained.values().cloned().collect();
    candidates.sort_by(|left, right| {
        right
            .jev_score
            .partial_cmp(&left.jev_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .fusion_score
                    .partial_cmp(&left.fusion_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.hit.id.cmp(&right.hit.id))
    });
    candidates
}

/// `{c["id"]: c for c in [*a, *b]}.values()`: `a`'s order, then `b`'s new ids.
fn merged(first: &[Candidate], second: &[Candidate]) -> Vec<Candidate> {
    let mut map = IdMap::from_candidates(first.iter().cloned());
    map.extend(second.iter().cloned());
    map.entries.into_iter().map(|(_, c)| c).collect()
}

/// The pool of a non-zero hop.
fn merged_pool(
    protected: &IdMap,
    carried: &IdMap,
    candidates: &[Candidate],
) -> Vec<Candidate> {
    let mut map = IdMap::from_candidates(protected.values().cloned());
    map.extend(carried.values().cloned());
    map.extend(candidates.iter().cloned());
    map.entries.into_iter().map(|(_, c)| c).collect()
}

fn seeds(frontier: &[Candidate]) -> Vec<Seed> {
    frontier
        .iter()
        .map(|candidate| Seed {
            document_id: candidate.hit.document_id,
            jev_score: candidate.jev_score,
            content: candidate.hit.content.clone(),
        })
        .collect()
}

fn retrieval_trace(
    stage: &'static str,
    hop: u32,
    candidates: &[Candidate],
    terms: &[ExpansionTerm],
    queries: Vec<ExpansionQuery>,
) -> TraceEntry {
    TraceEntry::Retrieval(RetrievalTrace {
        stage,
        hop,
        count: candidates.len(),
        candidate_document_ids: candidates
            .iter()
            .map(|candidate| candidate.hit.document_id)
            .collect(),
        candidate_ids: candidates.iter().map(|candidate| candidate.hit.id).collect(),
        terms: terms.iter().map(|term| term.term.clone()).collect(),
        term_scores: terms.to_vec(),
        expansion_queries: queries,
    })
}

/// `sag2.py:930-1043` `search_sag2`.
pub fn search(
    index: &CorpusIndex,
    judge: &dyn Judge,
    query: &str,
    options: &SearchOptions,
) -> Result<SearchResult, SearchError> {
    if options.max_hops > 2 {
        return Err(SearchError::Hops(options.max_hops));
    }
    let started = Instant::now();
    let mut retained = IdMap::new();
    let mut carried = IdMap::new();
    let mut protected = IdMap::new();
    let mut latest = IdMap::new();
    let mut expanded: BTreeSet<i64> = BTreeSet::new();
    let mut seen: BTreeSet<i64> = BTreeSet::new();
    let mut frontier: Vec<Candidate> = Vec::new();
    let mut trace: Vec<TraceEntry> = Vec::new();
    let mut baseline: Vec<Candidate> = Vec::new();
    let mut usage = Usage::default();

    let expression = index.match_query(query)?;
    if expression.is_empty() {
        return finish(index, &retained, baseline, trace, usage, started, options);
    }
    for hop in 0..=options.max_hops as u32 {
        if hop > 0 && frontier.is_empty() {
            break;
        }
        let terms = if hop > 0 {
            index.expansion_terms(query, &seeds(&frontier), options.expansion_term_limit)?
        } else {
            Vec::new()
        };
        if hop > 0 && terms.is_empty() {
            break;
        }
        let (hits, queries) = if hop > 0 {
            let (hits, queries) = index.retrieve_expansion(
                &expression,
                &terms,
                &seen,
                options.candidate_k,
            )?;
            (
                hits.into_iter().map(|hit| hit.hit).collect::<Vec<_>>(),
                queries,
            )
        } else {
            (
                index.retrieve(&expression, &seen, options.candidate_k, None)?,
                Vec::new(),
            )
        };
        let candidates: Vec<Candidate> = hits
            .into_iter()
            .enumerate()
            .map(|(position, hit)| Candidate {
                hit,
                hop,
                fusion_score: 1.0 / (61.0 + position as f64),
                jev_score: 0.0,
                jev_model: None,
                previous_jev_score: None,
            })
            .collect();
        trace.push(retrieval_trace(
            if hop > 0 { "expansion" } else { "bm25" },
            hop,
            &candidates,
            &terms,
            queries,
        ));
        if candidates.is_empty() {
            break;
        }
        if hop == 0 {
            baseline = candidates.clone();
            protected = IdMap::from_candidates(candidates.iter().take(5).cloned());
        }
        carried.extend(frontier.iter().cloned());
        let mut pool = if hop > 0 {
            merged_pool(&protected, &carried, &candidates)
        } else {
            candidates.clone()
        };
        let mut evidence = if hop > 0 {
            merged(&carried.values().cloned().collect::<Vec<_>>(), &candidates[..options.seed_k.min(candidates.len())])
        } else {
            Vec::new()
        };
        if hop == 2 {
            let carried_candidates: Vec<Candidate> = pool
                .iter()
                .filter(|candidate| latest.contains(candidate.hit.id))
                .cloned()
                .collect();
            let incoming: Vec<Candidate> = pool
                .iter()
                .filter(|candidate| !latest.contains(candidate.hit.id))
                .take(20)
                .cloned()
                .collect();
            let mut previous: Vec<Candidate> = latest.values().cloned().collect();
            previous.sort_by(|left, right| {
                right
                    .jev_score
                    .partial_cmp(&left.jev_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.hit.id.cmp(&right.hit.id))
            });
            let history = merged(&carried_candidates, &previous);
            let keep = pool
                .len()
                .saturating_sub(incoming.len())
                .min(history.len());
            let mut next = history[..keep].to_vec();
            next.extend(incoming);
            pool = next;
            evidence = ranked(&retained)
                .into_iter()
                .take(10)
                .map(|mut candidate| {
                    candidate.previous_jev_score = Some(candidate.jev_score);
                    candidate
                })
                .collect();
        }
        let (scored, round_usage) =
            judge.score(query, &pool, &evidence, options.context.as_ref())?;
        latest.extend(scored.iter().cloned());
        accumulate(&mut usage, hop, &round_usage);
        retained = IdMap::from_candidates(scored.iter().cloned());
        for candidate in &scored {
            seen.insert(candidate.hit.id);
        }
        trace.push(TraceEntry::Judge(JudgeTrace {
            hop,
            count: scored.len(),
            carried_seed_count: carried.len(),
            protected_bm25_count: protected.len(),
            evidence_ids: evidence
                .iter()
                .map(|candidate| candidate.hit.id)
                .collect(),
            scores: ranked(&retained)
                .iter()
                .map(|candidate| TraceScore {
                    id: candidate.hit.id,
                    score: candidate.jev_score,
                })
                .collect(),
            batches: round_usage.batches.clone(),
        }));
        frontier = ranked(&retained)
            .into_iter()
            .filter(|candidate| !expanded.contains(&candidate.hit.id))
            .take(options.seed_k)
            .collect();
        expanded.extend(frontier.iter().map(|candidate| candidate.hit.id));
    }
    finish(index, &retained, baseline, trace, usage, started, options)
}

/// `sag2.py:915-921` + `:946`: sum the counters and tag each batch with its hop.
fn accumulate(usage: &mut Usage, hop: u32, round: &RoundUsage) {
    usage.requests += round.requests;
    usage.judgments += round.judgments;
    usage.input_tokens += round.input_tokens;
    usage.output_tokens += round.output_tokens;
    usage.batches.extend(
        round
            .batches
            .iter()
            .cloned()
            .map(|detail| UsageBatch { hop, detail }),
    );
}

/// `sag2.py:1034-1043`.
fn finish(
    index: &CorpusIndex,
    retained: &IdMap,
    baseline: Vec<Candidate>,
    trace: Vec<TraceEntry>,
    usage: Usage,
    started: Instant,
    options: &SearchOptions,
) -> Result<SearchResult, SearchError> {
    let final_candidates: Vec<Candidate> = ranked(retained)
        .into_iter()
        .take(options.candidate_k)
        .collect();
    let head = final_candidates.len().min(10);
    let mut reordered = final_candidates[..head].to_vec();
    rank_dpp(&mut reordered, &index.statistics().diversity_idf)?;
    reordered.extend(final_candidates[head..].iter().cloned());
    Ok(SearchResult {
        candidates: reordered,
        baseline,
        trace,
        usage,
        elapsed_ms: started.elapsed().as_millis() as u64,
        algorithm: "SAG2",
    })
}
