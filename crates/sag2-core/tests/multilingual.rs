//! The multilingual recall fixture: word-rate and R@10 per language.
//!
//! `tests/data/multilingual.jsonl` holds twelve self-authored language packs
//! (ten documents, ten labelled queries each).  Nothing in it is copied from
//! UD, MIRACL or any other third-party corpus, so the numbers here are a
//! regression net rather than a benchmark of those corpora — see the plan's §7.
//!
//! For every pack this file checks:
//!
//! 1. **word rate** - the share of the pack's content words that tokenise to
//!    exactly one token.  The multilingual profile has to reach the pack's
//!    `target_word_rate`; `Profile::Sag2ZhV1` is pinned to its recorded
//!    `baseline_word_rate`, so the default path cannot move unnoticed.
//! 2. **R@10** - the pack's ten documents are indexed in a temporary
//!    directory and its ten queries are replayed in [`QueryMode::Lexical`] and
//!    [`QueryMode::Phrase3`].  Phrase3 must not trail Lexical, and the packs
//!    that ask for a gain (`target_query_gain`) must clear it.
//!
//! The packs are tiny, so the index is built with a sentence-sized
//! [`CHUNK_CHARS`] and a top-10 cut over the resulting chunks; with ten
//! one-chunk documents every R@10 would be 1.0 by construction.  The decoys
//! that make that cut discriminate are generated uniformly inside this file -
//! one rule for every pack, no per-language switch - and are not part of the
//! fixture.  The R@10 column is therefore a **regression signal** (it fails
//! when Phrase3 slips back to a plain character `OR`), not a benchmark
//! measurement: ten documents per pack, no third-party corpus, and no claim
//! about corpus-scale recall.
//!
//! Run with `--nocapture` to see the per-language table:
//!
//! ```sh
//! cargo test --offline -p sag2-core --test multilingual -- --nocapture
//! ```

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use sag2_core::{tokenize, tokenize_with, Profile};
use sag2_index::{CorpusIndex, Document, IndexOptions, QueryMode, RebuildPolicy};
use serde::Deserialize;

/// Chunk size the R@10 evaluation builds with: about one sentence per chunk,
/// so the ten-document packs produce roughly thirty chunks and a top-10 cut
/// actually discriminates.
const CHUNK_CHARS: usize = 40;

/// The top-10 cut the recall metric uses.
const RECALL_CUT: usize = 10;

/// How many word-salad decoys [`build_index`] adds per pack document.
///
/// Ten documents are too few for a top-10 cut to say anything, and a
/// character-level expression treats a decoy exactly like the sentence it came
/// from, which is the point: the decoy carries the same characters (so Lexical
/// retrieves it) and none of the original word order (so Phrase3 does not).
const DECOYS_PER_DOCUMENT: usize = 6;

/// The floating-point slack of the recall comparisons.
const EPSILON: f64 = 1e-9;

/// The word rates are pinned to three decimals, so they compare at that scale
/// (`11/12` is `0.916`, not `0.917`).
const WORD_RATE_EPSILON: f64 = 1e-3;

#[derive(Debug, Deserialize)]
struct Pack {
    lang: String,
    /// `"chars"` or `"words"`: which stream the decoys in [`decoys`] reorder.
    decoy: String,
    target_word_rate: f64,
    target_query_gain: f64,
    baseline_word_rate: f64,
    content_words: Vec<String>,
    docs: Vec<PackDocument>,
    queries: Vec<PackQuery>,
}

#[derive(Debug, Deserialize)]
struct PackDocument {
    id: i64,
    title: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct PackQuery {
    q: String,
    relevant: Vec<i64>,
}

/// Every pack, in fixture order.
fn packs() -> Vec<Pack> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("multilingual.jsonl");
    let text = fs::read_to_string(&path).expect("the fixture is committed");
    let mut packs = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        packs.push(
            serde_json::from_str(line)
                .unwrap_or_else(|error| panic!("{}:{}: {error}", path.display(), index + 1)),
        );
    }
    packs
}

/// The profile a pack's word rate is measured with.
///
/// Chinese and English are the reference pipeline's own languages; every other
/// pack names its language in the profile identifier, which also proves the
/// identifier parser agrees with the fixture.  Japanese and Korean segment with
/// a runtime dictionary that CI has no copy of, so their word rate here is the
/// documented dictionary-free fallback (`tokenize_with` without a segmenter).
fn profile(pack: &Pack) -> Profile {
    match pack.lang.as_str() {
        "zh" | "en" => Profile::Sag2ZhV1,
        other => format!("multilingual-v1-{other}")
            .parse()
            .unwrap_or_else(|error| panic!("{}: {error}", pack.lang)),
    }
}

/// The profile a pack's index is built with.
///
/// A `multilingual-v1-ja`/`-ko` index cannot be built without its dictionary,
/// and the fixture has to run offline, so those two packs are evaluated through
/// the reference pipeline (the character-level baseline for Korean and the
/// dictionary-free jieba stream for Japanese).  Their target only asks Phrase3
/// not to trail Lexical.
fn index_profile(pack: &Pack) -> Profile {
    match pack.lang.as_str() {
        "zh" | "en" | "ja" | "ko" => Profile::Sag2ZhV1,
        _ => profile(pack),
    }
}

/// `(hits, total)` for one profile: content words that come out as one token.
fn word_rate(profile: Profile, pack: &Pack) -> (usize, usize) {
    let hits = pack
        .content_words
        .iter()
        .filter(|word| tokenize_with(profile, word).len() == 1)
        .count();
    (hits, pack.content_words.len())
}

/// The decoy texts of one pack: the pack's own content, reordered.
///
/// Which stream is reordered is the `decoy` field of the pack, because the two
/// kinds of pipeline answer different questions:
///
/// * `"words"` (Thai, Vietnamese, Japanese, Chinese, German, French, English) -
///   the token stream is real words, so the decoy keeps every word and loses
///   the order.  A `Lexical` expression (the `OR` of the words) retrieves it as
///   readily as the document; a phrase expression does not.
/// * `"chars"` (Russian, Arabic, Hindi, Greek, Korean) - the token stream is
///   single characters, so the decoy is the character sequence rotated and
///   joined without spaces: every character is still there, no three-character
///   window of the original survives.
///
/// Decoy `k` of a document rotates its stream by `(k + 1) * 7` positions, which
/// is deterministic and keeps the same multiset of tokens.
fn decoys(pack: &Pack) -> Vec<(String, String)> {
    let profile = index_profile(pack);
    let mut decoys = Vec::new();
    for document in &pack.docs {
        let stream: Vec<String> = if pack.decoy == "chars" {
            document.text.chars().map(String::from).collect()
        } else {
            tokenize_with(profile, &document.text)
        };
        if stream.len() < 2 {
            continue;
        }
        for k in 0..DECOYS_PER_DOCUMENT {
            let shift = ((k + 1) * 7) % stream.len();
            let rotated: Vec<&str> = stream[shift..]
                .iter()
                .chain(stream[..shift].iter())
                .map(String::as_str)
                .collect();
            let separator = if pack.decoy == "chars" { "" } else { " " };
            decoys.push((
                format!("{}-decoy-{}", document.title, k + 1),
                rotated.join(separator),
            ));
        }
    }
    decoys
}

/// Build one pack's index (documents then decoys) and return it with its chunk
/// count.
fn build_index(directory: &Path, pack: &Pack) -> (CorpusIndex, usize) {
    let path: PathBuf = directory.join(format!("{}.sqlite", pack.lang));
    let decoys = decoys(pack);
    let mut documents: Vec<Document<'_>> = pack
        .docs
        .iter()
        .map(|document| Document {
            title: &document.title,
            text: &document.text,
            path: None,
            idx: None,
        })
        .collect();
    documents.extend(decoys.iter().map(|(title, text)| Document {
        title,
        text,
        path: None,
        idx: None,
    }));
    let (index, _) = CorpusIndex::open(
        &path,
        &documents,
        &format!("multilingual-fixture-{}", pack.lang),
        IndexOptions {
            chunk_chars: CHUNK_CHARS,
            profile: index_profile(pack),
            rebuild: RebuildPolicy::Always,
            ..IndexOptions::default()
        },
    )
    .expect("the fixture index builds");
    let chunks = index.statistics().total as usize;
    (index, chunks)
}

/// One query's outcome, for the failure report.
struct QueryOutcome {
    query: String,
    expression: String,
    hits: usize,
    recall: f64,
}

/// Mean R@10 over the pack's queries, plus every query's outcome.
fn recall_at_10(index: &CorpusIndex, pack: &Pack, mode: QueryMode) -> (f64, Vec<QueryOutcome>) {
    let mut outcomes = Vec::new();
    let mut total = 0.0;
    for query in &pack.queries {
        let expression = index
            .match_query_with(mode, &query.q)
            .expect("the query expression builds");
        let hits = index
            .retrieve(&expression, &BTreeSet::new(), RECALL_CUT, None)
            .expect("retrieval succeeds");
        let retrieved: BTreeSet<i64> = hits.iter().map(|hit| hit.document_id).collect();
        let found = query
            .relevant
            .iter()
            .filter(|id| retrieved.contains(id))
            .count();
        let recall = found as f64 / query.relevant.len() as f64;
        total += recall;
        outcomes.push(QueryOutcome {
            query: query.q.clone(),
            expression,
            hits: hits.len(),
            recall,
        });
    }
    (total / outcomes.len() as f64, outcomes)
}

#[test]
fn the_fixture_covers_twelve_languages() {
    let packs = packs();
    assert_eq!(packs.len(), 12, "one pack per language");
    let languages: BTreeSet<&str> = packs.iter().map(|pack| pack.lang.as_str()).collect();
    assert_eq!(
        languages,
        BTreeSet::from(["ar", "de", "el", "en", "fr", "hi", "ja", "ko", "ru", "th", "vi", "zh"])
    );
    for pack in &packs {
        assert_eq!(pack.docs.len(), 10, "{}: ten documents", pack.lang);
        assert_eq!(pack.queries.len(), 10, "{}: ten queries", pack.lang);
        assert!(
            pack.content_words.len() >= 10,
            "{}: at least ten content words",
            pack.lang
        );
        let ids: BTreeSet<i64> = pack.docs.iter().map(|document| document.id).collect();
        assert_eq!(ids.len(), pack.docs.len(), "{}: unique ids", pack.lang);
        for query in &pack.queries {
            assert!(!query.relevant.is_empty(), "{}: {query:?}", pack.lang);
            for id in &query.relevant {
                assert!(
                    ids.contains(id),
                    "{}: query {:?} names missing document {id}",
                    pack.lang,
                    query.q
                );
            }
        }
        assert!(
            !profile(pack).as_str().is_empty(),
            "{}: profile parses",
            pack.lang
        );
    }
}

#[test]
fn word_rates_meet_the_targets_and_the_baseline_is_pinned() {
    let mut table = String::from(
        "\nlanguage  profile                 words  hits  rate    target  baseline  profile-word-rate\n",
    );
    let mut failures = Vec::new();
    for pack in &packs() {
        let multilingual = profile(pack);
        let (hits, words) = word_rate(multilingual, pack);
        let rate = hits as f64 / words as f64;
        let (baseline_hits, baseline_words) = word_rate(Profile::Sag2ZhV1, pack);
        let baseline = baseline_hits as f64 / baseline_words as f64;
        table.push_str(&format!(
            "{:<9} {:<23} {words:>5}  {hits:>4}  {rate:<7.3} {:<7.3} {baseline:<9.3}\n",
            pack.lang,
            multilingual.as_str(),
            pack.target_word_rate,
        ));
        if rate + WORD_RATE_EPSILON < pack.target_word_rate {
            failures.push(format!(
                "{}: word rate {rate:.3} is below the target {:.3} (content words {:?})",
                pack.lang, pack.target_word_rate, pack.content_words
            ));
        }
        if (baseline - pack.baseline_word_rate).abs() > WORD_RATE_EPSILON {
            failures.push(format!(
                "{}: the Sag2ZhV1 baseline moved from {:.3} to {baseline:.3}",
                pack.lang, pack.baseline_word_rate
            ));
        }
    }
    println!("{table}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn phrase3_recall_clears_every_target() {
    let directory = tempfile::tempdir().expect("temp dir");
    let mut table = String::from(
        "\nlanguage  profile                 chunks  lexical  phrase3  gain    target\n",
    );
    let mut failures = Vec::new();
    for pack in &packs() {
        let evaluated = index_profile(pack);
        let (index, chunks) = build_index(directory.path(), pack);
        let (lexical, lexical_outcomes) = recall_at_10(&index, pack, QueryMode::Lexical);
        let (phrases, phrase_outcomes) = recall_at_10(&index, pack, QueryMode::Phrase3);
        let gain = phrases - lexical;
        table.push_str(&format!(
            "{:<9} {:<23} {chunks:>6}  {lexical:<8.3} {phrases:<8.3} {gain:<+7.3} {:.3}\n",
            pack.lang,
            evaluated.as_str(),
            pack.target_query_gain,
        ));
        if phrases + EPSILON < lexical {
            failures.push(format!(
                "{}: Phrase3 R@10 {phrases:.3} trails Lexical {lexical:.3}\nlexical:\n{}\nphrase3:\n{}",
                pack.lang,
                describe(&lexical_outcomes),
                describe(&phrase_outcomes)
            ));
        }
        if gain + EPSILON < pack.target_query_gain {
            failures.push(format!(
                "{}: Phrase3 gain {gain:+.3} is below the target {:.3}\nlexical:\n{}\nphrase3:\n{}",
                pack.lang,
                pack.target_query_gain,
                describe(&lexical_outcomes),
                describe(&phrase_outcomes)
            ));
        }
    }
    println!("{table}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn describe(outcomes: &[QueryOutcome]) -> String {
    let mut text = String::new();
    for outcome in outcomes {
        text.push_str(&format!(
            "  {:?} -> {:?} ({} hits, recall {:.2})\n",
            outcome.query, outcome.expression, outcome.hits, outcome.recall
        ));
    }
    text
}

#[test]
fn the_reference_profile_never_normalises_the_fixture() {
    // The default path is the certified one for every pack: `tokenize` is the
    // `Sag2ZhV1` entry point the goldens pin.
    for pack in &packs() {
        for word in &pack.content_words {
            assert_eq!(tokenize_with(Profile::Sag2ZhV1, word), tokenize(word));
        }
    }
}
