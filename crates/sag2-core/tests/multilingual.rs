//! The multilingual recall fixture: word-rate and R@10 per language.
//!
//! `tests/data/multilingual.jsonl` holds thirty self-authored language
//! packs (ten documents and at least ten labelled queries each).  Nothing in
//! it is copied from UD, MIRACL or any other third-party corpus, so the
//! numbers here are a regression net rather than a benchmark of those
//! corpora — see the plan's §7.
//!
//! Every pack is in exactly one state, and the fixture says which:
//!
//! * **measured** (`hi`, `th`, `de`, `fr`, `ko`, `pl`, `zh`, `en`, `it`, `es`,
//!   `ro`, `nl`, `sv`, `da`, `nb`, `pt`, `cs`, `sr`, `et`, `hu`, `fi`, `tr`,
//!   `mt`, `bg`, `uk`) - the profile has a word-level pipeline, so the word
//!   rate has a pinned target and the reference `Sag2ZhV1` rate is pinned as
//!   the baseline.
//! * **deferred** (`ru`, `ar`, `el`, `vi`) - the profile still needs the
//!   stemmer plan §5 batch 2 owns (Snowball for ru/el, an Arabic stemmer,
//!   Vietnamese word boundaries), so their word rate is printed but not
//!   asserted: a target would restate the gap, not measure it.  A whole word is
//!   one token for them since L1, but an inflected one is still a different
//!   term, which is what the stemmer has to fix.
//! * **dictionary required** (`ja`) - Japanese cannot tokenise without a
//!   runtime dictionary, and CI has none.  It must return
//!   [`AnalysisError::DictionaryMissing`](sag2_core::AnalysisError) instead of
//!   a character stream, so it contributes no word-rate or R@10 number here.
//!   Korean is *not* in this state: its eojeol are space-delimited, so the
//!   word-run pipeline reaches word-level tokens without a dictionary.
//!
//! For every measured pack this file checks:
//!
//! 1. **word rate** - the share of the pack's content words that survive as
//!    one whole token (see [`word_level`]).  The multilingual profile has to
//!    reach the pack's `target_word_rate`; `Profile::Sag2ZhV1` is pinned to its
//!    recorded `baseline_word_rate`, so the default path cannot move unnoticed.
//! 2. **R@10** - the pack's ten documents are indexed in a temporary
//!    directory and its queries are replayed in [`QueryMode::Lexical`] and
//!    [`QueryMode::Phrase3`].  Every labelled query must retrieve its document
//!    in Lexical mode (R@10 = 1.000); that pin is what makes a language rule
//!    observable, because [`word_level`] compares a word against its own
//!    normalisation and a letter substitution cancels out there, while the
//!    query spelling has to reach the stored one.  Phrase3 must not trail
//!    Lexical, and the packs that ask for a gain (`target_query_gain`) must
//!    clear it.
//!
//! 3. **fold pairs** - the spellings a language rule must collapse (`ѝ`/`и`,
//!    `Ħal`/`hal`, `İstanbul`/`istanbul`, the three Ukrainian apostrophes) are
//!    declared by the pack and must normalise and tokenise alike.  The rates
//!    cannot see an in-word substitution, so this is the check that fails when
//!    a rule is deleted.
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

use sag2_core::analysis::normalize::normalize;
use sag2_core::{tokenize, tokenize_with, AnalysisError, Profile};
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
    /// The profile still tokenises per character; plan §5 batch 2 owns the
    /// segmenter or stemmer, so the word rate is reported but not asserted.
    #[serde(default)]
    deferred: bool,
    /// The profile cannot tokenise without a runtime dictionary (Japanese
    /// only), so there is nothing to measure offline.
    #[serde(default)]
    dictionary_required: bool,
    /// The word rate the profile must reach - `Some` for measured packs only.
    #[serde(default)]
    target_word_rate: Option<f64>,
    /// The `Sag2ZhV1` word rate recorded for this pack - `Some` for measured
    /// packs only.
    #[serde(default)]
    baseline_word_rate: Option<f64>,
    /// The Phrase3 gain over Lexical the pack asks for - `Some` for every pack
    /// whose queries are evaluated, `None` for the dictionary packs.
    #[serde(default)]
    target_query_gain: Option<f64>,
    /// Spellings the profile's language rule must fold onto one token stream.
    ///
    /// The rates cannot see a pure in-word letter substitution: `word_level`
    /// normalises the word it compares (`ѝ` and `и` are each whole runs), and
    /// the corpus's code-point tokens make a cross-spelling query match anyway.
    /// The pack therefore states the pairs its rule owns and
    /// `the_declared_fold_pairs_reach_one_token_stream` asserts them directly.
    #[serde(default)]
    fold_pairs: Vec<FoldPair>,
    content_words: Vec<String>,
    docs: Vec<PackDocument>,
    queries: Vec<PackQuery>,
}

/// One `(from, to)` spelling pair a language rule must collapse.
#[derive(Debug, Deserialize)]
struct FoldPair {
    from: String,
    to: String,
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

/// The profile a pack is measured with.
///
/// Chinese and English are the reference pipeline's own languages; every other
/// pack names its language in the profile identifier, which also proves the
/// identifier parser agrees with the fixture.
fn profile(pack: &Pack) -> Profile {
    match pack.lang.as_str() {
        "zh" | "en" => Profile::Sag2ZhV1,
        other => format!("multilingual-v1-{other}")
            .parse()
            .unwrap_or_else(|error| panic!("{}: {error}", pack.lang)),
    }
}

/// Whether `word` survives as one whole token under `profile`.
///
/// The reference pipeline emits one token per word slice, so "one token" is
/// exactly "the word was not shredded".  A multilingual profile also emits the
/// code points of a run ([`sag2_core::analysis::Profile::Multilingual`]), so the
/// whole word is one token *in a longer stream*: `len() == 1` would score a
/// correctly segmented `bibliothèque` as a miss.  The token a multilingual
/// profile emits is the profile's own spelling of the word, so the comparison
/// is against [`normalize`], not against the raw fixture text (`szkoła` reaches
/// the stored token `szkola`).
fn word_level(profile: Profile, word: &str) -> bool {
    let Ok(tokens) = tokenize_with(profile, word) else {
        return false;
    };
    match profile {
        Profile::Sag2ZhV1 => tokens.len() == 1,
        Profile::Multilingual { language } => {
            let whole = normalize(language, word);
            tokens.contains(&whole)
        }
    }
}

/// `(hits, total)` for one profile: content words that survive as one token.
fn word_rate(profile: Profile, pack: &Pack) -> (usize, usize) {
    let hits = pack
        .content_words
        .iter()
        .filter(|word| word_level(profile, word))
        .count();
    (hits, pack.content_words.len())
}

/// The decoy texts of one pack: the pack's own content, reordered.
///
/// Which stream is reordered is the `decoy` field of the pack, because the two
/// kinds of pipeline answer different questions:
///
/// * `"words"` (Thai, Vietnamese, Japanese, Chinese, German, French, English,
///   Italian, Spanish, Romanian, Dutch, Swedish, Danish, Norwegian, Portuguese,
///   Czech, Estonian, Hungarian, Finnish, Turkish, Maltese) - the token stream
///   is real words, so the decoy keeps every word and loses the order.  A
///   `Lexical` expression (the `OR` of the words) retrieves it as readily as
///   the document; a phrase expression does not.
/// * `"chars"` (Russian, Arabic, Hindi, Greek, Polish, Korean, Serbian,
///   Ukrainian, Bulgarian) - the decoy is the character sequence rotated and
///   joined without spaces: every character is still there, no three-character
///   window of the original survives.  These packs carry code-point tokens
///   alongside their words, which is exactly the recall path the decoy has to
///   stress.
///
/// Decoy `k` of a document rotates its stream by `(k + 1) * 7` positions, which
/// is deterministic and keeps the same multiset of tokens.
fn decoys(pack: &Pack) -> Vec<(String, String)> {
    let profile = profile(pack);
    let mut decoys = Vec::new();
    for document in &pack.docs {
        let stream: Vec<String> = if pack.decoy == "chars" {
            document.text.chars().map(String::from).collect()
        } else {
            tokenize_with(profile, &document.text)
                .expect("a dictionary-free profile always tokenises")
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
            profile: profile(pack),
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
fn the_fixture_covers_thirty_languages_in_three_states() {
    let packs = packs();
    assert_eq!(packs.len(), 30, "one pack per language");
    let languages: BTreeSet<&str> = packs.iter().map(|pack| pack.lang.as_str()).collect();
    assert_eq!(
        languages,
        BTreeSet::from([
            "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hi", "hu", "it",
            "ja", "ko", "mt", "nb", "nl", "pl", "pt", "ro", "ru", "sr", "sv", "th", "tr", "uk",
            "vi", "zh"
        ])
    );
    for pack in &packs {
        assert_eq!(pack.docs.len(), 10, "{}: ten documents", pack.lang);
        assert!(
            pack.queries.len() >= 10,
            "{}: at least ten labelled queries",
            pack.lang
        );
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
        // Chinese has no multilingual variant: L1's word-character policy is a
        // `Profile::Multilingual` policy, and a Chinese pack must keep the
        // reference classifier rather than a `multilingual-v1-zh` id.
        if matches!(pack.lang.as_str(), "zh" | "en") {
            assert_eq!(
                profile(pack),
                Profile::Sag2ZhV1,
                "{}: the reference classifier",
                pack.lang
            );
        }
        // A pack is measured (both word-rate fields), deferred (neither, and
        // no gain target), or dictionary-required (no word rate at all).
        let measured = pack.target_word_rate.is_some() && pack.baseline_word_rate.is_some();
        if pack.dictionary_required {
            assert!(
                !measured,
                "{}: a dictionary pack measures nothing",
                pack.lang
            );
            assert!(!pack.deferred, "{}: dictionary is not deferred", pack.lang);
            assert_eq!(
                pack.target_query_gain, None,
                "{}: no R@10 target",
                pack.lang
            );
            assert!(matches!(pack.lang.as_str(), "ja"));
        } else if pack.deferred {
            assert!(
                !measured,
                "{}: a deferred pack does not pin a tautological target",
                pack.lang
            );
            assert_eq!(
                pack.target_query_gain,
                Some(0.0),
                "{}: a deferred pack asks only that Phrase3 not trail Lexical",
                pack.lang
            );
        } else {
            assert!(measured, "{}: a measured pack pins both rates", pack.lang);
            assert!(
                pack.target_query_gain.is_some(),
                "{}: a measured pack pins its gain",
                pack.lang
            );
        }
    }
}

#[test]
fn a_dictionary_profile_refuses_the_character_fallback() {
    // C2: a dictionary profile without a dictionary is a typed error, never the
    // character tokens the old wrapper handed back.  Japanese is the only such
    // profile left: Korean reaches word-level tokens through the word-run
    // pipeline, so it no longer needs a dictionary at all.
    let all = packs();
    let dictionary: Vec<&Pack> = all.iter().filter(|pack| pack.dictionary_required).collect();
    assert_eq!(dictionary.len(), 1, "Japanese only");
    for pack in dictionary {
        let profile = profile(pack);
        for word in &pack.content_words {
            match tokenize_with(profile, word) {
                Err(AnalysisError::DictionaryMissing { language }) => {
                    assert_eq!(
                        Profile::Multilingual { language },
                        profile,
                        "{}: the error names the profile's language",
                        pack.lang
                    );
                }
                other => panic!(
                    "{}: {word:?} must be DictionaryMissing, got {other:?}",
                    pack.lang
                ),
            }
        }
    }
}

#[test]
fn the_declared_fold_pairs_reach_one_token_stream() {
    // A letter substitution inside a word is invisible to `word_level`, and
    // the code-point tokens every non-ASCII run emits make a cross-spelling
    // query match either way, so the rule's own contract is asserted here: the
    // two spellings the pack declares must normalise alike and tokenise alike.
    // Deleting a language rule fails this test.
    let mut checked = 0;
    for pack in &packs() {
        let Profile::Multilingual { language } = profile(pack) else {
            continue;
        };
        let profile = Profile::Multilingual { language };
        for pair in &pack.fold_pairs {
            checked += 1;
            assert_eq!(
                normalize(language, &pair.from),
                normalize(language, &pair.to),
                "{}: {:?} and {:?} must normalise alike",
                pack.lang,
                pair.from,
                pair.to
            );
            assert_eq!(
                tokenize_with(profile, &pair.from).expect("a dictionary-free profile tokenises"),
                tokenize_with(profile, &pair.to).expect("a dictionary-free profile tokenises"),
                "{}: {:?} and {:?} must reach one token stream",
                pack.lang,
                pair.from,
                pair.to
            );
        }
    }
    assert_eq!(checked, 8, "every rule-bearing pack declares its pairs");
}

#[test]
fn word_rates_meet_the_targets_and_the_baseline_is_pinned() {
    let mut table = String::from(
        "\nlanguage  profile                 words  hits  rate    target  baseline  state\n",
    );
    let mut failures = Vec::new();
    for pack in &packs() {
        let multilingual = profile(pack);
        if pack.dictionary_required {
            table.push_str(&format!(
                "{:<9} {:<23} {:>5}  {:>4}  {:<7} {:<7} {:<9} dictionary required\n",
                pack.lang,
                multilingual.as_str(),
                pack.content_words.len(),
                "-",
                "-",
                "-",
                "-"
            ));
            continue;
        }
        let (hits, words) = word_rate(multilingual, pack);
        let rate = hits as f64 / words as f64;
        let (baseline_hits, baseline_words) = word_rate(Profile::Sag2ZhV1, pack);
        let baseline = baseline_hits as f64 / baseline_words as f64;
        let state = if pack.deferred {
            "deferred (§5 batch 2)"
        } else {
            "measured"
        };
        table.push_str(&format!(
            "{:<9} {:<23} {words:>5}  {hits:>4}  {rate:<7.3} {:<7} {baseline:<9.3} {state}\n",
            pack.lang,
            multilingual.as_str(),
            pack.target_word_rate
                .map(|target| format!("{target:.3}"))
                .unwrap_or_else(|| "-".to_owned()),
        ));
        let Some(target) = pack.target_word_rate else {
            continue;
        };
        if rate + WORD_RATE_EPSILON < target {
            failures.push(format!(
                "{}: word rate {rate:.3} is below the target {target:.3} (content words {:?})",
                pack.lang, pack.content_words
            ));
        }
        let Some(pinned) = pack.baseline_word_rate else {
            continue;
        };
        if (baseline - pinned).abs() > WORD_RATE_EPSILON {
            failures.push(format!(
                "{}: the Sag2ZhV1 baseline moved from {pinned:.3} to {baseline:.3}",
                pack.lang
            ));
        }
    }
    println!("{table}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn recall_meets_every_target() {
    let directory = tempfile::tempdir().expect("temp dir");
    let mut table = String::from(
        "\nlanguage  profile                 chunks  lexical  phrase3  gain    target\n",
    );
    let mut failures = Vec::new();
    for pack in &packs() {
        let multilingual = profile(pack);
        if pack.dictionary_required {
            table.push_str(&format!(
                "{:<9} {:<23} {:>6}  {:<8} {:<8} {:<7} dictionary required\n",
                pack.lang,
                multilingual.as_str(),
                "-",
                "-",
                "-",
                "-"
            ));
            continue;
        }
        let (index, chunks) = build_index(directory.path(), pack);
        let (lexical, lexical_outcomes) = recall_at_10(&index, pack, QueryMode::Lexical);
        let (phrases, phrase_outcomes) = recall_at_10(&index, pack, QueryMode::Phrase3);
        let gain = phrases - lexical;
        let target = pack.target_query_gain.unwrap_or(0.0);
        table.push_str(&format!(
            "{:<9} {:<23} {chunks:>6}  {lexical:<8.3} {phrases:<8.3} {gain:<+7.3} {target:.3}\n",
            pack.lang,
            multilingual.as_str(),
        ));
        if phrases + EPSILON < lexical {
            failures.push(format!(
                "{}: Phrase3 R@10 {phrases:.3} trails Lexical {lexical:.3}\nlexical:\n{}\nphrase3:\n{}",
                pack.lang,
                describe(&lexical_outcomes),
                describe(&phrase_outcomes)
            ));
        }
        // A labelled query has to retrieve its document.  This is the pin that
        // makes a normaliser rule observable: `word_level` normalises the word
        // it compares, so an in-word substitution (`ħ→h`, `ѝ→и`) cancels out
        // on both sides, while a query is the other spelling of the corpus
        // word and only reaches it when the rule ran at build time.
        if lexical + EPSILON < 1.0 {
            failures.push(format!(
                "{}: a labelled query misses its document (Lexical R@10 {lexical:.3})\n{}",
                pack.lang,
                describe(&lexical_outcomes)
            ));
        }
        if gain + EPSILON < target {
            failures.push(format!(
                "{}: Phrase3 gain {gain:+.3} is below the target {target:.3}\nlexical:\n{}\nphrase3:\n{}",
                pack.lang,
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
            assert_eq!(
                tokenize_with(Profile::Sag2ZhV1, word).expect("the reference profile tokenises"),
                tokenize(word)
            );
        }
    }
}
