//! Search-level behaviour the goldens cannot reach: `max_hops = 0` still calls
//! the judge (design §8.4), and the multi-turn context branches of the payload
//! builder (`sag2.py:831-846`), which no fixture exercises.

use std::path::Path;
use std::sync::Mutex;

use sag2::{
    search, Answer, BatchedJudge, Candidate, CorpusIndex, Hit, Judge, JudgeError, OracleJudge,
    Payload, Response, SearchContext, SearchOptions, Transport,
};
use sag2_index::{Document, IndexOptions, RebuildPolicy};
use sag2_json::Value;

fn corpus() -> [Document<'static>; 5] {
    [
        Document {
            title: "Anchor",
            text: "Anchor birthplace depicts the town.",
            path: None,
            idx: None,
        },
        Document {
            title: "Multiple",
            text: "Multiple relates to Anchor directly.",
            path: None,
            idx: None,
        },
        Document {
            title: "Bridge",
            text: "Bridge connects Multiple to the town.",
            path: None,
            idx: None,
        },
        Document {
            title: "High",
            text: "High value passage about the town.",
            path: None,
            idx: None,
        },
        Document {
            title: "Series",
            text: "Series orders the remaining passages.",
            path: None,
            idx: None,
        },
    ]
}

fn open_index(directory: &Path) -> CorpusIndex {
    let documents = corpus();
    let (index, _) = CorpusIndex::open(
        &directory.join("search.sqlite"),
        &documents,
        "fixture",
        IndexOptions {
            chunk_chars: 2200,
            index_workers: 1,
            source_dir: None,
            rebuild: RebuildPolicy::Always,
        },
    )
    .expect("build the fixture index");
    index
}

fn candidate(id: i64, title: &str, content: &str) -> Candidate {
    Candidate {
        hit: Hit {
            id,
            document_id: id,
            title: title.to_owned(),
            heading: String::new(),
            content: content.to_owned(),
            full_doc: None,
            spans: Vec::new(),
            characters: 0,
            source_id: String::new(),
            score: 0.0,
        },
        hop: 0,
        fusion_score: 0.0,
        jev_score: 0.0,
        jev_model: None,
        previous_jev_score: None,
    }
}

#[test]
fn max_hops_zero_still_scores_one_round() {
    let directory = tempfile::tempdir().expect("temp dir");
    let index = open_index(directory.path());
    let judge = BatchedJudge::new(OracleJudge::new());
    let result = search(
        &index,
        &judge,
        "Anchor",
        &SearchOptions {
            candidate_k: 3,
            seed_k: 1,
            max_hops: 0,
            expansion_term_limit: 2,
            context: None,
        },
    )
    .expect("search");
    assert_eq!(result.algorithm, "SAG2");
    let stages: Vec<&str> = result
        .trace
        .iter()
        .map(|entry| match entry {
            sag2::TraceEntry::Retrieval(trace) => trace.stage,
            sag2::TraceEntry::Judge(_) => "jev",
        })
        .collect();
    assert_eq!(stages, ["bm25", "jev"]);
    assert!(!result.candidates.is_empty());
    assert!(result
        .candidates
        .iter()
        .all(|candidate| candidate.jev_model.as_deref() == Some("oracle-jev")));
    assert_eq!(result.usage.requests, 1);
    assert_eq!(result.usage.judgments, result.usage.batches[0].detail.candidate_count);
}

#[test]
fn more_than_two_hops_is_refused() {
    let directory = tempfile::tempdir().expect("temp dir");
    let index = open_index(directory.path());
    let judge = BatchedJudge::new(OracleJudge::new());
    let error = search(
        &index,
        &judge,
        "Anchor",
        &SearchOptions {
            max_hops: 3,
            ..SearchOptions::default()
        },
    )
    .expect_err("three hops are not supported");
    assert!(error.to_string().contains("at most two expansion rounds"));
}

/// A transport that records payload bytes and answers every candidate.
struct Capture {
    payloads: Mutex<Vec<Vec<u8>>>,
    state: Mutex<Vec<String>>,
}

impl Capture {
    fn new() -> Self {
        Capture {
            payloads: Mutex::new(Vec::new()),
            state: Mutex::new(Vec::new()),
        }
    }
}

impl Transport for Capture {
    fn request(&self, payload: &Payload) -> Result<Response, JudgeError> {
        self.payloads
            .lock()
            .expect("payload log")
            .push(payload.to_bytes());
        self.state
            .lock()
            .expect("state log")
            .push(payload.state.clone());
        let answers = payload
            .questions
            .iter()
            .map(|(key, _)| {
                (
                    key.clone(),
                    Answer {
                        question_type: "noul".to_owned(),
                        noul: 0.5,
                    },
                )
            })
            .collect();
        Ok(Response {
            model: "capture".to_owned(),
            answers,
            input_tokens: 1,
            output_tokens: 1,
            request_attempts: 1,
            http_attempts: None,
            failed_attempts: sag2_json::Object::new(),
        })
    }
}

#[test]
fn context_state_uses_the_history_branches_in_reference_order() {
    let judge = BatchedJudge::new(Capture::new());
    let mut evidence = vec![candidate(2, "Earlier", "Earlier passage")];
    evidence[0].previous_jev_score = Some(0.4);
    let context = SearchContext {
        original_question: "Where was Anchor born?".to_owned(),
        evidence: vec![candidate(1, "FromContext", "Context passage")],
        search_history: vec![Value::Str("prior round".to_owned())],
        previous_decisions: Value::Str("tentative".to_owned()),
    };
    let candidates = vec![candidate(3, "New", "New passage")];
    let (scored, _usage) = judge
        .score("follow-up", &candidates, &evidence, Some(&context))
        .expect("score");
    assert_eq!(scored.len(), 1);
    let raw = judge.transport().state.lock().expect("state")[0].clone();
    // serde_json sorts object keys, so the insertion-order contract is checked
    // against the serialised string, which is what the payload carries.
    let mut cursor = 0;
    for key in [
        "original_question",
        "previously_retrieved_evidence",
        "current_search_query",
        "retrieval_history",
        "previous_decisions",
        "history_note",
        "evidence_score_note",
    ] {
        let needle = format!("\"{key}\":");
        let position = raw[cursor..]
            .find(&needle)
            .unwrap_or_else(|| panic!("{key} is missing or out of order in {raw}"))
            + cursor;
        cursor = position;
    }
    let state: serde_json::Value = serde_json::from_str(&raw).expect("state JSON");
    assert_eq!(state["original_question"], "Where was Anchor born?");
    assert_eq!(state["current_search_query"], "follow-up");
    let rows = state["previously_retrieved_evidence"]
        .as_array()
        .expect("evidence array");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["passage_id"], 1);
    assert_eq!(rows[1]["passage_id"], 2);
    // The `previous_jev_score` branch writes the note only for `evidence`.
    assert!(state["evidence_score_note"].as_str().expect("note").contains("tentative judgments"));
}
