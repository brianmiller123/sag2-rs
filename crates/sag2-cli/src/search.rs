//! `sag2-cli search`: query one index with a chosen query mode.
//!
//! ```text
//! sag2-cli search --index <path> --query <text> [--query-mode lexical|phrase3]
//!                 [--judge oracle] [--limit <n>] [--dict-dir <path>]
//! ```
//!
//! The index is opened as the profile and dictionary it records, so the query
//! is tokenised by the same pipeline that built the stored tokens.  Without
//! `--judge` this is BM25 retrieval only; `--judge oracle` runs the full SAG2
//! loop with the deterministic fixture judge (no network).

use std::path::PathBuf;
use std::process::ExitCode;

use sag2::{open_recorded, QueryMode, QueryOptions, SearchOptions};

use crate::options;

/// The usage line for this command.
pub const USAGE: &str = "sag2-cli search --index <path> --query <text> \
[--query-mode lexical|phrase3] [--judge oracle] [--limit <n>] [--dict-dir <path>]";

/// Run `search`.  Errors are returned as text; the caller prints them and
/// exits with code 3.
pub fn run(args: &[String]) -> Result<ExitCode, String> {
    let mut path: Option<PathBuf> = None;
    let mut query: Option<String> = None;
    let mut dictionary_dir: Option<PathBuf> = None;
    let mut mode = QueryMode::Lexical;
    let mut judge = false;
    let mut limit = 10_usize;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(ExitCode::SUCCESS);
            }
            "--index" => {
                path = Some(PathBuf::from(options::value(
                    args, &mut index, "--index", USAGE,
                )?))
            }
            "--query" => {
                query = Some(options::value(args, &mut index, "--query", USAGE)?.to_owned())
            }
            "--query-mode" => {
                let text = options::value(args, &mut index, "--query-mode", USAGE)?;
                mode = match text {
                    "lexical" => QueryMode::Lexical,
                    "phrase3" => QueryMode::Phrase3,
                    other => {
                        return Err(format!(
                            "--query-mode: unknown mode {other:?}; use lexical or phrase3"
                        ))
                    }
                };
            }
            "--judge" => {
                let text = options::value(args, &mut index, "--judge", USAGE)?;
                if text != "oracle" {
                    return Err(format!(
                        "--judge: unknown judge {text:?}; only the deterministic \"oracle\" \
                         fixture judge is available (omit the flag for BM25 retrieval only)"
                    ));
                }
                judge = true;
            }
            "--limit" => {
                let text = options::value(args, &mut index, "--limit", USAGE)?;
                limit = text
                    .parse()
                    .map_err(|_| format!("--limit: {text:?} is not a positive integer"))?;
                if limit == 0 {
                    return Err("--limit must be at least 1".to_owned());
                }
            }
            "--dict-dir" => {
                dictionary_dir = Some(PathBuf::from(options::value(
                    args,
                    &mut index,
                    "--dict-dir",
                    USAGE,
                )?))
            }
            flag => return Err(format!("unknown argument {flag}\n{USAGE}")),
        }
        index += 1;
    }
    let path = path.ok_or_else(|| format!("--index is required\n{USAGE}"))?;
    let query = query.ok_or_else(|| format!("--query is required\n{USAGE}"))?;

    let opened = open_recorded(
        &path,
        options::dictionary_dir(dictionary_dir).as_deref(),
        QueryOptions::default(),
    )
    .map_err(|error| format!("cannot open {}: {error}", path.display()))?;

    println!("index:      {}", path.display());
    println!("profile:    {}", opened.profile());
    println!("mode:       {}", mode_name(mode));
    let expression = opened
        .match_query_with(mode, &query)
        .map_err(|error| format!("cannot build the query expression: {error}"))?;
    println!("expression: {expression}");
    if judge {
        search_with_oracle(&opened, &query, mode, limit)
    } else {
        retrieve(&opened, &expression, limit)
    }
}

fn mode_name(mode: QueryMode) -> &'static str {
    match mode {
        QueryMode::Lexical => "lexical",
        QueryMode::Phrase3 => "phrase3",
    }
}

fn retrieve(
    opened: &sag2::CorpusIndex,
    expression: &str,
    limit: usize,
) -> Result<ExitCode, String> {
    let hits = opened
        .retrieve(expression, &Default::default(), limit, None)
        .map_err(|error| format!("retrieval failed: {error}"))?;
    println!("judge:      none (BM25 retrieval only)");
    println!("hits:       {}", hits.len());
    for (rank, hit) in hits.iter().enumerate() {
        println!(
            "{}\tdoc={}\tscore={:.6}\t{}",
            rank + 1,
            hit.document_id,
            hit.score,
            hit.title
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn search_with_oracle(
    opened: &sag2::CorpusIndex,
    query: &str,
    mode: QueryMode,
    limit: usize,
) -> Result<ExitCode, String> {
    let options = SearchOptions {
        candidate_k: limit,
        query_mode: mode,
        ..SearchOptions::default()
    };
    let judge = sag2::BatchedJudge::new(sag2::OracleJudge::new());
    let result = sag2::search(opened, &judge, query, &options)
        .map_err(|error| format!("search failed: {error}"))?;
    println!("judge:      oracle");
    println!("candidates: {}", result.candidates.len());
    for (rank, candidate) in result.candidates.iter().take(limit).enumerate() {
        println!(
            "{}\tdoc={}\tjev={:.6}\tfusion={:.6}\t{}",
            rank + 1,
            candidate.hit.document_id,
            candidate.jev_score,
            candidate.fusion_score,
            candidate.hit.title
        );
    }
    Ok(ExitCode::SUCCESS)
}
