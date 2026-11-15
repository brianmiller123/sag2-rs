//! `sag2-cli build`: turn a corpus into a SAG2 index.
//!
//! ```text
//! sag2-cli build --input <jsonl|dir> --output <index> [--profile <id>]
//!                [--dict-dir <path>] [--chunk-chars <n>]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use sag2::{CorpusIndex, IndexOptions, Profile, RebuildPolicy};

use crate::corpus::Corpus;
use crate::options;

/// The usage line for this command.
pub const USAGE: &str = "sag2-cli build --input <jsonl|dir> --output <index> \
[--profile sag2-zh-v1|multilingual-v1-<lang>] [--dict-dir <path>] [--chunk-chars <n>]";

/// Run `build`.  Errors are returned as text; the caller prints them and exits
/// with code 3, so a bad argument or a refused build never panics.
pub fn run(args: &[String]) -> Result<ExitCode, String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut profile = Profile::Sag2ZhV1;
    let mut dictionary_dir: Option<PathBuf> = None;
    let mut chunk_chars = sag2::IndexOptions::default().chunk_chars;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(ExitCode::SUCCESS);
            }
            "--input" => {
                input = Some(PathBuf::from(options::value(
                    args, &mut index, "--input", USAGE,
                )?))
            }
            "--output" => {
                output = Some(PathBuf::from(options::value(
                    args, &mut index, "--output", USAGE,
                )?))
            }
            "--profile" => {
                let text = options::value(args, &mut index, "--profile", USAGE)?;
                profile = text
                    .parse()
                    .map_err(|error: sag2_core::UnknownProfile| format!("--profile: {error}"))?;
            }
            "--dict-dir" => {
                dictionary_dir = Some(PathBuf::from(options::value(
                    args,
                    &mut index,
                    "--dict-dir",
                    USAGE,
                )?))
            }
            "--chunk-chars" => {
                let text = options::value(args, &mut index, "--chunk-chars", USAGE)?;
                chunk_chars = text
                    .parse()
                    .map_err(|_| format!("--chunk-chars: {text:?} is not a positive integer"))?;
                if chunk_chars == 0 {
                    return Err("--chunk-chars must be at least 1".to_owned());
                }
            }
            flag => return Err(format!("unknown argument {flag}\n{USAGE}")),
        }
        index += 1;
    }
    let input = input.ok_or_else(|| format!("--input is required\n{USAGE}"))?;
    let output = output.ok_or_else(|| format!("--output is required\n{USAGE}"))?;

    let corpus = Corpus::load(&input)?;
    let documents = corpus.documents();
    let options = IndexOptions {
        chunk_chars,
        profile,
        dictionary_dir: options::dictionary_dir(dictionary_dir),
        rebuild: RebuildPolicy::OnCorpusChange,
        ..IndexOptions::default()
    };
    let (opened, outcome) = CorpusIndex::open(&output, &documents, &corpus.fingerprint, options)
        .map_err(|error| format!("cannot build {}: {error}", output.display()))?;

    println!("index:       {}", opened.path().display());
    println!("profile:     {}", opened.profile());
    println!("fingerprint: {}", corpus.fingerprint);
    println!("documents:   {}", documents.len());
    println!("chunks:      {}", opened.statistics().total);
    println!("action:      {:?}", outcome.action);
    if let Some(reason) = outcome.rebuilt {
        println!("rebuilt:     {reason:?}");
    }
    Ok(ExitCode::SUCCESS)
}
