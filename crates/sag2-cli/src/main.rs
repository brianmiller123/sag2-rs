//! `sag2-cli` — command line entry points (design §2.1, §7.3).
//!
//! Subcommands: `verify` (the gate replay tooling), `build` (corpus → index)
//! and `search` (query one index).  Exit codes: `0` success, `2` a gate
//! comparison failure (`verify` only), `3` a usage or open/build failure.

mod build;
mod corpus;
mod options;
mod search;
mod verify;

use std::process::ExitCode;

fn usage() -> String {
    format!(
        "sag2-cli <command> [options]\n\n\
         commands:\n  \
         {}\n  \
         {}\n  \
         verify --gate A|B|C|D [--fixtures ...] [--golden <dir>]\n\n\
         Run `sag2-cli <command> --help` for a command's options.",
        build::USAGE,
        search::USAGE
    )
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else {
        eprintln!("{}", usage());
        return ExitCode::from(3);
    };
    let result = match command.as_str() {
        "verify" => return verify::verify(&args[1..]),
        "build" => build::run(&args[1..]),
        "search" => search::run(&args[1..]),
        "-h" | "--help" | "help" => {
            println!("{}", usage());
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown command {other}\n\n{}", usage())),
    };
    match result {
        Ok(code) => code,
        Err(message) => {
            eprintln!("sag2-cli: {message}");
            ExitCode::from(3)
        }
    }
}
