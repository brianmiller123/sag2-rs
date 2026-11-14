//! `sag2-cli` — command line entry points (design §2.1, §7.3).
//!
//! P2 ships one subcommand: `verify --gate A|B|C|D`, the replay half of the
//! gate tooling.  Index/search/bench/dump arrive with the plans that give them
//! something to run.

mod verify;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else {
        eprintln!("{}", verify::usage());
        return ExitCode::from(3);
    };
    match command.as_str() {
        "verify" => verify::verify(&args[1..]),
        other => {
            eprintln!("sag2-cli: unknown command {other}");
            eprintln!("{}", verify::usage());
            ExitCode::from(3)
        }
    }
}
