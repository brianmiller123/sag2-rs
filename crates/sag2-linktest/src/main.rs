//! Independent downstream package proving the full link path: `rusqlite`'s
//! bundled SQLite 3.50.2 plus the vendored `sag_rank.c` static archive that
//! `sag2-rank-sys` puts on the link line.
//!
//! This must stay a *separate* package. The archive reaches a downstream link
//! only through `sag2-rank-sys`'s propagated `cargo:rustc-link-lib=+whole-archive`
//! instruction: `cargo:rustc-link-arg` is scoped to the emitting package's own
//! targets (design §2.3 rule 2). A binary in another package is therefore the
//! only proof that the propagated form actually links.

use std::error::Error;

use sag2_rank_sys::sqlite3;

fn main() -> Result<(), Box<dyn Error>> {
    let connection = rusqlite::Connection::open_in_memory()?;

    // SAFETY: the handle refers to the live connection behind `connection`.
    let handle = unsafe { connection.handle() } as *mut sqlite3;
    sag2_rank_sys::register(handle)?;
    println!("register ok");

    let version: String = connection.query_row("SELECT sqlite_version()", [], |row| row.get(0))?;
    assert_eq!(
        version, "3.50.2",
        "bundled SQLite drifted from the design's layer-S contract"
    );
    println!("sqlite {version}");

    let options: Vec<String> = {
        let mut statement = connection.prepare("PRAGMA compile_options")?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<Result<_, _>>()?
    };
    assert!(
        options.iter().any(|option| option == "ENABLE_FTS5"),
        "FTS5 must be compiled into the bundled SQLite"
    );
    println!("compile options: {} (ENABLE_FTS5 present)", options.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn null_handle_maps_to_sqlite_misuse() {
        // `sag_register_rank` starts with `sqlite3_prepare_v2`, which rejects a
        // null database with SQLITE_MISUSE (21) before dereferencing it, so the
        // error mapping is observable without a live connection.
        let error = sag2_rank_sys::register(std::ptr::null_mut())
            .expect_err("a null connection must not be accepted");
        assert_eq!(error.code(), 21, "SQLITE_MISUSE");
    }
}
