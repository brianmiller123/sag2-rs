//! Bindings for the vendored `sag_rank.c` FTS5 scoring functions.
//!
//! `sag_rank.c` registers two FTS5 auxiliary functions through the extension
//! API: `sag_stats` caches the per-query statistics blob, and `sag_bm25`
//! computes BM25 from that blob so filtered searches reuse the exact
//! statistics of the unfiltered query (`sag2.py`'s ranking SQL and
//! `sag2_rank.py`'s runtime `load_extension` load the same C file). Here the
//! file is compiled into a static archive and linked into the process, which
//! removes the extension-extraction step while keeping the SQLite that serves
//! the queries identical to the one being extended.
//!
//! This crate intentionally has **no** SQLite dependency: the amalgamation
//! belongs to `sag2-index` through `rusqlite`'s `bundled` feature, and exactly
//! one crate may own it. Only the opaque [`sqlite3`] handle is defined here so
//! callers can pass a connection pointer through.

#![allow(non_camel_case_types)]

use core::ffi::c_int;
use core::fmt;

/// Opaque SQLite connection handle, layout-compatible with the C `sqlite3`.
#[repr(C)]
pub struct sqlite3 {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    /// Register `sag_stats` and `sag_bm25` on `db`; returns an SQLite result
    /// code (`SQLITE_OK` on success).
    pub fn sag_register_rank(db: *mut sqlite3) -> c_int;
}

/// `SQLITE_OK` (`sqlite3.h`).
pub const SQLITE_OK: c_int = 0;

/// A non-`SQLITE_OK` result code from [`register`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    code: c_int,
}

impl Error {
    /// The SQLite result code returned by `sag_register_rank`.
    pub fn code(self) -> c_int {
        self.code
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "sag_register_rank failed with SQLite result code {}",
            self.code
        )
    }
}

impl std::error::Error for Error {}

/// Register the `sag_stats` and `sag_bm25` FTS5 functions on `db`.
///
/// # Safety
///
/// `db` must be a valid, open connection handle obtained from the SQLite
/// instance this archive is linked against, and it must not be used after that
/// connection is closed. The pointer cannot be validated here; passing anything
/// else is undefined behaviour.
pub unsafe fn register(db: *mut sqlite3) -> Result<(), Error> {
    // SAFETY: the caller guarantees `db` is a live connection handle; the C
    // entry point only prepares one statement and registers two functions.
    let code = unsafe { sag_register_rank(db) };
    if code == SQLITE_OK {
        Ok(())
    } else {
        Err(Error { code })
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, SQLITE_OK};

    #[test]
    fn ok_code_is_zero() {
        assert_eq!(SQLITE_OK, 0);
    }

    #[test]
    fn error_reports_the_sqlite_code() {
        let error = Error { code: 5 };
        assert_eq!(error.code(), 5);
        assert!(error.to_string().contains('5'));
    }
}
