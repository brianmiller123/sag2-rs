//! Build the vendored `sag_rank.c` into a static archive and put that archive
//! on the link line.
//!
//! `sag_rank.c` is compiled with `-DSQLITE_CORE -DSAG_STATIC`, so it calls the
//! real `sqlite3_*`/FTS5 symbols instead of going through the loadable
//! extension API. That makes the *position* of the archive on the link line
//! load-bearing: a static archive is scanned once, left to right, and only
//! members that define symbols still undefined at that point are extracted.
//!
//! Why `#[link]` is not enough: `#[link(name = "sag_rank", kind = "static")]`
//! becomes a native-library entry whose position relative to the `-l` flags
//! Cargo emits for `libsqlite3-sys` is not controllable, and the P0 spike
//! observed exactly that arrangement failing with undefined `sqlite3_*`
//! symbols (the archive was scanned after the amalgamation). This build script
//! therefore emits two different instructions:
//!
//! * `cargo:rustc-link-search` + `cargo:rustc-link-lib=static:+whole-archive`
//!   -- the propagating form: every downstream link, including the separate
//!   `sag2-linktest` package, sees the archive. `+whole-archive` takes all
//!   members unconditionally, which decouples symbol extraction from scan
//!   order and leaves the `sqlite3_*` references to the amalgamation that
//!   Cargo orders after this dependency's `-l`.
//! * `cargo:rustc-link-arg=<archive>` -- applies only to this package's own
//!   targets (tests/examples/benches); it is the arrangement the P0 spike
//!   verified and does not depend on the `-l` search path.
//!
//! `cargo:rustc-link-arg` does not propagate to dependents (verified: the
//! downstream binary failed with an undefined `sag_register_rank` when it was
//! the only instruction), which is why the native-library pair above is
//! required and why the link proof lives in a separate package.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let vendor = manifest_dir.join("vendor");

    let build = {
        let mut build = cc::Build::new();
        build
            .file(vendor.join("sag_rank.c"))
            .include(&vendor)
            .define("SQLITE_CORE", None)
            .define("SAG_STATIC", None);
        build
    };

    let objects = build.compile_intermediates();

    let archive = out_dir.join("libsag_rank.a");
    let _ = std::fs::remove_file(&archive);
    let status = build
        .try_get_archiver()
        .expect("no archiver available for the sag_rank static archive")
        .arg("crs")
        .arg(&archive)
        .args(&objects)
        .status()
        .expect("failed to spawn the archiver");
    assert!(status.success(), "archiver failed: {status}");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=sag_rank");
    println!("cargo:rustc-link-arg={}", archive.display());
    println!("cargo:rerun-if-changed=vendor/sag_rank.c");
    println!("cargo:rerun-if-changed=vendor/sqlite3.h");
    println!("cargo:rerun-if-changed=vendor/sqlite3ext.h");
    println!("cargo:rerun-if-changed=build.rs");
}
