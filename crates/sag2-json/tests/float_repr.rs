//! Differential test: `py_repr` against CPython 3.12's `repr(float)`.
//!
//! `tests/fixtures/floats.jsonl` is produced by
//! `python3.12 crates/sag2-json/tests/fixtures/generate.py`. Each line carries
//! the raw bit pattern so the exact `f64` is reconstructed with `from_bits`,
//! never by re-parsing the decimal text.

use sag2_json::py_repr;

const FIXTURES: &str = include_str!("fixtures/floats.jsonl");

#[test]
fn py_repr_matches_cpython_repr() {
    let mut checked = 0usize;
    for (line_number, line) in FIXTURES.lines().enumerate() {
        let record: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|error| panic!("floats.jsonl:{}: {error}", line_number + 1));
        let bits = record["bits"].as_u64().expect("bits");
        let expected = record["repr"].as_str().expect("repr");
        let value = f64::from_bits(bits);
        assert_eq!(py_repr(value), expected, "bits={bits:#018x}");
        checked += 1;
    }
    assert!(
        checked >= 10_000,
        "expected at least 10_000 float fixtures, got {checked}"
    );
}

#[test]
fn fixtures_keep_the_special_cases() {
    // A regenerated fixture must not silently lose the values the formatter
    // has to special-case.
    for expected in ["inf", "-inf", "nan", "-0.0", "0.0"] {
        let needle = format!("\"repr\":\"{expected}\"");
        assert!(FIXTURES.contains(&needle), "fixture is missing {expected}");
    }
}
