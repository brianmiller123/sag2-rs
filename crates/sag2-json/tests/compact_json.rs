//! Differential test: `compact_json` against `sag2.py:65-66`'s helper
//! (`json.dumps(value, ensure_ascii=False, separators=(",", ":"))`).
//!
//! `tests/fixtures/json_cases.jsonl` is produced by
//! `python3.12 crates/sag2-json/tests/fixtures/generate.py`; the case
//! definitions below mirror the generator's `json_cases()` dict, including the
//! insertion order.

use std::collections::HashSet;

use sag2_json::{compact_json, Object, Value};

const FIXTURES: &str = include_str!("fixtures/json_cases.jsonl");

fn object(entries: &[(&str, Value)]) -> Value {
    let mut object = Object::new();
    for (key, value) in entries {
        object.insert(*key, value.clone());
    }
    Value::Object(object)
}

fn cases() -> Vec<(&'static str, Value)> {
    let mut updated = Object::new();
    updated.insert("a", Value::Int(1));
    updated.insert("b", Value::Int(2));
    updated.insert("a", Value::Int(3));

    vec![
        (
            "flat_object",
            object(&[
                ("a", Value::Int(1)),
                ("b", Value::Null),
                ("c", Value::Bool(true)),
                ("d", Value::Bool(false)),
            ]),
        ),
        (
            "empty_containers",
            object(&[("object", object(&[])), ("array", Value::Array(vec![]))]),
        ),
        ("empty_string", object(&[("s", Value::Str(String::new()))])),
        (
            "string_escapes",
            object(&[(
                "s",
                Value::Str("a\"b\\c\nd\re\tf\u{8}g\u{c}h\0i\u{1}j\u{1f}k\u{7f}l/m".to_owned()),
            )]),
        ),
        (
            "unicode_passthrough",
            object(&[(
                "s",
                Value::Str("中文 😀 café ẞ \u{2028}\u{2029}".to_owned()),
            )]),
        ),
        (
            "key_escapes",
            object(&[("a\"b", Value::Int(1)), ("k\\2", Value::Int(2))]),
        ),
        (
            "nested",
            object(&[
                (
                    "outer",
                    object(&[(
                        "inner",
                        Value::Array(vec![
                            Value::Int(1),
                            Value::Int(2),
                            object(&[("deep", Value::Str("x".to_owned()))]),
                        ]),
                    )]),
                ),
                (
                    "list",
                    Value::Array(vec![
                        Value::Array(vec![]),
                        Value::Array(vec![object(&[])]),
                        Value::Null,
                    ]),
                ),
            ]),
        ),
        (
            "object_order",
            object(&[
                ("z", Value::Int(1)),
                ("a", Value::Int(2)),
                ("m", Value::Int(3)),
            ]),
        ),
        ("key_order_after_update", Value::Object(updated)),
        (
            "floats",
            Value::Array(vec![
                Value::Float(0.1),
                Value::Float(1.0),
                Value::Float(-0.0),
                Value::Float(1e16),
                Value::Float(1e-7),
                Value::Float(1e300),
                Value::Float(1.7976931348623157e308),
                Value::Float(5e-324),
                Value::Float(2.5e-10),
                Value::Float(123456789012345678.0),
            ]),
        ),
        (
            "special_floats",
            Value::Array(vec![
                Value::Float(f64::NAN),
                Value::Float(f64::INFINITY),
                Value::Float(f64::NEG_INFINITY),
                Value::Float(-0.0),
                Value::Float(0.0),
            ]),
        ),
        (
            "ints",
            Value::Array(vec![
                Value::Int(0),
                Value::Int(-1),
                Value::Int(i64::MAX),
                Value::Int(i64::MIN),
            ]),
        ),
        (
            "bools_and_null",
            Value::Array(vec![Value::Bool(true), Value::Bool(false), Value::Null]),
        ),
    ]
}

#[test]
fn compact_json_matches_cpython() {
    let fixtures: Vec<(String, String)> = FIXTURES
        .lines()
        .map(|line| {
            let record: serde_json::Value =
                serde_json::from_str(line).expect("fixture line is valid JSON");
            (
                record["case"].as_str().expect("case").to_owned(),
                record["compact"].as_str().expect("compact").to_owned(),
            )
        })
        .collect();

    let cases = cases();
    assert_eq!(
        cases.len(),
        fixtures.len(),
        "fixture/case count mismatch: regenerate fixtures"
    );

    let mut seen = HashSet::new();
    for (name, value) in &cases {
        assert!(seen.insert(*name), "duplicate case name {name}");
        let (_, expected) = fixtures
            .iter()
            .find(|(case, _)| case == name)
            .unwrap_or_else(|| panic!("fixture is missing case {name}"));
        assert_eq!(&compact_json(value), expected, "case {name}");
    }
}

#[test]
fn compact_object_has_no_whitespace_and_keeps_order() {
    assert_eq!(compact_json(&object(&[("a", Value::Int(1))])), "{\"a\":1}");

    let nested = object(&[("z", object(&[("b", Value::Int(1)), ("a", Value::Int(2))]))]);
    assert_eq!(compact_json(&nested), "{\"z\":{\"b\":1,\"a\":2}}");
}
