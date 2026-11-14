//! Byte-exact port of the JSON codec the Python SAG2 implementation uses for
//! every outward-facing payload.
//!
//! `sag2.py:65-66` defines
//!
//! ```python
//! def compact_json(value: Any) -> str:
//!     return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
//! ```
//!
//! and the request journal keys its cache files on
//! `sha256(compact_json([endpoint, payload]))` (`model_api.py:48`). Gate C-1
//! therefore compares payload bytes, which makes three properties contractual
//! (design §1.3, §9 R8):
//!
//! * **Key order is the insertion order** of the Python `dict`. [`Object`] is
//!   an insertion-ordered map; no `HashMap` participates in encoding.
//! * **Floats are `float.__repr__` bytes**, produced by [`py_repr`]. `NaN`,
//!   `Infinity` and `-Infinity` are emitted as bare tokens because CPython's
//!   JSON encoder allows them (`allow_nan=True`).
//! * **Non-ASCII text passes through as UTF-8** (`ensure_ascii=False`); the
//!   only escaped characters are `"`, `\`, the five short escapes, and the
//!   remaining C0 controls (`\u00XX`, lowercase hex).
//!
//! Strings are the only JSON scalars Python can represent losslessly that Rust
//! cannot: a Python `str` may hold lone surrogates, which [`String`] cannot.
//! Such values are outside this contract. Integers are carried as `i64`, which
//! covers every integer the retrieval path can produce (ids, counts, ranks).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod float;

pub use float::py_repr;

use core::fmt;

/// Encode `value` exactly like `sag2.py`'s `compact_json`.
pub fn compact_json(value: &Value) -> String {
    let mut out = String::with_capacity(64);
    value
        .write_compact(&mut out)
        .expect("writing JSON into a String cannot fail");
    out
}

/// A JSON value that preserves object key insertion order.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// `None` / `null`.
    Null,
    /// `True` / `False`.
    Bool(bool),
    /// A Python `int` within `i64` range.
    Int(i64),
    /// A Python `float`, encoded through [`py_repr`].
    Float(f64),
    /// A string, escaped like Python's `json.dumps`.
    Str(String),
    /// A list.
    Array(Vec<Value>),
    /// An insertion-ordered dict.
    Object(Object),
}

impl Value {
    /// Append the compact JSON encoding of this value to `out`.
    pub fn write_compact(&self, out: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Value::Null => out.write_str("null"),
            Value::Bool(true) => out.write_str("true"),
            Value::Bool(false) => out.write_str("false"),
            Value::Int(value) => write!(out, "{value}"),
            Value::Float(value) => out.write_str(&json_float(*value)),
            Value::Str(value) => write_json_string(value, out),
            Value::Array(items) => {
                out.write_char('[')?;
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.write_char(',')?;
                    }
                    item.write_compact(out)?;
                }
                out.write_char(']')
            }
            Value::Object(object) => {
                out.write_char('{')?;
                for (index, (key, item)) in object.iter().enumerate() {
                    if index > 0 {
                        out.write_char(',')?;
                    }
                    write_json_string(key, out)?;
                    out.write_char(':')?;
                    item.write_compact(out)?;
                }
                out.write_char('}')
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_compact(formatter)
    }
}

/// `json.dumps` writes floats with `float.__repr__`, except that the three
/// non-finite values are spelled `NaN`, `Infinity` and `-Infinity`.
fn json_float(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_owned()
    } else if value == f64::INFINITY {
        "Infinity".to_owned()
    } else if value == f64::NEG_INFINITY {
        "-Infinity".to_owned()
    } else {
        py_repr(value)
    }
}

/// The escape set of CPython's JSON encoder with `ensure_ascii=False`
/// (`py_encode_basestring_ascii`'s non-ascii branch).
fn write_json_string(value: &str, out: &mut impl fmt::Write) -> fmt::Result {
    out.write_char('"')?;
    let mut literal_start = 0;
    for (index, character) in value.char_indices() {
        let escaped = match character {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            '\u{08}' => "\\b",
            '\u{0c}' => "\\f",
            control if (control as u32) < 0x20 => {
                out.write_str(&value[literal_start..index])?;
                write!(out, "\\u{:04x}", control as u32)?;
                literal_start = index + control.len_utf8();
                continue;
            }
            _ => continue,
        };
        out.write_str(&value[literal_start..index])?;
        out.write_str(escaped)?;
        literal_start = index + character.len_utf8();
    }
    out.write_str(&value[literal_start..])?;
    out.write_char('"')
}

/// An insertion-ordered JSON object, matching Python `dict` semantics: a key
/// keeps the position of its first insertion when its value is replaced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Object {
    entries: Vec<(String, Value)>,
}

impl Object {
    /// An empty object.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace `key`. Returns the previous value, if any.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) -> Option<Value> {
        let key = key.into();
        for entry in &mut self.entries {
            if entry.0 == key {
                return Some(core::mem::replace(&mut entry.1, value));
            }
        }
        self.entries.push((key, value));
        None
    }

    /// The value stored under `key`.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries
            .iter()
            .find(|(existing, _)| existing == key)
            .map(|(_, value)| value)
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the object has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate `(key, value)` in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Bool(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Int(value)
    }
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Value::Int(i64::from(value))
    }
}

impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Value::Int(i64::from(value))
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Float(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::Str(value.to_owned())
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::Str(value)
    }
}

impl From<Vec<Value>> for Value {
    fn from(value: Vec<Value>) -> Self {
        Value::Array(value)
    }
}

impl From<Object> for Value {
    fn from(value: Object) -> Self {
        Value::Object(value)
    }
}
