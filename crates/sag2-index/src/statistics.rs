//! Corpus statistics cached by the index (`sag2.py:328-401`, design §3.2).
//!
//! The Python implementation recomputes the statistics when the cached row is
//! missing; that path needs the tokenizer and the chunker, which belong to
//! `sag2-core` (P1).  A reader therefore requires the cached row and fails
//! loudly when it is absent instead of silently serving empty statistics.

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::{Connection, OptionalExtension};
use serde_json::Value as Json;

/// The seven keys `_load_statistics` writes (`sag2.py:388-396`).
const KEYS: [&str; 7] = [
    "frequencies",
    "total",
    "document_count",
    "title_phrases",
    "aliases",
    "diversity_idf",
    "source_statistics_complete",
];

/// A decode failure of the cached `index_statistics.data` row.
#[derive(Debug)]
pub enum StatisticsError {
    /// No `index_statistics` row for the current corpus signature.
    MissingRow,
    /// `index_statistics.data` is not valid JSON.
    Json(serde_json::Error),
    /// The SQLite read failed.
    Sqlite(crate::SqliteError),
    /// The data object is missing a key of the seven-key contract.
    MissingKey(&'static str),
    /// The data object has a key outside the seven-key contract (design §3.2.7).
    UnexpectedKey(String),
    /// A key holds the wrong JSON type.
    WrongType(&'static str),
    /// A `diversity_idf` value is not a finite number.
    NonFiniteIdf(String),
}

impl std::fmt::Display for StatisticsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatisticsError::MissingRow => write!(
                formatter,
                "index_statistics has no row for this corpus signature (recomputing \
                 statistics needs the tokenizer and belongs to index building)"
            ),
            StatisticsError::Json(error) => write!(formatter, "index_statistics.data: {error}"),
            StatisticsError::Sqlite(error) => write!(formatter, "index_statistics: {error}"),
            StatisticsError::MissingKey(key) => {
                write!(formatter, "index_statistics.data is missing key {key}")
            }
            StatisticsError::UnexpectedKey(key) => {
                write!(formatter, "index_statistics.data has unexpected key {key}")
            }
            StatisticsError::WrongType(key) => {
                write!(
                    formatter,
                    "index_statistics.data key {key} has the wrong type"
                )
            }
            StatisticsError::NonFiniteIdf(term) => {
                write!(formatter, "diversity_idf[{term}] is not a finite number")
            }
        }
    }
}

impl std::error::Error for StatisticsError {}

/// The corpus-wide statistics `_load_statistics` exposes on the index.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Statistics {
    /// `fts5vocab` body document frequencies.
    pub frequencies: BTreeMap<String, i64>,
    /// Number of chunks.
    pub total: i64,
    /// Number of distinct documents.
    pub document_count: i64,
    /// Tokenised titles with at least two tokens, as a set (Python stores
    /// `{tuple(words) ...}`).
    pub title_phrases: BTreeSet<Vec<String>>,
    /// Folded alias -> canonical title.
    pub aliases: BTreeMap<String, String>,
    /// DPP term weights.
    pub diversity_idf: BTreeMap<String, f64>,
    /// Whether `source_statistics_complete` was set when the row was written.
    pub source_statistics_complete: bool,
}

/// Load and validate the cached statistics for `signature`.
pub fn load(db: &Connection, signature: &str) -> Result<Statistics, StatisticsError> {
    let data: Option<String> = db
        .query_row(
            "SELECT data FROM index_statistics WHERE signature=?",
            [signature],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| StatisticsError::Sqlite(error.into()))?;
    let data = data.ok_or(StatisticsError::MissingRow)?;
    parse(&data)
}

fn parse(text: &str) -> Result<Statistics, StatisticsError> {
    let root: Json = serde_json::from_str(text).map_err(StatisticsError::Json)?;
    let object = root.as_object().ok_or(StatisticsError::WrongType("data"))?;
    for key in KEYS {
        if !object.contains_key(key) {
            return Err(StatisticsError::MissingKey(key));
        }
    }
    if let Some(key) = object.keys().find(|key| !KEYS.contains(&key.as_str())) {
        return Err(StatisticsError::UnexpectedKey(key.clone()));
    }

    let mut frequencies = BTreeMap::new();
    for (term, count) in as_object("frequencies", object.get("frequencies"))? {
        let count = count
            .as_i64()
            .ok_or(StatisticsError::WrongType("frequencies"))?;
        frequencies.insert(term.clone(), count);
    }

    let mut title_phrases = BTreeSet::new();
    for phrase in object["title_phrases"]
        .as_array()
        .ok_or(StatisticsError::WrongType("title_phrases"))?
    {
        let words = phrase
            .as_array()
            .ok_or(StatisticsError::WrongType("title_phrases"))?
            .iter()
            .map(|word| {
                word.as_str()
                    .map(str::to_owned)
                    .ok_or(StatisticsError::WrongType("title_phrases"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        title_phrases.insert(words);
    }

    let mut aliases = BTreeMap::new();
    for (alias, target) in as_object("aliases", object.get("aliases"))? {
        let target = target
            .as_str()
            .ok_or(StatisticsError::WrongType("aliases"))?;
        aliases.insert(alias.clone(), target.to_owned());
    }

    let mut diversity_idf = BTreeMap::new();
    for (term, weight) in as_object("diversity_idf", object.get("diversity_idf"))? {
        let weight = weight
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| StatisticsError::NonFiniteIdf(term.clone()))?;
        diversity_idf.insert(term.clone(), weight);
    }

    Ok(Statistics {
        frequencies,
        total: as_i64("total", object.get("total"))?,
        document_count: as_i64("document_count", object.get("document_count"))?,
        title_phrases,
        aliases,
        diversity_idf,
        source_statistics_complete: object["source_statistics_complete"]
            .as_bool()
            .ok_or(StatisticsError::WrongType("source_statistics_complete"))?,
    })
}

fn as_object<'a>(
    key: &'static str,
    value: Option<&'a Json>,
) -> Result<&'a serde_json::Map<String, Json>, StatisticsError> {
    value
        .and_then(Json::as_object)
        .ok_or(StatisticsError::WrongType(key))
}

fn as_i64(key: &'static str, value: Option<&Json>) -> Result<i64, StatisticsError> {
    value
        .and_then(Json::as_i64)
        .ok_or(StatisticsError::WrongType(key))
}

#[cfg(test)]
mod tests {
    use super::{parse, StatisticsError};

    const SAMPLE: &str = r#"{"frequencies":{"a":3},"total":1,"document_count":1,
        "title_phrases":[["alice","paul"]],"aliases":{},"diversity_idf":{"a":1.5},
        "source_statistics_complete":false}"#;

    #[test]
    fn converts_title_phrases_and_idf() {
        let statistics = parse(SAMPLE).expect("sample parses");
        assert_eq!(statistics.total, 1);
        assert_eq!(statistics.document_count, 1);
        assert!(statistics
            .title_phrases
            .contains(&vec!["alice".to_owned(), "paul".to_owned()]));
        assert_eq!(statistics.diversity_idf.get("a"), Some(&1.5));
        assert!(!statistics.source_statistics_complete);
    }

    #[test]
    fn rejects_an_extra_key() {
        let text = SAMPLE.replace("\"total\":1", "\"total\":1,\"extra\":0");
        assert!(matches!(parse(&text), Err(StatisticsError::UnexpectedKey(key)) if key == "extra"));
    }

    #[test]
    fn rejects_a_missing_key() {
        let text = SAMPLE.replace("\"total\":1,", "");
        assert!(matches!(
            parse(&text),
            Err(StatisticsError::MissingKey("total"))
        ));
    }
}
