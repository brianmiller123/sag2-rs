//! Gate A: `CorpusIndex.retrieve` (`sag2.py:454-549`).
//!
//! Every SQL string, the pruning boundary and the `chunks` materialisation are
//! copied from the reference implementation; design §13.4 makes the SQL text
//! itself part of the contract (`MATERIALIZED` and `+rowid` are load-bearing).

use std::collections::HashMap;

use rusqlite::{params, Connection};
use sag2_json::{compact_json, Value};

/// The ranking query (`sag2.py:484-490`), byte for byte.
pub const BASE_SQL: &str = concat!(
    "WITH top_hits AS MATERIALIZED (",
    "SELECT rowid,sag_bm25(chunk_fts,?,2,2,1) AS score FROM chunk_fts ",
    "WHERE chunk_fts MATCH ? ",
    "AND rowid NOT IN (SELECT value FROM json_each(?)) ",
    "ORDER BY score,rowid LIMIT ?) SELECT rowid,score FROM top_hits ORDER BY score,rowid"
);

/// The substitution `sag2.py:506-509` applies to [`BASE_SQL`] for pruning.
const PRUNE_NEEDLE: &str = "ORDER BY score,rowid LIMIT ?)";
const PRUNE_REPLACEMENT: &str = "AND +rowid IN (SELECT rowid FROM chunk_fts WHERE chunk_fts \
                                 MATCH ?) ORDER BY score,rowid LIMIT ?)";

/// The per-expression statistics blob (`sag2.py:487-492`).
pub const SAG_STATS_SQL: &str =
    "SELECT sag_stats(chunk_fts) FROM chunk_fts WHERE chunk_fts MATCH ? LIMIT 1";

const CHUNKS_SQL: &str = "SELECT id,document_id,title,heading,content,full_doc,spans,characters,\
                          source_id FROM chunks WHERE id IN (SELECT value FROM json_each(?))";

/// A retrieved chunk with its BM25 score (`sag2.py:525-532`).
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// The `chunks.id` row id.
    pub id: i64,
    /// The document the chunk belongs to.
    pub document_id: i64,
    /// The document title.
    pub title: String,
    /// The chunk's heading (`""` when none).
    pub heading: String,
    /// The chunk text (materialised from `spans` in `source_dir` mode).
    pub content: String,
    /// The whole document, when the index stores it.
    pub full_doc: Option<String>,
    /// Code-point spans into the source text.
    pub spans: Vec<(u64, u64)>,
    /// Character count the chunk reports.
    pub characters: u32,
    /// The source identifier the index built.
    pub source_id: String,
    /// The `sag_bm25` score (negative; lower is better).
    pub score: f64,
}

/// A failure of the read path.
#[derive(Debug)]
pub enum QueryError {
    /// SQLite rejected a statement or the file.
    Sqlite(crate::SqliteError),
    /// The `sag_stats` blob is not a sequence of `f64`s.
    InvalidRankStats,
    /// A `chunks` row does not decode into [`Hit`].
    MalformedRow {
        /// The column that failed to decode.
        column: &'static str,
    },
    /// A ranked id has no `chunks` row (Python would raise `KeyError`).
    MissingChunk(i64),
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryError::Sqlite(error) => write!(formatter, "{error}"),
            QueryError::InvalidRankStats => {
                write!(
                    formatter,
                    "sag_stats returned a blob that is not a sequence of f64"
                )
            }
            QueryError::MalformedRow { column } => {
                write!(formatter, "chunks.{column} does not decode")
            }
            QueryError::MissingChunk(id) => write!(formatter, "ranked chunk id {id} has no row"),
        }
    }
}

impl std::error::Error for QueryError {}

impl From<rusqlite::Error> for QueryError {
    fn from(error: rusqlite::Error) -> Self {
        QueryError::Sqlite(error.into())
    }
}

/// `re.findall(r'"((?:""|[^"])*)"', expression)` (`sag2.py:502`).
///
/// The captured text keeps doubled quotes as two characters, exactly like the
/// regex group, because the pruning subset rebuilds the phrases verbatim.
pub(crate) fn quoted_phrases(expression: &str) -> Vec<String> {
    let chars: Vec<char> = expression.chars().collect();
    let mut phrases = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] != '"' {
            index += 1;
            continue;
        }
        index += 1;
        let mut phrase = String::new();
        let mut closed = false;
        while index < chars.len() {
            if chars[index] == '"' {
                if chars.get(index + 1) == Some(&'"') {
                    phrase.push_str("\"\"");
                    index += 2;
                    continue;
                }
                index += 1;
                closed = true;
                break;
            }
            phrase.push(chars[index]);
            index += 1;
        }
        if !closed {
            break;
        }
        phrases.push(phrase);
    }
    phrases
}

/// `struct.unpack(f"{len(blob) // 8}d", blob)[1:]` (`sag2.py:504`).
pub(crate) fn decode_idfs(blob: &[u8]) -> Result<Vec<f64>, QueryError> {
    if blob.len() % 8 != 0 {
        return Err(QueryError::InvalidRankStats);
    }
    let values: Vec<f64> = blob
        .chunks_exact(8)
        .map(|chunk| {
            let bytes: [u8; 8] = chunk.try_into().expect("chunks_exact yields 8 bytes");
            f64::from_le_bytes(bytes)
        })
        .collect();
    Ok(values.get(1..).unwrap_or(&[]).to_vec())
}

/// `sorted(range(len(idfs)), key=lambda i: -idfs[i])[: math.ceil(len * 0.55)]`
/// (`sag2.py:503-505`).  `sorted` is stable, so equal IDFs keep index order.
pub(crate) fn top_idf_indices(idfs: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..idfs.len()).collect();
    order.sort_by(|&left, &right| {
        idfs[right]
            .partial_cmp(&idfs[left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate((idfs.len() as f64 * 0.55).ceil() as usize);
    order
}

/// Run one ranking query and collect `(rowid, score)` in result order.
pub(crate) fn run_query(
    db: &Connection,
    sql: &str,
    stats: &[u8],
    query: &str,
    excluded_json: &str,
    limit: usize,
    subset: Option<&str>,
) -> Result<Vec<(i64, f64)>, rusqlite::Error> {
    let mut statement = db.prepare_cached(sql)?;
    let mut rows = match subset {
        Some(subset) => {
            statement.query(params![stats, query, excluded_json, subset, limit as i64])?
        }
        None => statement.query(params![stats, query, excluded_json, limit as i64])?,
    };
    let mut ranked = Vec::new();
    while let Some(row) = rows.next()? {
        ranked.push((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?));
    }
    Ok(ranked)
}

/// The pruning variant of [`BASE_SQL`] with the unselected phrases excluded.
pub(crate) fn pruned_sql() -> String {
    BASE_SQL.replace(PRUNE_NEEDLE, PRUNE_REPLACEMENT)
}

/// `sag2.py:525-532`: read the `chunks` rows for `ranked` and attach scores.
pub(crate) fn materialize(db: &Connection, ranked: &[(i64, f64)]) -> Result<Vec<Hit>, QueryError> {
    if ranked.is_empty() {
        return Ok(Vec::new());
    }
    let ids: Vec<Value> = ranked.iter().map(|(id, _)| Value::Int(*id)).collect();
    let ids_json = compact_json(&Value::Array(ids));
    let mut statement = db.prepare(CHUNKS_SQL)?;
    let mut rows = statement.query([ids_json])?;
    let mut by_id = HashMap::with_capacity(ranked.len());
    while let Some(row) = rows.next()? {
        let id: i64 = row.get(0)?;
        let spans: String = row.get(6)?;
        let characters: i64 = row.get(7)?;
        let hit = Hit {
            id,
            document_id: row.get(1)?,
            title: row.get(2)?,
            heading: row.get(3)?,
            content: row.get(4)?,
            full_doc: Some(row.get(5)?),
            spans: serde_json::from_str(&spans)
                .map_err(|_| QueryError::MalformedRow { column: "spans" })?,
            characters: u32::try_from(characters).map_err(|_| QueryError::MalformedRow {
                column: "characters",
            })?,
            source_id: row.get(8)?,
            score: 0.0,
        };
        by_id.insert(id, hit);
    }
    let mut hits = Vec::with_capacity(ranked.len());
    for (id, score) in ranked {
        let mut hit = by_id.remove(id).ok_or(QueryError::MissingChunk(*id))?;
        hit.score = *score;
        hits.push(hit);
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::{decode_idfs, pruned_sql, quoted_phrases, top_idf_indices, BASE_SQL, PRUNE_NEEDLE};

    #[test]
    fn phrases_keep_doubled_quotes() {
        // CPython: re.findall(r'"((?:""|[^"])*)"', '"a""b"') == ['a""b'].
        assert_eq!(quoted_phrases(r#""a""b""#), vec!["a\"\"b".to_owned()]);
        // CPython: the unterminated quote and the missing quote both yield [].
        assert!(quoted_phrases("\"unterminated").is_empty());
        assert!(quoted_phrases("no quotes here").is_empty());
    }

    #[test]
    fn pruning_rewrites_only_the_expected_fragment() {
        let pruned = pruned_sql();
        assert_eq!(pruned.matches("+rowid IN").count(), 1);
        assert!(pruned.contains("WHERE chunk_fts MATCH ?) ORDER BY score,rowid LIMIT ?)"));
        assert!(BASE_SQL.contains("MATERIALIZED"));
        assert!(BASE_SQL.ends_with("ORDER BY score,rowid"));
        assert_eq!(BASE_SQL.matches(PRUNE_NEEDLE).count(), 1);
    }

    #[test]
    fn idfs_skip_the_leading_double() {
        let mut blob = Vec::new();
        for value in [7.5_f64, 1.0, 2.0] {
            blob.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(decode_idfs(&blob).expect("decodes"), vec![1.0, 2.0]);
        assert!(decode_idfs(&[0, 1, 2]).is_err());
        assert_eq!(decode_idfs(&[]).expect("empty"), Vec::<f64>::new());
    }

    #[test]
    fn selection_is_stable_and_takes_fifty_five_percent() {
        let idfs = [1.0, 3.0, 3.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5];
        assert_eq!(top_idf_indices(&idfs), vec![1, 2, 3, 0, 4]);
    }
}
