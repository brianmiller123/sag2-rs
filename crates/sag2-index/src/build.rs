//! Index construction: the `CorpusIndex.__init__` build branch
//! (`sag2.py:197-311`) and `_load_statistics` (`sag2.py:328-401`).
//!
//! The schema, the signature, the `chunks` row layout and the cached
//! statistics JSON are the Python ones (design §3.1 scheme A); only the
//! provenance keys live in the extra `rust_index_metadata` table (design
//! §3.2).  Row order is insertion order: `document_id` is the corpus position
//! plus one, duplicates (by `title + "\n" + text`) keep their gap, and chunk
//! ids are the global insertion counter.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, Transaction};
use sag2_core::{chunk_document, tokenize, utf16_length};
use sag2_json::{compact_json, Object, Value};
use sha2::{Digest, Sha256};

use crate::open::{IndexOptions, RuntimeVersions};
use crate::signature::signature;

/// One corpus row (`sag2.py:182-187`, design §2.5).
///
/// `path` is the **raw** row path.  The index build resolves `None` to
/// `"document.txt"` (`sag2.py:182`), while `chunk_document` resolves its own
/// `None` to `"document.md"` (`chunking.py:67`); the two defaults are
/// deliberately not unified (design §9 R13).
#[derive(Debug, Clone, Copy)]
pub struct Document<'a> {
    pub title: &'a str,
    pub text: &'a str,
    pub path: Option<&'a str>,
    /// `sag2.py:187`: `source_id` is `str(doc.get("idx", position)).rsplit("_", 1)[0]`.
    pub idx: Option<&'a str>,
}

/// The `source_dir` build requires the corpus row to name its file.
#[derive(Debug)]
pub enum BuildError {
    Sqlite(rusqlite::Error),
    Io {
        path: PathBuf,
        error: std::io::Error,
    },
    NotUtf8 {
        path: PathBuf,
    },
    MissingPath {
        position: usize,
    },
    EmptyDocument {
        position: usize,
    },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::Sqlite(error) => write!(formatter, "{error}"),
            BuildError::Io { path, error } => {
                write!(formatter, "cannot read {}: {error}", path.display())
            }
            BuildError::NotUtf8 { path } => {
                write!(formatter, "{} is not valid UTF-8", path.display())
            }
            BuildError::MissingPath { position } => write!(
                formatter,
                "corpus row {position} needs a path when source_dir is set"
            ),
            BuildError::EmptyDocument { position } => write!(
                formatter,
                "corpus row {position} needs non-empty title and text"
            ),
        }
    }
}

impl std::error::Error for BuildError {}

impl From<rusqlite::Error> for BuildError {
    fn from(error: rusqlite::Error) -> Self {
        BuildError::Sqlite(error)
    }
}

/// Build (or rebuild) the index at `path` from `corpus` (`sag2.py:197-311`).
pub fn build(
    path: &Path,
    corpus: &[Document<'_>],
    fingerprint: &str,
    options: &IndexOptions,
    versions: &RuntimeVersions,
) -> Result<(), BuildError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|error| BuildError::Io {
                path: parent.to_owned(),
                error,
            })?;
        }
    }
    let expected = signature(
        fingerprint,
        options.chunk_chars,
        options.source_dir.as_deref(),
    );
    let mut db = Connection::open(path)?;
    db.execute_batch("CREATE TABLE IF NOT EXISTS metadata (signature TEXT NOT NULL)")?;
    let tx = db.transaction()?;
    let contentless = options.source_dir.is_some();
    if contentless {
        // The `content=''` FTS table and `source_files` replace the
        // materialised `content`/`full_doc` columns (`sag2.py:243-252`).
        tx.execute_batch(
            "DROP TABLE IF EXISTS chunks;
             DROP TABLE IF EXISTS chunk_fts;
             DROP TABLE IF EXISTS source_files;
             CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL, \
             title TEXT NOT NULL, heading TEXT NOT NULL, \
             start_char INTEGER NOT NULL, end_char INTEGER NOT NULL, \
             spans TEXT NOT NULL, characters INTEGER NOT NULL, source_id TEXT NOT NULL);
             CREATE INDEX chunks_document ON chunks(document_id);
             CREATE VIRTUAL TABLE chunk_fts USING fts5(title,heading,body,content='');
             CREATE TABLE source_files (document_id INTEGER PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL);",
        )?;
    } else {
        tx.execute_batch(
            "DROP TABLE IF EXISTS chunks;
             DROP TABLE IF EXISTS chunk_fts;
             DROP TABLE IF EXISTS source_files;
             CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL, \
             title TEXT NOT NULL, heading TEXT NOT NULL, \
             content TEXT NOT NULL, full_doc TEXT NOT NULL, \
             spans TEXT NOT NULL, characters INTEGER NOT NULL, source_id TEXT NOT NULL);
             CREATE INDEX chunks_document ON chunks(document_id);
             CREATE VIRTUAL TABLE chunk_fts USING fts5(title,heading,body);",
        )?;
    }

    let mut chunk_statement = tx.prepare("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)")?;
    let mut fts_statement =
        tx.prepare("INSERT INTO chunk_fts(rowid,title,heading,body) VALUES(?,?,?,?)")?;
    let mut source_statement = if contentless {
        Some(tx.prepare("INSERT INTO source_files VALUES(?,?,?)")?)
    } else {
        None
    };

    let mut seen: HashSet<String> = HashSet::new();
    let mut statistics_documents: Vec<StatisticDocument> = Vec::new();
    let mut chunk_id = 0_i64;
    for (position, document) in corpus.iter().enumerate() {
        let title = py_strip(document.title).to_owned();
        let (content, source_sha256, source_path) = match options.source_dir.as_deref() {
            None => (document.text.to_owned(), None, None),
            Some(source_dir) => {
                let relative = document.path.ok_or(BuildError::MissingPath { position })?;
                let file = source_dir.join(relative);
                let bytes = fs::read(&file).map_err(|error| BuildError::Io {
                    path: file.clone(),
                    error,
                })?;
                let text = std::str::from_utf8(&bytes)
                    .map_err(|_| BuildError::NotUtf8 { path: file.clone() })?
                    .to_owned();
                (text, Some(sha256_hex(&bytes)), Some(relative))
            }
        };
        if title.is_empty() || py_strip(&content).is_empty() {
            return Err(BuildError::EmptyDocument { position });
        }
        let full_doc = format!("{title}\n{content}");
        let duplicate = !seen.insert(full_doc.clone());
        if contentless {
            if duplicate {
                continue;
            }
            // `source_files` and `_load_statistics` both iterate the
            // deduplicated corpus (`sag2.py:284-290`, `356-367`).
            source_statement
                .as_mut()
                .ok_or(BuildError::MissingPath { position })?
                .execute(rusqlite::params![
                    (position + 1) as i64,
                    source_path.ok_or(BuildError::MissingPath { position })?,
                    source_sha256
                ])?;
            statistics_documents.push(StatisticDocument {
                title: title.clone(),
                text: content.clone(),
                path: document.path.map(str::to_owned),
            });
        } else {
            // A materialised index caches statistics over the original corpus,
            // duplicates included (`sag2.py:328-401`).
            statistics_documents.push(StatisticDocument {
                title: document.title.to_owned(),
                text: content.clone(),
                path: document.path.map(str::to_owned),
            });
            if duplicate {
                continue;
            }
        }

        let source_index = match document.idx {
            Some(idx) => idx.to_owned(),
            None => position.to_string(),
        };
        let source_id = source_index
            .rsplit_once('_')
            .map_or_else(|| source_index.clone(), |(left, _)| left.to_owned());
        let chunks = chunk_document(
            &content,
            options.chunk_chars,
            document.path.or(Some("document.txt")),
        );
        for chunk in chunks {
            chunk_id += 1;
            let document_id = (position + 1) as i64;
            let spans = spans_json(&chunk.spans);
            let characters = utf16_length(&chunk.content) as i64;
            if contentless {
                let start = chunk.spans.first().map_or(0, |span| span.0) as i64;
                let end = chunk.spans.last().map_or(0, |span| span.1) as i64;
                chunk_statement.execute(rusqlite::params![
                    chunk_id,
                    document_id,
                    title,
                    chunk.heading,
                    start,
                    end,
                    spans,
                    characters,
                    source_id
                ])?;
            } else {
                chunk_statement.execute(rusqlite::params![
                    chunk_id,
                    document_id,
                    title,
                    chunk.heading,
                    chunk.content,
                    full_doc,
                    spans,
                    characters,
                    source_id
                ])?;
            }
            fts_statement.execute(rusqlite::params![
                chunk_id,
                join_tokens(&title),
                join_tokens(&chunk.heading),
                join_tokens(&chunk.content)
            ])?;
        }
    }
    drop(chunk_statement);
    drop(fts_statement);
    drop(source_statement);
    tx.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')", [])?;

    tx.execute("DELETE FROM metadata", [])?;
    tx.execute("INSERT INTO metadata VALUES(?)", [&expected])?;
    load_statistics(
        &tx,
        &statistics_documents,
        options.chunk_chars,
        contentless,
        &expected,
    )?;
    write_rust_metadata(&tx, versions)?;
    tx.commit()?;
    Ok(())
}

/// One row of the corpus iteration `_load_statistics` performs.
struct StatisticDocument {
    title: String,
    text: String,
    path: Option<String>,
}

fn spans_json(spans: &[(usize, usize)]) -> String {
    compact_json(&Value::Array(
        spans
            .iter()
            .map(|&(start, end)| {
                Value::Array(vec![Value::Int(start as i64), Value::Int(end as i64)])
            })
            .collect(),
    ))
}

fn join_tokens(text: &str) -> String {
    tokenize(text).join(" ")
}

/// `sag2.py:328-401` `_load_statistics`.
fn load_statistics(
    tx: &Transaction<'_>,
    corpus: &[StatisticDocument],
    chunk_chars: usize,
    contentless: bool,
    signature: &str,
) -> Result<(), BuildError> {
    tx.execute(
        "CREATE TABLE IF NOT EXISTS index_statistics \
         (signature TEXT PRIMARY KEY, data TEXT NOT NULL)",
        [],
    )?;

    // `self.frequencies` is the FTS5 `body` document frequency
    // (`sag2.py:348-352`).
    tx.execute(
        "CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main,chunk_fts,'col')",
        [],
    )?;
    let mut frequencies = Object::new();
    {
        let mut statement = tx.prepare("SELECT term, doc FROM temp.vocab WHERE col='body'")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            frequencies.insert(row.get::<_, String>(0)?, Value::Int(row.get::<_, i64>(1)?));
        }
    }
    let total: i64 = tx.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    let document_count: i64 = tx.query_row(
        "SELECT COUNT(DISTINCT document_id) FROM chunks",
        [],
        |row| row.get(0),
    )?;

    let mut title_phrases: BTreeSet<Vec<String>> = BTreeSet::new();
    let mut names: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut term_frequencies: BTreeMap<String, i64> = BTreeMap::new();
    let mut count = 0_i64;
    for document in corpus {
        let words = tokenize(&document.title);
        if words.len() >= 2 {
            title_phrases.insert(words);
        }
        let name = sag2_core::casefold(&sag2_core::nfkc(&document.title));
        names
            .entry(fold_latin_accents(&name))
            .or_default()
            .insert(name);
        let mut terms: BTreeSet<String> = diversity_tokens(&document.text).into_iter().collect();
        for chunk in chunk_document(&document.text, chunk_chars, document.path.as_deref()) {
            terms.extend(diversity_tokens(&chunk.content));
        }
        for term in terms {
            *term_frequencies.entry(term).or_insert(0) += 1;
        }
        count += 1;
    }
    let aliases: BTreeMap<String, String> = names
        .into_iter()
        .filter_map(|(alias, values)| {
            if values.len() == 1 && !values.contains(&alias) {
                values.into_iter().next().map(|value| (alias, value))
            } else {
                None
            }
        })
        .collect();

    let mut data = Object::new();
    data.insert("frequencies", Value::Object(frequencies));
    data.insert("total", Value::Int(total));
    data.insert("document_count", Value::Int(document_count));
    data.insert(
        "title_phrases",
        Value::Array(
            title_phrases
                .into_iter()
                .map(|phrase| Value::Array(phrase.into_iter().map(Value::Str).collect()))
                .collect(),
        ),
    );
    data.insert(
        "aliases",
        Value::Object(
            aliases
                .into_iter()
                .fold(Object::new(), |mut object, (alias, value)| {
                    object.insert(alias, Value::Str(value));
                    object
                }),
        ),
    );
    let mut diversity_idf = Object::new();
    for (term, frequency) in term_frequencies {
        let idf = ((1.0 + count as f64) / (1.0 + frequency as f64)).ln() + 1.0;
        diversity_idf.insert(term, Value::Float(idf));
    }
    data.insert("diversity_idf", Value::Object(diversity_idf));
    data.insert("source_statistics_complete", Value::Bool(contentless));

    tx.execute("DELETE FROM index_statistics", [])?;
    tx.execute(
        "INSERT INTO index_statistics VALUES(?,?)",
        rusqlite::params![signature, compact_json(&Value::Object(data))],
    )?;
    Ok(())
}

/// The v1 `rust_index_metadata` rows (design §3.3).
fn write_rust_metadata(
    tx: &Transaction<'_>,
    versions: &RuntimeVersions,
) -> Result<(), rusqlite::Error> {
    tx.execute(
        "CREATE TABLE IF NOT EXISTS rust_index_metadata \
         (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        [],
    )?;
    tx.execute("DELETE FROM rust_index_metadata", [])?;
    let rows: [(&str, &str); 10] = [
        ("format_version", crate::open::FORMAT_VERSION_STR),
        ("tokenizer_version", versions.tokenizer_version),
        ("chunk_policy_version", versions.chunk_policy_version),
        ("jieba_dict_sha256", versions.jieba_dict_sha256),
        ("unicode_version", versions.unicode_version),
        ("uca_version", versions.uca_version),
        ("uca_keys_sha256", versions.uca_keys_sha256),
        ("sqlite_version", versions.sqlite_version),
        ("sag_rank_sha256", versions.sag_rank_sha256),
        ("sag2_rs_version", versions.sag2_rs_version),
    ];
    let mut statement = tx.prepare("INSERT INTO rust_index_metadata VALUES(?,?)")?;
    for (key, value) in rows {
        statement.execute(rusqlite::params![key, value])?;
    }
    Ok(())
}

/// `sag2.py:122` `diversity_tokens`: `re.findall(r"[^\W_]+", text.casefold())`.
///
/// `[^\W_]` is Python's `str.isalnum()`; for every code point assigned in
/// Unicode 15.0 that equals `\p{L}` or `\p{N}`, the tables Gate D certifies.
fn diversity_tokens(text: &str) -> Vec<String> {
    let folded = sag2_core::casefold(text);
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in folded.chars() {
        let scalar = character as u32;
        if sag2_core::is_letter(scalar) || sag2_core::is_number(scalar) {
            current.push(character);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// `sag2.py:110-119` `fold_latin_accents`.
///
/// The only property the reference reads from the Unicode database beyond
/// decomposition is `"LATIN" in unicodedata.name(char, "")`.  The ranges below
/// are the exact answer for CPython 3.12.11 / Unicode 15.0.0 (51 ranges,
/// generated with `unicodedata.name`); everything else is NFKD + combining
/// class + NFKC from `sag2-core`.
fn fold_latin_accents(text: &str) -> String {
    let mut characters = String::new();
    let mut latin_base = false;
    for character in sag2_core::nfkd(text).chars() {
        if sag2_core::combining_class(character as u32) != 0 {
            if !latin_base {
                characters.push(character);
            }
        } else {
            characters.push(character);
            latin_base = is_latin_name(character as u32);
        }
    }
    sag2_core::nfkc(&characters)
}

/// `"LATIN" in unicodedata.name(cp, "")`, Unicode 15.0.0.
///
/// The exact 51 ranges CPython 3.12.11 returns, generated with
/// `unicodedata.name`.  Symbols whose names spell out a Latin letter (squared
/// CL, negative circled A, tags) belong to the set, which is why it is not a
/// script table.
fn is_latin_name(scalar: u32) -> bool {
    const RANGES: [(u32, u32); 51] = [
        (0x0041, 0x005A),
        (0x0061, 0x007A),
        (0x00C0, 0x00D6),
        (0x00D8, 0x00F6),
        (0x00F8, 0x02AF),
        (0x0363, 0x036F),
        (0x1ABF, 0x1AC0),
        (0x1ACC, 0x1ACE),
        (0x1D00, 0x1D25),
        (0x1D62, 0x1D65),
        (0x1D6B, 0x1D77),
        (0x1D79, 0x1D9A),
        (0x1DCA, 0x1DCA),
        (0x1DD3, 0x1DF4),
        (0x1E00, 0x1EFF),
        (0x2071, 0x2071),
        (0x207F, 0x207F),
        (0x2090, 0x209C),
        (0x2184, 0x2184),
        (0x249C, 0x24E9),
        (0x271D, 0x271F),
        (0x2C2E, 0x2C2E),
        (0x2C5E, 0x2C5E),
        (0x2C60, 0x2C7C),
        (0x2C7E, 0x2C7F),
        (0xA722, 0xA76F),
        (0xA771, 0xA787),
        (0xA78B, 0xA7CA),
        (0xA7D0, 0xA7D1),
        (0xA7D3, 0xA7D3),
        (0xA7D5, 0xA7D9),
        (0xA7F5, 0xA7F7),
        (0xA7FA, 0xA7FF),
        (0xAB30, 0xAB5A),
        (0xAB60, 0xAB64),
        (0xAB66, 0xAB68),
        (0xFB00, 0xFB06),
        (0xFF21, 0xFF3A),
        (0xFF41, 0xFF5A),
        (0x1DF00, 0x1DF1E),
        (0x1DF25, 0x1DF2A),
        (0x1F110, 0x1F12C),
        (0x1F130, 0x1F149),
        (0x1F150, 0x1F169),
        (0x1F170, 0x1F18A),
        (0x1F1A5, 0x1F1A5),
        (0x1F520, 0x1F521),
        (0x1F524, 0x1F524),
        (0x1F546, 0x1F547),
        (0xE0041, 0xE005A),
        (0xE0061, 0xE007A),
    ];
    RANGES
        .binary_search_by(|&(low, high)| {
            if scalar < low {
                std::cmp::Ordering::Greater
            } else if scalar > high {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn py_strip(text: &str) -> &str {
    text.trim_matches(|character: char| {
        character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
    })
}

#[cfg(test)]
mod tests {
    use rusqlite::Connection;

    use super::*;
    use crate::open::{IndexOptions, RebuildPolicy, RUNTIME_VERSIONS};
    use crate::{CorpusIndex, OpenAction};

    fn materialised(chunk_chars: usize) -> IndexOptions {
        IndexOptions {
            chunk_chars,
            source_dir: None,
            rebuild: RebuildPolicy::Always,
        }
    }

    #[test]
    fn builds_the_python_schema_and_statistics() {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("index.sqlite");
        let corpus = [
            Document {
                title: " Anchor ",
                text: "Alice Paul birthplace.",
                path: None,
                idx: None,
            },
            Document {
                title: "Project Aurora",
                text: "Bob Smith bridge target.",
                path: None,
                idx: Some("dsid_1"),
            },
            Document {
                title: "Project Aurora",
                text: "Bob Smith bridge target.",
                path: None,
                idx: Some("duplicate"),
            },
        ];
        let (index, outcome) =
            CorpusIndex::open(&path, &corpus, "fixture", materialised(2200)).expect("build");
        assert_eq!(outcome.action, OpenAction::Built);
        assert_eq!(index.statistics().total, 2);
        assert_eq!(index.statistics().document_count, 2);
        assert!(index
            .statistics()
            .title_phrases
            .contains(&vec!["project".to_owned(), "aurora".to_owned()]));
        drop(index);

        let db = Connection::open(&path).expect("reopen");
        let metadata_rows: i64 = db
            .query_row("SELECT COUNT(*) FROM metadata", [], |row| row.get(0))
            .expect("metadata rows");
        assert_eq!(metadata_rows, 1);
        let signature: String = db
            .query_row("SELECT signature FROM metadata", [], |row| row.get(0))
            .expect("signature");
        assert_eq!(signature, crate::signature("fixture", 2200, None));

        let columns: Vec<String> = db
            .prepare("PRAGMA table_info(chunks)")
            .expect("table info")
            .query_map([], |row| row.get(1))
            .expect("columns")
            .collect::<Result<_, _>>()
            .expect("column names");
        assert_eq!(
            columns,
            [
                "id",
                "document_id",
                "title",
                "heading",
                "content",
                "full_doc",
                "spans",
                "characters",
                "source_id"
            ]
        );
        let source_ids: Vec<String> = db
            .prepare("SELECT source_id FROM chunks ORDER BY id")
            .expect("source ids")
            .query_map([], |row| row.get(0))
            .expect("rows")
            .collect::<Result<_, _>>()
            .expect("values");
        assert_eq!(source_ids, ["0", "dsid"]);
        let title: String = db
            .query_row("SELECT title FROM chunks WHERE id=1", [], |row| row.get(0))
            .expect("title");
        assert_eq!(title, "Anchor");

        let rust_keys: i64 = db
            .query_row("SELECT COUNT(*) FROM rust_index_metadata", [], |row| {
                row.get(0)
            })
            .expect("rust keys");
        assert_eq!(rust_keys, 10);
        let fts_sql: String = db
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name='chunk_fts'",
                [],
                |row| row.get(0),
            )
            .expect("fts sql");
        assert!(!fts_sql.contains("content=''"));
        let source_tables: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='source_files'",
                [],
                |row| row.get(0),
            )
            .expect("source_files");
        assert_eq!(source_tables, 0);

        let data: String = db
            .query_row("SELECT data FROM index_statistics", [], |row| row.get(0))
            .expect("statistics");
        let data: serde_json::Value = serde_json::from_str(&data).expect("statistics json");
        assert_eq!(data["total"], 2);
        assert_eq!(data["document_count"], 2);
        assert_eq!(data["source_statistics_complete"], false);
        assert_eq!(data["frequencies"]["alice"], 1);
        assert_eq!(data["frequencies"]["target"], 1);
    }

    #[test]
    fn builds_a_source_dir_index_with_contentless_fts() {
        let directory = tempfile::tempdir().expect("temp dir");
        let file = directory.path().join("dsid_1__Guide.md");
        let source = "# 发布指南\n## 生产环境\n授权规则必须完整保留。";
        fs::write(&file, source).expect("write source");
        let corpus = [Document {
            title: "Guide",
            text: "ignored",
            path: Some("dsid_1__Guide.md"),
            idx: Some("dsid_1"),
        }];
        let path = directory.path().join("index.sqlite");
        let options = IndexOptions {
            chunk_chars: 512,
            source_dir: Some(directory.path().to_owned()),
            rebuild: RebuildPolicy::Always,
        };
        build(&path, &corpus, "fixture", &options, &RUNTIME_VERSIONS).expect("build");

        let db = Connection::open(&path).expect("open");
        let fts_sql: String = db
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name='chunk_fts'",
                [],
                |row| row.get(0),
            )
            .expect("fts sql");
        assert!(fts_sql.contains("content=''"));
        let (start, end): (i64, i64) = db
            .query_row("SELECT start_char, end_char FROM chunks", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .expect("ranges");
        assert_eq!(start, 15);
        assert_eq!(end as usize, source.chars().count());
        let (relative, digest): (String, String) = db
            .query_row("SELECT path, sha256 FROM source_files", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .expect("source_files");
        assert_eq!(relative, "dsid_1__Guide.md");
        assert_eq!(digest, sha256_hex(source.as_bytes()));
        let data: String = db
            .query_row("SELECT data FROM index_statistics", [], |row| row.get(0))
            .expect("statistics");
        let data: serde_json::Value = serde_json::from_str(&data).expect("statistics json");
        assert_eq!(data["source_statistics_complete"], true);
        let signature: String = db
            .query_row("SELECT signature FROM metadata", [], |row| row.get(0))
            .expect("signature");
        assert_eq!(
            signature,
            crate::signature("fixture", 512, options.source_dir.as_deref())
        );
    }

    #[test]
    fn folds_latin_accents_but_keeps_non_latin_marks() {
        assert_eq!(fold_latin_accents("zürich"), "zurich");
        assert_eq!(fold_latin_accents("Ångström"), "Angstrom");
        assert_eq!(fold_latin_accents("ΐ"), "ΐ");
        assert_eq!(fold_latin_accents("が"), "が");
        assert!(is_latin_name('A' as u32));
        assert!(is_latin_name('é' as u32));
        assert!(!is_latin_name('Ω' as u32));
    }

    #[test]
    fn diversity_tokens_are_alnum_runs_after_casefold() {
        assert_eq!(
            diversity_tokens("A.H.G. half-ounce_日本"),
            ["a", "h", "g", "half", "ounce", "日本"]
        );
    }
}
