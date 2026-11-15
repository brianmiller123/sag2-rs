//! SAG2 text core: the §4 tokenizer, the §1.4 D3 chunker and the §5 UCA 9.0.0
//! collator, byte-exact with the Python reference.
//!
//! The dictionary (`data/dict.txt`) is the certified jieba 0.42.1 file and the
//! DUCET table (`data/allkeys-9.0.0.txt`) is pyuca 1.2's UCA 9.0.0 data; the
//! FREQ table, the Unicode tables and the compiled collation table are all
//! generated at build time from those files and the Python exporter's
//! exhaustive goldens.

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod analysis;
mod ascii_cut;
mod chunk;
mod collate;
mod dict;
mod diversity;
pub mod names;
mod query;
mod tokenize;
mod unicode;

pub use analysis::{
    tokenize_with, tokenize_with_segmenter, AnalysisError, DictionaryAnalysis, Language, Profile,
    UnknownProfile,
};
pub use chunk::{chunk_document, source_content, utf16_length, Chunk, DEFAULT_CHUNK_CHARS};
pub use collate::sort_key;
pub use dict::{
    anchor_every, contains as freq_contains, max_key_chars, short_entries, short_entry_count,
};
pub use diversity::diversity_tokens;
pub use query::{match_query, match_query_with, match_query_with_segmenter, quote_term, QueryMode};
pub use tokenize::tokenize;
pub use unicode::{
    casefold, combining_class, is_case_ignorable, is_cased, is_letter, is_mark, is_number,
    is_space, is_uppercase, is_word, lower, nfd, nfkc, nfkd,
};
