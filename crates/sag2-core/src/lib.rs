//! SAG2 text core: the §4 tokenizer and the §1.4 D3 chunker, byte-exact with
//! the Python reference.
//!
//! The dictionary (`data/dict.txt`) is the certified jieba 0.42.1 file; the
//! FREQ table and the Unicode tables are generated at build time from that
//! dictionary and from the Python exporter's exhaustive goldens.

#![deny(unsafe_code)]

mod ascii_cut;
mod chunk;
mod dict;
mod tokenize;
mod unicode;

pub use chunk::{chunk_document, source_content, utf16_length, Chunk, DEFAULT_CHUNK_CHARS};
pub use tokenize::tokenize;
pub use unicode::{
    casefold, combining_class, is_letter, is_mark, is_number, is_uppercase, lower, nfkc, nfkd,
};
