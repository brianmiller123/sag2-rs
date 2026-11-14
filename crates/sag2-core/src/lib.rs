//! SAG2 text core: the §4 tokenizer, byte-exact with the Python reference.
//!
//! The dictionary (`data/dict.txt`) is the certified jieba 0.42.1 file; the
//! FREQ table and the Unicode tables are generated at build time from that
//! dictionary and from the Python exporter's exhaustive goldens.

#![deny(unsafe_code)]

mod ascii_cut;
mod dict;
mod tokenize;
mod unicode;

pub use tokenize::tokenize;
pub use unicode::{
    casefold, combining_class, is_letter, is_mark, is_number, is_uppercase, lower, nfkc,
};
