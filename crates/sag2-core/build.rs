//! Build-time generation for the tokenizer's data tables (design §4.3, §4.5).
//!
//! Two inputs, both produced by the Python exporter:
//!
//! * `data/dict.txt` - the certified jieba 0.42.1 dictionary, copied byte for
//!   byte.  The build extracts the codepoint-length <= 3 entries into a sorted
//!   NUL-separated blob with one `u32` anchor every 64 entries, and the
//!   (handful of) pure-ASCII entries the Python DAG can match.
//! * `golden/unicode_*.bin.gz` - the exporter's exhaustive per-codepoint
//!   tables.  The build keeps only non-identity mappings and rewrites the
//!   category masks as bitsets, so the crate only `include_bytes!`s compact
//!   read-only data and never parses anything at run time.

use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Rule 5 of design §4.1 only looks up 2- and 3-codepoint substrings.
const FREQ_MAX_CHARS: usize = 3;
/// One anchor per 64 entries (design §4.3).
const ANCHOR_EVERY: usize = 64;
const CODEPOINTS: usize = 0x110000;

const LOWER: (&str, &[u8], usize) = ("unicode_lower", b"SAG2LOWR", 17);
const CASEFOLD: (&str, &[u8], usize) = ("unicode_casefold", b"SAG2CASE", 17);
const COMBINING: (&str, &[u8], usize) = ("unicode_combining", b"SAG2COMB", 5);
const CATEGORIES: (&str, &[u8], usize) = ("unicode_categories", b"SAG2CATE", 5);
const SPARSE: [(&str, &[u8]); 3] = [
    ("unicode_nfkd", b"SAG2NFKD"),
    ("unicode_nfkc", b"SAG2NFKC"),
    ("unicode_compose", b"SAG2COMP"),
];

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let golden = crate_dir.join("../../golden");

    let dict_path = crate_dir.join("data/dict.txt");
    println!("cargo:rerun-if-changed={}", dict_path.display());
    for (name, _, _) in [LOWER, CASEFOLD, COMBINING, CATEGORIES] {
        println!(
            "cargo:rerun-if-changed={}",
            golden.join(format!("{name}.bin.gz")).display()
        );
    }
    for (name, _) in SPARSE {
        println!(
            "cargo:rerun-if-changed={}",
            golden.join(format!("{name}.bin.gz")).display()
        );
    }

    let (total, freq) = load_dict(&dict_path);
    write_freq_tables(&out_dir, total, &freq);
    write_unicode_tables(&golden, &out_dir);
}

/// Python's `jieba.Tokenizer.gen_pfdict` (`sag2.py:49` constructs the tokenizer
/// whose `FREQ` rule 5 consults): every line contributes its frequency to
/// `total` even when the word repeats, and the last frequency wins.
fn load_dict(path: &Path) -> (u64, BTreeMap<String, u32>) {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    let mut total = 0_u64;
    let mut freq: BTreeMap<String, u32> = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split(' ');
        let word = fields.next().expect("dictionary word");
        let value: u32 = fields
            .next()
            .unwrap_or_else(|| panic!("{}:{} has no frequency", path.display(), index + 1))
            .parse()
            .unwrap_or_else(|_| panic!("{}:{} has an invalid frequency", path.display(), index + 1));
        total += u64::from(value);
        freq.insert(word.to_owned(), value);
    }
    (total, freq)
}

fn write_freq_tables(out_dir: &Path, total: u64, freq: &BTreeMap<String, u32>) {
    let mut blob: Vec<u8> = Vec::new();
    let mut anchors: Vec<u32> = Vec::new();
    let mut entries = 0_usize;
    let mut ascii: Vec<(&str, u32)> = Vec::new();
    let mut crossing: Vec<&str> = Vec::new();
    // `BTreeMap` iterates in `str` order, which is UTF-8 byte order.
    for (word, value) in freq {
        if word.chars().count() <= FREQ_MAX_CHARS && *value > 0 {
            if entries % ANCHOR_EVERY == 0 {
                anchors.push(blob.len() as u32);
            }
            blob.extend_from_slice(word.as_bytes());
            blob.push(0);
            entries += 1;
        }
        if word.is_ascii() {
            ascii.push((word.as_str(), *value));
        }
        if crosses_ascii_cjk_boundary(word) {
            crossing.push(word.as_str());
        }
    }
    fs::write(out_dir.join("freq.bin"), &blob).expect("write freq blob");
    let mut anchor_bytes = Vec::with_capacity(anchors.len() * 4);
    for anchor in &anchors {
        anchor_bytes.extend_from_slice(&anchor.to_le_bytes());
    }
    fs::write(out_dir.join("freq_anchors.bin"), anchor_bytes).expect("write freq anchors");

    let mut source = String::new();
    writeln!(source, "pub const DICT_TOTAL: u64 = {total};").expect("write consts");
    writeln!(source, "pub const FREQ_ENTRIES: usize = {entries};").expect("write consts");
    writeln!(source, "pub const ANCHOR_EVERY: usize = {ANCHOR_EVERY};").expect("write consts");
    writeln!(source, "pub static ASCII_WORDS: &[(&str, u32)] = &[").expect("write consts");
    for (word, value) in ascii {
        writeln!(source, "    ({word:?}, {value}),").expect("write consts");
    }
    writeln!(source, "];").expect("write consts");
    // Python lowercases the text but not the dictionary, so only already
    // lowercase entries can match.  These are the only words that straddle the
    // ASCII/CJK boundary the tokenizer splits on (design §4.2).
    writeln!(
        source,
        "pub static CROSSING_WORDS: &[&str] = &["
    )
    .expect("write consts");
    for word in crossing {
        writeln!(source, "    {word:?},").expect("write consts");
    }
    writeln!(source, "];").expect("write consts");
    fs::write(out_dir.join("dict_consts.rs"), source).expect("write dict consts");
}

/// True for a dictionary word made only of `re_han_default` characters that
/// mixes the CJK range with the ASCII part and would survive `lower()`.
fn crosses_ascii_cjk_boundary(word: &str) -> bool {
    let mut has_cjk = false;
    let mut has_ascii = false;
    for character in word.chars() {
        let cp = character as u32;
        if (0x4E00..=0x9FD5).contains(&cp) {
            has_cjk = true;
        } else if character.is_ascii_alphanumeric()
            || matches!(character, '+' | '#' | '&' | '.' | '_' | '%' | '-')
        {
            has_ascii = true;
        } else {
            // Anything outside `re_han_default` splits jieba's blocks, so the
            // DAG can never match this word across a boundary either.
            return false;
        }
        if character.is_ascii_uppercase() {
            return false;
        }
    }
    has_cjk && has_ascii
}

fn gunzip(path: &Path) -> Vec<u8> {
    let compressed = fs::read(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    let mut data = Vec::new();
    flate2::read::GzDecoder::new(&compressed[..])
        .read_to_end(&mut data)
        .unwrap_or_else(|error| panic!("cannot decompress {}: {error}", path.display()));
    data
}

fn table_body<'a>(data: &'a [u8], magic: &[u8], width: usize, path: &Path) -> &'a [u8] {
    if data.len() < 8 || &data[..8] != magic {
        panic!("{} does not start with the documented magic tag", path.display());
    }
    let body = &data[8..];
    let expected = CODEPOINTS * width;
    if body.len() != expected {
        panic!(
            "{} holds {} record bytes, expected {expected}",
            path.display(),
            body.len()
        );
    }
    body
}

fn write_unicode_tables(golden: &Path, out_dir: &Path) {
    for (name, magic, width) in [LOWER, CASEFOLD] {
        let path = golden.join(format!("{name}.bin.gz"));
        let data = gunzip(&path);
        let body = table_body(&data, magic, width, &path);
        let mut sparse = Vec::new();
        let mut count = 0_u32;
        for cp in 0..CODEPOINTS {
            let base = cp * width;
            let length = body[base + 4] as usize;
            let identity = length == 1
                && u32::from_le_bytes(body[base + 5..base + 9].try_into().expect("mapping")) as usize
                    == cp;
            if identity {
                continue;
            }
            sparse.extend_from_slice(&body[base..base + width]);
            count += 1;
        }
        let mut file = Vec::with_capacity(4 + sparse.len());
        file.extend_from_slice(&count.to_le_bytes());
        file.extend_from_slice(&sparse);
        let output = name.strip_prefix("unicode_").unwrap_or(name);
        fs::write(out_dir.join(format!("{output}.bin")), file).expect("write mapping table");
    }

    let path = golden.join("unicode_combining.bin.gz");
    let data = gunzip(&path);
    let body = table_body(&data, b"SAG2COMB", 5, &path);
    let mut sparse = Vec::new();
    let mut count = 0_u32;
    for cp in 0..CODEPOINTS {
        let class = body[cp * 5 + 4];
        if class != 0 {
            sparse.extend_from_slice(&body[cp * 5..cp * 5 + 5]);
            count += 1;
        }
    }
    let mut file = Vec::with_capacity(4 + sparse.len());
    file.extend_from_slice(&count.to_le_bytes());
    file.extend_from_slice(&sparse);
    fs::write(out_dir.join("combining.bin"), file).expect("write combining table");

    let path = golden.join("unicode_categories.bin.gz");
    let data = gunzip(&path);
    let body = table_body(&data, b"SAG2CATE", 5, &path);
    let mut bitsets = [vec![0_u8; CODEPOINTS / 8], vec![0_u8; CODEPOINTS / 8], vec![0_u8; CODEPOINTS / 8], vec![0_u8; CODEPOINTS / 8]];
    for cp in 0..CODEPOINTS {
        let mask = body[cp * 5 + 4];
        for (bit, bitset) in bitsets.iter_mut().enumerate() {
            if mask & (1 << bit) != 0 {
                bitset[cp / 8] |= 1 << (cp % 8);
            }
        }
    }
    for (name, bitset) in ["letters", "numbers", "uppercase", "marks"]
        .iter()
        .zip(bitsets.iter())
    {
        fs::write(out_dir.join(format!("{name}.bin")), bitset).expect("write category bitset");
    }

    // Sparse tables keep the exporter's own header: the runtime walks them by
    // stride, so no generated Rust source is needed.
    for (name, magic) in SPARSE {
        let path = golden.join(format!("{name}.bin.gz"));
        let data = gunzip(&path);
        if data.len() < 8 || &data[..8] != magic {
            panic!("{} does not start with the documented magic tag", path.display());
        }
        let output = name.strip_prefix("unicode_").unwrap_or(name);
        fs::write(out_dir.join(format!("{output}.bin")), &data[8..])
            .expect("write sparse table");
    }
}
