//! Unicode tables generated from the Python 3.12.11 exporter output (design §4.5).
//!
//! Every table is a compile-time `include_bytes!` of data `build.rs` derived
//! from `golden/unicode_*.bin.gz`; nothing is parsed at run time.
//!
//! * `lower.bin` / `casefold.bin`: `u32 count` then `count` 17-byte records
//!   `cp:u32le, len:u8, cps:[u32le;3]`, holding only non-identity mappings and
//!   sorted by `cp`.
//! * `combining.bin`: `u32 count` then `count` 5-byte records
//!   `cp:u32le, class:u8`, holding only non-zero combining classes.
//! * `letters.bin` / `numbers.bin` / `uppercase.bin` / `marks.bin`: one bit per
//!   code point (LSB first) for `\p{L}`, `\p{N}`, `\p{Lu}` and `\p{M}`.
//! * `nfkd.bin`: `u32 count, u8 width` then `count` records
//!   `cp:u32le, len:u8, cps:[u32le;width]`, the fully decomposed compatibility
//!   mapping of every code point that changes (Hangul is algorithmic).
//! * `compose.bin`: `u32 count` then `count` records
//!   `a:u32le, b:u32le, composite:u32le`, sorted by `(a, b)`; the canonical
//!   composition pairs Python's `NFC` applies (Hangul is algorithmic).
//!
//! `nfkc` is computed from Python 15.0.0 data rather than from a crate:
//! `unicode-normalization` tracks Unicode 18.0.0 while CPython 3.12.11 ships
//! 15.0.0, and the difference is observable in the reference corpus.  Gate D
//! requires zero differences over all 0x110000 code points.

static LOWER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/lower.bin"));
static CASEFOLD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/casefold.bin"));
static COMBINING: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/combining.bin"));
static LETTERS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/letters.bin"));
static NUMBERS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/numbers.bin"));
static UPPERCASE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/uppercase.bin"));
static MARKS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/marks.bin"));
static NFKD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nfkd.bin"));
static COMPOSE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compose.bin"));

const MAPPING_RECORD: usize = 17;
const COMBINING_RECORD: usize = 5;
const COMPOSE_RECORD: usize = 12;

const HANGUL_S_BASE: u32 = 0xAC00;
const HANGUL_L_BASE: u32 = 0x1100;
const HANGUL_V_BASE: u32 = 0x1161;
const HANGUL_T_BASE: u32 = 0x11A7;
const HANGUL_L_COUNT: u32 = 19;
const HANGUL_V_COUNT: u32 = 21;
const HANGUL_T_COUNT: u32 = 28;
const HANGUL_N_COUNT: u32 = HANGUL_V_COUNT * HANGUL_T_COUNT;
const HANGUL_S_COUNT: u32 = HANGUL_L_COUNT * HANGUL_N_COUNT;

fn table_count(table: &[u8]) -> usize {
    u32::from_le_bytes(table[..4].try_into().expect("table count")) as usize
}

fn mapping(table: &[u8], cp: u32) -> Option<([u32; 3], usize)> {
    let count = table_count(table);
    let records = &table[4..];
    let mut low = 0_usize;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let base = middle * MAPPING_RECORD;
        let found = u32::from_le_bytes(
            records[base..base + 4]
                .try_into()
                .expect("mapping record cp"),
        );
        match found.cmp(&cp) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => {
                let length = records[base + 4] as usize;
                let mut cps = [0_u32; 3];
                for (slot, offset) in cps.iter_mut().take(length).zip((0..).step_by(4)) {
                    let at = base + 5 + offset;
                    *slot = u32::from_le_bytes(
                        records[at..at + 4].try_into().expect("mapping record cps"),
                    );
                }
                return Some((cps, length));
            }
        }
    }
    None
}

fn apply(table: &[u8], text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match mapping(table, character as u32) {
            Some((cps, length)) => {
                for &cp in &cps[..length] {
                    out.push(char::from_u32(cp).expect("generated mapping is a scalar value"));
                }
            }
            None => out.push(character),
        }
    }
    out
}

/// `str.lower()` (design §4.1 rule 1).
pub fn lower(text: &str) -> String {
    apply(LOWER, text)
}

/// `str.casefold()` (design §4.5; used by `diversity_tokens`).
pub fn casefold(text: &str) -> String {
    apply(CASEFOLD, text)
}

/// `unicodedata.combining(chr(cp))`.
pub fn combining_class(cp: u32) -> u8 {
    if cp >= 0x11_0000 {
        return 0;
    }
    let count = table_count(COMBINING);
    let records = &COMBINING[4..];
    let mut low = 0_usize;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let base = middle * COMBINING_RECORD;
        let found = u32::from_le_bytes(
            records[base..base + 4]
                .try_into()
                .expect("combining record cp"),
        );
        match found.cmp(&cp) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return records[base + 4],
        }
    }
    0
}

fn bit_is_set(bitset: &[u8], cp: u32) -> bool {
    let cp = cp as usize;
    match bitset.get(cp / 8) {
        Some(byte) => byte & (1 << (cp % 8)) != 0,
        None => false,
    }
}

/// `\p{L}` under the Python `regex` module's Unicode 15.0.0 tables.
pub fn is_letter(cp: u32) -> bool {
    bit_is_set(LETTERS, cp)
}

/// `\p{N}` under the Python `regex` module's Unicode 15.0.0 tables.
pub fn is_number(cp: u32) -> bool {
    bit_is_set(NUMBERS, cp)
}

/// `\p{Lu}` under the Python `regex` module's Unicode 15.0.0 tables.
pub fn is_uppercase(cp: u32) -> bool {
    bit_is_set(UPPERCASE, cp)
}

/// `\p{M}` under the Python `regex` module's Unicode 15.0.0 tables.
pub fn is_mark(cp: u32) -> bool {
    bit_is_set(MARKS, cp)
}

/// One record of `nfkd.bin` for `cp`, plus the table's record width.
fn nfkd_record(cp: u32) -> Option<(&'static [u8], usize)> {
    let count = table_count(NFKD);
    let width = NFKD[4] as usize;
    let stride = 5 + 4 * width;
    let records = &NFKD[5..];
    let mut low = 0_usize;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let base = middle * stride;
        let found = u32::from_le_bytes(
            records[base..base + 4]
                .try_into()
                .expect("nfkd record cp"),
        );
        match found.cmp(&cp) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return Some((&records[base..base + stride], width)),
        }
    }
    None
}

fn compose_hangul(left: u32, right: u32) -> Option<u32> {
    if (HANGUL_L_BASE..HANGUL_L_BASE + HANGUL_L_COUNT).contains(&left)
        && (HANGUL_V_BASE..HANGUL_V_BASE + HANGUL_V_COUNT).contains(&right)
    {
        return Some(
            HANGUL_S_BASE
                + ((left - HANGUL_L_BASE) * HANGUL_V_COUNT + (right - HANGUL_V_BASE))
                    * HANGUL_T_COUNT,
        );
    }
    if (HANGUL_S_BASE..HANGUL_S_BASE + HANGUL_S_COUNT).contains(&left)
        && (left - HANGUL_S_BASE) % HANGUL_T_COUNT == 0
        && (HANGUL_T_BASE + 1..HANGUL_T_BASE + HANGUL_T_COUNT).contains(&right)
    {
        return Some(left + (right - HANGUL_T_BASE));
    }
    None
}

fn compose_pair(left: u32, right: u32) -> Option<u32> {
    if let Some(composite) = compose_hangul(left, right) {
        return Some(composite);
    }
    let count = table_count(COMPOSE);
    let records = &COMPOSE[4..];
    let mut low = 0_usize;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let base = middle * COMPOSE_RECORD;
        let first = u32::from_le_bytes(
            records[base..base + 4]
                .try_into()
                .expect("compose record a"),
        );
        let second = u32::from_le_bytes(
            records[base + 4..base + 8]
                .try_into()
                .expect("compose record b"),
        );
        match (first, second).cmp(&(left, right)) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => {
                return Some(u32::from_le_bytes(
                    records[base + 8..base + 12]
                        .try_into()
                        .expect("compose record composite"),
                ))
            }
        }
    }
    None
}

fn decompose_char(character: char, out: &mut Vec<char>) {
    let cp = character as u32;
    if (HANGUL_S_BASE..HANGUL_S_BASE + HANGUL_S_COUNT).contains(&cp) {
        let index = cp - HANGUL_S_BASE;
        out.push(char::from_u32(HANGUL_L_BASE + index / HANGUL_N_COUNT).expect("hangul lead"));
        out.push(
            char::from_u32(HANGUL_V_BASE + (index % HANGUL_N_COUNT) / HANGUL_T_COUNT)
                .expect("hangul vowel"),
        );
        let trailing = index % HANGUL_T_COUNT;
        if trailing != 0 {
            out.push(char::from_u32(HANGUL_T_BASE + trailing).expect("hangul trailing"));
        }
        return;
    }
    match nfkd_record(cp) {
        Some((record, width)) => {
            let length = record[4] as usize;
            for index in 0..length {
                let at = 5 + index * 4;
                let mapped = u32::from_le_bytes(
                    record[at..at + 4].try_into().expect("nfkd mapping cp"),
                );
                out.push(char::from_u32(mapped).expect("nfkd mapping is a scalar value"));
            }
            debug_assert!(length <= width);
        }
        None => out.push(character),
    }
}

/// Canonical ordering (UAX #15): stable movement of combining marks by class.
fn canonical_order(characters: &mut [char]) {
    for index in 1..characters.len() {
        let class = combining_class(characters[index] as u32);
        if class == 0 {
            continue;
        }
        let mut position = index;
        while position > 0 {
            let previous = combining_class(characters[position - 1] as u32);
            if previous == 0 || previous <= class {
                break;
            }
            characters.swap(position - 1, position);
            position -= 1;
        }
    }
}

fn compose(characters: &[char]) -> String {
    let mut out: Vec<char> = Vec::with_capacity(characters.len());
    let mut starter: Option<usize> = None;
    let mut last_class: Option<u8> = None;
    for &character in characters {
        let class = combining_class(character as u32);
        if let Some(index) = starter {
            let blocked = last_class.is_some_and(|last| last >= class);
            if !blocked {
                if let Some(composite) = compose_pair(out[index] as u32, character as u32) {
                    out[index] =
                        char::from_u32(composite).expect("composition is a scalar value");
                    continue;
                }
            }
            if class == 0 {
                starter = Some(out.len());
                last_class = None;
                out.push(character);
                continue;
            }
            last_class = Some(class);
            out.push(character);
            continue;
        }
        if class == 0 {
            starter = Some(out.len());
            last_class = None;
        }
        out.push(character);
    }
    out.into_iter().collect()
}

/// `unicodedata.normalize("NFKC", text)`.
pub fn nfkc(text: &str) -> String {
    let mut characters: Vec<char> = Vec::with_capacity(text.len());
    for character in text.chars() {
        decompose_char(character, &mut characters);
    }
    canonical_order(&mut characters);
    compose(&characters)
}
