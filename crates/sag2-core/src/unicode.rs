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
//! * `cased.bin` / `case_ignorable.bin`: the two properties `str.lower()` uses
//!   for Greek `Final_Sigma`, measured from `str.lower()` itself rather than
//!   taken from a table whose Unicode version could differ from CPython's.
//! * `nfkd.bin`: `u32 count, u8 width` then `count` records
//!   `cp:u32le, len:u8, cps:[u32le;width]`, the fully decomposed compatibility
//!   mapping of every code point that changes (Hangul is algorithmic).
//! * `nfd.bin`: the same layout for the *canonical* decomposition pyuca's
//!   `sort_key` starts with (design §5.4 step 1).
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
static CASED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cased.bin"));
static CASE_IGNORABLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/case_ignorable.bin"));
static NFKD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nfkd.bin"));
static NFD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nfd.bin"));
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

fn push_mapping(table: &[u8], character: char, out: &mut String) {
    match mapping(table, character as u32) {
        Some((cps, length)) => {
            for &cp in &cps[..length] {
                out.push(char::from_u32(cp).expect("generated mapping is a scalar value"));
            }
        }
        None => out.push(character),
    }
}

fn apply(table: &[u8], text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        push_mapping(table, character, &mut out);
    }
    out
}

/// `str.lower()` (design §4.1 rule 1).
///
/// The per-code-point table covers every mapping except Greek `Final_Sigma`,
/// the one context-sensitive rule `str.lower()` applies: `Σ` lowercases to `ς`
/// when the preceding significant character is cased and the following one is
/// not, "significant" skipping `Case_Ignorable` (`SpecialCasing.txt`).
pub fn lower(text: &str) -> String {
    if !text.contains('\u{3A3}') {
        return apply(LOWER, text);
    }
    let characters: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (index, &character) in characters.iter().enumerate() {
        if character == '\u{3A3}' && final_sigma(&characters, index) {
            out.push('\u{3C2}');
        } else {
            push_mapping(LOWER, character, &mut out);
        }
    }
    out
}

/// Whether the `Σ` at `index` is word-final for `str.lower()`.
fn final_sigma(characters: &[char], index: usize) -> bool {
    let before = characters[..index]
        .iter()
        .rev()
        .find(|character| !is_case_ignorable(**character as u32))
        .is_some_and(|character| is_cased(*character as u32));
    if !before {
        return false;
    }
    match characters[index + 1..]
        .iter()
        .find(|character| !is_case_ignorable(**character as u32))
    {
        None => true,
        Some(character) => !is_cased(*character as u32),
    }
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

/// Unicode `Cased`, as CPython's `str.lower()` applies it to `Final_Sigma`.
pub fn is_cased(cp: u32) -> bool {
    bit_is_set(CASED, cp)
}

/// Unicode `Case_Ignorable`, as CPython's `str.lower()` applies it to
/// `Final_Sigma`.
pub fn is_case_ignorable(cp: u32) -> bool {
    bit_is_set(CASE_IGNORABLE, cp)
}

/// `str.isspace()`, which is also what Python's `\s` matches.
///
/// The four file/group/record/unit separators count as whitespace there;
/// Rust's `White_Space` property (behind `char::is_whitespace`) excludes them.
pub fn is_space(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

/// `\b`'s "word character" set (`_shared_word` in the reference modules).
///
/// Python's `regex` module and the `regex` crate both follow UTS#18 here:
/// letters, marks, decimal numbers, connector punctuation and join controls.
/// Join controls (U+200C/U+200D) are the only members that do not appear in
/// the four generated tables, so they are spelled out.
pub fn is_word(cp: u32) -> bool {
    is_letter(cp) || is_mark(cp) || is_number(cp) || cp == 0x5F || cp == 0x200C || cp == 0x200D
}

/// One record of a sparse decomposition table for `cp`, plus the table's
/// record width.  `table` is `nfd.bin` or `nfkd.bin`, which share the layout.
fn sparse_record(table: &'static [u8], cp: u32) -> Option<(&'static [u8], usize)> {
    let count = table_count(table);
    let width = table[4] as usize;
    let stride = 5 + 4 * width;
    let records = &table[5..];
    let mut low = 0_usize;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let base = middle * stride;
        let found = u32::from_le_bytes(records[base..base + 4].try_into().expect("nfkd record cp"));
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

fn decompose_char(table: &'static [u8], character: char, out: &mut Vec<char>) {
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
    match sparse_record(table, cp) {
        Some((record, width)) => {
            let length = record[4] as usize;
            for index in 0..length {
                let at = 5 + index * 4;
                let mapped =
                    u32::from_le_bytes(record[at..at + 4].try_into().expect("decomposition cp"));
                out.push(char::from_u32(mapped).expect("decomposition is a scalar value"));
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
                    out[index] = char::from_u32(composite).expect("composition is a scalar value");
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

/// `unicodedata.normalize("NFD", text)`.
///
/// pyuca's `sort_key` normalises with NFD before the trie lookup (design §5.4
/// step 1); the canonical table holds only the non-identity mappings, the
/// Hangul syllables decompose algorithmically and the ordering is UAX #15.
pub fn nfd(text: &str) -> String {
    let mut characters: Vec<char> = Vec::with_capacity(text.len());
    for character in text.chars() {
        decompose_char(NFD, character, &mut characters);
    }
    canonical_order(&mut characters);
    characters.into_iter().collect()
}

/// `unicodedata.normalize("NFKD", text)`.
pub fn nfkd(text: &str) -> String {
    let mut characters: Vec<char> = Vec::with_capacity(text.len());
    for character in text.chars() {
        decompose_char(NFKD, character, &mut characters);
    }
    canonical_order(&mut characters);
    characters.into_iter().collect()
}

/// `unicodedata.normalize("NFKC", text)`.
pub fn nfkc(text: &str) -> String {
    let mut characters: Vec<char> = Vec::with_capacity(text.len());
    for character in text.chars() {
        decompose_char(NFKD, character, &mut characters);
    }
    canonical_order(&mut characters);
    compose(&characters)
}

#[cfg(test)]
mod tests {
    use super::lower;

    /// Every vector is `str.lower()`'s output on CPython 3.12.11 (Unicode
    /// 15.0.0); `Final_Sigma` is the only context-sensitive entry.
    #[test]
    fn lower_applies_the_final_sigma_rule() {
        for (text, expected) in [
            ("ΟΔΟΣ", "οδος"),
            ("ΣΣ", "σς"),
            ("Σ", "σ"),
            ("ΑΣ.", "ας."),
            ("ΑΣΣ", "ασς"),
            ("ΑΣ\u{0301}", "ας\u{0301}"),
            ("ΑΣ'", "ας'"),
            ("ΑΣΑ", "ασα"),
            ("ΑΣΣΑ", "ασσα"),
            ("Η ΣΟΦΙΑ ΤΗΣ ΑΘΗΝΑΣ", "η σοφια της αθηνας"),
            ("İSTANBUL", "i\u{0307}stanbul"),
            ("ß", "ß"),
        ] {
            assert_eq!(lower(text), expected, "{text:?}");
        }
    }
}
