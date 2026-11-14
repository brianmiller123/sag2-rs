//! UCA 9.0.0 collation, byte-exact with `pyuca` 1.2 (design §5).
//!
//! The reference sorts expansion terms with `COLLATOR.sort_key`
//! (`sag2.py:661-663`), and pyuca 1.2 is the only UCA 9.0.0 implementation the
//! goldens can come from (design §5.1), so this module is a line-for-line port
//! of `pyuca/collator.py` and `pyuca/trie.py` rather than an independent
//! reading of the standard (design §5.4 item 3):
//!
//! 1. `nfd` normalises the string - pyuca's first step;
//! 2. the lookup takes the longest prefix that has an entry, which is how the
//!    868 contractions (`ch`, `L·`, ...) work;
//! 3. a following non-starter that is not blocked from the matched segment may
//!    be folded into it (pyuca's unlabelled "handle non-starters" loop);
//! 4. code points with no entry take the UCA §7.1.3 implicit weights, with the
//!    CJK ranges and the `@implicitweights` directives as the base;
//! 5. the key is the four levels joined by `0` separators with zero weights
//!    dropped - pyuca's `sort_key_from_collation_elements`.
//!
//! `sort_key` returns the weights as `Vec<u32>`: Rust's `Vec` ordering is
//! Python's tuple ordering (lexicographic, shorter prefix first), so callers
//! compare keys directly and get pyuca's semantics.
//!
//! The data is compiled by `build.rs` from `data/allkeys-9.0.0.txt`
//! (`crates/sag2-core/data/PROVENANCE.md`); nothing is parsed at run time.

use crate::unicode::{combining_class, nfd};

/// Packed DUCET keys, ascending: `cp0 << 42 | cp1 << 21 | cp2` (absent = 0).
static KEYS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ducet_keys.bin"));
/// `u32le` start of each key's collation elements in [`CES`], plus a sentinel.
static OFFSETS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ducet_offsets.bin"));
/// `u16le` primary, secondary, tertiary weights - three per collation element.
static CES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ducet_ces.bin"));
/// `start:u32le, end:u32le, base:u32le` per `@implicitweights` directive.
static IMPLICIT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ducet_implicit.bin"));

const KEY_BYTES: usize = 8;
const OFFSET_BYTES: usize = 4;
const CE_BYTES: usize = 6;
const IMPLICIT_BYTES: usize = 12;
/// DUCET 9.0.0's longest entry is three code points (design §5.4 item 2).
const MAX_KEY: usize = 3;

/// The UCA 9.0.0 collator's flags for the CJK implicit weights, as
/// `Collator_9_0_0` sets them: 8.0.0 ideographs and ext-E on, 10.0.0 and
/// ext-F off (`pyuca/collator.py`).
const CJK_IDEOGRAPH_EXCEPTIONS: [u32; 12] = [
    0xFA0E, 0xFA0F, 0xFA11, 0xFA13, 0xFA14, 0xFA1F, 0xFA21, 0xFA23, 0xFA24, 0xFA27, 0xFA28, 0xFA29,
];

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("generated table slot"),
    )
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("generated table slot"),
    )
}

fn entry_count() -> usize {
    KEYS.len() / KEY_BYTES
}

fn key_at(index: usize) -> u64 {
    u64::from_le_bytes(
        KEYS[index * KEY_BYTES..(index + 1) * KEY_BYTES]
            .try_into()
            .expect("generated key slot"),
    )
}

/// `(start, length)` of the entry's collation elements in [`CES`].
fn element_range(index: usize) -> (u32, u32) {
    let start = read_u32(OFFSETS, index * OFFSET_BYTES);
    let end = read_u32(OFFSETS, (index + 1) * OFFSET_BYTES);
    (start, end - start)
}

/// The entry for exactly `packed`, if the DUCET lists it.
fn find_entry(packed: u64) -> Option<(u32, u32)> {
    let mut low = 0_usize;
    let mut high = entry_count();
    while low < high {
        let middle = (low + high) / 2;
        match key_at(middle).cmp(&packed) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return Some(element_range(middle)),
        }
    }
    None
}

fn pack(cps: &[u32]) -> u64 {
    let mut packed = 0_u64;
    for (index, &cp) in cps.iter().enumerate() {
        packed |= u64::from(cp) << (21 * (MAX_KEY - 1 - index));
    }
    packed
}

/// `Trie.find_prefix`: the longest prefix that carries a value.  `pyuca`
/// breaks traversal when the child link is missing, which is exactly "no entry
/// for any longer prefix".
fn find_prefix(cps: &[u32]) -> (usize, Option<(u32, u32)>) {
    let mut length = MAX_KEY.min(cps.len());
    while length > 0 {
        if let Some(found) = find_entry(pack(&cps[..length])) {
            return (length, Some(found));
        }
        length -= 1;
    }
    (0, None)
}

fn extend(elements: &mut Vec<[u32; 3]>, start: u32, count: u32) {
    for index in 0..count as usize {
        let base = (start as usize + index) * CE_BYTES;
        let mut element = [0_u32; 3];
        for (slot, offset) in element.iter_mut().zip((0..).step_by(2)) {
            *slot = u32::from(read_u16(CES, base + offset));
        }
        elements.push(element);
    }
}

/// UCA §7.1.3 implicit weights for a code point the DUCET does not list.
fn implicit_weight(cp: u32) -> [[u32; 3]; 2] {
    let (aaaa, bbbb) = implicit_weights(cp);
    [[aaaa, 0x0020, 0x0002], [bbbb, 0x0000, 0x0000]]
}

fn implicit_weights(cp: u32) -> (u32, u32) {
    let bbbb = (cp & 0x7FFF) | 0x8000;
    let unified = (0x4E00..=0x9FCC).contains(&cp)
        // `CJK_IDEOGRAPHS_8_0_0` (the 9.0.0 collator sets it).
        || (0x9FCD..=0x9FD5).contains(&cp)
        || CJK_IDEOGRAPH_EXCEPTIONS.contains(&cp);
    if unified {
        return (0xFB40 + (cp >> 15), bbbb);
    }
    let extension = (0x3400..=0x4DB5).contains(&cp)      // ext A (3.0)
        || (0x20000..=0x2A6D6).contains(&cp)             // ext B (3.1)
        || (0x2A700..=0x2B734).contains(&cp)             // ext C (5.2)
        || (0x2B740..=0x2B81D).contains(&cp)             // ext D (6.0)
        || (0x2B820..=0x2CEAF).contains(&cp); // ext E (8.0)
    if extension {
        // pyuca's 8.0.0 test workaround: one ext-E code point starts at FBC0.
        let base = if cp == 0x2CEA2 { 0xFBC0 } else { 0xFB80 };
        return (base + (cp >> 15), bbbb);
    }
    for index in 0..IMPLICIT.len() / IMPLICIT_BYTES {
        let base = index * IMPLICIT_BYTES;
        let start = read_u32(IMPLICIT, base);
        let end = read_u32(IMPLICIT, base + 4);
        if start <= cp && cp <= end {
            return (read_u32(IMPLICIT, base + 8), (cp - start) | 0x8000);
        }
    }
    (0xFBC0 + (cp >> 15), bbbb)
}

/// `BaseCollator.collation_elements`: the collation elements of a normalised
/// string, in order, with the non-starter folding pyuca performs.
fn collation_elements(text: &str) -> Vec<[u32; 3]> {
    let mut lookup: Vec<u32> = text.chars().map(|character| character as u32).collect();
    let mut index = 0_usize;
    let mut elements: Vec<[u32; 3]> = Vec::new();
    while index < lookup.len() {
        let (consumed, mut value) = find_prefix(&lookup[index..]);
        let segment_start = index;

        // "handle non-starters": `C` joins `S` when the two-code-point probe is
        // itself an entry, the classes are non-zero and strictly descending,
        // and `C` is not blocked by an earlier mark of the same class.
        let mut scan = index + consumed;
        let mut last_class: Option<u8> = None;
        while scan < lookup.len() {
            let class = combining_class(lookup[scan]);
            if class == 0 || last_class == Some(class) {
                break;
            }
            last_class = Some(class);
            let mut probe: Vec<u32> = lookup[segment_start..segment_start + consumed].to_vec();
            probe.push(lookup[scan]);
            if probe.len() <= MAX_KEY {
                if let Some(found) = find_entry(pack(&probe)) {
                    lookup.remove(scan);
                    value = Some(found);
                    break;
                }
            }
            scan += 1;
        }

        index += consumed;
        match value {
            Some((start, count)) if count > 0 => extend(&mut elements, start, count),
            _ => {
                if index >= lookup.len() {
                    // pyuca would raise `IndexError` here.  No DUCET 9.0.0
                    // entry has an empty element list, so a zero-length entry
                    // that consumes the whole key is a corrupt-table invariant.
                    panic!("DUCET entry with no weights consumed the remaining key");
                }
                let cp = lookup.remove(index);
                elements.extend_from_slice(&implicit_weight(cp));
            }
        }
    }
    elements
}

/// `BaseCollator.sort_key`: NFD, collation elements, then the layered key.
///
/// The four levels are separated by `0` and zero weights are dropped, so an
/// empty string has exactly three separators (design §5.5).
pub fn sort_key(text: &str) -> Vec<u32> {
    let elements = collation_elements(&nfd(text));
    let mut key: Vec<u32> = Vec::with_capacity(elements.len() * 3 + 3);
    for level in 0..4 {
        if level > 0 {
            key.push(0);
        }
        for element in &elements {
            // `len(element) > level` in pyuca: the elements carry three
            // weights, so the fourth level is always empty.
            if let Some(&weight) = element.get(level) {
                if weight != 0 {
                    key.push(weight);
                }
            }
        }
    }
    key
}

#[cfg(test)]
mod tests {
    use super::sort_key;

    /// Design §5.5: the empty string's key is only the level separators.
    #[test]
    fn empty_string_is_three_separators() {
        assert_eq!(sort_key(""), vec![0, 0, 0]);
    }

    /// Design §5.1 S4: the case pair differs only after the first two levels.
    #[test]
    fn case_pair_keeps_the_first_two_levels() {
        let lower = sort_key("alice");
        let upper = sort_key("Alice");
        assert!(lower < upper, "{lower:?} < {upper:?}");
        let prefix = |key: &[u32]| {
            let mut ends = key.iter().enumerate().filter(|(_, weight)| **weight == 0);
            let second = ends.nth(1).map(|(index, _)| index).expect("two separators");
            key[..second].to_vec()
        };
        assert_eq!(prefix(&lower), prefix(&upper));
    }

    /// Design §5.5: composed and decomposed accents are the same term, and the
    /// accent itself is a secondary difference (`cafe < café`).
    #[test]
    fn accents_are_secondary_and_composition_insensitive() {
        assert_eq!(sort_key("café"), sort_key("cafe\u{0301}"));
        assert!(sort_key("cafe") < sort_key("café"));
    }

    /// UAX #15 canonical ordering: mark order in the input does not matter.
    #[test]
    fn combining_marks_are_canonically_ordered() {
        assert_eq!(sort_key("e\u{0301}\u{0327}"), sort_key("e\u{0327}\u{0301}"));
    }

    /// Design §5.5: CJK ideographs take the 0xFB40 implicit base, and the key
    /// head is the published fact.
    #[test]
    fn cjk_implicit_weights() {
        assert_eq!(sort_key("一")[..7], [64320, 52736, 0, 32, 0, 2, 0]);
        assert!(sort_key("一") < sort_key("二"));
    }

    /// Design §5.5: astral and punctuation code points must not panic.
    #[test]
    fn adversarial_code_points_produce_keys() {
        for text in ["\u{20000}", "👍", "👨‍👩‍👧‍👦", "%", "_", ".", "-", "\u{0301}"]
        {
            assert!(!sort_key(text).is_empty(), "{text:?}");
        }
    }
}
