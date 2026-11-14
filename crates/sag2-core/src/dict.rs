//! The §4.3 deterministic FREQ table.
//!
//! `build.rs` extracts the codepoint-length <= 3 dictionary entries into a
//! sorted, NUL-separated blob with one `u32` anchor every 64 entries.  Rule 5
//! of design §4.1 only asks whether a 2- or 3-codepoint substring is in jieba's
//! `FREQ` with a positive frequency, so membership is all this table stores.
//!
//! Lookup is allocation-free: binary search over ~4 000 anchors, then a linear
//! scan of at most 64 NUL-terminated entries.

include!(concat!(env!("OUT_DIR"), "/dict_consts.rs"));

static FREQ_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/freq.bin"));
static FREQ_ANCHORS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/freq_anchors.bin"));

/// The entry starting at `offset` and the offset of the next entry.
fn entry(offset: usize) -> (&'static [u8], usize) {
    let tail = &FREQ_BLOB[offset..];
    let length = tail
        .iter()
        .position(|byte| *byte == 0)
        .expect("freq blob entries are NUL terminated");
    (&tail[..length], offset + length + 1)
}

fn anchor(index: usize) -> usize {
    let at = index * 4;
    u32::from_le_bytes(
        FREQ_ANCHORS[at..at + 4]
            .try_into()
            .expect("freq anchor record"),
    ) as usize
}

fn contains_bytes(word: &[u8]) -> bool {
    let anchor_count = FREQ_ENTRIES.div_ceil(ANCHOR_EVERY);
    // First anchor whose entry sorts at or after `word`.
    let mut low = 0_usize;
    let mut high = anchor_count;
    while low < high {
        let middle = (low + high) / 2;
        if entry(anchor(middle)).0 < word {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    // The entry is in the window ending at `low` (when it lands before that
    // anchor's own entry) or in the window starting at `low` (when it *is*
    // that anchor's entry).  Both have to be scanned: an anchor's own entry is
    // the first one the binary search reports as `>= word`, so a
    // previous-window-only scan drops every 64th entry.
    let window_start = low.saturating_sub(1) * ANCHOR_EVERY;
    let window_end = ((low + 1) * ANCHOR_EVERY).min(FREQ_ENTRIES);
    let mut offset = anchor(window_start / ANCHOR_EVERY);
    for _ in window_start..window_end {
        let (candidate, next) = entry(offset);
        if candidate == word {
            return true;
        }
        offset = next;
    }
    false
}

/// `TOKENIZER.FREQ.get(word, 0) > 0` for one word.
pub fn contains(word: &str) -> bool {
    contains_bytes(word.as_bytes())
}

/// Every entry the table stores, in its own byte order (`str` order).
///
/// Gate D compares this set against the reference's short `TOKENIZER.FREQ`
/// keys, so a change to the table generation cannot hide: the whole 257k-entry
/// set is checked, not a tokenizer sample.  Membership *lookups* are covered
/// separately by that gate's anchor-position check.
pub fn short_entries() -> Vec<&'static str> {
    let mut entries = Vec::with_capacity(FREQ_ENTRIES);
    let mut offset = 0_usize;
    for _ in 0..FREQ_ENTRIES {
        let (candidate, next) = entry(offset);
        entries.push(std::str::from_utf8(candidate).expect("dictionary entries are UTF-8"));
        offset = next;
    }
    entries
}

/// `FREQ_ENTRIES`: how many short keys the table holds.
pub fn short_entry_count() -> usize {
    FREQ_ENTRIES
}

/// `ANCHOR_EVERY`: the stride of the binary search's anchor records.
pub fn anchor_every() -> usize {
    ANCHOR_EVERY
}

/// `FREQ_MAX_CHARS`: the codepoint length cap rule 5 looks up.
pub fn max_key_chars() -> usize {
    FREQ_MAX_CHARS
}

/// Same question for a 2- or 3-codepoint substring, without allocating.
pub fn contains_chars(chars: &[char]) -> bool {
    let mut buffer = [0_u8; 16];
    let mut length = 0_usize;
    for character in chars {
        length += character.encode_utf8(&mut buffer[length..]).len();
    }
    contains_bytes(&buffer[..length])
}

/// `FREQ[word] or 1` for the pure-ASCII dictionary entries the Python DAG can
/// match; `None` means "not a dictionary word", which is frequency 1 there.
pub fn ascii_freq(word: &str) -> Option<u32> {
    ASCII_WORDS
        .iter()
        .find(|(candidate, _)| *candidate == word)
        .map(|(_, value)| *value)
}

#[cfg(test)]
mod tests {
    use super::contains;

    /// Rule 5 asks `FREQ.get(word, 0) > 0` for every 2- and 3-codepoint
    /// substring, so `contains` must agree with the dictionary for *every*
    /// entry the table stores - including the ones that land exactly on an
    /// anchor boundary (`FREQ_ENTRIES / ANCHOR_EVERY` ~ 4 000 of them).
    #[test]
    fn every_stored_short_entry_is_found() {
        let mut missing = Vec::new();
        let mut checked = 0_usize;
        for line in include_str!("../data/dict.txt").lines() {
            let mut fields = line.split_whitespace();
            let (Some(word), Some(frequency)) = (fields.next(), fields.next()) else {
                continue;
            };
            if word.chars().count() > 3 || frequency.parse::<u32>().unwrap_or(0) == 0 {
                continue;
            }
            checked += 1;
            if !contains(word) {
                missing.push(word);
            }
        }
        assert!(checked > 100_000, "only {checked} entries were checked");
        assert!(
            missing.is_empty(),
            "{} entries are missing, e.g. {:?}",
            missing.len(),
            &missing[..missing.len().min(5)]
        );
    }
}
