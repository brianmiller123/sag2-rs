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
    // `word` can only live in the 64-entry window ending at that anchor.
    let window_start = low.saturating_sub(1) * ANCHOR_EVERY;
    let window_end = (window_start + ANCHOR_EVERY).min(FREQ_ENTRIES);
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

/// Longest dictionary word that mixes the CJK range with ASCII and therefore
/// spans the boundary `tokenize` splits on (design §4.2's equivalence premise).
pub fn crossing_word_at(suffix: &str) -> Option<&'static str> {
    let mut best: Option<&'static str> = None;
    for &word in CROSSING_WORDS {
        if suffix.starts_with(word) && best.is_none_or(|current| word.len() > current.len()) {
            best = Some(word);
        }
    }
    best
}
