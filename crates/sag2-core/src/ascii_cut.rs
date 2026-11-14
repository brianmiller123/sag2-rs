//! The connected-ASCII half of the jieba cut (design §4.2).
//!
//! Python replaces `jieba.finalseg.re_skip` with
//! `([a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*%?)` (`sag2.py:45-48`), so every stretch
//! of ASCII that jieba's DAG cannot cover leaves `finalseg` as one atomic run:
//! a leftmost scan that stops at the first `%` (the `%` stays attached to the
//! run before it) and never crosses one.
//!
//! `jieba-rs` 0.7.4 ships finalseg with the *unpatched* skip pattern, which is
//! why this module reproduces the exact Python behaviour for pure-ASCII runs:
//! the DAG over the dictionary's ASCII entries, then the patched finalseg.  The
//! reference corpus has no per-character difference here; Gate D proves it.

use crate::dict;

/// `__cut_DAG` for one pure-ASCII run, including the patched `finalseg`.
pub(crate) fn cut_run(run: &str, out: &mut Vec<String>) {
    debug_assert!(run.is_ascii(), "imported runs are pure ASCII");
    let bytes = run.as_bytes();
    let length = bytes.len();
    if length == 0 {
        return;
    }

    // DAG[x] = ascending exclusive end offsets of dictionary words starting at
    // x, or the single-character fallback (Python `get_DAG`).
    let mut ends: Vec<Vec<usize>> = vec![Vec::new(); length];
    for start in 0..length {
        let suffix = &bytes[start..];
        for &(word, _) in dict::ASCII_WORDS {
            if suffix.starts_with(word.as_bytes()) {
                ends[start].push(start + word.len());
            }
        }
        if ends[start].is_empty() {
            ends[start].push(start + 1);
        }
    }

    // `calc`: `max((score, end))`, so on an equal score the larger end wins.
    let log_total = (dict::DICT_TOTAL as f64).ln();
    let mut route = vec![(0_f64, 0_usize); length + 1];
    for start in (0..length).rev() {
        let mut best = f64::NEG_INFINITY;
        let mut best_end = start + 1;
        for &end in &ends[start] {
            let frequency = dict::ascii_freq(&run[start..end]).unwrap_or(1);
            let score = (frequency as f64).ln() - log_total + route[end].0;
            if score > best || (score == best && end > best_end) {
                best = score;
                best_end = end;
            }
        }
        route[start] = (best, best_end);
    }

    let mut buffer = String::new();
    let mut position = 0_usize;
    while position < length {
        let end = route[position].1;
        if end - position == 1 {
            buffer.push_str(&run[position..end]);
        } else {
            flush(&mut buffer, out);
            out.push(run[position..end].to_owned());
        }
        position = end;
    }
    flush(&mut buffer, out);
}

/// The buffer flush of `__cut_DAG`.
fn flush(buffer: &mut String, out: &mut Vec<String>) {
    if buffer.is_empty() {
        return;
    }
    if buffer.len() == 1 {
        out.push(std::mem::take(buffer));
    } else if dict::contains(buffer) {
        out.extend(buffer.chars().map(|character| character.to_string()));
        buffer.clear();
    } else {
        finalseg(buffer, out);
        buffer.clear();
    }
}

/// `finalseg.cut` for a pure-ASCII buffer under the patched skip pattern.
fn finalseg(buffer: &str, out: &mut Vec<String>) {
    let bytes = buffer.as_bytes();
    let length = bytes.len();
    let mut position = 0_usize;
    while position < length {
        if is_alphanumeric(bytes[position]) {
            let start = position;
            position += 1;
            while position < length && is_alphanumeric(bytes[position]) {
                position += 1;
            }
            loop {
                let connector = position < length && is_connector(bytes[position]);
                let next_is_alphanumeric =
                    position + 1 < length && is_alphanumeric(bytes[position + 1]);
                if !(connector && next_is_alphanumeric) {
                    break;
                }
                position += 2;
                while position < length && is_alphanumeric(bytes[position]) {
                    position += 1;
                }
            }
            if position < length && bytes[position] == b'%' {
                position += 1;
            }
            out.push(buffer[start..position].to_owned());
        } else {
            // `re.split` yields the text between matches whole.
            let start = position;
            while position < length && !is_alphanumeric(bytes[position]) {
                position += 1;
            }
            out.push(buffer[start..position].to_owned());
        }
    }
}

fn is_alphanumeric(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
}

fn is_connector(byte: u8) -> bool {
    matches!(byte, b'.' | b'_' | b'-')
}
