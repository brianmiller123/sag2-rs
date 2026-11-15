//! `sag2.py:69-86` `_cached_tokenize` - the §4 tokenizer.
//!
//! Steps, in the reference order (design §4.1):
//!
//! 1. `NFKC(text).lower()`
//! 2. jieba `cut(HMM=True)` over the certified dictionary; its `re_han` blocks
//!    are split at the ASCII/CJK boundary so the ASCII half can use the patched
//!    finalseg (`ascii_cut`) and the CJK half the crate's DAG + HMM.
//! 3. `[._-]` extra splits (length >= 2 and an ASCII lowercase letter)
//! 4. 2- and 3-codepoint substrings whose `FREQ` is positive
//! 5. the word itself
//! 6. `HAS_CONTENT` and `STOPWORDS` filters

use std::sync::LazyLock;

use jieba_rs::Jieba;

use crate::{ascii_cut, dict, unicode};

/// The certified jieba 0.42.1 dictionary, byte for byte (design §4.3).
static DICTIONARY: &[u8] = include_bytes!("../data/dict.txt");

/// `\p{L}` or `\p{N}` - `sag2.py:61` `HAS_CONTENT`.
fn has_content(word: &str) -> bool {
    word.chars().any(|character| {
        unicode::is_letter(character as u32) || unicode::is_number(character as u32)
    })
}

/// The 61 literal stopwords of `sag2.py:40-44`, sorted for `binary_search`.
static STOPWORDS: &[&str] = &[
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "一个",
    "一种",
    "与",
    "为什么",
    "了",
    "什么",
    "他",
    "你",
    "吗",
    "吧",
    "呢",
    "和",
    "哪",
    "哪个",
    "哪些",
    "在",
    "地",
    "如何",
    "它",
    "得",
    "怎么",
    "我",
    "或",
    "是",
    "是否",
    "的",
    "请",
    "这",
    "那",
];

static JIEBA: LazyLock<Jieba> = LazyLock::new(|| {
    let mut dictionary: &[u8] = DICTIONARY;
    Jieba::with_dict(&mut dictionary)
        .expect("crates/sag2-core/data/dict.txt must be a valid jieba dictionary")
});

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    /// `[\u4E00-\u9FD5]` - the CJK range `re_han_default` covers.
    Cjk,
    /// `[a-zA-Z0-9+#&._%-]` - the rest of `re_han_default`.
    Ascii,
    /// Everything else.
    Other,
}

fn class(character: char) -> Class {
    let cp = character as u32;
    if (0x4E00..=0x9FD5).contains(&cp) {
        Class::Cjk
    } else if character.is_ascii_alphanumeric()
        || matches!(character, '+' | '#' | '&' | '.' | '_' | '%' | '-')
    {
        Class::Ascii
    } else {
        Class::Other
    }
}

/// Tokenize `text` exactly like `sag2.tokenize` (design §4.1, §4.2).
///
/// This is [`crate::analysis::tokenize_with`] with
/// [`Profile::Sag2ZhV1`](crate::analysis::Profile::Sag2ZhV1), the certified
/// reference pipeline; it is the one entry point whose bytes the goldens pin.
pub fn tokenize(text: &str) -> Vec<String> {
    tokenize_default(text)
}

/// `NFKC` -> lower, then [`tokenize_normalised`].
///
/// This is the §4 pipeline a profile enters: [`Profile::Sag2ZhV1`] stops after
/// these two steps, and a multilingual profile runs its own
/// [`crate::analysis::normalize`] first and then calls
/// [`tokenize_normalised`], so no input is normalised twice (C4).
///
/// [`Profile::Sag2ZhV1`]: crate::analysis::Profile::Sag2ZhV1
pub(crate) fn tokenize_default(text: &str) -> Vec<String> {
    tokenize_normalised(&unicode::lower(&unicode::nfkc(text)))
}

/// The §4 pipeline itself, over text the caller has already normalised
/// (design §4.1-§4.3).
pub(crate) fn tokenize_normalised(normalised: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut block_start = 0_usize;
    let mut block_inside: Option<bool> = None;
    // `re_han_default` blocks: maximal runs of CJK or interface characters.
    for (index, character) in normalised.char_indices() {
        let inside = in_han_class(character);
        match block_inside {
            Some(state) if state == inside => {}
            Some(state) => {
                emit_block(&normalised[block_start..index], state, &mut words);
                block_start = index;
                block_inside = Some(inside);
            }
            None => {
                block_start = index;
                block_inside = Some(inside);
            }
        }
    }
    if let Some(state) = block_inside {
        emit_block(&normalised[block_start..], state, &mut words);
    }

    let mut tokens = Vec::new();
    for word in &words {
        post_process(word, &mut tokens);
    }
    tokens.retain(|token| keep(token));
    tokens
}

fn in_han_class(character: char) -> bool {
    matches!(class(character), Class::Cjk | Class::Ascii)
}

/// One `re_han_default` block (or the characters outside one).
fn emit_block(block: &str, inside: bool, out: &mut Vec<String>) {
    if !inside {
        // Python yields every code point of a skipped block (`sag2.py:53-58`).
        out.extend(block.chars().map(|character| character.to_string()));
        return;
    }
    // A block that mixes CJK and ASCII runs must be cut as one block: Python's
    // `__cut_DAG` buffers consecutive single-character routes and runs
    // `finalseg` over the whole buffer, so splitting at the script boundary
    // changes the HMM input ("最多20个" is one word + "20" + "个" only when the
    // digits stay in the buffer).
    if block
        .chars()
        .any(|character| matches!(class(character), Class::Cjk))
    {
        // Cut the block as one unit (the HMM buffer may span the script
        // boundary), then hand every contiguous ASCII stretch to the patched
        // `finalseg` model.  Python's finalseg splits a buffer at CJK runs and
        // treats each ASCII stretch atomically (`sag2.py:45-48`), which is
        // exactly `ascii_cut`; the DAG route inside an ASCII stretch cannot
        // depend on the CJK text after it, so re-cutting the stretch is exact.
        let mut ascii_run = String::new();
        for word in JIEBA.cut(block, true) {
            if word
                .chars()
                .all(|character| class(character) == Class::Ascii)
            {
                ascii_run.push_str(word);
                continue;
            }
            if !ascii_run.is_empty() {
                ascii_cut::cut_run(&ascii_run, out);
                ascii_run.clear();
            }
            out.push(word.to_owned());
        }
        if !ascii_run.is_empty() {
            ascii_cut::cut_run(&ascii_run, out);
        }
        return;
    }
    // No CJK means no crossing dictionary word, only ASCII text; Python's
    // `__cut_DAG` over it is exactly the ported ASCII model (dictionary DAG +
    // the patched `finalseg`).
    ascii_cut::cut_run(block, out);
}

/// Rules 3-5 of design §4.1 for one cut word.
fn post_process(word: &str, out: &mut Vec<String>) {
    if word.bytes().any(|byte| matches!(byte, b'.' | b'_' | b'-')) {
        for part in word.split(['.', '_', '-']) {
            if part.chars().count() >= 2 && part.bytes().any(|byte| byte.is_ascii_lowercase()) {
                out.push(part.to_owned());
            }
        }
    }
    let characters: Vec<char> = word.chars().collect();
    for length in [2_usize, 3] {
        if characters.len() > length {
            for start in 0..=(characters.len() - length) {
                if dict::contains_chars(&characters[start..start + length]) {
                    out.push(characters[start..start + length].iter().collect());
                }
            }
        }
    }
    out.push(word.to_owned());
}

/// `HAS_CONTENT` and `STOPWORDS`, the final filter every profile ends in.
pub(crate) fn keep(token: &str) -> bool {
    has_content(token) && STOPWORDS.binary_search(&token).is_err()
}
