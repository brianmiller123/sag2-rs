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
//!
//! Steps 1-6 are the certified pipeline and the only one [`tokenize`] uses.
//! [`tokenize_multilingual`] shares steps 3-6 and the jieba path for CJK text,
//! but takes its word boundaries from [`is_word_character`] instead of
//! `re_han_default`, so a language that does not write in ASCII or Han reaches
//! word-level tokens without losing the code-point tokens the reference
//! pipeline emits (plan §5.1 L1).

use std::sync::LazyLock;

use jieba_rs::Jieba;

use crate::analysis::normalize;
use crate::analysis::Language;
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

/// The multilingual word character (plan §5.1 L1): a letter or a mark that
/// attaches to the word, or a number.
///
/// [`class`] accepts ASCII and `U+4E00-U+9FD5` only, so every other script used
/// to reach [`emit_block`] one code point at a time.  This set keeps a word
/// together in every script, and it is built from the pinned Unicode 15.0.0
/// tables of [`crate::unicode`], so the classification cannot drift with the
/// compiler's Unicode version.  A stacked mark (ccc != 0) normally marks the
/// *previous* code point and therefore ends a run; the two exceptions are the
/// attachment classes - ccc 7 (`Nukta`) and ccc 9 (`Virama`) - which attach the
/// following code point and keep the run together.  Those classes are assigned
/// in several script families, not only the Indic ones (ccc 9 also in Thai,
/// Lao, Tibetan, Myanmar, Khmer, Sinhala, Balinese and Sundanese; ccc 7 also in
/// Myanmar, Balinese, Batak, Javanese and Adlam), which is why the rule is a
/// class rule and why the word set it changes is pipeline-wide: the `ml-2` bump
/// applies to every `multilingual-v1-<lang>` profile, not only `-hi`, because
/// the version names the pipeline rather than the language.
fn is_word_character(character: char) -> bool {
    let cp = character as u32;
    if unicode::is_number(cp) {
        return true;
    }
    // The attachment classes: ccc 7 (Nukta) and ccc 9 (Virama) join the next
    // code point to the current word instead of ending it.
    let joins = matches!(unicode::combining_class(cp), 0 | 7 | 9);
    joins && (unicode::is_letter(cp) || unicode::is_mark(cp))
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
/// This is the §4 pipeline the reference profile enters after its own
/// normalisation: `NFKC` -> lower here, then [`tokenize_normalised`].  A
/// multilingual profile runs [`crate::analysis::normalize`] first and then
/// [`tokenize_multilingual`], so no input is normalised twice (C4).
///
/// [`Profile::Sag2ZhV1`]: crate::analysis::Profile::Sag2ZhV1
pub(crate) fn tokenize_default(text: &str) -> Vec<String> {
    tokenize_normalised(&unicode::lower(&unicode::nfkc(text)))
}

/// The certified §4 pipeline, over text the caller has already normalised
/// (design §4.1-§4.3).
fn tokenize_normalised(normalised: &str) -> Vec<String> {
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

    finish(words)
}

/// The multilingual word pipeline (plan §5.1 L1, L2b).
///
/// Same post-rules and filters as [`tokenize_normalised`], but the word
/// boundaries are [`is_word_character`] and a word run the reference classifier
/// would have shredded emits its code points *and* the whole run: the
/// code-point tokens are what ru/ar/hi recall depends on, so they are never
/// replaced.  CJK runs still go through the jieba path of [`emit_block`].
///
/// `original` is the text before [`crate::analysis::normalize`] and is read for
/// [`Language::De`] only: its word run also carries the deaccented spelling
/// FTS5 `unicode61` indexes for an umlaut, which the umlaut expansion in the
/// fold would otherwise cost (see [`emit_segment`]).
pub(crate) fn tokenize_multilingual(
    language: Language,
    normalised: &str,
    original: &str,
) -> Vec<String> {
    // The paired spelling is the pre-fold word run.  That pairing is exact:
    // `fold_script`'s maps rewrite letters as letters and never create or
    // delete a boundary, and `normalised` and `accented` are both built from
    // the same `lower(nfkc(original))`, so they have the same segments.
    //
    // The deaccented spelling is derived from each paired run instead of being
    // read from a re-segmented deaccented text: `strip_marks` deletes combining
    // marks, which can merge two word runs (`x` + U+0308 + `y` is two runs
    // before the strip and one after), and a second walk would silently shift
    // every later word's variants.
    let accented_text = if language == Language::De {
        unicode::lower(&unicode::nfkc(original))
    } else {
        String::new()
    };
    let accented = word_runs(&accented_text);
    let mut variants = accented.iter();
    let mut words: Vec<String> = Vec::new();
    for_each_segment(normalised, |kind, segment| {
        let variant = if kind == Segment::Word {
            variants.next().copied()
        } else {
            None
        };
        emit_segment(segment, kind, variant, &mut words);
    });
    // Debug builds (and every test) fail on a desync instead of dropping the
    // variants of the words after it.
    debug_assert!(
        variants.next().is_none(),
        "the German variants must pair one for one with the word runs"
    );
    finish(words)
}

/// Walk `text` as its maximal multilingual segments, in order.
fn for_each_segment<'a>(text: &'a str, mut emit: impl FnMut(Segment, &'a str)) {
    let mut block_start = 0_usize;
    let mut block_kind: Option<Segment> = None;
    for (index, character) in text.char_indices() {
        let kind = segment(character);
        match block_kind {
            Some(state) if state == kind => {}
            Some(state) => {
                emit(state, &text[block_start..index]);
                block_start = index;
                block_kind = Some(kind);
            }
            None => {
                block_start = index;
                block_kind = Some(kind);
            }
        }
    }
    if let Some(kind) = block_kind {
        emit(kind, &text[block_start..]);
    }
}

/// The word runs of `text`, in order, without emitting anything.
fn word_runs(text: &str) -> Vec<&str> {
    let mut runs = Vec::new();
    for_each_segment(text, |kind, segment| {
        if kind == Segment::Word {
            runs.push(segment);
        }
    });
    runs
}

/// One multilingual segment: `re_han_default`'s CJK range, a word run, or a
/// code point that is neither.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Segment {
    /// A CJK run: the jieba path, exactly as in the reference pipeline.
    ///
    /// The run ends where the classifier leaves the CJK range, so a mixed
    /// Han+ASCII run (`最多20个`) reaches jieba as `最多` and `个` instead of the
    /// one block [`emit_block`] cuts for the reference pipeline.  The
    /// multilingual pipeline takes its word boundaries from
    /// [`is_word_character`] and does not reproduce Python's `finalseg` buffer
    /// for the ASCII part, so splitting at the script boundary is this
    /// profile's decision, not an accident.
    Cjk,
    /// A maximal run of [`is_word_character`] code points.
    Word,
    /// Anything else: one code point, one token, as the reference does it.
    Other,
}

fn segment(character: char) -> Segment {
    if class(character) == Class::Cjk {
        Segment::Cjk
    } else if is_word_character(character) {
        Segment::Word
    } else {
        Segment::Other
    }
}

/// Emit one multilingual segment.
fn emit_segment(segment: &str, kind: Segment, variant: Option<&str>, out: &mut Vec<String>) {
    match kind {
        Segment::Cjk => emit_block(segment, true, out),
        Segment::Word => {
            // L2b: a German word the umlaut expansion transliterated also
            // carries the spelling FTS `unicode61` indexes for the umlaut, so
            // both ASCII spellings reach the word.  Order: the pre-fold code
            // points, the transliteration, the deaccented spelling.
            if let Some(accented) = variant {
                if has_german_umlaut(accented) {
                    out.extend(accented.chars().map(String::from));
                    out.push(segment.to_owned());
                    let deaccented = normalize::deaccent(accented);
                    if deaccented != segment {
                        out.push(deaccented);
                    }
                    return;
                }
            }
            // An ASCII run already reaches the word level in the reference
            // pipeline, so it has no code-point tokens to preserve; emitting
            // them would only spend the query expression budget on characters.
            if segment.is_ascii() {
                out.push(segment.to_owned());
                return;
            }
            // Dual emission: the code points the reference pipeline would have
            // produced, then the whole run.  A one-code-point run is not
            // repeated.
            out.extend(segment.chars().map(String::from));
            if segment.chars().count() > 1 {
                out.push(segment.to_owned());
            }
        }
        Segment::Other => out.extend(segment.chars().map(String::from)),
    }
}

/// Whether `word` carries a character the German fold rewrites.
///
/// `word` is a word run of the profile's `lower(nfkc(text))` spelling, so only
/// the lowercase forms can occur (`ẞ` has already become `ß`).  `ä/ö/ü` expand
/// into two letters, so the transliteration and the deaccented spelling differ
/// and both are emitted.  `ß` also triggers the emission, but the shared Latin
/// fold maps it to `ss` on both sides, so its two spellings are the same string
/// and only one token results.
fn has_german_umlaut(word: &str) -> bool {
    word.chars()
        .any(|character| matches!(character, 'ä' | 'ö' | 'ü' | 'ß'))
}

/// Rules 3-5 and the final filter, over the words a splitter produced.
fn finish(words: Vec<String>) -> Vec<String> {
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
