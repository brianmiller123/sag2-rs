//! Product-behaviour checks for the §4 tokenizer.  Every expectation here was
//! produced by the Python reference (jieba 0.42.1 + the patched finalseg); the
//! exhaustive comparison lives in `sag2-cli verify --gate D --fixtures
//! tokenizer`.

use sag2_core::{
    casefold, combining_class, is_letter, is_mark, is_number, is_uppercase, lower, nfkc, tokenize,
};

fn tokens(text: &str) -> Vec<String> {
    tokenize(text)
}

#[test]
fn frozen_reference_tokenizer_lines() {
    assert_eq!(
        tokens("half-ounce café. A.H.G. Alston"),
        ["half", "ounce", "half-ounce", "caf", "é", "a.h.g", "alston"]
    );
    assert_eq!(
        tokens("Chester A. Arthur's Jean-Luc Picard"),
        ["chester", "arthur", "s", "jean", "luc", "jean-luc", "picard"]
    );
    assert_eq!(
        tokens("Anchor 北京数据库"),
        ["anchor", "北京", "数据", "据库", "数据库"]
    );
    assert_eq!(
        tokens("foo-bar-baz a_b foo.bar +test #tag C++ 3.14 1,000 -foo-"),
        [
            "foo",
            "bar",
            "baz",
            "foo-bar-baz",
            "a_b",
            "foo",
            "bar",
            "foo.bar",
            "test",
            "tag",
            "c++",
            "3.14",
            "1",
            "000",
            "foo"
        ]
    );
}

#[test]
fn mixed_script_blocks_keep_one_hmm_buffer() {
    // Python's `__cut_DAG` buffers consecutive single-character routes and runs
    // `finalseg` over the whole buffer; "最多20个" is therefore one word, "20"
    // and "个", not "最"+"多" followed by a separate ASCII cut.
    assert_eq!(tokens("选出最多20个"), ["选出", "最多", "20", "个"]);
    assert_eq!(
        tokens("方法流程 选出最多20个扩展词项"),
        ["方法", "流程", "选出", "最多", "20", "个", "扩展", "词项"]
    );
    // A contiguous ASCII stretch inside a mixed block still follows the patched
    // `finalseg` (`%` left-end rule applies here too).
    assert_eq!(
        tokens("enterpriserag-bench共511 962"),
        [
            "enterpriserag",
            "bench",
            "enterpriserag-bench",
            "共",
            "511",
            "962"
        ]
    );
}

#[test]
fn a_percent_ends_the_leftmost_ascii_run() {
    assert_eq!(
        tokens("paper-arXiv%3A2610.22588-18181b"),
        [
            "paper",
            "arxiv%",
            "paper-arxiv%",
            "3a2610",
            "18181b",
            "3a2610.22588-18181b"
        ]
    );
    assert_eq!(
        tokens("See https://example.com/a%20b?q=1%2F2 for the mirrored copy."),
        [
            "see",
            "https",
            "example",
            "com",
            "example.com",
            "a%",
            "20b",
            "q",
            "1%",
            "2f2",
            "mirrored",
            "copy"
        ]
    );
}

#[test]
fn emoji_only_tokens_fail_the_has_content_filter() {
    assert_eq!(tokens("台风😀登陆"), ["台风", "登陆"]);
}

#[test]
fn stopwords_are_dropped_after_the_substrings_are_added() {
    assert_eq!(tokens("the anchor and the data"), ["anchor", "data"]);
    assert_eq!(tokens("Beijing 北京 2024"), ["beijing", "北京", "2024"]);
}

#[test]
fn mixed_ascii_cjk_dictionary_words_stay_whole() {
    assert_eq!(tokens("4S店 and 1号店"), ["4s店", "1号店"]);
    assert_eq!(tokens("x4s店"), ["x", "4s店"]);
    assert_eq!(tokens("江南style"), ["江南", "江南style"]);
    assert_eq!(tokens("x 江南style y"), ["x", "江南", "江南style", "y"]);
}

#[test]
fn ascii_dictionary_words_survive_the_connected_run() {
    // "AT&T" is in the dictionary only as `AT&T`; lowering the text leaves
    // `at&t` unmatched, so only the finalseg run pieces survive the filters.
    assert_eq!(tokens("AT&T and C#"), ["t", "c#"]);
}

#[test]
fn combining_split_of_a_non_latin_neighbour() {
    assert_eq!(
        tokens("Ünïcode İstanbul"),
        ["ü", "n", "ï", "code", "i", "stanbul"]
    );
}

#[test]
fn lower_and_casefold_match_python() {
    assert_eq!(lower("\u{130}"), "i\u{307}");
    assert_eq!(lower("ZÜRICH"), "zürich");
    assert_eq!(casefold("ß"), "ss");
    assert_eq!(casefold("ς"), "σ");
    assert_eq!(casefold("ﬁ"), "fi");
    assert_eq!(casefold("İ"), "i\u{307}");
}

#[test]
fn combining_classes_match_python() {
    assert_eq!(combining_class(0x0301), 230);
    assert_eq!(combining_class(0x0308), 230);
    assert_eq!(combining_class(0x00E9), 0);
    assert_eq!(combining_class('a' as u32), 0);
    assert_eq!(combining_class(0x11_0000), 0);
}

#[test]
fn categories_match_python_regex() {
    assert!(is_letter('中' as u32));
    assert!(is_letter('a' as u32));
    assert!(!is_letter(' ' as u32));
    assert!(is_number('7' as u32));
    assert!(!is_number('a' as u32));
    assert!(is_uppercase('A' as u32));
    assert!(!is_uppercase('a' as u32));
    assert!(is_mark(0x0301));
    assert!(!is_mark('a' as u32));
}

#[test]
fn nfkc_uses_the_python_15_tables() {
    assert_eq!(nfkc("x\u{b2}"), "x2");
    assert_eq!(nfkc("\u{fb01}"), "fi");
    assert_eq!(nfkc("a\u{301}"), "\u{e1}");
    assert_eq!(nfkc("\u{2126}"), "\u{3a9}");
    assert_eq!(nfkc("\u{130}"), "\u{130}");
    // U+1DFE7 gained a decomposition after Unicode 15.0.0; CPython 3.12.11
    // leaves it alone, and so must the port.
    assert_eq!(nfkc("\u{1DFE7}"), "\u{1DFE7}");
    assert_eq!(nfkc("\u{1DFE7}\u{301}"), "\u{1DFE7}\u{301}");
}

#[test]
fn nfkc_round_trips_hangul() {
    assert_eq!(nfkc("\u{AC01}"), "\u{AC01}");
    assert_eq!(nfkc("\u{1100}\u{1161}"), "\u{AC00}");
    assert_eq!(nfkc("\u{1100}\u{1161}\u{11A8}"), "\u{AC01}");
}
