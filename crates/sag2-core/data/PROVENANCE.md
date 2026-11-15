# Third-party data: jieba `dict.txt`

`dict.txt` is a byte-for-byte copy of the dictionary that ships with
**jieba 0.42.1** (the reference implementation's tokenizer, `sag2.py:49`):

| field | value |
| --- | --- |
| source | `site-packages/jieba/dict.txt` from `jieba==0.42.1` |
| sha256 | `7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8` |
| bytes | 5,071,852 |
| lines | 349,046 |

It is vendored because design §4.3 forbids using jieba-rs's own bundled
dictionary (its version is unknown and would drift both the DAG and `FREQ`).
The port reads it in two places, both at build time:

* `build.rs` extracts the codepoint-length <= 3 entries into a sorted,
  NUL-separated `FREQ` blob with one `u32` anchor every 64 entries (rule 5 of
  design §4.1) plus the pure-ASCII and ASCII/CJK-crossing dictionary words.
* `tokenize` hands the same bytes to `jieba_rs::Jieba::with_dict`, which builds
  the prefix tree used for the DAG.

The file is unmodified; any local edit invalidates the Gate D evidence and the
`manifest.json` provenance recorded by `tools/export_golden.py`.

License: jieba is MIT-licensed (see the upstream repository).

# Third-party data: `allkeys-9.0.0.txt`

`allkeys-9.0.0.txt` is the DUCET 9.0.0 table that ships with **pyuca 1.2**, the
collation implementation the reference imports (`sag2.py:50`, design §5.1):

| field | value |
| --- | --- |
| source | `site-packages/pyuca/allkeys-9.0.0.txt` from `pyuca==1.2` |
| sha256 | `0633f4520c99f249b0c53aa1442cd2521702041fb00a32df944fec13c9da3ed5` |
| bytes | 1,808,188 |
| entries | 30,677 (29,809 single code points, 864 two, 4 three) |
| collation elements | 36,624 (largest entry 18; maximum weight 65533) |
| `@implicitweights` | `17000..18AFF; FB00` (Tangut and Tangut Components) |

It is vendored because no crate implements UCA 9.0.0 (design §5.1), and the
gate requires byte-exact parity with pyuca 1.2 rather than a newer UCA. The
port reads it at build time only: `build.rs` compiles it into the sorted
longest-prefix table `sag2-core` embeds (design §5.4 item 2).

Two properties of this file matter to the port:

* every collation element carries exactly three weights (no `[.x.x.x.x]`
  fourth weight), so pyuca's fourth level is always empty;
* `[*` marks 7,123 entries as variable, and pyuca assigns them the same
  three weights as `[.` entries — the port must do the same (design §5.4
  item 5).

The file is unmodified; any local edit changes the sort keys Gate B certifies.
The digest is also recorded as `uca_keys_sha256` in the index provenance
(`crates/sag2-index/src/open.rs`).

License: the DUCET file is distributed under the Unicode Terms of Use
(https://www.unicode.org/copyright.html).

# Third-party data: Thai `thai/words_th.txt`

`thai/words_th.txt` is the `words_th` corpus word list that ships with
**PyThaiNLP**, the dictionary the reference `newmm` tokenizer reads:

| field | value |
| --- | --- |
| source | `https://raw.githubusercontent.com/PyThaiNLP/pythainlp/dev/pythainlp/corpus/words_th.txt` |
| upstream commit | `PyThaiNLP/pythainlp` `dev` @ `269d0f9e940b71e1f4cfb8913cdef622d3bd378c` (2026-11-13) |
| pinned URL | `https://raw.githubusercontent.com/PyThaiNLP/pythainlp/269d0f9e940b71e1f4cfb8913cdef622d3bd378c/pythainlp/corpus/words_th.txt` |
| sha256 | `12bf4d04e8d59a8b235eb9e7ac59b18e5f4d56320f25e75e9c9f8f565543b481` |
| bytes | 1,520,294 |
| lines | 62,109 |
| license | CC0-1.0 (public domain dedication) |

The commit is the `dev` tip the file was captured from; that tree and its
descendants carry the digest above, while the release branch's copy differs
(`35f65d43…`, 62,107 lines), which is why the URL is pinned to a commit and not
to a branch name.  `crates/sag2-core/src/analysis/segment/thai.rs` asserts the
digest, the byte count and the line count, so an edit to the vendored file is a
test failure rather than a silent re-segmentation.

It is vendored because `nlpo3` 1.4.0 (Apache-2.0) implements Thai `newmm`
segmentation but ships no dictionary. One word per line; 611 of the lines are
multi-word expressions that keep their inner spaces (`newmm` matches them as one
unit). The port reads the file at build time only: `include_str!` in
`crates/sag2-core/src/analysis/segment/thai.rs` builds the tokenizer trie once
per process, so `multilingual-v1-th` builds and runs offline and every build
segments with the same list.

The file is unmodified; any local edit changes every `multilingual-v1-th` token
stream and invalidates the pinned segmentations in
`crates/sag2-core/tests/thai_segmentation.rs`.

# Test fixture: `tests/data/multilingual.jsonl`

`multilingual.jsonl` is the multilingual recall fixture the T7 regression test
reads (`crates/sag2-core/tests/multilingual.rs`). It is **self-authored**: every
document, query and content word in it was written for this repository. No text
is copied from UD, MIRACL, Tatoeba or any other corpus, so there is no
third-party licence or attribution attached to it.

| field | value |
| --- | --- |
| languages | `ru ar hi el vi th pl ja ko de fr zh en` (one JSON line each) |
| per language | 10 documents (2-4 sentences), 10 labelled queries, 12 content words |
| states | `measured` (th de fr zh en: pinned word-rate target and baseline), `deferred` (ru ar hi el vi pl: plan §5 batch 2 has the segmenter/stemmer, so only the rate is printed), `dictionary required` (ja ko: asserted to return `AnalysisError::DictionaryMissing`, never a character stream) |
| used for | word-rate targets, the pinned `Sag2ZhV1` baseline, and the Lexical-vs-Phrase3 R@10 comparison |

The R@10 evaluation adds decoy documents derived deterministically from the
fixture itself (its own token stream, rotated: words for the word-level
profiles, characters for the character-level ones). The decoys are unlabelled
negatives generated inside the test - they are not part of this file and are
not a corpus.
