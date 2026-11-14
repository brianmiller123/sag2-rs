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
