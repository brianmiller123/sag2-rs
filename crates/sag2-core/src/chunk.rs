//! Sentence-preserving chunks, byte-exact with `chunking.py` (design §9 R12).
//!
//! Spans are **Python string indices**: code point offsets into `source`, not
//! UTF-8 byte offsets.  The reference slices `source[start:end]` and writes the
//! pairs straight into `chunks.spans` (`sag2.py:302`), and Gate A compares those
//! serialised pairs byte for byte, so the unit is part of the contract.
//!
//! `path` follows `chunking.py:67`: `None` means `document.md`.  The index path
//! has its own default (`document.txt`, `sag2.py:182`); callers there must pass
//! it explicitly (design §9 R13).

/// `chunking.py:6` `DEFAULT_CHUNK_CHARS`.
pub const DEFAULT_CHUNK_CHARS: usize = 2200;

/// `chunking.py:8-13` `CODE_SUFFIXES`, all 45 entries in reference order.
///
/// `tools/export_golden.py` re-parses this literal and refuses to export when
/// it drifts from `chunking.CODE_SUFFIXES`.
const CODE_SUFFIXES: [&str; 45] = [
    ".rs", ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cc", ".cs", ".rb", ".php", ".swift", ".kt", ".kts", ".scala", ".sh", ".bash", ".zsh",
    ".ps1", ".sql", ".html", ".htm", ".css", ".scss", ".vue", ".svelte", ".r", ".lua", ".dart",
    ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml",
];

/// `chunking.py:14` `CLOSERS` (the characters that may follow a period).
const CLOSERS: [char; 8] = ['"', '\'', '”', '’', '）', ')', ']', '}'];

/// One `chunk_document` result (`chunking.py:153-178`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub heading: String,
    pub spans: Vec<(usize, usize)>,
    pub content: String,
}

/// `len(text.encode("utf-16-le")) // 2` (`chunking.py:18-19`).
pub fn utf16_length(text: &str) -> usize {
    text.chars()
        .map(|character| if character as u32 >= 0x1_0000 { 2 } else { 1 })
        .sum()
}

/// `chunking.py:53-60` `source_content`.
pub fn source_content(source: &str, spans: &[(usize, usize)]) -> String {
    let characters: Vec<char> = source.chars().collect();
    let mut result = String::new();
    for &(start, end) in spans {
        let part: String = characters[start..end].iter().collect();
        if !result.is_empty() && !result.ends_with('\n') && !part.starts_with('\n') {
            result.push_str("\n\n");
        }
        result.push_str(&part);
    }
    result
}

/// `chunking.py:63-178` `chunk_document`.
///
/// # Panics
///
/// Panics when `target < 2`, mirroring the reference `ValueError`
/// (`chunking.py:66`); the public signature is the design §2.5 contract.
pub fn chunk_document(source: &str, target: usize, path: Option<&str>) -> Vec<Chunk> {
    assert!(
        target >= 2,
        "chunk target must be at least two characters (chunking.py:66)"
    );
    let file = path.unwrap_or("document.md");
    let suffix = suffix_lower(file);
    let markdown = suffix == ".md" || suffix == ".markdown";
    let code = CODE_SUFFIXES.contains(&suffix.as_str())
        || matches!(
            file_name(file),
            "Dockerfile" | "Containerfile" | "Makefile" | "CMakeLists.txt"
        );

    let characters: Vec<char> = source.chars().collect();
    let length = characters.len();
    let mut lines: Vec<(usize, usize)> = Vec::new();
    let mut start = 0_usize;
    let mut index = 0_usize;
    while index < length {
        if characters[index] == '\r' || characters[index] == '\n' {
            lines.push((start, index));
            if index + 1 < length && characters[index] == '\r' && characters[index + 1] == '\n' {
                index += 1;
            }
            start = index + 1;
        }
        index += 1;
    }
    lines.push((start, length));

    let mut blocks: Vec<Block> = Vec::new();
    let mut headings: Vec<String> = Vec::new();
    let mut first: Option<usize> = None;
    let mut fence: Option<String> = None;
    let mut front = markdown && py_strip(&slice(&characters, lines[0].0, lines[0].1)).eq("---");

    if !markdown {
        if !py_strip(source).is_empty() {
            blocks.push(Block {
                heading: String::new(),
                start: 0,
                end: length,
                code,
            });
        }
    } else {
        for (line_index, &(line_start, line_end)) in lines.iter().enumerate() {
            let line: String = characters[line_start..line_end].iter().collect();
            if front {
                if line_index > 0 && matches!(py_strip(&line), "---" | "...") {
                    front = false;
                }
                continue;
            }
            if let Some(marker) = fence_match(&line) {
                match &fence {
                    None => {
                        if line_index > 0 {
                            flush(
                                &characters,
                                &lines,
                                &mut blocks,
                                &headings,
                                &mut first,
                                line_index - 1,
                                false,
                            );
                        }
                        first = Some(line_index);
                        fence = Some(marker);
                    }
                    Some(open) => {
                        let open_char = open.chars().next().expect("fence marker");
                        let marker_char = marker.chars().next().expect("fence marker");
                        if open_char == marker_char
                            && marker.chars().count() >= open.chars().count()
                        {
                            fence = None;
                            flush(
                                &characters,
                                &lines,
                                &mut blocks,
                                &headings,
                                &mut first,
                                line_index,
                                true,
                            );
                        }
                    }
                }
                continue;
            }
            if fence.is_some() {
                continue;
            }
            if let Some((level, heading)) = heading_match(&line) {
                if line_index > 0 {
                    flush(
                        &characters,
                        &lines,
                        &mut blocks,
                        &headings,
                        &mut first,
                        line_index - 1,
                        false,
                    );
                }
                headings.resize(level, String::new());
                headings[level - 1] = heading;
            } else if py_strip(&line).is_empty() {
                if line_index > 0 {
                    flush(
                        &characters,
                        &lines,
                        &mut blocks,
                        &headings,
                        &mut first,
                        line_index - 1,
                        false,
                    );
                }
            } else if first.is_none() {
                first = Some(line_index);
            }
        }
        flush(
            &characters,
            &lines,
            &mut blocks,
            &headings,
            &mut first,
            lines.len() - 1,
            fence.is_some(),
        );
    }

    let overlap = if target >= 160 { 80 } else { 0 };
    let limit = target - if overlap > 0 { overlap + 2 } else { 0 };
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut units: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut chunk_length = 0_usize;
    for block in &blocks {
        let sentences = sentence_ranges(&characters, block.start, block.end, block.code);
        let mut groups: Vec<(usize, usize)> = Vec::new();
        let mut cursor = block.start;
        let mut used = 0_usize;
        for &(sentence_start, sentence_end) in &sentences {
            let size = utf16_length(&slice(&characters, sentence_start, sentence_end));
            if used > 0 && used + size > limit {
                groups.push((cursor, sentence_start));
                cursor = sentence_start;
                used = 0;
            }
            used += size;
        }
        if cursor < block.end {
            groups.push((cursor, block.end));
        }
        for &(group_start, group_end) in &groups {
            let size = utf16_length(&slice(&characters, group_start, group_end));
            let group_units = sentence_ranges(&characters, group_start, group_end, block.code);
            let extend = match chunks.last() {
                None => false,
                Some(last) => last.heading == block.heading && chunk_length + 2 + size <= limit,
            };
            if extend {
                let last = chunks.last_mut().expect("checked above");
                last.spans.push((group_start, group_end));
                units
                    .last_mut()
                    .expect("units parallel chunks")
                    .extend(group_units);
                chunk_length += 2 + size;
            } else {
                chunks.push(Chunk {
                    heading: block.heading.clone(),
                    spans: vec![(group_start, group_end)],
                    content: String::new(),
                });
                units.push(group_units);
                chunk_length = size;
            }
        }
    }

    for chunk_index in (1..chunks.len()).rev() {
        if overlap == 0 || chunks[chunk_index - 1].heading != chunks[chunk_index].heading {
            continue;
        }
        let mut prefix: Vec<(usize, usize)> = Vec::new();
        let mut used = 0_usize;
        for &(start, end) in units[chunk_index - 1].iter().rev() {
            let size = utf16_length(&slice(&characters, start, end));
            if !prefix.is_empty() && used + size > overlap {
                break;
            }
            prefix.push((start, end));
            used += size;
            if used >= overlap {
                break;
            }
        }
        prefix.reverse();
        prefix.extend(chunks[chunk_index].spans.iter().copied());
        chunks[chunk_index].spans = prefix;
    }

    for chunk in &mut chunks {
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for &(start, end) in &chunk.spans {
            match merged.last_mut() {
                Some(last) if start <= last.1 => {
                    last.1 = last.1.max(end);
                }
                _ => merged.push((start, end)),
            }
        }
        chunk.content = source_content_from(&characters, &merged);
        chunk.spans = merged;
    }
    chunks
}

/// One flushed Markdown block (`chunking.py:88-93`).
struct Block {
    heading: String,
    start: usize,
    end: usize,
    code: bool,
}

/// `chunking.py:88-94` `flush`; `first` is cleared whether or not a block is
/// emitted, exactly like the reference.
fn flush(
    characters: &[char],
    lines: &[(usize, usize)],
    blocks: &mut Vec<Block>,
    headings: &[String],
    first: &mut Option<usize>,
    last: usize,
    code: bool,
) {
    let Some(first_index) = *first else {
        return;
    };
    let start = lines[first_index].0;
    let end = lines[last].1;
    let text: String = characters[start..end].iter().collect();
    if !py_strip(&text).is_empty() {
        let heading = headings
            .iter()
            .filter(|heading| !heading.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(" / ");
        blocks.push(Block {
            heading,
            start,
            end,
            code,
        });
    }
    *first = None;
}

/// `chunking.py:22-51` `sentence_ranges`.
fn sentence_ranges(
    characters: &[char],
    start: usize,
    end: usize,
    code: bool,
) -> Vec<(usize, usize)> {
    let mut cursor = start;
    let mut position = start;
    let mut result = Vec::new();
    while position < end {
        let character = characters[position];
        let boundary = if code {
            character == '\n'
        } else {
            matches!(character, '。' | '！' | '？' | '!' | '?')
                || (character == '.'
                    && (position + 1 == end
                        || is_space(characters[position + 1])
                        || CLOSERS.contains(&characters[position + 1])))
        };
        position += 1;
        if !boundary {
            continue;
        }
        if !code {
            while position < end
                && (is_space(characters[position])
                    || matches!(characters[position], '。' | '！' | '？' | '!' | '?')
                    || CLOSERS.contains(&characters[position]))
            {
                position += 1;
            }
        }
        result.push((cursor, position));
        cursor = position;
    }
    if cursor < end {
        result.push((cursor, end));
    }
    result
}

/// `chunking.py:29-44` boundary predicate folded into [`sentence_ranges`].
fn is_space(character: char) -> bool {
    // `str.isspace()` also treats the four file/group/record/unit separators
    // as whitespace; Rust's `White_Space` property (used by
    // `char::is_whitespace`) excludes them.
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

/// `str.strip()` with no arguments.
fn py_strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

fn slice(characters: &[char], start: usize, end: usize) -> String {
    characters[start..end].iter().collect()
}

/// Content assembly for the already-collected character vector.
fn source_content_from(characters: &[char], spans: &[(usize, usize)]) -> String {
    let mut result = String::new();
    for &(start, end) in spans {
        let part = slice(characters, start, end);
        if !result.is_empty() && !result.ends_with('\n') && !part.starts_with('\n') {
            result.push_str("\n\n");
        }
        result.push_str(&part);
    }
    result
}

/// The final path component, as `Path.name` on Linux.
fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or("")
}

/// `PurePath.suffix.lower()` on Linux.
fn suffix_lower(path: &str) -> String {
    let name = file_name(path);
    let length = name.chars().count();
    let Some(dot) = name.rfind('.') else {
        return String::new();
    };
    let dot_chars = name[..dot].chars().count();
    if dot_chars == 0 || dot_chars + 1 == length {
        return String::new();
    }
    name[dot..].to_lowercase()
}

/// `FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")` (`chunking.py:15`).
fn fence_match(line: &str) -> Option<String> {
    let characters: Vec<char> = line.chars().collect();
    let mut index = 0_usize;
    while index < characters.len() && index < 3 && is_space(characters[index]) {
        index += 1;
    }
    if index == 3 && index < characters.len() && is_space(characters[index]) {
        return None;
    }
    let marker = characters.get(index)?;
    if *marker != '`' && *marker != '~' {
        return None;
    }
    let mut end = index;
    while end < characters.len() && characters[end] == *marker {
        end += 1;
    }
    if end - index < 3 {
        return None;
    }
    Some(characters[index..end].iter().collect())
}

/// `HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")`
/// (`chunking.py:14`), including the regex engine's backtracking order.
fn heading_match(line: &str) -> Option<(usize, String)> {
    let characters: Vec<char> = line.chars().collect();
    let mut index = 0_usize;
    while index < characters.len() && index < 3 && is_space(characters[index]) {
        index += 1;
    }
    if index == 3 && index < characters.len() && is_space(characters[index]) {
        return None;
    }
    let hash_start = index;
    while index < characters.len() && characters[index] == '#' && index - hash_start < 6 {
        index += 1;
    }
    let level = index - hash_start;
    if level == 0 || index >= characters.len() || !is_space(characters[index]) {
        return None;
    }
    let rest = &characters[index..];
    let mut whitespace = 0_usize;
    while whitespace < rest.len() && is_space(rest[whitespace]) {
        whitespace += 1;
    }
    // `\s+` is greedy; the lazy group starts as late as possible and backs off
    // one whitespace at a time until `(.+?)\s*#*\s*$` can match.
    for start in (1..=whitespace).rev() {
        for length in 1..=(rest.len() - start) {
            if suffix_matches(&rest[start + length..]) {
                return Some((
                    level,
                    rest[start..start + length].iter().collect::<String>(),
                ));
            }
        }
    }
    None
}

/// Membership in `\s*#*\s*$`.
fn suffix_matches(characters: &[char]) -> bool {
    let mut index = 0_usize;
    while index < characters.len() && is_space(characters[index]) {
        index += 1;
    }
    while index < characters.len() && characters[index] == '#' {
        index += 1;
    }
    while index < characters.len() && is_space(characters[index]) {
        index += 1;
    }
    index == characters.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf16_length_counts_surrogate_pairs() {
        assert_eq!(utf16_length(""), 0);
        assert_eq!(utf16_length("abc"), 3);
        assert_eq!(utf16_length("😀"), 2);
        assert_eq!(utf16_length("a😀𠀀"), 5);
    }

    #[test]
    fn source_content_joins_non_newline_spans_with_two_newlines() {
        assert_eq!(source_content("abcdef", &[(0, 2), (4, 6)]), "ab\n\nef");
        assert_eq!(source_content("ab\ncdef", &[(0, 3), (3, 7)]), "ab\ncdef");
        assert_eq!(source_content("\nabc", &[(0, 1), (1, 4)]), "\nabc");
    }

    #[test]
    fn path_default_is_document_md_not_document_txt() {
        let source = "# heading\nbody";
        let markdown = chunk_document(source, 512, None);
        let plain = chunk_document(source, 512, Some("document.txt"));
        assert_eq!(markdown.len(), 1);
        assert_eq!(markdown[0].heading, "heading");
        assert_eq!(plain.len(), 1);
        assert_eq!(plain[0].heading, "");
        assert_eq!(plain[0].content, source);
    }

    #[test]
    fn long_sentence_and_overlap_keep_the_heading() {
        let first = "完整条件".repeat(100) + "。";
        let long = "超长句子".repeat(150) + "。";
        let last = "只有满足上述条件才能发布。".to_owned();
        let source = format!("# 发布指南\n## 生产环境\n{first}{long}{last}");
        let chunks = chunk_document(&source, 512, None);
        let contents: Vec<&str> = chunks.iter().map(|chunk| chunk.content.as_str()).collect();
        assert_eq!(
            contents,
            [
                first.as_str(),
                &format!("{first}{long}"),
                &format!("{long}{last}")
            ]
        );
        assert!(chunks
            .iter()
            .all(|chunk| chunk.heading == "发布指南 / 生产环境"));
    }

    #[test]
    fn chunk_length_counts_returned_text_not_the_gap() {
        let source = format!("# A\nfirst.\n{}second.", "\n".repeat(10000));
        let chunks = chunk_document(&source, 2200, Some("guide.md"));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "first.\n\nsecond.");
        assert_eq!(utf16_length(&chunks[0].content), 15);
        assert_eq!(chunks[0].spans, vec![(4, 10), (10011, 10018)]);
    }

    #[test]
    #[should_panic(expected = "at least two characters")]
    fn target_below_two_panics_like_the_reference() {
        let _ = chunk_document("body", 1, None);
    }
}
