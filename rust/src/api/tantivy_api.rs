use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use std::path::PathBuf;
use std::sync::Mutex;
use tantivy::collector::TopDocs;
use tantivy::query::{AllQuery, QueryParser, TermQuery};
use tantivy::schema::*;
use tantivy::{
    Index, IndexReader, IndexWriter, Order, ReloadPolicy, Searcher, TantivyDocument, Term,
};

// Flutter에서 사용할 문서 구조체
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub content: String,
    pub preview_text: String,
    pub is_pinned: bool,
    pub updated_at: i64,
}

// Flutter에서 사용할 검색 결과 구조체
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
    pub doc: Document,
    pub snippet: Option<String>,
}

// Tantivy의 핵심 로직을 관리하는 구조체
struct TantivyApi {
    index: Index,
    writer: Mutex<IndexWriter>,
    reader: IndexReader,
    id_field: Field,
    title_field: Field,
    content_field: Field,
    preview_text_field: Field,
    is_pinned_field: Field,
    updated_at_field: Field,
    sort_key_field: Field,
}

impl TantivyApi {
    /// Reloads the reader so the searcher sees the latest commit.
    fn searcher(&self) -> Result<Searcher> {
        self.reader.reload()?;
        Ok(self.reader.searcher())
    }

    fn id_term(&self, id: &str) -> Term {
        Term::from_field_text(self.id_field, id)
    }

    fn extract_document(&self, retrieved_doc: &TantivyDocument) -> Document {
        let text = |field: Field| {
            retrieved_doc
                .get_first(field)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string()
        };
        Document {
            id: text(self.id_field),
            title: text(self.title_field),
            content: text(self.content_field),
            preview_text: text(self.preview_text_field),
            is_pinned: retrieved_doc
                .get_first(self.is_pinned_field)
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            updated_at: retrieved_doc
                .get_first(self.updated_at_field)
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
        }
    }

    fn to_tantivy_doc(&self, doc: &Document) -> TantivyDocument {
        let mut tantivy_doc = TantivyDocument::new();
        tantivy_doc.add_text(self.id_field, &doc.id);
        tantivy_doc.add_text(self.title_field, &doc.title);
        tantivy_doc.add_text(self.content_field, &doc.content);
        tantivy_doc.add_text(self.preview_text_field, &doc.preview_text);
        tantivy_doc.add_bool(self.is_pinned_field, doc.is_pinned);
        tantivy_doc.add_i64(self.updated_at_field, doc.updated_at);

        let sort_key: i64 =
            if doc.is_pinned { 1_i64 << 60 } else { 0 } | (doc.updated_at & 0x0FFF_FFFF_FFFF_FFFF);
        tantivy_doc.add_i64(self.sort_key_field, sort_key);

        tantivy_doc
    }

    /// 동일한 ID의 문서가 있다면 삭제 후 추가 (Update-or-Insert). Does not commit.
    fn upsert(&self, writer: &IndexWriter, doc: &Document) -> Result<()> {
        writer.delete_term(self.id_term(&doc.id));
        writer.add_document(self.to_tantivy_doc(doc))?;
        Ok(())
    }

    fn find_by_id(&self, searcher: &Searcher, id: &str) -> Result<Option<Document>> {
        let query = TermQuery::new(self.id_term(id), IndexRecordOption::Basic);
        let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;
        match top_docs.first() {
            Some((_, doc_address)) => {
                let retrieved_doc = searcher.doc::<TantivyDocument>(*doc_address)?;
                Ok(Some(self.extract_document(&retrieved_doc)))
            }
            None => Ok(None),
        }
    }
}

fn build_schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field("id", STRING | STORED);
    builder.add_text_field("title", TEXT | STORED);
    builder.add_text_field("content", TEXT | STORED);
    builder.add_text_field("preview_text", STORED);
    builder.add_bool_field("is_pinned", STORED | FAST);
    builder.add_i64_field("updated_at", STORED | FAST);
    builder.add_i64_field("sort_key", FAST);
    builder.build()
}

/// False for indexes written by older versions of the schema.
fn is_current_schema(schema: &Schema) -> bool {
    schema.get_field("preview_text").is_ok()
        && schema.get_field("sort_key").is_ok()
        && schema
            .get_field("content")
            .is_ok_and(|f| schema.get_field_entry(f).is_stored())
        && schema.get_field("file_name").is_err()
        && schema.get_field("created_at").is_err()
}

static STATE: Lazy<Mutex<Option<TantivyApi>>> = Lazy::new(|| Mutex::new(None));

/// Runs `f` with the initialized API, holding the global state lock.
fn with_api<T>(f: impl FnOnce(&TantivyApi) -> Result<T>) -> Result<T> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;
    f(api)
}

// Tantivy 인덱스를 초기화하는 함수
// 초기화는 빠른 작업이므로 sync로 처리
#[flutter_rust_bridge::frb(sync)]
pub fn init_tantivy(dir_path: String) -> Result<()> {
    let mut state_lock = STATE.lock().unwrap();
    if state_lock.is_some() {
        // 이미 초기화된 경우
        return Ok(());
    }

    let index_dir = PathBuf::from(dir_path);
    std::fs::create_dir_all(&index_dir)?;

    // 기존 인덱스가 현재 스키마와 맞으면 열고, 아니면 지우고 새로 생성
    let existing = if index_dir.join("meta.json").exists() {
        let index = Index::open_in_dir(&index_dir)?;
        if is_current_schema(&index.schema()) {
            Some(index)
        } else {
            drop(index);
            std::fs::remove_dir_all(&index_dir)?;
            std::fs::create_dir_all(&index_dir)?;
            None
        }
    } else {
        None
    };
    let index = match existing {
        Some(index) => index,
        None => Index::create_in_dir(&index_dir, build_schema())?,
    };

    let schema = index.schema();
    let field = |name: &str| {
        schema
            .get_field(name)
            .map_err(|_| anyhow!("'{name}' field not found"))
    };

    let writer = index.writer_with_num_threads(1, 15_000_000)?; // 1 thread, 15MB heap for embedded efficiency

    // Manual 정책: 각 읽기 함수가 searcher()에서 직접 reload
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()?;

    let api = TantivyApi {
        id_field: field("id")?,
        title_field: field("title")?,
        content_field: field("content")?,
        preview_text_field: field("preview_text")?,
        is_pinned_field: field("is_pinned")?,
        updated_at_field: field("updated_at")?,
        sort_key_field: field("sort_key")?,
        index,
        writer: Mutex::new(writer),
        reader,
    };

    *state_lock = Some(api);

    Ok(())
}

// [CREATE] 새 문서를 추가하는 함수 (같은 ID가 있으면 교체)
// 즉시 commit하므로 단일 문서 추가에 적합
// 대량 추가는 add_documents_batch 사용 권장
pub fn add_document(doc: Document) -> Result<()> {
    add_documents_batch(vec![doc])
}

use std::ops::Range;

#[inline]
fn edit_distance_le_one(a: &[u8], b: &[u8]) -> bool {
    let m = a.len();
    let n = b.len();

    if (m as isize - n as isize).abs() > 1 {
        return false;
    }

    let mut i = 0;
    let mut j = 0;
    let mut diff = 0;

    while i < m && j < n {
        if a[i].to_ascii_lowercase() == b[j].to_ascii_lowercase() {
            i += 1;
            j += 1;
        } else {
            diff += 1;
            if diff > 1 {
                return false;
            }
            if m > n {
                i += 1;
            } else if n > m {
                j += 1;
            } else {
                // Check single transposition (e.g. "teh" vs "the")
                if i + 1 < m
                    && j + 1 < n
                    && a[i].to_ascii_lowercase() == b[j + 1].to_ascii_lowercase()
                    && a[i + 1].to_ascii_lowercase() == b[j].to_ascii_lowercase()
                {
                    i += 2;
                    j += 2;
                } else {
                    i += 1;
                    j += 1;
                }
            }
        }
    }

    if i < m || j < n {
        diff += 1;
    }

    diff <= 1
}

fn is_word_match(doc_word: &str, query_word: &str) -> bool {
    let d = doc_word.as_bytes();
    let q = query_word.as_bytes();

    if d.is_empty() || q.is_empty() {
        return false;
    }

    // 1. Direct prefix match (e.g. "microservice" starts with "micro")
    if d.len() >= q.len() && d[..q.len()].eq_ignore_ascii_case(q) {
        return true;
    }

    // 2. Fuzzy prefix match with Levenshtein edit distance <= 1 (e.g. "archtecture" vs "architecture")
    let q_len = q.len();
    let min_k = q_len.saturating_sub(1);
    let max_k = (q_len + 1).min(d.len());

    for k in min_k..=max_k {
        if edit_distance_le_one(&d[..k], q) {
            return true;
        }
    }
    false
}

fn find_matches_in_text(text: &str, query: &str) -> Vec<Range<usize>> {
    let query_words: Vec<&str> = query.split_whitespace().collect();
    if query_words.is_empty() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    let mut word_start: Option<usize> = None;

    for (byte_idx, ch) in text.char_indices() {
        if ch.is_alphanumeric() {
            if word_start.is_none() {
                word_start = Some(byte_idx);
            }
        } else if let Some(start) = word_start {
            let word = &text[start..byte_idx];
            if query_words.iter().any(|qw| is_word_match(word, qw)) {
                matches.push(start..byte_idx);
            }
            word_start = None;
        }
    }
    if let Some(start) = word_start {
        let word = &text[start..];
        if query_words.iter().any(|qw| is_word_match(word, qw)) {
            matches.push(start..text.len());
        }
    }

    matches
}

fn push_sanitized(out: &mut String, text: &str) {
    let mut prev_space = out.ends_with(' ');
    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_space && !out.is_empty() {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(c);
            prev_space = false;
        }
    }
}

/// Formats a single-line preview centered around the first match
fn format_centered_snippet(text: &str, matches: &[Range<usize>]) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }

    if matches.is_empty() {
        let trimmed = text.trim();
        let cleaned = trimmed.trim_start_matches(|c: char| c == '#' || c == '-' || c == '>' || c == ' ');
        if cleaned.is_empty() {
            return None;
        }
        let mut snippet = String::with_capacity(84);
        let mut chars = cleaned.chars();
        for _ in 0..80 {
            if let Some(c) = chars.next() {
                snippet.push(if c == '\n' || c == '\r' { ' ' } else { c });
            } else {
                break;
            }
        }
        if chars.next().is_some() {
            snippet.push_str("...");
        }
        return Some(snippet);
    }

    let first_match = &matches[0];
    let total_bytes = text.len();

    let start_target = first_match.start.saturating_sub(30);
    let end_target = (first_match.end + 60).min(total_bytes);

    let mut snippet_start = start_target;
    while snippet_start > 0 && !text.is_char_boundary(snippet_start) {
        snippet_start -= 1;
    }
    if snippet_start > 0 {
        if let Some(pos) = text[snippet_start..first_match.start].find(char::is_whitespace) {
            snippet_start += pos + 1;
        }
    }

    let mut snippet_end = end_target;
    while snippet_end < total_bytes && !text.is_char_boundary(snippet_end) {
        snippet_end += 1;
    }
    if snippet_end < total_bytes {
        if let Some(pos) = text[first_match.end..snippet_end].rfind(char::is_whitespace) {
            snippet_end = first_match.end + pos;
        }
    }

    let mut result = String::with_capacity(128);
    if snippet_start > 0 {
        result.push_str("... ");
    }

    let mut curr = snippet_start;
    for m in matches {
        if m.end <= snippet_start || m.start >= snippet_end {
            continue;
        }
        let term_start = m.start.max(snippet_start);
        let term_end = m.end.min(snippet_end);

        if term_start > curr {
            push_sanitized(&mut result, &text[curr..term_start]);
        }
        result.push_str("<b>");
        result.push_str(&text[term_start..term_end]);
        result.push_str("</b>");
        curr = term_end;
    }

    if curr < snippet_end {
        push_sanitized(&mut result, &text[curr..snippet_end]);
    }

    if snippet_end < total_bytes {
        result.push_str(" ...");
    }

    Some(result.trim().to_string())
}

// [READ] 쿼리로 문서를 검색하는 함수 (관련도 순)
pub fn search_documents(query: String, top_k: usize) -> Result<Vec<SearchResult>> {
    with_api(|api| {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let searcher = api.searcher()?;
        let mut query_parser =
            QueryParser::for_index(&api.index, vec![api.title_field, api.content_field]);
        query_parser.set_field_fuzzy(api.title_field, true, 1, true);
        query_parser.set_field_fuzzy(api.content_field, true, 1, true);
        let parsed_query = query_parser.parse_query(&query)?;
        let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(top_k))?;

        top_docs
            .into_iter()
            .map(|(score, doc_address)| {
                let retrieved_doc = searcher.doc::<TantivyDocument>(doc_address)?;
                let doc = api.extract_document(&retrieved_doc);
                let matches = find_matches_in_text(&doc.content, &query);
                let snippet = format_centered_snippet(&doc.content, &matches);
                Ok(SearchResult {
                    score,
                    doc,
                    snippet,
                })
            })
            .collect()
    })
}

/// Notes ordered pinned first, then most recently updated.
pub fn get_notes(skip: usize, take: usize) -> Result<Vec<Document>> {
    with_api(|api| {
        let searcher = api.searcher()?;
        let top_docs = TopDocs::with_limit(take)
            .and_offset(skip)
            .order_by_fast_field::<i64>("sort_key", Order::Desc);
        let results = searcher.search(&AllQuery, &top_docs)?;

        results
            .into_iter()
            .map(|(_sort_key, doc_address)| {
                let retrieved_doc = searcher.doc::<TantivyDocument>(doc_address)?;
                Ok(api.extract_document(&retrieved_doc))
            })
            .collect()
    })
}

// [READ] ID로 특정 문서를 가져오는 함수
// ID 조회는 비교적 빠른 작업이므로 sync로 처리
#[flutter_rust_bridge::frb(sync)]
pub fn get_document_by_id(id: String) -> Result<Option<Document>> {
    with_api(|api| api.find_by_id(&api.searcher()?, &id))
}

pub fn toggle_pin_document(id: String) -> Result<bool> {
    with_api(|api| {
        let mut doc = api
            .find_by_id(&api.searcher()?, &id)?
            .ok_or_else(|| anyhow!("Note with id {} not found", id))?;
        doc.is_pinned = !doc.is_pinned;

        let mut writer = api.writer.lock().unwrap();
        api.upsert(&writer, &doc)?;
        writer.commit()?;

        Ok(doc.is_pinned)
    })
}

pub fn update_document(doc: Document) -> Result<()> {
    // add_document가 내부적으로 delete & add 로직을 수행하므로 그대로 호출
    add_document(doc)
}

// [DELETE] 문서를 삭제하는 함수
pub fn delete_document(id: String) -> Result<()> {
    delete_documents_batch(vec![id])
}

// [BATCH] 여러 문서를 한 번에 추가하는 함수 (성능 최적화)
pub fn add_documents_batch(docs: Vec<Document>) -> Result<()> {
    with_api(|api| {
        let mut writer = api.writer.lock().unwrap();
        for doc in &docs {
            api.upsert(&writer, doc)?;
        }
        // 모든 문서를 추가한 후 한 번만 commit
        writer.commit()?;
        Ok(())
    })
}

// [BATCH] 여러 문서를 한 번에 삭제하는 함수 (성능 최적화)
pub fn delete_documents_batch(ids: Vec<String>) -> Result<()> {
    with_api(|api| {
        let mut writer = api.writer.lock().unwrap();
        for id in &ids {
            writer.delete_term(api.id_term(id));
        }
        // 모든 삭제 작업 후 한 번만 commit
        writer.commit()?;
        Ok(())
    })
}

// [UTILITY] 명시적으로 commit을 수행하는 함수
// add_document_no_commit과 함께 사용하여 수동으로 트랜잭션 제어
#[flutter_rust_bridge::frb(sync)]
pub fn commit() -> Result<()> {
    with_api(|api| {
        api.writer.lock().unwrap().commit()?;
        Ok(())
    })
}

// [CREATE] commit 없이 문서를 추가하는 함수 (고급 사용자용)
// 여러 작업을 수행한 후 commit()을 호출하여 성능 최적화
pub fn add_document_no_commit(doc: Document) -> Result<()> {
    with_api(|api| {
        let writer = api.writer.lock().unwrap();
        api.upsert(&writer, &doc)
    })
}

pub fn delete_document_no_commit(id: String) -> Result<()> {
    with_api(|api| {
        let writer = api.writer.lock().unwrap();
        writer.delete_term(api.id_term(&id));
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_test_dir(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("tantivy_test_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.to_string_lossy().to_string()
    }

    #[test]
    fn test_fuzzy_and_prefix_search() {
        let path = temp_test_dir("fuzzy");
        init_tantivy(path.clone()).unwrap();

        let doc = Document {
            id: "doc1".to_string(),
            title: "Meeting Notes".to_string(),
            content: "Discussing quarterly flutter release".to_string(),
            preview_text: "Meeting notes preview".to_string(),
            is_pinned: false,
            updated_at: 100,
        };
        add_document(doc).unwrap();

        // 1. Exact match
        let res = search_documents("meeting".to_string(), 10).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].doc.id, "doc1");

        // 2. Prefix search (search-as-you-type)
        let res = search_documents("meet".to_string(), 10).unwrap();
        assert_eq!(res.len(), 1, "prefix 'meet' should match 'Meeting'");

        let res = search_documents("flutt".to_string(), 10).unwrap();
        assert_eq!(res.len(), 1, "prefix 'flutt' should match 'flutter'");

        // 3. Fuzzy typo search (1 edit distance)
        let res = search_documents("meating".to_string(), 10).unwrap();
        assert_eq!(res.len(), 1, "typo 'meating' should match 'Meeting'");

        let res = search_documents("fluttr".to_string(), 10).unwrap();
        assert_eq!(res.len(), 1, "typo 'fluttr' should match 'flutter'");

        // 4. top_k = 0 should return empty list without crashing
        let res = search_documents("meeting".to_string(), 0).unwrap();
        assert_eq!(res.len(), 0);

        // 5. Snippet generation test
        let doc2 = Document {
            id: "doc2".to_string(),
            title: "Project Design".to_string(),
            content: "First paragraph. Second paragraph talking about system modularity details. Third conclusion.".to_string(),
            preview_text: "Project design preview".to_string(),
            is_pinned: false,
            updated_at: 101,
        };
        add_document(doc2).unwrap();
        let res = search_documents("modularity".to_string(), 10).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].doc.id, "doc2");
        assert!(res[0].snippet.is_some(), "Snippet should be generated");
        let snip = res[0].snippet.as_ref().unwrap();
        assert!(snip.contains("<b>modularity</b>"), "Snippet must contain the tagged matched term");

        // 6. Prefix & Typo snippet tests
        let doc3 = Document {
            id: "doc3".to_string(),
            title: "Database Performance".to_string(),
            content: "# Database Performance\nHigh performance search engines rely on inverted indexes.\n\nMicroservice architecture enables horizontal scalability. In addition, microservice reliability is critical.\n\n# Benchmarks".to_string(),
            preview_text: "# Database Performance preview".to_string(),
            is_pinned: false,
            updated_at: 102,
        };
        add_document(doc3).unwrap();

        // 6a. Prefix "micro" matches "Microservice" and produces centered snippet
        let res_prefix = search_documents("micro".to_string(), 10).unwrap();
        assert_eq!(res_prefix.len(), 1);
        assert!(res_prefix[0].snippet.is_some());
        let snip_prefix = res_prefix[0].snippet.as_ref().unwrap();
        assert!(snip_prefix.contains("<b>Microservice</b>"));
        assert!(!snip_prefix.starts_with("# Database Performance"), "Snippet should center on the match, not the doc start");

        // 6b. Typo "archtecture" (missing 'i') matches "architecture" and produces centered snippet
        let res_typo = search_documents("archtecture".to_string(), 10).unwrap();
        assert_eq!(res_typo.len(), 1);
        assert!(res_typo[0].snippet.is_some());
        let snip_typo = res_typo[0].snippet.as_ref().unwrap();
        assert!(snip_typo.contains("<b>architecture</b>"));
        assert!(!snip_typo.starts_with("# Database Performance"));

        let _ = std::fs::remove_dir_all(path);
    }
}
