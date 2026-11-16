use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

// Flutter에서 사용할 문서 구조체
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub preview_text: String,
    pub file_name: Option<String>,
    pub is_pinned: bool,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Everything the notes list and search results show.
#[derive(Debug, Clone)]
pub struct NoteMeta {
    pub id: String,
    pub title: String,
    pub preview_text: String,
    pub file_name: Option<String>,
    pub is_pinned: bool,
    pub created_at: i64,
    pub updated_at: i64,
}

// Flutter에서 사용할 검색 결과 구조체
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
    pub doc: Document,
}

// Tantivy의 핵심 로직을 관리하는 구조체
struct TantivyApi {
    index: Index,
    writer: Mutex<IndexWriter>,
    reader: IndexReader,
    id_field: Field,
    title_field: Field,
    preview_text_field: Field,
    file_name_field: Field,
    is_pinned_field: Field,
    created_at_field: Field,
    updated_at_field: Field,
    sort_key_field: Field,
}

impl TantivyApi {
    fn extract_meta(&self, retrieved_doc: &TantivyDocument) -> NoteMeta {
        let text = |field: Field| {
            retrieved_doc
                .get_first(field)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string()
        };
        NoteMeta {
            id: text(self.id_field),
            title: text(self.title_field),
            preview_text: text(self.preview_text_field),
            file_name: retrieved_doc
                .get_first(self.file_name_field)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            is_pinned: retrieved_doc
                .get_first(self.is_pinned_field)
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            created_at: retrieved_doc
                .get_first(self.created_at_field)
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
            updated_at: retrieved_doc
                .get_first(self.updated_at_field)
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
        }
    }

    fn extract_document(&self, retrieved_doc: &TantivyDocument) -> Document {
        let meta = self.extract_meta(retrieved_doc);
        Document {
            id: meta.id,
            title: meta.title,
            preview_text: meta.preview_text,
            file_name: meta.file_name,
            is_pinned: meta.is_pinned,
            created_at: meta.created_at,
            updated_at: meta.updated_at,
        }
    }

    fn to_tantivy_doc(&self, doc: &Document) -> TantivyDocument {
        let mut tantivy_doc = TantivyDocument::new();
        tantivy_doc.add_text(self.id_field, &doc.id);
        tantivy_doc.add_text(self.title_field, &doc.title);
        tantivy_doc.add_text(self.preview_text_field, &doc.preview_text);
        if let Some(ref file_name) = doc.file_name {
            tantivy_doc.add_text(self.file_name_field, file_name);
        }
        tantivy_doc.add_bool(self.is_pinned_field, doc.is_pinned);
        tantivy_doc.add_i64(self.created_at_field, doc.created_at);
        tantivy_doc.add_i64(self.updated_at_field, doc.updated_at);

        let sort_key: i64 =
            if doc.is_pinned { 1_i64 << 60 } else { 0 } | (doc.updated_at & 0x0FFF_FFFF_FFFF_FFFF);
        tantivy_doc.add_i64(self.sort_key_field, sort_key);

        tantivy_doc
    }
}

fn build_schema(builder: &mut SchemaBuilder) {
    builder.add_text_field("id", STRING | STORED);
    builder.add_text_field("title", TEXT | STORED);
    builder.add_text_field("preview_text", STORED);
    builder.add_text_field("file_name", STORED);
    builder.add_bool_field("is_pinned", STORED | FAST);
    builder.add_i64_field("created_at", STORED);
    builder.add_i64_field("updated_at", STORED | FAST);
    builder.add_i64_field("sort_key", FAST);
}

static STATE: Lazy<Arc<Mutex<Option<TantivyApi>>>> = Lazy::new(|| Arc::new(Mutex::new(None)));

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

    let (index, schema) = if index_dir.join("meta.json").exists() {
        // 기존 인덱스 열기
        let index = Index::open_in_dir(&index_dir)?;
        let schema = index.schema();
        if schema.get_field("preview_text").is_err()
            || schema.get_field("sort_key").is_err()
            || schema.get_field("content").is_ok()
        {
            drop(index);
            std::fs::remove_dir_all(&index_dir)?;
            std::fs::create_dir_all(&index_dir)?;
            let mut schema_builder = Schema::builder();
            build_schema(&mut schema_builder);
            let schema = schema_builder.build();
            let index = Index::create_in_dir(&index_dir, schema.clone())?;
            (index, schema)
        } else {
            (index, schema)
        }
    } else {
        // 새 인덱스 생성
        let mut schema_builder = Schema::builder();
        build_schema(&mut schema_builder);
        let schema = schema_builder.build();
        let index = Index::create_in_dir(&index_dir, schema.clone())?;
        (index, schema)
    };

    let id_field = schema
        .get_field("id")
        .map_err(|_| anyhow!("'id' field not found"))?;
    let title_field = schema
        .get_field("title")
        .map_err(|_| anyhow!("'title' field not found"))?;
    let preview_text_field = schema
        .get_field("preview_text")
        .map_err(|_| anyhow!("'preview_text' field not found"))?;
    let file_name_field = schema
        .get_field("file_name")
        .map_err(|_| anyhow!("'file_name' field not found"))?;
    let is_pinned_field = schema
        .get_field("is_pinned")
        .map_err(|_| anyhow!("'is_pinned' field not found"))?;
    let created_at_field = schema
        .get_field("created_at")
        .map_err(|_| anyhow!("'created_at' field not found"))?;
    let updated_at_field = schema
        .get_field("updated_at")
        .map_err(|_| anyhow!("'updated_at' field not found"))?;
    let sort_key_field = schema
        .get_field("sort_key")
        .map_err(|_| anyhow!("'sort_key' field not found"))?;

    let writer = index.writer_with_num_threads(1, 15_000_000)?; // 1 thread, 15MB heap for embedded efficiency

    // Reader를 생성하고 OnCommit 정책으로 자동 리로드
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()?;

    let api = TantivyApi {
        index,
        writer: Mutex::new(writer),
        reader,
        id_field,
        title_field,
        preview_text_field,
        file_name_field,
        is_pinned_field,
        created_at_field,
        updated_at_field,
        sort_key_field,
    };

    *state_lock = Some(api);

    Ok(())
}

// [CREATE] 새 문서를 추가하는 함수
// 즉시 commit하므로 단일 문서 추가에 적합
// 대량 추가는 add_documents_batch 사용 권장
pub fn add_document(doc: Document) -> Result<()> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    let mut writer = api.writer.lock().unwrap();

    // 추가하기 전에 동일한 ID의 문서가 있다면 삭제 (Update-or-Insert)
    let id_term = Term::from_field_text(api.id_field, &doc.id);
    writer.delete_term(id_term);

    let tantivy_doc = api.to_tantivy_doc(&doc);
    writer.add_document(tantivy_doc)?;
    writer.commit()?;

    Ok(())
}

// [READ] 쿼리로 문서를 검색하는 함수
pub fn search_documents(query: String, top_k: usize) -> Result<Vec<SearchResult>> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    // reader를 리로드하여 최신 변경사항을 반영
    api.reader.reload()?;

    // 전역 reader 재사용
    let searcher = api.reader.searcher();

    let query_parser = QueryParser::for_index(&api.index, vec![api.title_field]);
    let parsed_query = query_parser.parse_query(&query)?;

    let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(top_k))?;

    let mut results = Vec::new();
    for (score, doc_address) in top_docs {
        let retrieved_doc = searcher.doc::<TantivyDocument>(doc_address)?;
        results.push(SearchResult {
            score,
            doc: api.extract_document(&retrieved_doc),
        });
    }

    Ok(results)
}

pub fn get_notes(skip: usize, take: usize) -> Result<Vec<Document>> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    api.reader.reload()?;
    let searcher = api.reader.searcher();

    let query = tantivy::query::AllQuery;
    let top_docs = TopDocs::with_limit(take)
        .and_offset(skip)
        .order_by_fast_field::<i64>("sort_key", tantivy::Order::Desc);

    let results = searcher.search(&query, &top_docs)?;

    let mut docs = Vec::new();
    for (_score, doc_address) in results {
        let retrieved_doc = searcher.doc::<TantivyDocument>(doc_address)?;
        docs.push(api.extract_document(&retrieved_doc));
    }

    Ok(docs)
}

/// Like [get_notes] (pinned first, then most recently updated), but returns
/// only metadata.
pub fn get_notes_metadata(skip: usize, take: usize) -> Result<Vec<NoteMeta>> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    api.reader.reload()?;
    let searcher = api.reader.searcher();

    let top_docs = TopDocs::with_limit(take)
        .and_offset(skip)
        .order_by_fast_field::<i64>("sort_key", tantivy::Order::Desc);
    let results = searcher.search(&tantivy::query::AllQuery, &top_docs)?;

    let mut metas = Vec::with_capacity(results.len());
    for (_sort_key, doc_address) in results {
        let retrieved_doc = searcher.doc::<TantivyDocument>(doc_address)?;
        metas.push(api.extract_meta(&retrieved_doc));
    }
    Ok(metas)
}

/// Like [search_documents] (ordered by relevance), but returns only metadata.
pub fn search_notes_metadata(query: String, top_k: usize) -> Result<Vec<NoteMeta>> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    api.reader.reload()?;
    let searcher = api.reader.searcher();

    let query_parser = QueryParser::for_index(&api.index, vec![api.title_field]);
    let parsed_query = query_parser.parse_query(&query)?;
    let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(top_k))?;

    let mut metas = Vec::with_capacity(top_docs.len());
    for (_score, doc_address) in top_docs {
        let retrieved_doc = searcher.doc::<TantivyDocument>(doc_address)?;
        metas.push(api.extract_meta(&retrieved_doc));
    }
    Ok(metas)
}

// [READ] ID로 특정 문서를 가져오는 함수
// ID 조회는 비교적 빠른 작업이므로 sync로 처리
#[flutter_rust_bridge::frb(sync)]
pub fn get_document_by_id(id: String) -> Result<Option<Document>> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    // 전역 reader 재사용
    api.reader.reload()?;
    let searcher = api.reader.searcher();

    let id_term = Term::from_field_text(api.id_field, &id);
    let query = tantivy::query::TermQuery::new(id_term, IndexRecordOption::Basic);

    let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;

    if let Some((_, doc_address)) = top_docs.first() {
        let retrieved_doc = searcher.doc::<TantivyDocument>(*doc_address)?;
        return Ok(Some(api.extract_document(&retrieved_doc)));
    }

    Ok(None)
}

pub fn toggle_pin_document(id: String) -> Result<bool> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    api.reader.reload()?;
    let searcher = api.reader.searcher();
    let id_term = Term::from_field_text(api.id_field, &id);
    let query = tantivy::query::TermQuery::new(id_term.clone(), IndexRecordOption::Basic);

    let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;

    if let Some((_, doc_address)) = top_docs.first() {
        let retrieved_doc = searcher.doc::<TantivyDocument>(*doc_address)?;
        let mut doc = api.extract_document(&retrieved_doc);
        doc.is_pinned = !doc.is_pinned;
        let new_status = doc.is_pinned;

        let mut writer = api.writer.lock().unwrap();
        writer.delete_term(id_term);
        let tantivy_doc = api.to_tantivy_doc(&doc);
        writer.add_document(tantivy_doc)?;
        writer.commit()?;

        return Ok(new_status);
    }

    Err(anyhow!("Note with id {} not found", id))
}

pub fn update_document(doc: Document) -> Result<()> {
    // add_document가 내부적으로 delete & add 로직을 수행하므로 그대로 호출
    add_document(doc)
}

// [DELETE] 문서를 삭제하는 함수
pub fn delete_document(id: String) -> Result<()> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    let mut writer = api.writer.lock().unwrap();
    let id_term = Term::from_field_text(api.id_field, &id);

    writer.delete_term(id_term);
    writer.commit()?;

    Ok(())
}

// [BATCH] 여러 문서를 한 번에 추가하는 함수 (성능 최적화)
pub fn add_documents_batch(docs: Vec<Document>) -> Result<()> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    let mut writer = api.writer.lock().unwrap();

    for doc in docs {
        // 기존 문서가 있다면 삭제 (Update-or-Insert)
        let id_term = Term::from_field_text(api.id_field, &doc.id);
        writer.delete_term(id_term);
        let tantivy_doc = api.to_tantivy_doc(&doc);
        writer.add_document(tantivy_doc)?;
    }

    // 모든 문서를 추가한 후 한 번만 commit
    writer.commit()?;

    Ok(())
}

// [BATCH] 여러 문서를 한 번에 삭제하는 함수 (성능 최적화)
pub fn delete_documents_batch(ids: Vec<String>) -> Result<()> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    let mut writer = api.writer.lock().unwrap();

    for id in ids {
        let id_term = Term::from_field_text(api.id_field, &id);
        writer.delete_term(id_term);
    }

    // 모든 삭제 작업 후 한 번만 commit
    writer.commit()?;

    Ok(())
}

// [UTILITY] 명시적으로 commit을 수행하는 함수
// add_document_no_commit과 함께 사용하여 수동으로 트랜잭션 제어
#[flutter_rust_bridge::frb(sync)]
pub fn commit() -> Result<()> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    let mut writer = api.writer.lock().unwrap();
    writer.commit()?;

    Ok(())
}

// [CREATE] commit 없이 문서를 추가하는 함수 (고급 사용자용)
// 여러 작업을 수행한 후 commit()을 호출하여 성능 최적화
pub fn add_document_no_commit(doc: Document) -> Result<()> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    let writer = api.writer.lock().unwrap();

    let id_term = Term::from_field_text(api.id_field, &doc.id);
    writer.delete_term(id_term);
    let tantivy_doc = api.to_tantivy_doc(&doc);

    writer.add_document(tantivy_doc)?;

    Ok(())
}

pub fn delete_document_no_commit(id: String) -> Result<()> {
    let state_lock = STATE.lock().unwrap();
    let api = state_lock
        .as_ref()
        .ok_or_else(|| anyhow!("Tantivy not initialized"))?;

    let writer = api.writer.lock().unwrap();
    let id_term = Term::from_field_text(api.id_field, &id);

    writer.delete_term(id_term);

    Ok(())
}
