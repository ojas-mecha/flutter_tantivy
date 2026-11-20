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
    builder.add_text_field("content", TEXT);
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
            .is_ok_and(|f| !schema.get_field_entry(f).is_stored())
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

// [READ] 쿼리로 문서를 검색하는 함수 (관련도 순)
pub fn search_documents(query: String, top_k: usize) -> Result<Vec<SearchResult>> {
    with_api(|api| {
        let searcher = api.searcher()?;
        let query_parser =
            QueryParser::for_index(&api.index, vec![api.title_field, api.content_field]);
        let parsed_query = query_parser.parse_query(&query)?;
        let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(top_k))?;

        top_docs
            .into_iter()
            .map(|(score, doc_address)| {
                let retrieved_doc = searcher.doc::<TantivyDocument>(doc_address)?;
                Ok(SearchResult {
                    score,
                    doc: api.extract_document(&retrieved_doc),
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
