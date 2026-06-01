use crate::error::{MemxError, Result};
use crate::store::Store;
use crate::types::{
    EntryId, MatchSource, MemoryEntry, SearchFilter, SearchResult, Section, SessionLog,
    TranscriptChunk,
};
use chrono::{NaiveDate, Utc};
use rusqlite::{Connection, ffi::sqlite3_auto_extension, params};
use std::path::Path;
use std::sync::Once;

static VEC_INIT: Once = Once::new();

fn ensure_vec_extension() {
    VEC_INIT.call_once(|| unsafe {
        // SAFETY: `sqlite3_vec_init` has the exact signature required by
        // `sqlite3_auto_extension` — it is a valid SQLite extension entry point.
        // The transmute converts between compatible function-pointer types (both
        // are nullable pointers to C functions with the same ABI). This block
        // executes exactly once via `Once::call_once`.
        #[allow(clippy::missing_transmute_annotations)]
        sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

pub struct SqliteStore {
    conn: Connection,
}

impl SqliteStore {
    pub fn open(path: impl AsRef<Path>, dims: usize) -> Result<Self> {
        ensure_vec_extension();
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.migrate_with_dimensions(dims)?;
        Ok(store)
    }

    pub fn open_in_memory(dims: usize) -> Result<Self> {
        ensure_vec_extension();
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.migrate_with_dimensions(dims)?;
        Ok(store)
    }

    fn migrate_with_dimensions(&self, dims: usize) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                section TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                session_number INTEGER NOT NULL,
                goal TEXT,
                deliverables TEXT NOT NULL DEFAULT '[]',
                decisions TEXT NOT NULL DEFAULT '[]',
                open_threads TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS transcripts (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );",
        )?;

        let exists: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master
             WHERE type='table' AND name='entries_vec'",
            [],
            |row| row.get(0),
        )?;

        if !exists {
            // SAFETY: dims is usize — no SQL injection risk
            self.conn.execute_batch(&format!(
                "CREATE VIRTUAL TABLE entries_vec USING vec0(
                    id TEXT PRIMARY KEY,
                    embedding float[{dims}]
                );"
            ))?;
        }

        Ok(())
    }
}

fn embedding_blob(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn parse_rfc3339(
    s: &str,
    col: usize,
) -> std::result::Result<chrono::DateTime<Utc>, rusqlite::Error> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(col, rusqlite::types::Type::Text, Box::new(e))
        })
}

fn parse_ulid(s: &str, col: usize) -> std::result::Result<EntryId, rusqlite::Error> {
    s.parse().map_err(|e: ulid::DecodeError| {
        rusqlite::Error::FromSqlConversionFailure(col, rusqlite::types::Type::Text, Box::new(e))
    })
}

fn parse_json_vec(s: &str, col: usize) -> std::result::Result<Vec<String>, rusqlite::Error> {
    serde_json::from_str(s).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(col, rusqlite::types::Type::Text, Box::new(e))
    })
}

impl Store for SqliteStore {
    fn insert_entry(&self, entry: &MemoryEntry, embedding: &[f32]) -> Result<()> {
        let id = entry.id.to_string();
        self.conn.execute(
            "INSERT INTO entries (id, section, content, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                id,
                entry.section.as_str(),
                entry.content,
                entry.created_at.to_rfc3339(),
                entry.updated_at.to_rfc3339()
            ],
        )?;
        self.conn.execute(
            "INSERT INTO entries_vec (id, embedding) VALUES (?1, ?2)",
            params![id, embedding_blob(embedding)],
        )?;
        Ok(())
    }

    fn update_entry(&self, entry: &MemoryEntry, embedding: &[f32]) -> Result<()> {
        let id = entry.id.to_string();
        let changed = self.conn.execute(
            "UPDATE entries SET section = ?1, content = ?2, updated_at = ?3
             WHERE id = ?4",
            params![
                entry.section.as_str().to_string(),
                entry.content,
                Utc::now().to_rfc3339(),
                id
            ],
        )?;
        require_affected(changed, entry.id)?;
        self.conn.execute(
            "UPDATE entries_vec SET embedding = ?1 WHERE id = ?2",
            params![embedding_blob(embedding), id],
        )?;
        Ok(())
    }

    fn delete_entry(&self, id: EntryId) -> Result<()> {
        let id_str = id.to_string();
        let changed = self
            .conn
            .execute("DELETE FROM entries WHERE id = ?1", params![id_str])?;
        require_affected(changed, id)?;
        self.conn
            .execute("DELETE FROM entries_vec WHERE id = ?1", params![id_str])?;
        Ok(())
    }

    fn get_entry(&self, id: EntryId) -> Result<MemoryEntry> {
        let id_str = id.to_string();
        self.conn
            .query_row(
                "SELECT id, section, content, created_at, updated_at
                 FROM entries WHERE id = ?1",
                params![id_str],
                |row| {
                    let section_str: String = row.get(1)?;
                    Ok(MemoryEntry {
                        id,
                        section: section_str
                            .parse()
                            .unwrap_or(Section::Custom(section_str.clone())),
                        content: row.get(2)?,
                        created_at: parse_rfc3339(&row.get::<_, String>(3)?, 3)?,
                        updated_at: parse_rfc3339(&row.get::<_, String>(4)?, 4)?,
                    })
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => MemxError::NotFound(id),
                other => MemxError::Storage(other),
            })
    }

    fn list_entries(&self, section: Option<&Section>) -> Result<Vec<MemoryEntry>> {
        let mut entries = Vec::new();
        match section {
            Some(s) => {
                let mut stmt = self.conn.prepare(
                    "SELECT id, section, content, created_at, updated_at
                     FROM entries WHERE section = ?1 ORDER BY created_at",
                )?;
                let rows = stmt.query_map(params![s.as_str()], row_to_entry)?;
                for row in rows {
                    entries.push(row?);
                }
            }
            None => {
                let mut stmt = self.conn.prepare(
                    "SELECT id, section, content, created_at, updated_at
                     FROM entries ORDER BY created_at",
                )?;
                let rows = stmt.query_map([], row_to_entry)?;
                for row in rows {
                    entries.push(row?);
                }
            }
        }
        Ok(entries)
    }

    fn total_chars(&self, section: Option<&Section>) -> Result<usize> {
        let total: i64 = match section {
            Some(s) => self.conn.query_row(
                "SELECT COALESCE(SUM(LENGTH(content)), 0)
                 FROM entries WHERE section = ?1",
                params![s.as_str()],
                |row| row.get(0),
            )?,
            None => self.conn.query_row(
                "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM entries",
                [],
                |row| row.get(0),
            )?,
        };
        Ok(total as usize)
    }

    fn insert_session(&self, session: &SessionLog) -> Result<()> {
        let deliverables = serde_json::to_string(&session.deliverables)
            .map_err(|e| MemxError::Serialization(e.to_string()))?;
        let decisions = serde_json::to_string(&session.decisions)
            .map_err(|e| MemxError::Serialization(e.to_string()))?;
        let open_threads = serde_json::to_string(&session.open_threads)
            .map_err(|e| MemxError::Serialization(e.to_string()))?;

        self.conn.execute(
            "INSERT INTO sessions
             (id, date, session_number, goal, deliverables, decisions, open_threads)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session.id.to_string(),
                session.date.to_string(),
                session.session_number,
                session.goal,
                deliverables,
                decisions,
                open_threads,
            ],
        )?;
        Ok(())
    }

    fn get_sessions_for_date(&self, date: NaiveDate) -> Result<Vec<SessionLog>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, date, session_number, goal,
                    deliverables, decisions, open_threads
             FROM sessions WHERE date = ?1
             ORDER BY session_number",
        )?;
        let rows = stmt.query_map(params![date.to_string()], |row| {
            let date_str: String = row.get(1)?;
            let parsed_date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d").map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;

            Ok(SessionLog {
                id: parse_ulid(&row.get::<_, String>(0)?, 0)?,
                date: parsed_date,
                session_number: row.get(2)?,
                goal: row.get(3)?,
                deliverables: parse_json_vec(&row.get::<_, String>(4)?, 4)?,
                decisions: parse_json_vec(&row.get::<_, String>(5)?, 5)?,
                open_threads: parse_json_vec(&row.get::<_, String>(6)?, 6)?,
            })
        })?;
        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row?);
        }
        Ok(sessions)
    }

    fn insert_transcript(&self, chunk: &TranscriptChunk) -> Result<()> {
        self.conn.execute(
            "INSERT INTO transcripts (id, session_id, timestamp, content)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                chunk.id.to_string(),
                chunk.session_id.to_string(),
                chunk.timestamp.to_rfc3339(),
                chunk.content,
            ],
        )?;
        Ok(())
    }

    fn search_similar(
        &self,
        embedding: &[f32],
        top_k: usize,
        filter: Option<&SearchFilter>,
    ) -> Result<Vec<SearchResult>> {
        if filter.is_some() {
            return Err(MemxError::Other(anyhow::anyhow!(
                "search filters not yet implemented"
            )));
        }

        let mut stmt = self.conn.prepare(
            "SELECT v.id, v.distance, e.content
             FROM entries_vec v
             JOIN entries e ON e.id = v.id
             WHERE v.embedding MATCH ?1
             AND k = ?2
             ORDER BY v.distance",
        )?;

        let rows = stmt.query_map(params![embedding_blob(embedding), top_k as i64], |row| {
            let distance: f64 = row.get(1)?;
            Ok(SearchResult {
                source: MatchSource::Memory(parse_ulid(&row.get::<_, String>(0)?, 0)?),
                content: row.get(2)?,
                score: 1.0 - distance as f32,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}

fn require_affected(changed: usize, id: EntryId) -> Result<()> {
    if changed == 0 {
        return Err(MemxError::NotFound(id));
    }
    Ok(())
}

// Column indices for SELECT id, section, content, created_at, updated_at
const COL_ID: usize = 0;
const COL_SECTION: usize = 1;
const COL_CONTENT: usize = 2;
const COL_CREATED_AT: usize = 3;
const COL_UPDATED_AT: usize = 4;

fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryEntry> {
    let id_str: String = row.get(COL_ID)?;
    let section_str: String = row.get(COL_SECTION)?;

    Ok(MemoryEntry {
        id: parse_ulid(&id_str, COL_ID)?,
        section: section_str
            .parse()
            .unwrap_or(Section::Custom(section_str.clone())),
        content: row.get(COL_CONTENT)?,
        created_at: parse_rfc3339(&row.get::<_, String>(COL_CREATED_AT)?, COL_CREATED_AT)?,
        updated_at: parse_rfc3339(&row.get::<_, String>(COL_UPDATED_AT)?, COL_UPDATED_AT)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MemxError;

    fn test_store() -> SqliteStore {
        SqliteStore::open_in_memory(4).expect("in-memory store should open")
    }

    fn emb() -> Vec<f32> {
        vec![0.1_f32; 4]
    }

    // ── Unit tests ──────────────────────────────────────────────

    #[test]
    fn sqlite_store_open_creates_db_file() {
        let dir = tempfile::tempdir().expect("tempdir should create");
        let path = dir.path().join("test.db");
        let store = SqliteStore::open(&path, 4).expect("open should succeed");
        assert_eq!(store.total_chars(None).expect("total chars"), 0);
        assert!(path.exists());
    }

    #[test]
    fn sqlite_store_open_and_migrate() {
        let store = test_store();
        let count = store.total_chars(None).expect("empty store");
        assert_eq!(count, 0);
    }

    #[test]
    fn sqlite_store_insert_and_retrieve_entry() {
        let store = test_store();
        let entry = MemoryEntry::new(Section::ActiveThreads, "working on memx".into());
        store
            .insert_entry(&entry, &emb())
            .expect("insert should succeed");

        let retrieved = store.get_entry(entry.id).expect("get should succeed");
        assert_eq!(retrieved.content, "working on memx");
        assert_eq!(retrieved.section, Section::ActiveThreads);
    }

    #[test]
    fn sqlite_store_update_entry_happy_path() {
        let store = test_store();
        let mut entry = MemoryEntry::new(Section::ActiveThreads, "original".into());
        store
            .insert_entry(&entry, &emb())
            .expect("insert should succeed");

        entry.content = "updated".into();
        entry.section = Section::EnvironmentNotes;
        store
            .update_entry(&entry, &emb())
            .expect("update should succeed");

        let retrieved = store.get_entry(entry.id).expect("get should succeed");
        assert_eq!(retrieved.content, "updated");
        assert_eq!(retrieved.section, Section::EnvironmentNotes);
    }

    #[test]
    fn sqlite_store_update_entry_not_found() {
        let store = test_store();
        let entry = MemoryEntry::new(Section::ActiveThreads, "ghost".into());
        let result = store.update_entry(&entry, &emb());
        assert!(matches!(result, Err(MemxError::NotFound(_))));
    }

    #[test]
    fn sqlite_store_delete_entry_existing() {
        let store = test_store();
        let entry = MemoryEntry::new(Section::ActiveThreads, "temp".into());
        store
            .insert_entry(&entry, &emb())
            .expect("insert should succeed");
        store.delete_entry(entry.id).expect("delete should succeed");

        let result = store.get_entry(entry.id);
        assert!(matches!(result, Err(MemxError::NotFound(_))));
    }

    #[test]
    fn sqlite_store_delete_entry_not_found() {
        let store = test_store();
        let id = EntryId::new();
        let result = store.delete_entry(id);
        assert!(matches!(result, Err(MemxError::NotFound(_))));
    }

    #[test]
    fn sqlite_store_get_entry_not_found() {
        let store = test_store();
        let id = EntryId::new();
        let result = store.get_entry(id);
        assert!(matches!(result, Err(MemxError::NotFound(_))));
    }

    #[test]
    fn sqlite_store_list_entries_all() {
        let store = test_store();
        let e1 = MemoryEntry::new(Section::ActiveThreads, "first".into());
        let e2 = MemoryEntry::new(Section::EnvironmentNotes, "second".into());
        store.insert_entry(&e1, &emb()).expect("insert e1");
        store.insert_entry(&e2, &emb()).expect("insert e2");

        let all = store.list_entries(None).expect("list should succeed");
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn sqlite_store_list_entries_by_section() {
        let store = test_store();
        let e1 = MemoryEntry::new(Section::ActiveThreads, "a".into());
        let e2 = MemoryEntry::new(Section::EnvironmentNotes, "b".into());
        let e3 = MemoryEntry::new(Section::ActiveThreads, "c".into());
        store.insert_entry(&e1, &emb()).expect("insert e1");
        store.insert_entry(&e2, &emb()).expect("insert e2");
        store.insert_entry(&e3, &emb()).expect("insert e3");

        let threads = store
            .list_entries(Some(&Section::ActiveThreads))
            .expect("list filtered");
        assert_eq!(threads.len(), 2);
        assert!(threads.iter().all(|e| e.section == Section::ActiveThreads));
    }

    #[test]
    fn sqlite_store_list_entries_empty() {
        let store = test_store();
        let entries = store.list_entries(None).expect("list empty should succeed");
        assert!(entries.is_empty());
    }

    #[test]
    fn sqlite_store_total_chars_by_section() {
        let store = test_store();
        let e1 = MemoryEntry::new(Section::ActiveThreads, "hello".into());
        let e2 = MemoryEntry::new(Section::EnvironmentNotes, "world!!!".into());
        store.insert_entry(&e1, &emb()).expect("insert e1");
        store.insert_entry(&e2, &emb()).expect("insert e2");

        let total = store.total_chars(None).expect("total chars");
        assert_eq!(total, 13); // "hello" + "world!!!"

        let threads_only = store
            .total_chars(Some(&Section::ActiveThreads))
            .expect("section chars");
        assert_eq!(threads_only, 5);
    }

    #[test]
    fn sqlite_store_session_roundtrip() {
        let store = test_store();
        let date = NaiveDate::from_ymd_opt(2026, 5, 31).expect("valid date");
        let mut session = SessionLog::new(date, 1);
        session.goal = Some("implement tests".into());
        session.deliverables = vec!["unit tests".into(), "property tests".into()];
        session.decisions = vec!["use proptest".into()];
        session.open_threads = vec!["fuzz targets".into()];

        store.insert_session(&session).expect("insert session");

        let sessions = store.get_sessions_for_date(date).expect("get sessions");
        assert_eq!(sessions.len(), 1);

        let s = &sessions[0];
        assert_eq!(s.session_number, 1);
        assert_eq!(s.goal.as_deref(), Some("implement tests"));
        assert_eq!(s.deliverables.len(), 2);
        assert_eq!(s.decisions, vec!["use proptest"]);
        assert_eq!(s.open_threads, vec!["fuzz targets"]);
    }

    #[test]
    fn sqlite_store_session_for_date_empty() {
        let store = test_store();
        let date = NaiveDate::from_ymd_opt(2026, 1, 1).expect("valid date");
        let sessions = store.get_sessions_for_date(date).expect("empty date");
        assert!(sessions.is_empty());
    }

    #[test]
    fn sqlite_store_insert_transcript_happy_path() {
        let store = test_store();
        let date = NaiveDate::from_ymd_opt(2026, 5, 31).expect("valid date");
        let session = SessionLog::new(date, 1);
        store.insert_session(&session).expect("insert session");

        let chunk = TranscriptChunk {
            id: EntryId::new(),
            session_id: session.id,
            timestamp: Utc::now(),
            content: "user asked about testing".into(),
        };
        store.insert_transcript(&chunk).expect("insert transcript");
    }

    #[test]
    fn sqlite_store_search_with_filter_returns_error() {
        let store = test_store();
        let filter = SearchFilter::Section(Section::ActiveThreads);
        let result = store.search_similar(&[0.1; 4], 5, Some(&filter));
        assert!(matches!(result, Err(MemxError::Other(_))));
    }

    #[test]
    fn sqlite_store_vector_search_returns_results() {
        let store = test_store();
        let e1 = MemoryEntry::new(Section::ActiveThreads, "rust programming".into());
        let e2 = MemoryEntry::new(Section::ActiveThreads, "python scripting".into());
        store
            .insert_entry(&e1, &[1.0, 0.0, 0.0, 0.0])
            .expect("insert e1");
        store
            .insert_entry(&e2, &[0.0, 1.0, 0.0, 0.0])
            .expect("insert e2");

        let results = store
            .search_similar(&[0.9, 0.1, 0.0, 0.0], 2, None)
            .expect("search should succeed");
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
        assert_eq!(results[0].content, "rust programming");
    }

    #[test]
    fn sqlite_store_vector_search_empty() {
        let store = test_store();
        let results = store
            .search_similar(&[1.0, 0.0, 0.0, 0.0], 5, None)
            .expect("search empty");
        assert!(results.is_empty());
    }

    // ── Conformance: Store trait contract ────────────────────────

    fn assert_store_contract(store: &dyn Store) {
        let entry = MemoryEntry::new(Section::ActiveThreads, "contract test".into());
        let embedding = vec![0.5_f32; 4];
        store
            .insert_entry(&entry, &embedding)
            .expect("contract: insert");

        let retrieved = store.get_entry(entry.id).expect("contract: get");
        assert_eq!(retrieved.id, entry.id);
        assert_eq!(retrieved.content, "contract test");

        let all = store.list_entries(None).expect("contract: list all");
        assert!(all.iter().any(|e| e.id == entry.id));

        let filtered = store
            .list_entries(Some(&Section::ActiveThreads))
            .expect("contract: list filtered");
        assert!(filtered.iter().any(|e| e.id == entry.id));

        let chars = store.total_chars(None).expect("contract: total chars");
        assert!(chars >= "contract test".len());

        let results = store
            .search_similar(&embedding, 1, None)
            .expect("contract: search");
        assert!(!results.is_empty());

        store.delete_entry(entry.id).expect("contract: delete");
        let gone = store.get_entry(entry.id);
        assert!(
            matches!(gone, Err(MemxError::NotFound(_))),
            "contract: deleted entry should be NotFound"
        );
    }

    #[test]
    fn sqlite_store_satisfies_store_contract() {
        let store = test_store();
        assert_store_contract(&store);
    }

    // ── Unit edge cases ────────────────────────────────────────────

    #[test]
    fn sqlite_store_transcript_without_session_fk_fails() {
        let store = test_store();
        store
            .conn
            .execute_batch("PRAGMA foreign_keys = ON")
            .expect("pragma");
        let chunk = TranscriptChunk {
            id: EntryId::new(),
            session_id: EntryId::new(),
            timestamp: Utc::now(),
            content: "orphan chunk".into(),
        };
        let result = store.insert_transcript(&chunk);
        assert!(result.is_err(), "FK violation should fail with PRAGMA on");
    }

    #[test]
    fn sqlite_store_search_score_in_zero_one_range() {
        let store = test_store();
        let e = MemoryEntry::new(Section::ActiveThreads, "test".into());
        store
            .insert_entry(&e, &[1.0, 0.0, 0.0, 0.0])
            .expect("insert");
        let results = store
            .search_similar(&[1.0, 0.0, 0.0, 0.0], 1, None)
            .expect("search");
        assert_eq!(results.len(), 1);
        assert!(
            results[0].score >= 0.0 && results[0].score <= 1.0,
            "score {} out of [0,1] range",
            results[0].score
        );
    }

    #[test]
    fn sqlite_store_session_with_unicode_roundtrips() {
        let store = test_store();
        let date = NaiveDate::from_ymd_opt(2026, 6, 1).expect("valid");
        let mut session = SessionLog::new(date, 1);
        session.goal = Some("implement \u{1F980} crab tests".into());
        session.deliverables = vec!["\u{00E9}l\u{00E8}ve".into()];
        session.decisions = vec!["\u{4F60}\u{597D}".into()];
        store.insert_session(&session).expect("insert");

        let sessions = store.get_sessions_for_date(date).expect("get");
        assert_eq!(
            sessions[0].goal.as_deref(),
            Some("implement \u{1F980} crab tests")
        );
        assert_eq!(sessions[0].deliverables[0], "\u{00E9}l\u{00E8}ve");
        assert_eq!(sessions[0].decisions[0], "\u{4F60}\u{597D}");
    }

    #[test]
    fn sqlite_store_insert_entry_with_empty_content() {
        let store = test_store();
        let e = MemoryEntry::new(Section::ActiveThreads, String::new());
        store.insert_entry(&e, &emb()).expect("insert empty");
        let retrieved = store.get_entry(e.id).expect("get");
        assert_eq!(retrieved.content, "");
        assert_eq!(store.total_chars(None).expect("chars"), 0);
    }

    #[test]
    fn sqlite_store_duplicate_insert_fails() {
        let store = test_store();
        let e = MemoryEntry::new(Section::ActiveThreads, "dup".into());
        store.insert_entry(&e, &emb()).expect("first insert");
        let result = store.insert_entry(&e, &emb());
        assert!(result.is_err(), "duplicate PK insert should fail");
    }

    // ── Property tests ─────────────────────────────────────────────

    mod property {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn sqlite_store_insert_get_roundtrip_content(content in "\\PC{0,500}") {
                let store = test_store();
                let entry = MemoryEntry::new(Section::ActiveThreads, content.clone());
                store.insert_entry(&entry, &emb()).expect("insert");
                let retrieved = store.get_entry(entry.id).expect("get");
                prop_assert_eq!(retrieved.content, content);
            }

            #[test]
            fn sqlite_store_total_chars_matches_sum(
                contents in proptest::collection::vec("\\PC{1,100}", 1..10)
            ) {
                let store = test_store();
                let mut expected_chars = 0usize;
                for c in &contents {
                    let entry = MemoryEntry::new(Section::ActiveThreads, c.clone());
                    store.insert_entry(&entry, &emb()).expect("insert");
                    expected_chars += c.chars().count();
                }
                let total = store.total_chars(None).expect("total");
                prop_assert_eq!(total, expected_chars);
            }

            #[test]
            fn sqlite_store_list_filtered_never_leaks(
                a_count in 1..5usize,
                b_count in 1..5usize,
            ) {
                let store = test_store();
                for _ in 0..a_count {
                    let e = MemoryEntry::new(Section::ActiveThreads, "a".into());
                    store.insert_entry(&e, &emb()).expect("insert a");
                }
                for _ in 0..b_count {
                    let e = MemoryEntry::new(Section::EnvironmentNotes, "b".into());
                    store.insert_entry(&e, &emb()).expect("insert b");
                }
                let filtered = store
                    .list_entries(Some(&Section::ActiveThreads))
                    .expect("list");
                prop_assert_eq!(filtered.len(), a_count);
                for e in &filtered {
                    prop_assert_eq!(&e.section, &Section::ActiveThreads);
                }
            }

            #[test]
            fn sqlite_store_delete_then_get_is_not_found(
                content in "\\PC{1,100}",
                section in prop_oneof![
                    Just(Section::ActiveThreads),
                    Just(Section::EnvironmentNotes),
                ],
            ) {
                let store = test_store();
                let entry = MemoryEntry::new(section, content);
                store.insert_entry(&entry, &emb()).expect("insert");
                store.delete_entry(entry.id).expect("delete");
                let result = store.get_entry(entry.id);
                prop_assert!(matches!(result, Err(MemxError::NotFound(_))));
            }
        }
    }
}
