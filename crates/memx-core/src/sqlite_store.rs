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

impl Store for SqliteStore {
    fn insert_entry(&self, entry: &MemoryEntry, embedding: &[f32]) -> Result<()> {
        let id = entry.id.to_string();
        let created = entry.created_at.to_rfc3339();
        let updated = entry.updated_at.to_rfc3339();

        self.conn.execute(
            "INSERT INTO entries (id, section, content, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, entry.section.as_str(), entry.content, created, updated],
        )?;

        // sqlite-vec expects embedding as a raw byte blob of little-endian f32s
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        self.conn.execute(
            "INSERT INTO entries_vec (id, embedding) VALUES (?1, ?2)",
            params![id, blob],
        )?;

        Ok(())
    }

    fn update_entry(&self, entry: &MemoryEntry, embedding: &[f32]) -> Result<()> {
        let id = entry.id.to_string();
        let section = entry.section.as_str().to_string();
        let updated = Utc::now().to_rfc3339();

        let changed = self.conn.execute(
            "UPDATE entries SET section = ?1, content = ?2, updated_at = ?3
             WHERE id = ?4",
            params![section, entry.content, updated, id],
        )?;

        require_affected(changed, entry.id)?;

        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        self.conn.execute(
            "UPDATE entries_vec SET embedding = ?1 WHERE id = ?2",
            params![blob, id],
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
                    let created_str: String = row.get(3)?;
                    let updated_str: String = row.get(4)?;

                    let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                        .map_err(|e| {
                            rusqlite::Error::FromSqlConversionFailure(
                                3,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            )
                        })?
                        .with_timezone(&Utc);
                    let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_str)
                        .map_err(|e| {
                            rusqlite::Error::FromSqlConversionFailure(
                                4,
                                rusqlite::types::Type::Text,
                                Box::new(e),
                            )
                        })?
                        .with_timezone(&Utc);

                    Ok(MemoryEntry {
                        id,
                        section: section_str
                            .parse()
                            .unwrap_or(Section::Custom(section_str.clone())),
                        content: row.get(2)?,
                        created_at,
                        updated_at,
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
            let id_str: String = row.get(0)?;
            let date_str: String = row.get(1)?;
            let deliverables_str: String = row.get(4)?;
            let decisions_str: String = row.get(5)?;
            let threads_str: String = row.get(6)?;

            let id = id_str.parse().map_err(|e: ulid::DecodeError| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            let parsed_date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d").map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            let deliverables: Vec<String> =
                serde_json::from_str(&deliverables_str).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        4,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
            let decisions: Vec<String> = serde_json::from_str(&decisions_str).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    5,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            let open_threads: Vec<String> = serde_json::from_str(&threads_str).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    6,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;

            Ok(SessionLog {
                id,
                date: parsed_date,
                session_number: row.get(2)?,
                goal: row.get(3)?,
                deliverables,
                decisions,
                open_threads,
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
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut stmt = self.conn.prepare(
            "SELECT v.id, v.distance, e.content
             FROM entries_vec v
             JOIN entries e ON e.id = v.id
             WHERE v.embedding MATCH ?1
             AND k = ?2
             ORDER BY v.distance",
        )?;

        let rows = stmt.query_map(params![blob, top_k as i64], |row| {
            let id_str: String = row.get(0)?;
            let distance: f64 = row.get(1)?;
            let content: String = row.get(2)?;
            let entry_id = id_str.parse().map_err(|e: ulid::DecodeError| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            Ok(SearchResult {
                source: MatchSource::Memory(entry_id),
                content,
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

fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryEntry> {
    let id_str: String = row.get(0)?;
    let section_str: String = row.get(1)?;
    let created_str: String = row.get(3)?;
    let updated_str: String = row.get(4)?;

    let id = id_str.parse().map_err(|e: ulid::DecodeError| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
        .map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e))
        })?
        .with_timezone(&Utc);
    let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_str)
        .map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(e))
        })?
        .with_timezone(&Utc);

    Ok(MemoryEntry {
        id,
        section: section_str
            .parse()
            .unwrap_or(Section::Custom(section_str.clone())),
        content: row.get(2)?,
        created_at,
        updated_at,
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
    fn open_creates_db_file() {
        let dir = tempfile::tempdir().expect("tempdir should create");
        let path = dir.path().join("test.db");
        let store = SqliteStore::open(&path, 4).expect("open should succeed");
        assert_eq!(store.total_chars(None).expect("total chars"), 0);
        assert!(path.exists());
    }

    #[test]
    fn open_and_migrate() {
        let store = test_store();
        let count = store.total_chars(None).expect("empty store");
        assert_eq!(count, 0);
    }

    #[test]
    fn insert_and_retrieve_entry() {
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
    fn update_entry_happy_path() {
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
    fn update_entry_not_found() {
        let store = test_store();
        let entry = MemoryEntry::new(Section::ActiveThreads, "ghost".into());
        let result = store.update_entry(&entry, &emb());
        assert!(matches!(result, Err(MemxError::NotFound(_))));
    }

    #[test]
    fn delete_entry_existing() {
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
    fn delete_entry_not_found() {
        let store = test_store();
        let id = EntryId::new();
        let result = store.delete_entry(id);
        assert!(matches!(result, Err(MemxError::NotFound(_))));
    }

    #[test]
    fn get_entry_not_found() {
        let store = test_store();
        let id = EntryId::new();
        let result = store.get_entry(id);
        assert!(matches!(result, Err(MemxError::NotFound(_))));
    }

    #[test]
    fn list_entries_all() {
        let store = test_store();
        let e1 = MemoryEntry::new(Section::ActiveThreads, "first".into());
        let e2 = MemoryEntry::new(Section::EnvironmentNotes, "second".into());
        store.insert_entry(&e1, &emb()).expect("insert e1");
        store.insert_entry(&e2, &emb()).expect("insert e2");

        let all = store.list_entries(None).expect("list should succeed");
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn list_entries_by_section() {
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
    fn list_entries_empty() {
        let store = test_store();
        let entries = store.list_entries(None).expect("list empty should succeed");
        assert!(entries.is_empty());
    }

    #[test]
    fn total_chars_by_section() {
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
    fn session_roundtrip() {
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
    fn session_for_date_empty() {
        let store = test_store();
        let date = NaiveDate::from_ymd_opt(2026, 1, 1).expect("valid date");
        let sessions = store.get_sessions_for_date(date).expect("empty date");
        assert!(sessions.is_empty());
    }

    #[test]
    fn insert_transcript_happy_path() {
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
    fn search_with_filter_returns_error() {
        let store = test_store();
        let filter = SearchFilter::Section(Section::ActiveThreads);
        let result = store.search_similar(&[0.1; 4], 5, Some(&filter));
        assert!(matches!(result, Err(MemxError::Other(_))));
    }

    #[test]
    fn vector_search_returns_results() {
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
    fn vector_search_empty_store() {
        let store = test_store();
        let results = store
            .search_similar(&[1.0, 0.0, 0.0, 0.0], 5, None)
            .expect("search empty");
        assert!(results.is_empty());
    }

    // ── Conformance: Store trait contract ────────────────────────

    fn assert_store_contract(store: &dyn Store) {
        // Insert
        let entry = MemoryEntry::new(Section::ActiveThreads, "contract test".into());
        let embedding = vec![0.5_f32; 4];
        store
            .insert_entry(&entry, &embedding)
            .expect("contract: insert");

        // Get
        let retrieved = store.get_entry(entry.id).expect("contract: get");
        assert_eq!(retrieved.id, entry.id);
        assert_eq!(retrieved.content, "contract test");

        // List
        let all = store.list_entries(None).expect("contract: list all");
        assert!(all.iter().any(|e| e.id == entry.id));

        let filtered = store
            .list_entries(Some(&Section::ActiveThreads))
            .expect("contract: list filtered");
        assert!(filtered.iter().any(|e| e.id == entry.id));

        // Total chars
        let chars = store.total_chars(None).expect("contract: total chars");
        assert!(chars >= "contract test".len());

        // Search
        let results = store
            .search_similar(&embedding, 1, None)
            .expect("contract: search");
        assert!(!results.is_empty());

        // Delete
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
}
