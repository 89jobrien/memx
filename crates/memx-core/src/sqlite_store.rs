use crate::error::{MemxError, Result};
use crate::store::Store;
use crate::types::*;
use chrono::{NaiveDate, Utc};
use rusqlite::{Connection, ffi::sqlite3_auto_extension, params};
use std::path::Path;
use std::sync::Once;

static VEC_INIT: Once = Once::new();

fn ensure_vec_extension() {
    VEC_INIT.call_once(|| unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

pub struct SqliteStore {
    conn: Connection,
}

impl SqliteStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        ensure_vec_extension();
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    pub fn open_in_memory() -> Result<Self> {
        ensure_vec_extension();
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    pub fn open_with_dimensions(path: impl AsRef<Path>, dims: usize) -> Result<Self> {
        ensure_vec_extension();
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.migrate_with_dimensions(dims)?;
        Ok(store)
    }

    pub fn open_in_memory_with_dimensions(dims: usize) -> Result<Self> {
        ensure_vec_extension();
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.migrate_with_dimensions(dims)?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.migrate_with_dimensions(4)
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
        let section = entry.section.as_str().to_string();
        let created = entry.created_at.to_rfc3339();
        let updated = entry.updated_at.to_rfc3339();

        self.conn.execute(
            "INSERT INTO entries (id, section, content, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, section, entry.content, created, updated],
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

        if changed == 0 {
            return Err(MemxError::NotFound(entry.id));
        }

        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        self.conn.execute(
            "UPDATE entries_vec SET embedding = ?1 WHERE id = ?2",
            params![blob, id],
        )?;

        Ok(())
    }

    fn delete_entry(&self, id: EntryId) -> Result<()> {
        let id_str = id.to_string();
        self.conn
            .execute("DELETE FROM entries WHERE id = ?1", params![id_str])?;
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

                    Ok(MemoryEntry {
                        id,
                        section: parse_section(&section_str),
                        content: row.get(2)?,
                        created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                            .unwrap()
                            .with_timezone(&Utc),
                        updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
                            .unwrap()
                            .with_timezone(&Utc),
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
                let rows = stmt.query_map(params![s.as_str()], |row| row_to_entry(row))?;
                for row in rows {
                    entries.push(row?);
                }
            }
            None => {
                let mut stmt = self.conn.prepare(
                    "SELECT id, section, content, created_at, updated_at
                     FROM entries ORDER BY created_at",
                )?;
                let rows = stmt.query_map([], |row| row_to_entry(row))?;
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
        self.conn.execute(
            "INSERT INTO sessions
             (id, date, session_number, goal, deliverables, decisions, open_threads)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session.id.to_string(),
                session.date.to_string(),
                session.session_number,
                session.goal,
                serde_json::to_string(&session.deliverables).unwrap(),
                serde_json::to_string(&session.decisions).unwrap(),
                serde_json::to_string(&session.open_threads).unwrap(),
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
            Ok(SessionLog {
                id: id_str.parse().unwrap(),
                date: NaiveDate::parse_from_str(&date_str, "%Y-%m-%d").unwrap(),
                session_number: row.get(2)?,
                goal: row.get(3)?,
                deliverables: serde_json::from_str(&deliverables_str).unwrap(),
                decisions: serde_json::from_str(&decisions_str).unwrap(),
                open_threads: serde_json::from_str(&threads_str).unwrap(),
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
        _filter: Option<&SearchFilter>,
    ) -> Result<Vec<SearchResult>> {
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
            Ok(SearchResult {
                source: MatchSource::Memory(id_str.parse().unwrap()),
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

fn parse_section(s: &str) -> Section {
    match s {
        "active_threads" => Section::ActiveThreads,
        "environment_notes" => Section::EnvironmentNotes,
        "pending_decisions" => Section::PendingDecisions,
        other => Section::Custom(other.to_string()),
    }
}

fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryEntry> {
    let id_str: String = row.get(0)?;
    let section_str: String = row.get(1)?;
    let created_str: String = row.get(3)?;
    let updated_str: String = row.get(4)?;

    Ok(MemoryEntry {
        id: id_str.parse().unwrap(),
        section: parse_section(&section_str),
        content: row.get(2)?,
        created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
            .unwrap()
            .with_timezone(&Utc),
        updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
            .unwrap()
            .with_timezone(&Utc),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> SqliteStore {
        SqliteStore::open_in_memory().unwrap()
    }

    #[test]
    fn open_and_migrate() {
        let store = test_store();
        let count = store.total_chars(None).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn insert_and_retrieve_entry() {
        let store = test_store();
        let entry = MemoryEntry::new(Section::ActiveThreads, "working on memx".into());
        let fake_embedding = vec![0.1_f32; 4];
        store.insert_entry(&entry, &fake_embedding).unwrap();

        let retrieved = store.get_entry(entry.id).unwrap();
        assert_eq!(retrieved.content, "working on memx");
        assert_eq!(retrieved.section, Section::ActiveThreads);
    }

    #[test]
    fn total_chars_by_section() {
        let store = test_store();
        let e1 = MemoryEntry::new(Section::ActiveThreads, "hello".into());
        let e2 = MemoryEntry::new(Section::EnvironmentNotes, "world!!!".into());
        let emb = vec![0.1_f32; 4];
        store.insert_entry(&e1, &emb).unwrap();
        store.insert_entry(&e2, &emb).unwrap();

        let total = store.total_chars(None).unwrap();
        assert_eq!(total, 13); // "hello" + "world!!!"

        let threads_only = store.total_chars(Some(&Section::ActiveThreads)).unwrap();
        assert_eq!(threads_only, 5);
    }

    #[test]
    fn delete_entry() {
        let store = test_store();
        let entry = MemoryEntry::new(Section::ActiveThreads, "temp".into());
        let emb = vec![0.1_f32; 4];
        store.insert_entry(&entry, &emb).unwrap();
        store.delete_entry(entry.id).unwrap();

        let result = store.get_entry(entry.id);
        assert!(result.is_err());
    }

    #[test]
    fn vector_search_returns_results() {
        let store = test_store();
        let e1 = MemoryEntry::new(Section::ActiveThreads, "rust programming".into());
        let e2 = MemoryEntry::new(Section::ActiveThreads, "python scripting".into());
        store.insert_entry(&e1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        store.insert_entry(&e2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let results = store
            .search_similar(&[0.9, 0.1, 0.0, 0.0], 2, None)
            .unwrap();
        assert_eq!(results.len(), 2);
        // First result should be closer to the query vector
        assert!(results[0].score >= results[1].score);
        assert_eq!(results[0].content, "rust programming");
    }
}
