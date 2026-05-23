# Plan: Foundational Type System

## Goal

Define memx's core types, storage layer (SQLite + sqlite-vec), and embedding
trait with two backends (fastembed, ort) — enough to store, embed, and search
memory entries from Rust.

## Architecture

- Crates affected: `memx-core`, `memx-embed`
- New traits/types:
  - `memx-core`: `EntryId`, `Section`, `MemoryEntry`, `SessionLog`,
    `TranscriptChunk`, `UserProfile`, `Budget`, `WriteAction`, `WriteResult`,
    `SearchQuery`, `SearchFilter`, `SearchResult`, `SearchMatch`, `MatchSource`,
    `SourceKind`, `Store` trait, `SqliteStore` impl
  - `memx-embed`: `Embedder` trait, `FastEmbedBackend`, `OrtBackend`
- Data flow: text -> Embedder::embed() -> Vec<f32> -> SqliteStore::upsert()
  -> vec0 virtual table; query -> embed -> vec_distance_cosine -> SearchResult
- DB: single file at `~/.memx/memx.db`

## Tech Stack

- Rust edition 2024
- `rusqlite` (bundled, loadable_extension) + `sqlite-vec` crate
- `fastembed` 5.x (default embedder, bundles bge-m3)
- `ort` 2.x + `tokenizers` (alt embedder for custom ONNX)
- `ulid` for EntryId (temporal ordering)
- `chrono` for timestamps
- `serde` + `serde_json` for serialization
- `thiserror` for error types
- `clap` (cli, later plan)

## Tasks

### Task 1: Scaffold workspace

**Crate**: workspace root
**File(s)**: `Cargo.toml`, `crates/memx-core/Cargo.toml`,
`crates/memx-core/src/lib.rs`, `crates/memx-embed/Cargo.toml`,
`crates/memx-embed/src/lib.rs`

1. Create workspace Cargo.toml:

   ```toml
   [workspace]
   resolver = "3"
   members = ["crates/*"]

   [workspace.package]
   edition = "2024"
   license = "MIT OR Apache-2.0"
   authors = ["Joseph O'Brien"]

   [workspace.dependencies]
   chrono = { version = "0.4", features = ["serde"] }
   serde = { version = "1", features = ["derive"] }
   serde_json = "1"
   thiserror = "2"
   ulid = { version = "1", features = ["serde"] }
   rusqlite = { version = "0.34", features = ["bundled", "loadable_extension"] }
   anyhow = "1"
   ```

2. Create `crates/memx-core/Cargo.toml`:

   ```toml
   [package]
   name = "memx-core"
   version = "0.1.0"
   edition.workspace = true
   license.workspace = true
   authors.workspace = true

   [dependencies]
   chrono.workspace = true
   serde.workspace = true
   serde_json.workspace = true
   thiserror.workspace = true
   ulid.workspace = true
   rusqlite.workspace = true
   anyhow.workspace = true

   [dev-dependencies]
   tempfile = "3"
   ```

3. Create `crates/memx-core/src/lib.rs`:

   ```rust
   pub mod types;
   pub mod error;
   ```

4. Create `crates/memx-embed/Cargo.toml`:

   ```toml
   [package]
   name = "memx-embed"
   version = "0.1.0"
   edition.workspace = true
   license.workspace = true
   authors.workspace = true

   [dependencies]
   anyhow.workspace = true

   [dev-dependencies]
   ```

5. Create `crates/memx-embed/src/lib.rs`:

   ```rust
   pub mod embedder;
   ```

6. Verify:

   ```
   cargo check --workspace   -> compiles
   ```

7. Commit: `git commit -m "chore: scaffold memx workspace with core and embed crates"`

---

### Task 2: Define core identity and section types

**Crate**: `memx-core`
**File(s)**: `crates/memx-core/src/types.rs`
**Run**: `cargo nextest run -p memx-core`

1. Write failing test:

   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn entry_id_is_unique() {
           let a = EntryId::new();
           let b = EntryId::new();
           assert_ne!(a, b);
       }

       #[test]
       fn entry_id_roundtrips_string() {
           let id = EntryId::new();
           let s = id.to_string();
           let parsed: EntryId = s.parse().unwrap();
           assert_eq!(id, parsed);
       }

       #[test]
       fn section_display() {
           assert_eq!(Section::ActiveThreads.as_str(), "active_threads");
           assert_eq!(Section::EnvironmentNotes.as_str(), "environment_notes");
           assert_eq!(Section::PendingDecisions.as_str(), "pending_decisions");
           let custom = Section::Custom("my_section".into());
           assert_eq!(custom.as_str(), "my_section");
       }
   }
   ```

   Run: `cargo nextest run -p memx-core -- tests`
   Expected: FAIL (types don't exist)

2. Implement in `crates/memx-core/src/types.rs`:

   ```rust
   use chrono::{DateTime, NaiveDate, Utc};
   use serde::{Deserialize, Serialize};
   use std::fmt;
   use std::str::FromStr;
   use ulid::Ulid;

   #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
   pub struct EntryId(Ulid);

   impl EntryId {
       pub fn new() -> Self {
           Self(Ulid::new())
       }

       pub fn from_ulid(ulid: Ulid) -> Self {
           Self(ulid)
       }
   }

   impl Default for EntryId {
       fn default() -> Self {
           Self::new()
       }
   }

   impl fmt::Display for EntryId {
       fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
           write!(f, "{}", self.0)
       }
   }

   impl FromStr for EntryId {
       type Err = ulid::DecodeError;

       fn from_str(s: &str) -> Result<Self, Self::Err> {
           Ok(Self(Ulid::from_str(s)?))
       }
   }

   #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
   pub enum Section {
       ActiveThreads,
       EnvironmentNotes,
       PendingDecisions,
       Custom(String),
   }

   impl Section {
       pub fn as_str(&self) -> &str {
           match self {
               Self::ActiveThreads => "active_threads",
               Self::EnvironmentNotes => "environment_notes",
               Self::PendingDecisions => "pending_decisions",
               Self::Custom(s) => s.as_str(),
           }
       }
   }

   impl fmt::Display for Section {
       fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
           f.write_str(self.as_str())
       }
   }
   ```

3. Verify:

   ```
   cargo nextest run -p memx-core    -> all green
   cargo clippy -p memx-core -- -D warnings  -> zero warnings
   ```

4. Commit: `git commit -m "feat(core): add EntryId and Section types"`

---

### Task 3: Define memory entry, session, transcript, and profile types

**Crate**: `memx-core`
**File(s)**: `crates/memx-core/src/types.rs`
**Run**: `cargo nextest run -p memx-core`

1. Write failing test:

   ```rust
   #[test]
   fn memory_entry_creation() {
       let entry = MemoryEntry::new(
           Section::ActiveThreads,
           "working on memx type system".into(),
       );
       assert_eq!(entry.section, Section::ActiveThreads);
       assert_eq!(entry.content, "working on memx type system");
       assert!(entry.created_at <= Utc::now());
   }

   #[test]
   fn budget_check() {
       let budget = Budget::new(2500);
       assert!(budget.allows(2000));
       assert!(budget.allows(2500));
       assert!(!budget.allows(2501));
   }

   #[test]
   fn session_log_creation() {
       let log = SessionLog::new(NaiveDate::from_ymd_opt(2026, 5, 22).unwrap(), 1);
       assert_eq!(log.session_number, 1);
       assert!(log.goal.is_none());
   }
   ```

   Expected: FAIL

2. Implement (append to `types.rs`):

   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct MemoryEntry {
       pub id: EntryId,
       pub section: Section,
       pub content: String,
       pub created_at: DateTime<Utc>,
       pub updated_at: DateTime<Utc>,
   }

   impl MemoryEntry {
       pub fn new(section: Section, content: String) -> Self {
           let now = Utc::now();
           Self {
               id: EntryId::new(),
               section,
               content,
               created_at: now,
               updated_at: now,
           }
       }
   }

   #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
   pub struct Budget {
       pub max_chars: usize,
   }

   impl Budget {
       pub fn new(max_chars: usize) -> Self {
           Self { max_chars }
       }

       pub fn allows(&self, current_chars: usize) -> bool {
           current_chars <= self.max_chars
       }

       pub fn memory_default() -> Self {
           Self { max_chars: 2500 }
       }

       pub fn user_profile_default() -> Self {
           Self { max_chars: 1375 }
       }
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct SessionLog {
       pub id: EntryId,
       pub date: NaiveDate,
       pub session_number: u32,
       pub goal: Option<String>,
       pub deliverables: Vec<String>,
       pub decisions: Vec<String>,
       pub open_threads: Vec<String>,
   }

   impl SessionLog {
       pub fn new(date: NaiveDate, session_number: u32) -> Self {
           Self {
               id: EntryId::new(),
               date,
               session_number,
               goal: None,
               deliverables: Vec::new(),
               decisions: Vec::new(),
               open_threads: Vec::new(),
           }
       }
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct TranscriptChunk {
       pub id: EntryId,
       pub session_id: EntryId,
       pub timestamp: DateTime<Utc>,
       pub content: String,
   }

   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct UserProfile {
       pub about: String,
       pub preferences: Vec<String>,
       pub working_style: String,
       pub budget: Budget,
   }

   impl Default for UserProfile {
       fn default() -> Self {
           Self {
               about: String::new(),
               preferences: Vec::new(),
               working_style: String::new(),
               budget: Budget::user_profile_default(),
           }
       }
   }
   ```

3. Verify:

   ```
   cargo nextest run -p memx-core    -> all green
   cargo clippy -p memx-core -- -D warnings  -> zero warnings
   ```

4. Commit: `git commit -m "feat(core): add MemoryEntry, SessionLog, TranscriptChunk, UserProfile, Budget"`

---

### Task 4: Define search and write action types

**Crate**: `memx-core`
**File(s)**: `crates/memx-core/src/types.rs`
**Run**: `cargo nextest run -p memx-core`

1. Write failing test:

   ```rust
   #[test]
   fn search_query_defaults() {
       let q = SearchQuery::new("test query".into());
       assert_eq!(q.top_k, 5);
       assert!(q.filter.is_none());
   }

   #[test]
   fn write_action_variants() {
       let add = WriteAction::Add {
           section: Section::ActiveThreads,
           content: "hello".into(),
       };
       assert!(matches!(add, WriteAction::Add { .. }));

       let id = EntryId::new();
       let replace = WriteAction::Replace {
           target: id,
           content: "updated".into(),
       };
       assert!(matches!(replace, WriteAction::Replace { .. }));

       let remove = WriteAction::Remove { target: id };
       assert!(matches!(remove, WriteAction::Remove { .. }));
   }
   ```

   Expected: FAIL

2. Implement (append to `types.rs`):

   ```rust
   #[derive(Debug, Clone)]
   pub struct SearchQuery {
       pub text: String,
       pub top_k: usize,
       pub filter: Option<SearchFilter>,
   }

   impl SearchQuery {
       pub fn new(text: String) -> Self {
           Self {
               text,
               top_k: 5,
               filter: None,
           }
       }

       pub fn with_top_k(mut self, top_k: usize) -> Self {
           self.top_k = top_k;
           self
       }

       pub fn with_filter(mut self, filter: SearchFilter) -> Self {
           self.filter = Some(filter);
           self
       }
   }

   #[derive(Debug, Clone)]
   pub enum SearchFilter {
       Section(Section),
       DateRange {
           from: NaiveDate,
           to: NaiveDate,
       },
       Source(SourceKind),
   }

   #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
   pub enum SourceKind {
       Memory,
       Session,
       Transcript,
   }

   #[derive(Debug, Clone)]
   pub struct SearchResult {
       pub source: MatchSource,
       pub content: String,
       pub score: f32,
   }

   #[derive(Debug, Clone)]
   pub enum MatchSource {
       Memory(EntryId),
       Session {
           date: NaiveDate,
           session_number: u32,
       },
       Transcript {
           date: NaiveDate,
           timestamp: DateTime<Utc>,
       },
   }

   #[derive(Debug, Clone)]
   pub enum WriteAction {
       Add {
           section: Section,
           content: String,
       },
       Replace {
           target: EntryId,
           content: String,
       },
       Remove {
           target: EntryId,
       },
   }

   #[derive(Debug, Clone)]
   pub struct WriteResult {
       pub entry_id: EntryId,
       pub action: WriteActionKind,
       pub chars_used: usize,
       pub chars_remaining: usize,
       pub deduplicated: bool,
   }

   #[derive(Debug, Clone, Copy, PartialEq, Eq)]
   pub enum WriteActionKind {
       Added,
       Replaced,
       Removed,
   }
   ```

3. Verify:

   ```
   cargo nextest run -p memx-core    -> all green
   cargo clippy -p memx-core -- -D warnings  -> zero warnings
   ```

4. Commit: `git commit -m "feat(core): add SearchQuery, SearchFilter, WriteAction, WriteResult"`

---

### Task 5: Define error types

**Crate**: `memx-core`
**File(s)**: `crates/memx-core/src/error.rs`
**Run**: `cargo nextest run -p memx-core`

1. Write failing test:

   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn error_display() {
           let err = MemxError::BudgetExceeded {
               current: 2600,
               max: 2500,
           };
           let msg = err.to_string();
           assert!(msg.contains("2600"));
           assert!(msg.contains("2500"));
       }

       #[test]
       fn error_from_rusqlite() {
           let sql_err = rusqlite::Error::QueryReturnedNoRows;
           let err: MemxError = sql_err.into();
           assert!(matches!(err, MemxError::Storage(_)));
       }
   }
   ```

   Expected: FAIL

2. Implement in `crates/memx-core/src/error.rs`:

   ```rust
   use thiserror::Error;

   #[derive(Debug, Error)]
   pub enum MemxError {
       #[error("budget exceeded: {current} chars exceeds {max} char limit")]
       BudgetExceeded { current: usize, max: usize },

       #[error("entry not found: {0}")]
       NotFound(crate::types::EntryId),

       #[error("duplicate entry detected")]
       Duplicate,

       #[error("storage error: {0}")]
       Storage(#[from] rusqlite::Error),

       #[error("embedding error: {0}")]
       Embedding(String),

       #[error("{0}")]
       Other(#[from] anyhow::Error),
   }

   pub type Result<T> = std::result::Result<T, MemxError>;
   ```

3. Verify:

   ```
   cargo nextest run -p memx-core    -> all green
   cargo clippy -p memx-core -- -D warnings  -> zero warnings
   ```

4. Commit: `git commit -m "feat(core): add MemxError type with thiserror"`

---

### Task 6: Define Embedder trait

**Crate**: `memx-embed`
**File(s)**: `crates/memx-embed/src/embedder.rs`
**Run**: `cargo nextest run -p memx-embed`

1. Write failing test:

   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       struct MockEmbedder;

       impl Embedder for MockEmbedder {
           fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
               Ok(texts
                   .iter()
                   .map(|_| vec![0.1, 0.2, 0.3, 0.4])
                   .collect())
           }

           fn embed_one(&self, text: &str) -> anyhow::Result<Vec<f32>> {
               Ok(vec![0.1, 0.2, 0.3, 0.4])
           }

           fn dimensions(&self) -> usize {
               4
           }

           fn model_id(&self) -> &str {
               "mock-4d"
           }
       }

       #[test]
       fn mock_embedder_dimensions() {
           let e = MockEmbedder;
           assert_eq!(e.dimensions(), 4);
       }

       #[test]
       fn mock_embedder_batch() {
           let e = MockEmbedder;
           let results = e.embed(&["hello", "world"]).unwrap();
           assert_eq!(results.len(), 2);
           assert_eq!(results[0].len(), 4);
       }

       #[test]
       fn mock_embedder_single() {
           let e = MockEmbedder;
           let result = e.embed_one("hello").unwrap();
           assert_eq!(result.len(), 4);
       }
   }
   ```

   Expected: FAIL

2. Implement in `crates/memx-embed/src/embedder.rs`:

   ```rust
   pub trait Embedder: Send + Sync {
       fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;

       fn embed_one(&self, text: &str) -> anyhow::Result<Vec<f32>> {
           let mut results = self.embed(&[text])?;
           results
               .pop()
               .ok_or_else(|| anyhow::anyhow!("embed returned empty results"))
       }

       fn dimensions(&self) -> usize;

       fn model_id(&self) -> &str;
   }
   ```

3. Verify:

   ```
   cargo nextest run -p memx-embed    -> all green
   cargo clippy -p memx-embed -- -D warnings  -> zero warnings
   ```

4. Commit: `git commit -m "feat(embed): add Embedder trait with batch and single-text methods"`

---

### Task 7: Define Store trait

**Crate**: `memx-core`
**File(s)**: `crates/memx-core/src/store.rs`
**Run**: `cargo nextest run -p memx-core`

1. Write failing test:

   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       use crate::types::*;

       #[test]
       fn trait_is_object_safe() {
           // Compile-time check: Store can be used as dyn Store
           fn _assert_object_safe(_: &dyn Store) {}
       }
   }
   ```

   Expected: FAIL (Store doesn't exist)

2. Implement in `crates/memx-core/src/store.rs`:

   ```rust
   use crate::error::Result;
   use crate::types::*;
   use chrono::NaiveDate;

   pub trait Store: Send + Sync {
       // Memory entries
       fn insert_entry(&self, entry: &MemoryEntry, embedding: &[f32]) -> Result<()>;
       fn update_entry(&self, entry: &MemoryEntry, embedding: &[f32]) -> Result<()>;
       fn delete_entry(&self, id: EntryId) -> Result<()>;
       fn get_entry(&self, id: EntryId) -> Result<MemoryEntry>;
       fn list_entries(&self, section: Option<&Section>) -> Result<Vec<MemoryEntry>>;
       fn total_chars(&self, section: Option<&Section>) -> Result<usize>;

       // Sessions
       fn insert_session(&self, session: &SessionLog) -> Result<()>;
       fn get_sessions_for_date(&self, date: NaiveDate) -> Result<Vec<SessionLog>>;

       // Transcripts
       fn insert_transcript(&self, chunk: &TranscriptChunk) -> Result<()>;

       // Vector search
       fn search_similar(
           &self,
           embedding: &[f32],
           top_k: usize,
           filter: Option<&SearchFilter>,
       ) -> Result<Vec<SearchResult>>;
   }
   ```

3. Update `crates/memx-core/src/lib.rs`:

   ```rust
   pub mod error;
   pub mod store;
   pub mod types;
   ```

4. Verify:

   ```
   cargo nextest run -p memx-core    -> all green
   cargo clippy -p memx-core -- -D warnings  -> zero warnings
   ```

5. Commit: `git commit -m "feat(core): add Store trait for persistence abstraction"`

---

### Task 8: Implement SqliteStore with sqlite-vec

**Crate**: `memx-core`
**File(s)**: `crates/memx-core/src/sqlite_store.rs`
**Run**: `cargo nextest run -p memx-core`

1. Add `sqlite-vec` to workspace deps in root `Cargo.toml`:

   ```toml
   sqlite-vec = "0.1"
   ```

   Add to `crates/memx-core/Cargo.toml`:

   ```toml
   sqlite-vec.workspace = true
   ```

2. Write failing test:

   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       use crate::types::*;

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
           let entry = MemoryEntry::new(
               Section::ActiveThreads,
               "working on memx".into(),
           );
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

           let threads_only =
               store.total_chars(Some(&Section::ActiveThreads)).unwrap();
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
   ```

   Expected: FAIL

3. Implement in `crates/memx-core/src/sqlite_store.rs`:

   ```rust
   use crate::error::{MemxError, Result};
   use crate::store::Store;
   use crate::types::*;
   use chrono::{NaiveDate, Utc};
   use rusqlite::{params, Connection};
   use std::path::Path;

   pub struct SqliteStore {
       conn: Connection,
   }

   impl SqliteStore {
       pub fn open(path: impl AsRef<Path>) -> Result<Self> {
           let conn = Connection::open(path)?;
           let store = Self { conn };
           store.load_vec_extension()?;
           store.migrate()?;
           Ok(store)
       }

       pub fn open_in_memory() -> Result<Self> {
           let conn = Connection::open_in_memory()?;
           let store = Self { conn };
           store.load_vec_extension()?;
           store.migrate()?;
           Ok(store)
       }

       fn load_vec_extension(&self) -> Result<()> {
           unsafe {
               self.conn.load_extension_enable()?;
               sqlite_vec::load(&self.conn)?;
               self.conn.load_extension_disable()?;
           }
           Ok(())
       }

       fn migrate(&self) -> Result<()> {
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

           // vec0 virtual table — dimension set at creation time.
           // We use 4 for tests; real usage re-creates with model dim.
           // Check if table exists first.
           let exists: bool = self.conn.query_row(
               "SELECT COUNT(*) > 0 FROM sqlite_master
                WHERE type='table' AND name='entries_vec'",
               [],
               |row| row.get(0),
           )?;

           if !exists {
               self.conn.execute_batch(
                   "CREATE VIRTUAL TABLE entries_vec USING vec0(
                       id TEXT PRIMARY KEY,
                       embedding float[4]
                   );",
               )?;
           }

           Ok(())
       }

       pub fn open_with_dimensions(
           path: impl AsRef<Path>,
           dims: usize,
       ) -> Result<Self> {
           let conn = Connection::open(path)?;
           let store = Self { conn };
           store.load_vec_extension()?;
           store.migrate_with_dimensions(dims)?;
           Ok(store)
       }

       pub fn open_in_memory_with_dimensions(dims: usize) -> Result<Self> {
           let conn = Connection::open_in_memory()?;
           let store = Self { conn };
           store.load_vec_extension()?;
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

           self.conn.execute(
               "INSERT INTO entries_vec (id, embedding) VALUES (?1, ?2)",
               params![id, embedding.to_vec()],
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

           self.conn.execute(
               "UPDATE entries_vec SET embedding = ?1 WHERE id = ?2",
               params![embedding.to_vec(), id],
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
                   let rows = stmt.query_map(params![s.as_str()], |row| {
                       row_to_entry(row)
                   })?;
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

       fn get_sessions_for_date(
           &self,
           date: NaiveDate,
       ) -> Result<Vec<SessionLog>> {
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
           let mut stmt = self.conn.prepare(
               "SELECT v.id, v.distance, e.content
                FROM entries_vec v
                JOIN entries e ON e.id = v.id
                WHERE v.embedding MATCH ?1
                ORDER BY v.distance
                LIMIT ?2",
           )?;

           let rows = stmt.query_map(
               params![embedding.to_vec(), top_k as i64],
               |row| {
                   let id_str: String = row.get(0)?;
                   let distance: f64 = row.get(1)?;
                   let content: String = row.get(2)?;
                   Ok(SearchResult {
                       source: MatchSource::Memory(id_str.parse().unwrap()),
                       content,
                       score: 1.0 - distance as f32,
                   })
               },
           )?;

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
   ```

4. Update `crates/memx-core/src/lib.rs`:

   ```rust
   pub mod error;
   pub mod sqlite_store;
   pub mod store;
   pub mod types;
   ```

5. Verify:

   ```
   cargo nextest run -p memx-core    -> all green
   cargo clippy -p memx-core -- -D warnings  -> zero warnings
   ```

6. Commit: `git commit -m "feat(core): implement SqliteStore with sqlite-vec vector search"`

---

## Risk

- **sqlite-vec vec0 parameter binding**: The `MATCH` syntax for vec0 queries
  requires passing the embedding as a blob. If `params![embedding.to_vec()]`
  doesn't work, we may need `rusqlite::types::Value::Blob` with manual
  serialization to little-endian f32 bytes.
- **sqlite-vec version compatibility**: `sqlite-vec 0.1.x` is pre-1.0. API may
  shift. Pin exact version in workspace deps.
- **In-memory vec0 dimension**: Tests use 4-dim vectors; production uses 1024.
  The `open_with_dimensions` constructor handles this, but migration must not
  clash if the DB already has a vec0 table with different dimensions.

## Follow-up Plans (not in scope)

- `memx-cli`: clap CLI with `store`, `search`, `remember`, `forget` commands
- `memx-mcp`: MCP server exposing Store + Embedder as tools
- `memx-embed` backends: `FastEmbedBackend`, `OrtBackend` implementations
- Budget enforcement as a `BudgetGuard` wrapper around Store
- Deduplication logic (substring match before insert)
