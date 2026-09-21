//! Capability-based persistence interfaces for memory data and vector search.

use crate::error::Result;
use crate::types::{
    EntryId, MemoryEntry, SearchFilter, SearchResult, Section, SessionLog, TranscriptChunk,
};
use chrono::NaiveDate;

pub trait EntryStore {
    /// Persists a memory entry and its embedding.
    fn insert_entry(&self, entry: &MemoryEntry, embedding: &[f32]) -> Result<()>;
    /// Replaces a memory entry and its embedding.
    fn update_entry(&self, entry: &MemoryEntry, embedding: &[f32]) -> Result<()>;
    /// Deletes the memory entry identified by `id`.
    fn delete_entry(&self, id: EntryId) -> Result<()>;
    /// Loads the memory entry identified by `id`.
    fn get_entry(&self, id: EntryId) -> Result<MemoryEntry>;
    /// Lists memory entries, optionally restricted to one section.
    fn list_entries(&self, section: Option<&Section>) -> Result<Vec<MemoryEntry>>;
    /// Counts stored content characters, optionally within one section.
    fn total_chars(&self, section: Option<&Section>) -> Result<usize>;
}

pub trait SessionStore {
    /// Persists a session log.
    fn insert_session(&self, session: &SessionLog) -> Result<()>;
    /// Loads all session logs recorded on `date`.
    fn get_sessions_for_date(&self, date: NaiveDate) -> Result<Vec<SessionLog>>;
}

pub trait TranscriptStore {
    /// Persists a transcript chunk.
    fn insert_transcript(&self, chunk: &TranscriptChunk) -> Result<()>;
}

pub trait VectorSearch {
    /// Returns the nearest stored items to `embedding`, subject to the optional filter.
    fn search_similar(
        &self,
        embedding: &[f32],
        top_k: usize,
        filter: Option<&SearchFilter>,
    ) -> Result<Vec<SearchResult>>;
}

/// Storage backend supporting every persistence capability.
pub trait Store: EntryStore + SessionStore + TranscriptStore + VectorSearch {}
impl<T: EntryStore + SessionStore + TranscriptStore + VectorSearch> Store for T {}

#[cfg(test)]
mod tests {
    use super::*;

    // qual:allow(test) reason: "compile-time proof, no callable SUT"
    #[test]
    fn dyn_entry_store_is_object_safe() {
        let _: fn(&dyn EntryStore) = |_s: &dyn EntryStore| {};
        assert!(std::mem::size_of::<&dyn EntryStore>() > 0);
    }

    #[test]
    fn dyn_session_store_is_object_safe() {
        let _: fn(&dyn SessionStore) = |_s: &dyn SessionStore| {};
        assert!(std::mem::size_of::<&dyn SessionStore>() > 0);
    }

    #[test]
    fn dyn_transcript_store_is_object_safe() {
        let _: fn(&dyn TranscriptStore) = |_s: &dyn TranscriptStore| {};
        assert!(std::mem::size_of::<&dyn TranscriptStore>() > 0);
    }

    #[test]
    fn dyn_vector_search_is_object_safe() {
        let _: fn(&dyn VectorSearch) = |_s: &dyn VectorSearch| {};
        assert!(std::mem::size_of::<&dyn VectorSearch>() > 0);
    }
}
