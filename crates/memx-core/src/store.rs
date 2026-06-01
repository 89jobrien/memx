use crate::error::Result;
use crate::types::{
    EntryId, MemoryEntry, SearchFilter, SearchResult, Section, SessionLog, TranscriptChunk,
};
use chrono::NaiveDate;

// Thread-safety bounds deferred to concrete implementations
pub trait Store {
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

#[cfg(test)]
mod tests {
    use super::*;

    // qual:allow(test) reason: "compile-time proof, no callable SUT"
    #[test]
    fn dyn_store_is_object_safe() {
        let _: fn(&dyn Store) = |_store: &dyn Store| {};
        assert!(std::mem::size_of::<&dyn Store>() > 0);
    }
}
