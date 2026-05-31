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
}

impl Default for EntryId {
    fn default() -> Self {
        Self::new()
    }
}

// qual:allow(dry) reason: "trivial newtype delegation, no derive available"
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
    // qual:allow(iosp) reason: "pure match dispatch, no I/O"
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

impl FromStr for Section {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "active_threads" => Section::ActiveThreads,
            "environment_notes" => Section::EnvironmentNotes,
            "pending_decisions" => Section::PendingDecisions,
            other => Section::Custom(other.to_string()),
        })
    }
}

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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserProfile {
    pub about: String,
    pub preferences: Vec<String>,
    pub working_style: String,
}

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
    DateRange { from: NaiveDate, to: NaiveDate },
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
    Add { section: Section, content: String },
    Replace { target: EntryId, content: String },
    Remove { target: EntryId },
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
        let parsed: EntryId = s.parse().expect("valid ULID should parse");
        assert_eq!(id, parsed);
    }

    #[test]
    fn entry_id_from_str_rejects_invalid() {
        let result = EntryId::from_str("not-a-ulid");
        assert!(result.is_err());
    }

    #[test]
    fn section_display() {
        assert_eq!(Section::ActiveThreads.as_str(), "active_threads");
        assert_eq!(Section::EnvironmentNotes.as_str(), "environment_notes");
        assert_eq!(Section::PendingDecisions.as_str(), "pending_decisions");
        let custom = Section::Custom("my_section".into());
        assert_eq!(custom.as_str(), "my_section");
    }

    #[test]
    fn section_from_str_known_variants() {
        assert_eq!(
            "active_threads".parse::<Section>().expect("known variant"),
            Section::ActiveThreads
        );
        assert_eq!(
            "environment_notes"
                .parse::<Section>()
                .expect("known variant"),
            Section::EnvironmentNotes
        );
        assert_eq!(
            "pending_decisions"
                .parse::<Section>()
                .expect("known variant"),
            Section::PendingDecisions
        );
    }

    #[test]
    fn section_from_str_custom() {
        let s: Section = "my_custom".parse().expect("custom always succeeds");
        assert_eq!(s, Section::Custom("my_custom".into()));
    }

    #[test]
    fn section_roundtrip_display_fromstr() {
        let variants = [
            Section::ActiveThreads,
            Section::EnvironmentNotes,
            Section::PendingDecisions,
            Section::Custom("user_notes".into()),
        ];
        for v in &variants {
            let s = v.to_string();
            let parsed: Section = s.parse().expect("roundtrip should succeed");
            assert_eq!(&parsed, v);
        }
    }

    #[test]
    fn memory_entry_creation() {
        let entry = MemoryEntry::new(Section::ActiveThreads, "working on memx type system".into());
        assert_eq!(entry.section, Section::ActiveThreads);
        assert_eq!(entry.content, "working on memx type system");
        assert!(entry.created_at <= Utc::now());
    }

    #[test]
    fn session_log_creation() {
        let log = SessionLog::new(NaiveDate::from_ymd_opt(2026, 5, 22).expect("valid date"), 1);
        assert_eq!(log.session_number, 1);
        assert!(log.goal.is_none());
    }

    #[test]
    fn search_query_defaults() {
        let q = SearchQuery::new("test query".into());
        assert_eq!(q.top_k, 5);
        assert!(q.filter.is_none());
    }

    #[test]
    fn search_query_builder() {
        let q = SearchQuery::new("test".into())
            .with_top_k(10)
            .with_filter(SearchFilter::Source(SourceKind::Memory));
        assert_eq!(q.top_k, 10);
        assert!(q.filter.is_some());
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

    mod property {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn entry_id_string_roundtrip(_ in 0..1000u32) {
                let id = EntryId::new();
                let s = id.to_string();
                let parsed: EntryId = s.parse()
                    .expect("ULID roundtrip must succeed");
                prop_assert_eq!(id, parsed);
            }

            #[test]
            fn section_display_fromstr_roundtrip(
                custom in "[a-z_]{1,50}"
            ) {
                let section = Section::Custom(custom);
                let s = section.to_string();
                let parsed: Section = s.parse()
                    .expect("Section roundtrip must succeed");
                prop_assert_eq!(section, parsed);
            }
        }
    }
}
