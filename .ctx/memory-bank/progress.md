# Progress: memx

## What works

- MemoryEntry CRUD (insert, get, update, delete, list, total_chars)
- Section filtering on list and total_chars
- SessionLog insert and date-based retrieval
- TranscriptChunk insert (with FK to sessions)
- Vector similarity search (unfiltered) via sqlite-vec
- File-based and in-memory SQLite store opening
- Embedder trait with default `embed_one` delegation
- Full test coverage: unit, property (proptest), conformance contracts
- Rustqual score: 97.3%

## In progress

- Nothing active

## Not started

- Search filter implementation (Section, DateRange, Source variants)
- Deduplication logic (Duplicate error exists but not enforced beyond PK)
- Budget enforcement (BudgetExceeded error exists, no enforcement logic)
- CLI binary crate
- CI/CD pipeline (GitHub Actions)
- README / documentation
- Real Embedder implementations (only MockEmbedder in tests)
- Fuzz target integration into CI
- Thread-safety (Send/Sync) -- deferred per trait comments
