# Active Context: memx

## Current focus

Memory bank generation (this session). No active feature work.

## Recent work (session 1, 2026-05-31)

- Scaffolded workspace from scratch
- Implemented full type system, Store trait, SqliteStore with sqlite-vec
- Resolved 11 GitHub issues (parse panics, delete semantics, search filter
  guard, required dims, trait bounds, Section::FromStr, dead Budget removal)
- 38+ test suite: unit, property, conformance
- Rustqual 77.9% -> 97.3% in 3 iterations
- Published to github.com/89jobrien/memx

## Open questions

- Search filters (Section, DateRange, Source) not yet implemented
- No CLI or binary crate — library only
- No CI pipeline
- Fuzz targets exist (untracked) but not integrated

## Decisions made

- ULID over UUID for time-sortable IDs
- sqlite-vec over pgvector — single-file, no external DB
- Trait-based Store/Embedder for backend swappability
- Edition 2024 with resolver 3
