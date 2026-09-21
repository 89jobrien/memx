# memx

A Rust library for storing and semantically searching AI-agent memory:
structured notes, session logs, and transcript chunks, backed by SQLite +
vector search.

## What it does

`memx` provides the domain types and storage/embedding traits for a
memory system, plus a SQLite-backed implementation using `sqlite-vec` for
similarity search:

- **Entries**: `MemoryEntry` records tagged with a `Section`
  (`ActiveThreads`, `EnvironmentNotes`, `PendingDecisions`, or a custom
  section), addressed by a ULID-based `EntryId`.
- **Sessions & transcripts**: `SessionLog` (goal, deliverables, decisions,
  open threads per date/session number) and `TranscriptChunk` (raw
  timestamped content tied to a session).
- **Writes via `MemxService`**: a generic orchestrator (`MemxService<Store,
Embedder>`) that executes `WriteAction::{Add, Replace, Remove}`, embeds
  content on write, and enforces an optional character budget
  (`with_budget`) across all sections.
- **Search**: embeds a query string and searches stored vectors via
  `VectorSearch`, returning `SearchResult`s with a `MatchSource` (memory
  entry, session, or transcript) and similarity score.

Everything is trait-based (`EntryStore`, `SessionStore`, `TranscriptStore`,
`VectorSearch`, combined as `Store`; `Embedder`), so storage and embedding
backends are pluggable.

## Architecture

Two-crate Cargo workspace:

- **`crates/memx-core`** — domain types (`types.rs`), storage port traits
  (`store.rs`), the `SqliteStore` adapter (`sqlite_store.rs`, using
  `rusqlite` + the `sqlite-vec` extension for a `vec0` virtual table),
  `MemxService` orchestrator (`service.rs`), and error types
  (`error.rs`, `MemxError` via `thiserror`).
- **`crates/memx-embed`** — the `Embedder` trait (`embedder.rs`): batch
  `embed`, a default `embed_one`, `dimensions()`, `model_id()`. No concrete
  embedding backend is implemented yet (`fastembed`/`ort` backends are
  planned — see `docs/plans/2026-05-22-foundational-type-system.md`).

There is currently no CLI or binary crate — this is a library workspace.

## Build / test

```sh
cargo build
cargo test
```

Both crates have unit tests, edge-case tests, and `proptest`-based property
tests (e.g. `EntryId`/`Section` string roundtrips, budget invariants in
`MemxService`, the `Embedder` and `Store` trait contracts). `memx-core` also
has an integration test at `crates/memx-core/tests/integration.rs`.

A `fuzz/` directory exists for fuzz targets (cargo-fuzz).

## Conventions

- Rust edition 2024, dual MIT/Apache-2.0 licensed via
  `[workspace.package]`.
- `rustqual.toml` configures code-quality checks for this workspace: dead
  code detection is disabled (public API is consumed cross-crate), and
  `sqlite_store.rs` is allowed up to 400 lines (single-adapter file).
- Storage is a single SQLite file (planned default: `~/.memx/memx.db`);
  `SqliteStore::open_in_memory` is used for tests.
