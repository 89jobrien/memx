# Tech Context: memx

## Stack

- Rust edition 2024, resolver 3
- SQLite via `rusqlite` 0.34 (bundled)
- Vector search via `sqlite-vec` 0.1
- IDs: `ulid` 1.x with serde
- Time: `chrono` 0.4 with serde
- Errors: `thiserror` 2 + `anyhow` 1
- Serialization: `serde` 1 + `serde_json` 1

## Dev dependencies

- `tempfile` 3 (SqliteStore file-based tests)
- `proptest` 1 (property tests in memx-core)

## Build commands

```
cargo build          # build all crates
cargo test           # run all tests
cargo clippy         # lint
```

## Constraints

- No CI workflows yet (no `.github/workflows/`)
- No README or CLAUDE.md in repo
- `search_similar` with filters returns `Err` — filters not yet implemented
- sqlite-vec extension registered via `sqlite3_auto_extension` with unsafe
  transmute (documented safety comment in sqlite_store.rs:15-26)
- Embedding dimensions are set at store open time, not configurable after

## Repo structure

```
Cargo.toml                          # workspace root
crates/
  memx-core/
    src/lib.rs                      # re-exports: error, sqlite_store, store, types
    src/types.rs                    # EntryId, Section, MemoryEntry, SessionLog, etc.
    src/store.rs                    # Store trait (object-safe)
    src/sqlite_store.rs             # SqliteStore impl + migration
    src/error.rs                    # MemxError enum + Result alias
    tests/                          # (untracked) external test files
    proptest-regressions/           # (untracked)
  memx-embed/
    src/lib.rs                      # re-exports embedder module
    src/embedder.rs                 # Embedder trait + MockEmbedder tests
fuzz/                               # (untracked) fuzz targets
.ctx/                               # handoff, session state
```
