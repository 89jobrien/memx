# System Patterns: memx

## Architecture

Hexagonal / ports-and-adapters:

- **Port**: `Store` trait (`store.rs`) — object-safe, defines all storage ops
- **Port**: `Embedder` trait (`embedder.rs`) — pluggable embedding backend
- **Adapter**: `SqliteStore` (`sqlite_store.rs`) — SQLite + sqlite-vec impl

## Data flow

1. Caller creates `MemoryEntry` with section + content
2. Caller obtains embedding via `Embedder::embed_one`
3. Caller passes entry + embedding to `Store::insert_entry`
4. `SqliteStore` writes metadata to `entries` table, vector to `entries_vec`
5. Search: caller embeds query text, calls `Store::search_similar` with vector
6. sqlite-vec performs KNN, results joined back to `entries` for content

## Key conventions

- ULID-based `EntryId` — newtype over `ulid::Ulid`, Display/FromStr roundtrip
- `Section` enum with `FromStr`/`Display` — known variants + `Custom(String)`
- Error type `MemxError` with `thiserror` — variants: BudgetExceeded, NotFound,
  Duplicate, Storage, Embedding, Parse, Serialization, Other
- `Result<T>` alias to `std::result::Result<T, MemxError>`
- Embeddings stored as raw little-endian f32 blobs in sqlite-vec
- `require_affected` helper guards update/delete against missing rows
- `row_to_entry` shared helper for SELECT-to-MemoryEntry mapping

## Testing patterns

- Conformance tests: `assert_store_contract`, `assert_embedder_contract`
- Property tests via proptest: roundtrip, invariant, leak checks
- In-memory SQLite for fast isolated tests (`SqliteStore::open_in_memory`)
- rustqual annotations: `qual:allow(dry)`, `qual:allow(iosp)` with reasons
