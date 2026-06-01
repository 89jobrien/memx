# Project Brief: memx

## What

A Rust library for persistent, vector-searchable memory storage for AI agents.
Stores memory entries, session logs, and transcript chunks in SQLite with
sqlite-vec for embedding-based similarity search.

## Who

Joseph O'Brien (89jobrien). Personal project.

## Crates

- `memx-core` (v0.1.0) — types, `Store` trait, `SqliteStore` implementation, error types
- `memx-embed` (v0.1.0) — `Embedder` trait for pluggable embedding backends

## Done criteria

- CRUD operations on memory entries with vector embeddings
- Session log and transcript storage
- Vector similarity search via sqlite-vec
- Trait-based design (Store, Embedder) for swappable backends

## Status

Foundational implementation complete. 38+ tests across unit, property, and
conformance dimensions. Rustqual score 97.3%. Published to GitHub.

## Source evidence

- Workspace: `Cargo.toml` (resolver 3, MIT OR Apache-2.0)
- Crates: `crates/memx-core/`, `crates/memx-embed/`
- HANDOFF: `.ctx/HANDOFF.memx.memx.yaml` (session 1 log)
- Git: 12 commits, b5b2885..bee3634
