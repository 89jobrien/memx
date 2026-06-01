# Product Context: memx

## Why it exists

Provides structured, searchable long-term memory for AI coding agents. Agents
need to persist context across sessions — active threads, environment notes,
pending decisions — and retrieve relevant memories via semantic similarity.

## UX principles

- Trait-based pluggability: swap storage backends (SqliteStore) and embedding
  providers (Embedder trait) independently
- Budget-aware: `BudgetExceeded` error prevents unbounded memory growth
- Section-based organization: `ActiveThreads`, `EnvironmentNotes`,
  `PendingDecisions`, `Custom(String)`
- ULID-based IDs for time-ordered, unique entry identification

## Users

AI agents and tooling that need persistent, searchable context memory.
