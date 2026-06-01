//! Integration test: full embed -> store -> search pipeline.
//!
//! Uses a trivial deterministic embedder stub to verify the complete
//! flow without requiring a real model.

use memx_core::sqlite_store::SqliteStore;
use memx_core::store::Store;
use memx_core::types::{MemoryEntry, Section};

const DIMS: usize = 4;

/// Deterministic "embedder" that hashes content into a fixed-length vector.
fn fake_embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIMS];
    for (i, b) in text.bytes().enumerate() {
        v[i % DIMS] += b as f32;
    }
    // Normalize to unit vector
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[test]
fn embed_store_search_pipeline() {
    let store = SqliteStore::open_in_memory(DIMS).expect("open store");

    let entries = [
        ("rust programming language", Section::ActiveThreads),
        ("python data science", Section::EnvironmentNotes),
        ("cargo build system", Section::ActiveThreads),
        ("javascript web development", Section::EnvironmentNotes),
    ];

    for (content, section) in &entries {
        let entry = MemoryEntry::new(section.clone(), content.to_string());
        let embedding = fake_embed(content);
        store
            .insert_entry(&entry, &embedding)
            .expect("insert should succeed");
    }

    // Search for something similar to "rust"
    let query_emb = fake_embed("rust cargo build");
    let results = store
        .search_similar(&query_emb, 2, None)
        .expect("search should succeed");

    assert_eq!(results.len(), 2, "should return top_k results");
    // Scores should be ordered descending
    assert!(results[0].score >= results[1].score);
    // All scores in valid range
    for r in &results {
        assert!(
            r.score >= -1.0 && r.score <= 1.01,
            "score {} out of range",
            r.score
        );
    }
}

#[test]
fn embed_store_update_search_reflects_change() {
    let store = SqliteStore::open_in_memory(DIMS).expect("open store");

    let mut entry = MemoryEntry::new(Section::ActiveThreads, "original topic".into());
    let emb1 = fake_embed("original topic");
    store.insert_entry(&entry, &emb1).expect("insert");

    // Update content and embedding
    entry.content = "completely different subject".into();
    let emb2 = fake_embed("completely different subject");
    store.update_entry(&entry, &emb2).expect("update");

    // Search should find the updated content
    let results = store.search_similar(&emb2, 1, None).expect("search");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "completely different subject");
}

#[test]
fn embed_store_delete_removes_from_search() {
    let store = SqliteStore::open_in_memory(DIMS).expect("open store");

    let entry = MemoryEntry::new(Section::ActiveThreads, "deleteme".into());
    let emb = fake_embed("deleteme");
    store.insert_entry(&entry, &emb).expect("insert");

    // Verify it appears in search
    let before = store.search_similar(&emb, 1, None).expect("search before");
    assert_eq!(before.len(), 1);

    // Delete and verify search is empty
    store.delete_entry(entry.id).expect("delete");
    let after = store.search_similar(&emb, 1, None).expect("search after");
    assert!(after.is_empty());
}
