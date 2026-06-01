#![no_main]
//! Fuzz target: exercises the sqlite-vec extension (initialized via unsafe
//! transmute in `ensure_vec_extension`) with arbitrary embedding data and
//! content strings through the full insert -> search pipeline.

use libfuzzer_sys::fuzz_target;
use memx_core::sqlite_store::SqliteStore;
use memx_core::store::Store;
use memx_core::types::{MemoryEntry, Section};

const DIMS: usize = 4;

fuzz_target!(|data: &[u8]| {
    // Need at least DIMS * 4 bytes for one embedding + 1 byte for content
    if data.len() < DIMS * 4 + 1 {
        return;
    }

    let store = match SqliteStore::open_in_memory(DIMS) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Extract embedding from first DIMS*4 bytes
    let mut embedding = [0.0f32; DIMS];
    for (i, chunk) in data[..DIMS * 4].chunks_exact(4).enumerate() {
        embedding[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }

    // Skip NaN/Inf — sqlite-vec may not handle them
    if embedding.iter().any(|f| !f.is_finite()) {
        return;
    }

    // Rest is content
    let content = match std::str::from_utf8(&data[DIMS * 4..]) {
        Ok(s) => s.to_string(),
        Err(_) => return,
    };

    let entry = MemoryEntry::new(Section::ActiveThreads, content);

    // Insert should not panic
    if store.insert_entry(&entry, &embedding).is_err() {
        return;
    }

    // Search should not panic
    let _ = store.search_similar(&embedding, 1, None);

    // Get should not panic
    let _ = store.get_entry(entry.id);

    // Delete should not panic
    let _ = store.delete_entry(entry.id);
});
