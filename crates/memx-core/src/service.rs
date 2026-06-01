use crate::error::{MemxError, Result};
use crate::store::Store;
use crate::types::{
    EntryId, MemoryEntry, SearchResult, Section, WriteAction, WriteActionKind, WriteResult,
};
use memx_embed::embedder::Embedder;

pub struct MemxService<S, E> {
    store: S,
    embedder: E,
    budget: Option<usize>,
}

impl<S: Store, E: Embedder> MemxService<S, E> {
    pub fn new(store: S, embedder: E) -> Self {
        Self {
            store,
            embedder,
            budget: None,
        }
    }

    pub fn with_budget(mut self, max_chars: usize) -> Self {
        self.budget = Some(max_chars);
        self
    }

    pub fn execute(&self, action: WriteAction) -> Result<WriteResult> {
        match action {
            WriteAction::Add { section, content } => self.add(section, content),
            WriteAction::Replace { target, content } => self.replace(target, content),
            WriteAction::Remove { target } => self.remove(target),
        }
    }

    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let embedding = self
            .embedder
            .embed_one(query)
            .map_err(|e| MemxError::Embedding(e.to_string()))?;
        self.store.search_similar(&embedding, top_k, None)
    }

    fn add(&self, section: Section, content: String) -> Result<WriteResult> {
        let content_len = content.chars().count();
        self.check_budget(content_len)?;

        let entry = MemoryEntry::new(section, content);
        let embedding = self
            .embedder
            .embed_one(&entry.content)
            .map_err(|e| MemxError::Embedding(e.to_string()))?;

        let id = entry.id;
        self.store.insert_entry(&entry, &embedding)?;

        let chars_used = self.store.total_chars(None)?;
        Ok(WriteResult {
            entry_id: id,
            action: WriteActionKind::Added,
            chars_used,
            chars_remaining: self.remaining(chars_used),
            deduplicated: false,
        })
    }

    fn replace(&self, target: EntryId, content: String) -> Result<WriteResult> {
        let mut entry = self.store.get_entry(target)?;
        let old_len = entry.content.chars().count();
        let new_len = content.chars().count();

        if new_len > old_len {
            self.check_budget(new_len - old_len)?;
        }

        entry.content = content;
        let embedding = self
            .embedder
            .embed_one(&entry.content)
            .map_err(|e| MemxError::Embedding(e.to_string()))?;

        self.store.update_entry(&entry, &embedding)?;

        let chars_used = self.store.total_chars(None)?;
        Ok(WriteResult {
            entry_id: target,
            action: WriteActionKind::Replaced,
            chars_used,
            chars_remaining: self.remaining(chars_used),
            deduplicated: false,
        })
    }

    fn remove(&self, target: EntryId) -> Result<WriteResult> {
        self.store.delete_entry(target)?;

        let chars_used = self.store.total_chars(None)?;
        Ok(WriteResult {
            entry_id: target,
            action: WriteActionKind::Removed,
            chars_used,
            chars_remaining: self.remaining(chars_used),
            deduplicated: false,
        })
    }

    fn check_budget(&self, additional: usize) -> Result<()> {
        if let Some(max) = self.budget {
            let current = self.store.total_chars(None)?;
            if current + additional > max {
                return Err(MemxError::BudgetExceeded {
                    current: current + additional,
                    max,
                });
            }
        }
        Ok(())
    }

    fn remaining(&self, chars_used: usize) -> usize {
        self.budget
            .map(|max| max.saturating_sub(chars_used))
            .unwrap_or(usize::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqlite_store::SqliteStore;

    struct FixedEmbedder(usize);

    impl Embedder for FixedEmbedder {
        fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.1; self.0]).collect())
        }

        fn dimensions(&self) -> usize {
            self.0
        }

        fn model_id(&self) -> &str {
            "fixed-4d"
        }
    }

    fn test_service() -> MemxService<SqliteStore, FixedEmbedder> {
        let store = SqliteStore::open_in_memory(4).expect("store");
        let embedder = FixedEmbedder(4);
        MemxService::new(store, embedder)
    }

    #[test]
    fn add_returns_write_result() {
        let svc = test_service();
        let result = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: "hello world".into(),
            })
            .expect("add");
        assert_eq!(result.action, WriteActionKind::Added);
        assert_eq!(result.chars_used, 11);
        assert_eq!(result.chars_remaining, usize::MAX);
    }

    #[test]
    fn add_with_budget_enforced() {
        let svc = test_service().with_budget(10);
        let result = svc.execute(WriteAction::Add {
            section: Section::ActiveThreads,
            content: "12345678901".into(), // 11 chars > 10 budget
        });
        assert!(matches!(result, Err(MemxError::BudgetExceeded { .. })));
    }

    #[test]
    fn add_within_budget_succeeds() {
        let svc = test_service().with_budget(20);
        let result = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: "hello".into(),
            })
            .expect("add");
        assert_eq!(result.chars_used, 5);
        assert_eq!(result.chars_remaining, 15);
    }

    #[test]
    fn budget_accumulates_across_adds() {
        let svc = test_service().with_budget(10);
        svc.execute(WriteAction::Add {
            section: Section::ActiveThreads,
            content: "12345".into(),
        })
        .expect("first add");

        let result = svc.execute(WriteAction::Add {
            section: Section::ActiveThreads,
            content: "123456".into(), // 5 + 6 = 11 > 10
        });
        assert!(matches!(result, Err(MemxError::BudgetExceeded { .. })));
    }

    #[test]
    fn replace_updates_content() {
        let svc = test_service();
        let added = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: "original".into(),
            })
            .expect("add");

        let replaced = svc
            .execute(WriteAction::Replace {
                target: added.entry_id,
                content: "updated".into(),
            })
            .expect("replace");
        assert_eq!(replaced.action, WriteActionKind::Replaced);
        assert_eq!(replaced.entry_id, added.entry_id);
    }

    #[test]
    fn replace_checks_budget_on_growth() {
        let svc = test_service().with_budget(10);
        let added = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: "12345".into(), // 5 chars
            })
            .expect("add");

        let result = svc.execute(WriteAction::Replace {
            target: added.entry_id,
            content: "12345678901234".into(), // grows by 9, total 14 > 10
        });
        assert!(matches!(result, Err(MemxError::BudgetExceeded { .. })));
    }

    #[test]
    fn replace_shrink_no_budget_check() {
        let svc = test_service().with_budget(10);
        let added = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: "1234567890".into(), // exactly at budget
            })
            .expect("add");

        let result = svc
            .execute(WriteAction::Replace {
                target: added.entry_id,
                content: "short".into(), // shrinks, should succeed
            })
            .expect("replace shrink");
        assert_eq!(result.chars_used, 5);
    }

    #[test]
    fn remove_deletes_entry() {
        let svc = test_service();
        let added = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: "to delete".into(),
            })
            .expect("add");

        let removed = svc
            .execute(WriteAction::Remove {
                target: added.entry_id,
            })
            .expect("remove");
        assert_eq!(removed.action, WriteActionKind::Removed);
        assert_eq!(removed.chars_used, 0);
    }

    #[test]
    fn remove_nonexistent_fails() {
        let svc = test_service();
        let result = svc.execute(WriteAction::Remove {
            target: EntryId::new(),
        });
        assert!(matches!(result, Err(MemxError::NotFound(_))));
    }

    #[test]
    fn search_delegates_to_store() {
        let svc = test_service();
        svc.execute(WriteAction::Add {
            section: Section::ActiveThreads,
            content: "rust programming".into(),
        })
        .expect("add");

        let results = svc.search("rust", 5).expect("search");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_empty_store() {
        let svc = test_service();
        let results = svc.search("anything", 5).expect("search");
        assert!(results.is_empty());
    }

    // ── Edge cases ────────────────────────────────────────────────

    #[test]
    fn add_empty_content() {
        let svc = test_service();
        let result = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: String::new(),
            })
            .expect("add empty");
        assert_eq!(result.chars_used, 0);
        assert_eq!(result.action, WriteActionKind::Added);
    }

    #[test]
    fn zero_budget_rejects_any_content() {
        let svc = test_service().with_budget(0);
        let result = svc.execute(WriteAction::Add {
            section: Section::ActiveThreads,
            content: "a".into(),
        });
        assert!(matches!(result, Err(MemxError::BudgetExceeded { .. })));
    }

    #[test]
    fn zero_budget_allows_empty_content() {
        let svc = test_service().with_budget(0);
        let result = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: String::new(),
            })
            .expect("empty at zero budget");
        assert_eq!(result.chars_used, 0);
        assert_eq!(result.chars_remaining, 0);
    }

    #[test]
    fn replace_nonexistent_fails() {
        let svc = test_service();
        let result = svc.execute(WriteAction::Replace {
            target: EntryId::new(),
            content: "ghost".into(),
        });
        assert!(matches!(result, Err(MemxError::NotFound(_))));
    }

    #[test]
    fn remove_frees_budget() {
        let svc = test_service().with_budget(10);
        let added = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: "1234567890".into(),
            })
            .expect("add at limit");
        assert_eq!(added.chars_remaining, 0);

        svc.execute(WriteAction::Remove {
            target: added.entry_id,
        })
        .expect("remove");

        // budget is now free again
        let second = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: "reuse".into(),
            })
            .expect("re-add after remove");
        assert_eq!(second.chars_used, 5);
        assert_eq!(second.chars_remaining, 5);
    }

    #[test]
    fn replace_same_length_at_budget_limit() {
        let svc = test_service().with_budget(5);
        let added = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: "abcde".into(),
            })
            .expect("add at limit");

        // same length replacement should succeed (no growth)
        let result = svc
            .execute(WriteAction::Replace {
                target: added.entry_id,
                content: "fghij".into(),
            })
            .expect("replace same len");
        assert_eq!(result.chars_used, 5);
        assert_eq!(result.chars_remaining, 0);
    }

    #[test]
    fn multiple_sections_share_budget() {
        let svc = test_service().with_budget(10);
        svc.execute(WriteAction::Add {
            section: Section::ActiveThreads,
            content: "aaaaa".into(),
        })
        .expect("add threads");

        svc.execute(WriteAction::Add {
            section: Section::EnvironmentNotes,
            content: "bbbbb".into(),
        })
        .expect("add env");

        // budget is full across sections
        let result = svc.execute(WriteAction::Add {
            section: Section::PendingDecisions,
            content: "c".into(),
        });
        assert!(matches!(result, Err(MemxError::BudgetExceeded { .. })));
    }

    #[test]
    fn search_top_k_zero_returns_empty() {
        let svc = test_service();
        svc.execute(WriteAction::Add {
            section: Section::ActiveThreads,
            content: "something".into(),
        })
        .expect("add");

        let results = svc.search("something", 0).expect("search k=0");
        assert!(results.is_empty());
    }

    // ── Conformance: MemxService contract ─────────────────────────

    fn assert_service_contract(svc: &MemxService<SqliteStore, FixedEmbedder>) {
        // Add
        let added = svc
            .execute(WriteAction::Add {
                section: Section::ActiveThreads,
                content: "contract entry".into(),
            })
            .expect("contract: add");
        assert_eq!(added.action, WriteActionKind::Added);
        assert!(added.chars_used >= "contract entry".len());

        // Replace
        let replaced = svc
            .execute(WriteAction::Replace {
                target: added.entry_id,
                content: "replaced content".into(),
            })
            .expect("contract: replace");
        assert_eq!(replaced.action, WriteActionKind::Replaced);
        assert_eq!(replaced.entry_id, added.entry_id);

        // Search finds replaced content
        let results = svc.search("replaced", 10).expect("contract: search");
        assert!(
            !results.is_empty(),
            "contract: search should find replaced entry"
        );

        // Remove
        let removed = svc
            .execute(WriteAction::Remove {
                target: added.entry_id,
            })
            .expect("contract: remove");
        assert_eq!(removed.action, WriteActionKind::Removed);

        // Search no longer finds it
        let after_remove = svc
            .search("replaced", 10)
            .expect("contract: search after remove");
        assert!(
            after_remove.is_empty(),
            "contract: removed entry should not appear in search"
        );

        // Remove again fails
        let double_remove = svc.execute(WriteAction::Remove {
            target: added.entry_id,
        });
        assert!(
            matches!(double_remove, Err(MemxError::NotFound(_))),
            "contract: double remove should be NotFound"
        );

        // Replace nonexistent fails
        let ghost_replace = svc.execute(WriteAction::Replace {
            target: EntryId::new(),
            content: "ghost".into(),
        });
        assert!(
            matches!(ghost_replace, Err(MemxError::NotFound(_))),
            "contract: replace nonexistent should be NotFound"
        );
    }

    #[test]
    fn service_satisfies_contract() {
        let svc = test_service();
        assert_service_contract(&svc);
    }

    #[test]
    fn service_with_budget_satisfies_contract() {
        let svc = test_service().with_budget(10_000);
        assert_service_contract(&svc);
    }

    // ── Property tests ────────────────────────────────────────────

    mod property {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn budget_never_exceeded(
                contents in proptest::collection::vec("\\PC{1,50}", 1..20),
                budget in 10..500usize,
            ) {
                let svc = test_service().with_budget(budget);
                let mut total = 0usize;
                for c in &contents {
                    let result = svc.execute(WriteAction::Add {
                        section: Section::ActiveThreads,
                        content: c.clone(),
                    });
                    match result {
                        Ok(r) => {
                            total += c.chars().count();
                            prop_assert!(r.chars_used <= budget,
                                "chars_used {} exceeded budget {}", r.chars_used, budget);
                            prop_assert_eq!(r.chars_used, total);
                        }
                        Err(MemxError::BudgetExceeded { current, max }) => {
                            prop_assert!(current > max);
                            // total in store must not have changed
                            break;
                        }
                        Err(e) => prop_assert!(false, "unexpected error: {e}"),
                    }
                }
            }

            #[test]
            fn chars_remaining_plus_used_equals_budget(
                content_len in 1..100usize,
                budget in 100..1000usize,
            ) {
                let svc = test_service().with_budget(budget);
                let content: String = "x".repeat(content_len);
                let result = svc.execute(WriteAction::Add {
                    section: Section::ActiveThreads,
                    content,
                }).expect("should fit in budget");
                prop_assert_eq!(
                    result.chars_used + result.chars_remaining,
                    budget,
                    "used + remaining must equal budget"
                );
            }

            #[test]
            fn add_then_remove_restores_zero(
                content in "\\PC{1,100}"
            ) {
                let svc = test_service();
                let added = svc.execute(WriteAction::Add {
                    section: Section::ActiveThreads,
                    content,
                }).expect("add");
                let removed = svc.execute(WriteAction::Remove {
                    target: added.entry_id,
                }).expect("remove");
                prop_assert_eq!(removed.chars_used, 0);
            }

            #[test]
            fn replace_preserves_entry_id(
                original in "\\PC{1,50}",
                replacement in "\\PC{1,50}",
            ) {
                let svc = test_service();
                let added = svc.execute(WriteAction::Add {
                    section: Section::ActiveThreads,
                    content: original,
                }).expect("add");
                let replaced = svc.execute(WriteAction::Replace {
                    target: added.entry_id,
                    content: replacement,
                }).expect("replace");
                prop_assert_eq!(replaced.entry_id, added.entry_id);
            }

            #[test]
            fn no_budget_means_unlimited(
                contents in proptest::collection::vec("\\PC{1,200}", 1..50),
            ) {
                let svc = test_service(); // no budget
                for c in &contents {
                    let result = svc.execute(WriteAction::Add {
                        section: Section::ActiveThreads,
                        content: c.clone(),
                    });
                    prop_assert!(result.is_ok(), "unbounded service should never reject");
                    prop_assert_eq!(result.unwrap().chars_remaining, usize::MAX);
                }
            }
        }
    }
}
