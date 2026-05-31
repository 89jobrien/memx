// Thread-safety bounds deferred to concrete implementations
pub trait Embedder {
    fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;

    fn embed_one(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut results = self.embed(&[text])?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embed returned empty results"))
    }

    fn dimensions(&self) -> usize;

    fn model_id(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEmbedder;

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect())
        }

        fn dimensions(&self) -> usize {
            4
        }

        fn model_id(&self) -> &str {
            "mock-4d"
        }
    }

    #[test]
    fn mock_embedder_dimensions() {
        let e = MockEmbedder;
        assert_eq!(e.dimensions(), 4);
    }

    #[test]
    fn mock_embedder_batch() {
        let e = MockEmbedder;
        let results = e.embed(&["hello", "world"]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 4);
    }

    #[test]
    fn mock_embedder_single() {
        let e = MockEmbedder;
        let result = e.embed_one("hello").unwrap();
        assert_eq!(result.len(), 4);
    }

    // ── Unit edge cases ────────────────────────────────────────────

    struct EmptyEmbedder;

    impl Embedder for EmptyEmbedder {
        fn embed(&self, _texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(vec![])
        }
        fn dimensions(&self) -> usize {
            4
        }
        fn model_id(&self) -> &str {
            "empty"
        }
    }

    #[test]
    fn embed_one_errors_on_empty_results() {
        let e = EmptyEmbedder;
        let result = e.embed_one("hello");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty results"));
    }

    #[test]
    fn embed_empty_batch() {
        let e = MockEmbedder;
        let results = e.embed(&[]).unwrap();
        assert!(results.is_empty());
    }

    // ── Conformance: Embedder trait contract ───────────────────────

    fn assert_embedder_contract(embedder: &dyn Embedder) {
        // dimensions() is consistent
        let dims = embedder.dimensions();
        assert!(dims > 0, "dimensions must be positive");

        // model_id() is non-empty
        assert!(
            !embedder.model_id().is_empty(),
            "model_id must be non-empty"
        );

        // embed_one returns vector of correct length
        let single = embedder.embed_one("test input").expect("embed_one");
        assert_eq!(
            single.len(),
            dims,
            "embed_one length must equal dimensions()"
        );

        // embed batch returns correct count with correct lengths
        let batch_input = &["alpha", "beta", "gamma"];
        let batch = embedder.embed(batch_input).expect("embed batch");
        assert_eq!(
            batch.len(),
            batch_input.len(),
            "embed batch length must equal input length"
        );
        for (i, vec) in batch.iter().enumerate() {
            assert_eq!(
                vec.len(),
                dims,
                "embed batch[{i}] length must equal dimensions()"
            );
        }

        // embed empty batch returns empty
        let empty = embedder.embed(&[]).expect("embed empty");
        assert!(empty.is_empty(), "embed empty batch must return empty");
    }

    #[test]
    fn mock_embedder_satisfies_contract() {
        assert_embedder_contract(&MockEmbedder);
    }
}
