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
}
