use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemxError {
    #[error("budget exceeded: {current} chars exceeds {max} char limit")]
    BudgetExceeded { current: usize, max: usize },

    #[error("entry not found: {0}")]
    NotFound(crate::types::EntryId),

    #[error("duplicate entry detected")]
    Duplicate,

    #[error("storage error: {0}")]
    Storage(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, MemxError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_exceeded_includes_values() {
        let err = MemxError::BudgetExceeded {
            current: 2600,
            max: 2500,
        };
        let msg = MemxError::to_string(&err);
        assert!(msg.contains("2600"));
        assert!(msg.contains("2500"));
    }

    #[test]
    fn storage_error_contains_message() {
        let err = MemxError::Storage("connection failed".into());
        assert!(err.to_string().contains("connection failed"));
    }
}
