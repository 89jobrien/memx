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
    Storage(#[from] rusqlite::Error),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, MemxError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = MemxError::BudgetExceeded {
            current: 2600,
            max: 2500,
        };
        let msg = err.to_string();
        assert!(msg.contains("2600"));
        assert!(msg.contains("2500"));
    }

    #[test]
    fn error_from_rusqlite() {
        let sql_err = rusqlite::Error::QueryReturnedNoRows;
        let err: MemxError = sql_err.into();
        assert!(matches!(err, MemxError::Storage(_)));
    }
}
