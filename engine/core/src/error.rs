//! Engine-wide error types.
//!
//! Every subsystem in Genovo funnels errors through [`EngineError`] so that
//! callers can handle failures uniformly. Subsystem-specific variants carry
//! enough context for diagnostics without leaking internal details.

use thiserror::Error;

/// Convenience alias used throughout the engine.
pub type EngineResult<T> = Result<T, EngineError>;

/// Top-level error type for the Genovo engine.
///
/// Each variant corresponds to a broad failure category. Subsystems may wrap
/// their own error types into the appropriate variant via `From` impls.
#[derive(Debug, Error)]
pub enum EngineError {
    /// An invalid argument was passed to an engine API.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A resource (asset, buffer, handle, etc.) was not found.
    #[error("resource not found: {0}")]
    NotFound(String),

    /// A handle was stale (generation mismatch) or otherwise invalid.
    #[error("invalid handle: index={index}, generation={generation}")]
    InvalidHandle {
        /// Slot index within the handle pool.
        index: u32,
        /// Expected generation.
        generation: u32,
    },

    /// A memory allocation failed.
    #[error("allocation failed: {reason} (requested {requested_bytes} bytes)")]
    AllocationFailed {
        /// Human-readable explanation.
        reason: String,
        /// Number of bytes that were requested.
        requested_bytes: usize,
    },

    /// An I/O operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A timeout expired before the operation completed.
    #[error("operation timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// The engine or a subsystem is in an invalid state for the requested
    /// operation.
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// A platform or graphics API call failed.
    #[error("platform error: {0}")]
    Platform(String),

    /// Catch-all for errors that do not fit other categories.
    #[error("engine error: {0}")]
    Other(String),
}
