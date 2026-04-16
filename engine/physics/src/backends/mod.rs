//! Physics backend implementations.
//!
//! Provides pluggable physics engine backends. The `CustomBackend` is the real,
//! fully-implemented pure-Rust backend wrapping `PhysicsWorld`.

use glam::Vec3;

use crate::interface::{PhysicsBackend, PhysicsResult, PhysicsWorld};

// ===========================================================================
// PhysX Backend (stub -- requires C++ FFI bindings)
// ===========================================================================

/// NVIDIA PhysX backend (stub).
///
/// Would wrap PhysX 5.x via C++ FFI bindings. Not yet implemented;
/// use `CustomBackend` for a working pure-Rust implementation.
pub struct PhysXBackend {
    _private: (),
}

impl PhysXBackend {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl PhysicsBackend for PhysXBackend {
    fn name(&self) -> &str {
        "PhysX 5.x (stub)"
    }

    fn create_world(&self, gravity: Vec3) -> PhysicsResult<PhysicsWorld> {
        // PhysX FFI not available; fall back to custom backend
        log::warn!("PhysX backend not available, use CustomBackend instead");
        Ok(PhysicsWorld::new(gravity))
    }

    fn is_available(&self) -> bool {
        false
    }
}

// ===========================================================================
// Bullet Physics Backend (stub -- requires C++ FFI bindings)
// ===========================================================================

/// Bullet Physics backend (stub).
pub struct BulletBackend {
    _private: (),
}

impl BulletBackend {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl PhysicsBackend for BulletBackend {
    fn name(&self) -> &str {
        "Bullet 3.x (stub)"
    }

    fn create_world(&self, gravity: Vec3) -> PhysicsResult<PhysicsWorld> {
        log::warn!("Bullet backend not available, use CustomBackend instead");
        Ok(PhysicsWorld::new(gravity))
    }

    fn is_available(&self) -> bool {
        false
    }
}

// ===========================================================================
// Custom Backend -- Fully Implemented Pure-Rust Physics
// ===========================================================================

/// Custom in-house physics backend.
///
/// A pure-Rust physics implementation with no external C/C++ dependencies.
/// Uses spatial hash broadphase, SAT/analytical narrow phase, sequential impulse
/// constraint solver, semi-implicit Euler integration, and body sleeping.
pub struct CustomBackend;

impl CustomBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CustomBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsBackend for CustomBackend {
    fn name(&self) -> &str {
        "Genovo Custom Physics"
    }

    fn create_world(&self, gravity: Vec3) -> PhysicsResult<PhysicsWorld> {
        Ok(PhysicsWorld::new(gravity))
    }

    fn is_available(&self) -> bool {
        true // Always available -- pure Rust, no external deps
    }
}

// ===========================================================================
// Backend registry
// ===========================================================================

/// Returns all known physics backends ordered by preference.
pub fn available_backends() -> Vec<Box<dyn PhysicsBackend>> {
    vec![
        Box::new(CustomBackend::new()),
        Box::new(PhysXBackend::new()),
        Box::new(BulletBackend::new()),
    ]
}

/// Select the best available physics backend.
///
/// Currently always returns `CustomBackend` since it is always available.
pub fn select_best_backend() -> Option<Box<dyn PhysicsBackend>> {
    available_backends().into_iter().find(|b| b.is_available())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_backend_available() {
        let backend = CustomBackend::new();
        assert!(backend.is_available());
        assert_eq!(backend.name(), "Genovo Custom Physics");
    }

    #[test]
    fn test_custom_backend_create_world() {
        let backend = CustomBackend::new();
        let world = backend.create_world(Vec3::new(0.0, -9.81, 0.0));
        assert!(world.is_ok());
    }

    #[test]
    fn test_select_best_backend() {
        let backend = select_best_backend();
        assert!(backend.is_some());
        assert_eq!(backend.unwrap().name(), "Genovo Custom Physics");
    }

    #[test]
    fn test_phsx_backend_not_available() {
        let backend = PhysXBackend::new();
        assert!(!backend.is_available());
    }

    #[test]
    fn test_bullet_backend_not_available() {
        let backend = BulletBackend::new();
        assert!(!backend.is_available());
    }
}
