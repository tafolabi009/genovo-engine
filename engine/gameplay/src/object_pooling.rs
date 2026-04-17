// engine/gameplay/src/object_pooling.rs
//
// Object pool for gameplay entities in the Genovo engine.
//
// Provides pre-allocated, reusable entity pools to reduce runtime allocation:
//
// - **Pre-allocate entities** -- Create a pool of inactive entities on load.
// - **Acquire/release** -- Acquire an entity from the pool and release it back.
// - **Auto-grow** -- Optionally grow the pool when exhausted.
// - **Warm pool on scene load** -- Pre-populate pools for expected entity types.
// - **Pool statistics** -- Track usage, peak demand, and allocation patterns.
// - **Per-type pools** -- Separate pools for different entity types.

use std::collections::{HashMap, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Pool type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolTypeId(pub u32);

impl fmt::Display for PoolTypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PoolType({})", self.0)
    }
}

/// Pooled entity handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PooledHandle {
    pub pool_type: PoolTypeId,
    pub index: u32,
    pub generation: u32,
}

impl fmt::Display for PooledHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pooled({}:{}/{})", self.pool_type.0, self.index, self.generation)
    }
}

// ---------------------------------------------------------------------------
// Pool entry state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolEntryState {
    Available,
    InUse,
    Warming,
}

/// A single entry in the pool.
#[derive(Debug, Clone)]
pub struct PoolEntry {
    pub handle: PooledHandle,
    pub state: PoolEntryState,
    pub acquire_time: Option<f64>,
    pub lifetime: f32,
    pub auto_release_time: Option<f32>,
    pub user_data: u64,
}

impl PoolEntry {
    fn new(pool_type: PoolTypeId, index: u32) -> Self {
        Self {
            handle: PooledHandle { pool_type, index, generation: 0 },
            state: PoolEntryState::Available,
            acquire_time: None,
            lifetime: 0.0,
            auto_release_time: None,
            user_data: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Pool configuration
// ---------------------------------------------------------------------------

/// Configuration for a single entity pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Pool type identifier.
    pub pool_type: PoolTypeId,
    /// Display name for debugging.
    pub name: String,
    /// Initial pool size (pre-allocated).
    pub initial_size: u32,
    /// Maximum pool size (0 = unlimited).
    pub max_size: u32,
    /// Whether the pool auto-grows when exhausted.
    pub auto_grow: bool,
    /// Number of entries to add when auto-growing.
    pub grow_amount: u32,
    /// Whether to auto-release entries after a timeout.
    pub auto_release: bool,
    /// Auto-release timeout in seconds.
    pub auto_release_timeout: f32,
    /// Whether to warm the pool on creation.
    pub warm_on_create: bool,
}

impl PoolConfig {
    pub fn new(pool_type: PoolTypeId, name: &str, initial_size: u32) -> Self {
        Self {
            pool_type,
            name: name.to_string(),
            initial_size,
            max_size: initial_size * 4,
            auto_grow: true,
            grow_amount: initial_size / 2,
            auto_release: false,
            auto_release_timeout: 30.0,
            warm_on_create: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Pool statistics
// ---------------------------------------------------------------------------

/// Statistics for a single pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Pool type name.
    pub name: String,
    /// Total entries (available + in use).
    pub total_entries: u32,
    /// Currently available entries.
    pub available: u32,
    /// Currently in use.
    pub in_use: u32,
    /// Peak in-use count.
    pub peak_in_use: u32,
    /// Total acquisitions.
    pub total_acquires: u64,
    /// Total releases.
    pub total_releases: u64,
    /// Total auto-releases.
    pub total_auto_releases: u64,
    /// Number of times the pool grew.
    pub grow_count: u32,
    /// Number of failed acquisitions (pool exhausted).
    pub exhaustion_count: u64,
    /// Average lifetime of acquired entities.
    pub avg_lifetime: f32,
}

impl fmt::Display for PoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}/{} in use (peak {}), {} acquires, {} releases",
            self.name, self.in_use, self.total_entries, self.peak_in_use,
            self.total_acquires, self.total_releases,
        )
    }
}

// ---------------------------------------------------------------------------
// Object pool
// ---------------------------------------------------------------------------

/// A single typed object pool.
pub struct ObjectPool {
    config: PoolConfig,
    entries: Vec<PoolEntry>,
    available_indices: VecDeque<u32>,
    stats: PoolStats,
    game_time: f64,
    lifetime_sum: f64,
    lifetime_count: u64,
}

impl ObjectPool {
    /// Create a new pool.
    pub fn new(config: PoolConfig) -> Self {
        let mut pool = Self {
            stats: PoolStats { name: config.name.clone(), ..Default::default() },
            entries: Vec::with_capacity(config.initial_size as usize),
            available_indices: VecDeque::with_capacity(config.initial_size as usize),
            config,
            game_time: 0.0,
            lifetime_sum: 0.0,
            lifetime_count: 0,
        };

        // Pre-allocate entries.
        for _ in 0..pool.config.initial_size {
            pool.add_entry();
        }

        pool
    }

    /// Add a new entry to the pool.
    fn add_entry(&mut self) -> u32 {
        let index = self.entries.len() as u32;
        self.entries.push(PoolEntry::new(self.config.pool_type, index));
        self.available_indices.push_back(index);
        index
    }

    /// Grow the pool by the configured amount.
    fn grow(&mut self) -> bool {
        let max = self.config.max_size;
        if max > 0 && self.entries.len() as u32 >= max {
            return false;
        }
        let grow_amount = self.config.grow_amount.max(1);
        let can_grow = if max > 0 {
            max - self.entries.len() as u32
        } else {
            grow_amount
        };
        let actual_grow = grow_amount.min(can_grow);
        for _ in 0..actual_grow {
            self.add_entry();
        }
        self.stats.grow_count += 1;
        true
    }

    /// Acquire an entity from the pool.
    pub fn acquire(&mut self) -> Option<PooledHandle> {
        if let Some(index) = self.available_indices.pop_front() {
            let handle = {
                let entry = &mut self.entries[index as usize];
                entry.state = PoolEntryState::InUse;
                entry.acquire_time = Some(self.game_time);
                entry.lifetime = 0.0;
                entry.handle.generation += 1;
                if self.config.auto_release {
                    entry.auto_release_time = Some(self.config.auto_release_timeout);
                }
                entry.handle
            };
            self.stats.total_acquires += 1;
            self.update_stats();
            Some(handle)
        } else if self.config.auto_grow {
            if self.grow() {
                self.acquire()
            } else {
                self.stats.exhaustion_count += 1;
                None
            }
        } else {
            self.stats.exhaustion_count += 1;
            None
        }
    }

    /// Release an entity back to the pool.
    pub fn release(&mut self, handle: PooledHandle) -> bool {
        let index = handle.index as usize;
        if index >= self.entries.len() {
            return false;
        }
        let entry = &mut self.entries[index];
        if entry.handle.generation != handle.generation || entry.state != PoolEntryState::InUse {
            return false;
        }
        let lifetime = self.game_time - entry.acquire_time.unwrap_or(self.game_time);
        self.lifetime_sum += lifetime;
        self.lifetime_count += 1;

        entry.state = PoolEntryState::Available;
        entry.acquire_time = None;
        entry.auto_release_time = None;
        entry.user_data = 0;
        self.available_indices.push_back(handle.index);
        self.stats.total_releases += 1;
        self.update_stats();
        true
    }

    /// Check if a handle is valid and in use.
    pub fn is_active(&self, handle: PooledHandle) -> bool {
        let index = handle.index as usize;
        if index >= self.entries.len() {
            return false;
        }
        let entry = &self.entries[index];
        entry.handle.generation == handle.generation && entry.state == PoolEntryState::InUse
    }

    /// Get the user data for a handle.
    pub fn user_data(&self, handle: PooledHandle) -> Option<u64> {
        let index = handle.index as usize;
        if index < self.entries.len() && self.entries[index].handle.generation == handle.generation {
            Some(self.entries[index].user_data)
        } else {
            None
        }
    }

    /// Set user data for a handle.
    pub fn set_user_data(&mut self, handle: PooledHandle, data: u64) -> bool {
        let index = handle.index as usize;
        if index < self.entries.len() && self.entries[index].handle.generation == handle.generation {
            self.entries[index].user_data = data;
            true
        } else {
            false
        }
    }

    /// Update the pool (handle auto-release, update stats).
    pub fn update(&mut self, dt: f32) {
        self.game_time += dt as f64;

        if self.config.auto_release {
            let mut to_release = Vec::new();
            for entry in &mut self.entries {
                if entry.state == PoolEntryState::InUse {
                    entry.lifetime += dt;
                    if let Some(ref mut timeout) = entry.auto_release_time {
                        *timeout -= dt;
                        if *timeout <= 0.0 {
                            to_release.push(entry.handle);
                        }
                    }
                }
            }
            for handle in to_release {
                self.release(handle);
                self.stats.total_auto_releases += 1;
            }
        }

        self.update_stats();
    }

    /// Update statistics.
    fn update_stats(&mut self) {
        self.stats.total_entries = self.entries.len() as u32;
        self.stats.available = self.available_indices.len() as u32;
        self.stats.in_use = self.stats.total_entries - self.stats.available;
        if self.stats.in_use > self.stats.peak_in_use {
            self.stats.peak_in_use = self.stats.in_use;
        }
        self.stats.avg_lifetime = if self.lifetime_count > 0 {
            (self.lifetime_sum / self.lifetime_count as f64) as f32
        } else {
            0.0
        };
    }

    /// Get pool statistics.
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Get the config.
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// Release all entities.
    pub fn release_all(&mut self) {
        self.available_indices.clear();
        for (i, entry) in self.entries.iter_mut().enumerate() {
            if entry.state == PoolEntryState::InUse {
                entry.state = PoolEntryState::Available;
                self.stats.total_releases += 1;
            }
            self.available_indices.push_back(i as u32);
        }
        self.update_stats();
    }
}

// ---------------------------------------------------------------------------
// Pool manager
// ---------------------------------------------------------------------------

/// Manages multiple typed pools.
pub struct PoolManager {
    pools: HashMap<PoolTypeId, ObjectPool>,
    next_type_id: u32,
}

impl PoolManager {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            next_type_id: 0,
        }
    }

    /// Create a new pool.
    pub fn create_pool(&mut self, name: &str, initial_size: u32) -> PoolTypeId {
        let id = PoolTypeId(self.next_type_id);
        self.next_type_id += 1;
        let config = PoolConfig::new(id, name, initial_size);
        self.pools.insert(id, ObjectPool::new(config));
        id
    }

    /// Create a pool with custom config.
    pub fn create_pool_with_config(&mut self, config: PoolConfig) -> PoolTypeId {
        let id = config.pool_type;
        self.pools.insert(id, ObjectPool::new(config));
        id
    }

    /// Get a pool.
    pub fn pool(&self, id: PoolTypeId) -> Option<&ObjectPool> {
        self.pools.get(&id)
    }

    /// Get a mutable pool.
    pub fn pool_mut(&mut self, id: PoolTypeId) -> Option<&mut ObjectPool> {
        self.pools.get_mut(&id)
    }

    /// Acquire from a specific pool.
    pub fn acquire(&mut self, pool_type: PoolTypeId) -> Option<PooledHandle> {
        self.pools.get_mut(&pool_type)?.acquire()
    }

    /// Release to the appropriate pool.
    pub fn release(&mut self, handle: PooledHandle) -> bool {
        self.pools
            .get_mut(&handle.pool_type)
            .map(|p| p.release(handle))
            .unwrap_or(false)
    }

    /// Update all pools.
    pub fn update(&mut self, dt: f32) {
        for pool in self.pools.values_mut() {
            pool.update(dt);
        }
    }

    /// Get statistics for all pools.
    pub fn all_stats(&self) -> Vec<&PoolStats> {
        self.pools.values().map(|p| p.stats()).collect()
    }

    /// Get aggregate stats.
    pub fn total_in_use(&self) -> u32 {
        self.pools.values().map(|p| p.stats().in_use).sum()
    }

    /// Get aggregate stats.
    pub fn total_available(&self) -> u32 {
        self.pools.values().map(|p| p.stats().available).sum()
    }
}

impl Default for PoolManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_release() {
        let config = PoolConfig::new(PoolTypeId(0), "test", 10);
        let mut pool = ObjectPool::new(config);

        let h = pool.acquire().unwrap();
        assert!(pool.is_active(h));
        assert_eq!(pool.stats().in_use, 1);

        pool.release(h);
        assert!(!pool.is_active(h));
        assert_eq!(pool.stats().in_use, 0);
    }

    #[test]
    fn test_auto_grow() {
        let mut config = PoolConfig::new(PoolTypeId(0), "test", 2);
        config.auto_grow = true;
        config.grow_amount = 2;
        config.max_size = 10;
        let mut pool = ObjectPool::new(config);

        pool.acquire().unwrap();
        pool.acquire().unwrap();
        // Pool should auto-grow.
        let h = pool.acquire().unwrap();
        assert!(pool.is_active(h));
        assert!(pool.stats().grow_count > 0);
    }

    #[test]
    fn test_exhaustion() {
        let mut config = PoolConfig::new(PoolTypeId(0), "test", 1);
        config.auto_grow = false;
        config.max_size = 1;
        let mut pool = ObjectPool::new(config);

        pool.acquire().unwrap();
        assert!(pool.acquire().is_none());
        assert_eq!(pool.stats().exhaustion_count, 1);
    }

    #[test]
    fn test_pool_manager() {
        let mut mgr = PoolManager::new();
        let pt = mgr.create_pool("bullets", 100);

        let h = mgr.acquire(pt).unwrap();
        assert_eq!(mgr.total_in_use(), 1);

        mgr.release(h);
        assert_eq!(mgr.total_in_use(), 0);
    }

    #[test]
    fn test_generation() {
        let config = PoolConfig::new(PoolTypeId(0), "test", 2);
        let mut pool = ObjectPool::new(config);

        let h1 = pool.acquire().unwrap();
        pool.release(h1);

        let h2 = pool.acquire().unwrap();
        // Same index but different generation.
        assert_eq!(h1.index, h2.index);
        assert_ne!(h1.generation, h2.generation);
        assert!(!pool.is_active(h1));
        assert!(pool.is_active(h2));
    }
}
