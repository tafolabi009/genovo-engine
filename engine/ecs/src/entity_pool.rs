// engine/ecs/src/entity_pool.rs
//
// Entity recycling pool for the Genovo ECS.
//
// Pre-allocates entity IDs and recycles despawned entities to reduce
// allocation overhead during gameplay:
//
// - **Pre-allocate entity IDs** -- Reserve a block of entity IDs upfront.
// - **Recycle despawned entities** -- Reuse IDs with generation increment.
// - **Reduce allocation overhead** -- Avoid frequent allocator calls.
// - **Pool warm-up** -- Pre-populate the pool at scene load.
// - **Statistics** -- Track pool usage and recycling efficiency.

use std::collections::VecDeque;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_POOL_SIZE: u32 = 1024;
const MAX_GENERATION: u32 = u32::MAX - 1;

// ---------------------------------------------------------------------------
// Entity ID
// ---------------------------------------------------------------------------

/// A recycled entity identifier with generation tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PooledEntity {
    pub index: u32,
    pub generation: u32,
}

impl PooledEntity {
    /// Create a new entity with generation 0.
    pub fn new(index: u32) -> Self {
        Self { index, generation: 0 }
    }

    /// Create with a specific generation.
    pub fn with_generation(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Pack into a single u64 for serialization.
    pub fn to_u64(&self) -> u64 {
        ((self.generation as u64) << 32) | (self.index as u64)
    }

    /// Unpack from u64.
    pub fn from_u64(packed: u64) -> Self {
        Self {
            index: (packed & 0xFFFFFFFF) as u32,
            generation: (packed >> 32) as u32,
        }
    }
}

impl fmt::Display for PooledEntity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}v{})", self.index, self.generation)
    }
}

// ---------------------------------------------------------------------------
// Slot state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct EntitySlot {
    generation: u32,
    alive: bool,
}

impl EntitySlot {
    fn new() -> Self {
        Self {
            generation: 0,
            alive: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Pool configuration
// ---------------------------------------------------------------------------

/// Configuration for the entity pool.
#[derive(Debug, Clone)]
pub struct EntityPoolConfig {
    /// Initial number of pre-allocated entity slots.
    pub initial_capacity: u32,
    /// Maximum number of entities (0 = unlimited).
    pub max_entities: u32,
    /// Number of entities to pre-warm.
    pub warm_count: u32,
    /// Whether to auto-grow when exhausted.
    pub auto_grow: bool,
    /// Growth amount when auto-growing.
    pub grow_amount: u32,
}

impl Default for EntityPoolConfig {
    fn default() -> Self {
        Self {
            initial_capacity: DEFAULT_POOL_SIZE,
            max_entities: 0,
            warm_count: 0,
            auto_grow: true,
            grow_amount: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// Pool statistics
// ---------------------------------------------------------------------------

/// Statistics for the entity pool.
#[derive(Debug, Clone, Default)]
pub struct EntityPoolStats {
    /// Total slots allocated.
    pub total_slots: u32,
    /// Currently alive entities.
    pub alive_count: u32,
    /// Available (recyclable) entity slots.
    pub available_count: u32,
    /// Total entities ever spawned.
    pub total_spawned: u64,
    /// Total entities ever despawned.
    pub total_despawned: u64,
    /// Total recycled (reused) entity IDs.
    pub total_recycled: u64,
    /// Number of pool growths.
    pub grow_count: u32,
    /// Peak alive count.
    pub peak_alive: u32,
    /// Average generation across all slots.
    pub avg_generation: f32,
    /// Maximum generation seen.
    pub max_generation: u32,
}

impl fmt::Display for EntityPoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EntityPool: {}/{} alive ({} available), {} spawned, {} recycled, peak {}",
            self.alive_count, self.total_slots, self.available_count,
            self.total_spawned, self.total_recycled, self.peak_alive
        )
    }
}

// ---------------------------------------------------------------------------
// Entity pool
// ---------------------------------------------------------------------------

/// Entity recycling pool with generational indices.
pub struct EntityPool {
    config: EntityPoolConfig,
    slots: Vec<EntitySlot>,
    free_list: VecDeque<u32>,
    stats: EntityPoolStats,
    next_fresh_index: u32,
}

impl EntityPool {
    /// Create a new entity pool.
    pub fn new(config: EntityPoolConfig) -> Self {
        let cap = config.initial_capacity as usize;
        let mut pool = Self {
            config,
            slots: Vec::with_capacity(cap),
            free_list: VecDeque::with_capacity(cap),
            stats: EntityPoolStats::default(),
            next_fresh_index: 0,
        };

        // Pre-allocate slots.
        for _ in 0..pool.config.initial_capacity {
            pool.allocate_slot();
        }

        pool
    }

    /// Create with default configuration.
    pub fn with_capacity(capacity: u32) -> Self {
        Self::new(EntityPoolConfig {
            initial_capacity: capacity,
            ..Default::default()
        })
    }

    /// Allocate a new slot (not yet alive).
    fn allocate_slot(&mut self) -> u32 {
        let index = self.slots.len() as u32;
        self.slots.push(EntitySlot::new());
        self.free_list.push_back(index);
        self.next_fresh_index = index + 1;
        self.stats.total_slots = self.slots.len() as u32;
        index
    }

    /// Grow the pool.
    fn grow(&mut self) -> bool {
        if self.config.max_entities > 0
            && self.slots.len() as u32 >= self.config.max_entities
        {
            return false;
        }
        let amount = self.config.grow_amount.min(
            if self.config.max_entities > 0 {
                self.config.max_entities - self.slots.len() as u32
            } else {
                self.config.grow_amount
            },
        );
        for _ in 0..amount {
            self.allocate_slot();
        }
        self.stats.grow_count += 1;
        true
    }

    /// Spawn a new entity (acquire from pool).
    pub fn spawn(&mut self) -> Option<PooledEntity> {
        let index = if let Some(idx) = self.free_list.pop_front() {
            idx
        } else if self.config.auto_grow && self.grow() {
            self.free_list.pop_front()?
        } else {
            return None;
        };

        let slot = &mut self.slots[index as usize];
        slot.alive = true;

        let entity = PooledEntity::with_generation(index, slot.generation);

        self.stats.total_spawned += 1;
        self.stats.alive_count += 1;
        if self.stats.alive_count > self.stats.peak_alive {
            self.stats.peak_alive = self.stats.alive_count;
        }
        self.stats.available_count = self.free_list.len() as u32;

        Some(entity)
    }

    /// Despawn an entity (release back to pool).
    pub fn despawn(&mut self, entity: PooledEntity) -> bool {
        let index = entity.index as usize;
        if index >= self.slots.len() {
            return false;
        }

        let slot = &mut self.slots[index];
        if !slot.alive || slot.generation != entity.generation {
            return false;
        }

        slot.alive = false;
        // Increment generation to invalidate stale references.
        if slot.generation < MAX_GENERATION {
            slot.generation += 1;
        } else {
            // Generation overflow: this slot cannot be reused.
            // In practice this would take billions of reuses.
            return true;
        }

        self.free_list.push_back(entity.index);
        self.stats.total_despawned += 1;
        self.stats.total_recycled += 1;
        self.stats.alive_count = self.stats.alive_count.saturating_sub(1);
        self.stats.available_count = self.free_list.len() as u32;

        if slot.generation > self.stats.max_generation {
            self.stats.max_generation = slot.generation;
        }

        true
    }

    /// Check if an entity is alive.
    pub fn is_alive(&self, entity: PooledEntity) -> bool {
        let index = entity.index as usize;
        if index >= self.slots.len() {
            return false;
        }
        let slot = &self.slots[index];
        slot.alive && slot.generation == entity.generation
    }

    /// Get the current generation of a slot.
    pub fn generation(&self, index: u32) -> Option<u32> {
        self.slots.get(index as usize).map(|s| s.generation)
    }

    /// Get pool statistics.
    pub fn stats(&self) -> &EntityPoolStats {
        &self.stats
    }

    /// Get the number of alive entities.
    pub fn alive_count(&self) -> u32 {
        self.stats.alive_count
    }

    /// Get the number of available slots.
    pub fn available_count(&self) -> u32 {
        self.free_list.len() as u32
    }

    /// Get the total capacity.
    pub fn capacity(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Warm the pool by pre-spawning and immediately despawning entities.
    pub fn warm(&mut self, count: u32) {
        let mut warmed = Vec::new();
        for _ in 0..count {
            if let Some(e) = self.spawn() {
                warmed.push(e);
            } else {
                break;
            }
        }
        for e in warmed {
            self.despawn(e);
        }
    }

    /// Reset the pool (despawn all entities).
    pub fn reset(&mut self) {
        self.free_list.clear();
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if slot.alive {
                slot.alive = false;
                slot.generation = slot.generation.saturating_add(1);
            }
            self.free_list.push_back(i as u32);
        }
        self.stats.alive_count = 0;
        self.stats.available_count = self.free_list.len() as u32;
    }

    /// Collect alive entities.
    pub fn alive_entities(&self) -> Vec<PooledEntity> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.alive)
            .map(|(i, s)| PooledEntity::with_generation(i as u32, s.generation))
            .collect()
    }

    /// Update average generation stat.
    pub fn update_stats(&mut self) {
        if self.slots.is_empty() {
            self.stats.avg_generation = 0.0;
            return;
        }
        let sum: u64 = self.slots.iter().map(|s| s.generation as u64).sum();
        self.stats.avg_generation = sum as f32 / self.slots.len() as f32;
    }
}

impl Default for EntityPool {
    fn default() -> Self {
        Self::new(EntityPoolConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_despawn() {
        let mut pool = EntityPool::with_capacity(10);
        let e = pool.spawn().unwrap();
        assert!(pool.is_alive(e));
        assert_eq!(pool.alive_count(), 1);

        pool.despawn(e);
        assert!(!pool.is_alive(e));
        assert_eq!(pool.alive_count(), 0);
    }

    #[test]
    fn test_generation() {
        let mut pool = EntityPool::with_capacity(1);
        let e1 = pool.spawn().unwrap();
        pool.despawn(e1);
        let e2 = pool.spawn().unwrap();

        assert_eq!(e1.index, e2.index);
        assert!(e2.generation > e1.generation);
        assert!(!pool.is_alive(e1));
        assert!(pool.is_alive(e2));
    }

    #[test]
    fn test_auto_grow() {
        let mut pool = EntityPool::new(EntityPoolConfig {
            initial_capacity: 2,
            auto_grow: true,
            grow_amount: 2,
            max_entities: 10,
            ..Default::default()
        });

        pool.spawn().unwrap();
        pool.spawn().unwrap();
        let e3 = pool.spawn().unwrap(); // Should grow.
        assert!(pool.is_alive(e3));
        assert!(pool.stats().grow_count > 0);
    }

    #[test]
    fn test_exhaustion() {
        let mut pool = EntityPool::new(EntityPoolConfig {
            initial_capacity: 1,
            auto_grow: false,
            max_entities: 1,
            ..Default::default()
        });

        pool.spawn().unwrap();
        assert!(pool.spawn().is_none());
    }

    #[test]
    fn test_packed_id() {
        let e = PooledEntity::with_generation(42, 7);
        let packed = e.to_u64();
        let unpacked = PooledEntity::from_u64(packed);
        assert_eq!(e, unpacked);
    }

    #[test]
    fn test_reset() {
        let mut pool = EntityPool::with_capacity(5);
        pool.spawn();
        pool.spawn();
        pool.spawn();
        assert_eq!(pool.alive_count(), 3);
        pool.reset();
        assert_eq!(pool.alive_count(), 0);
        assert_eq!(pool.available_count(), 5);
    }
}
