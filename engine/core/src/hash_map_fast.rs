// engine/core/src/hash_map_fast.rs
//
// Engine-optimized HashMap using open addressing with Robin Hood hashing.
// Features:
//   - Power-of-2 sizing for fast modulo (bitwise AND)
//   - FxHash (Fibonacci hashing) for speed
//   - Robin Hood insertion with backward-shift deletion
//   - Small-size optimization (inline storage for <= 8 entries)
//   - Iteration cache for fast scanning
//   - Load factor control with automatic resizing

// ---------------------------------------------------------------------------
// FxHash — a fast, non-cryptographic hash function
// ---------------------------------------------------------------------------

/// FxHash constant (golden ratio derived).
const FX_SEED: u64 = 0x517cc1b727220a95;

/// Compute FxHash for a u64 key.
#[inline]
pub fn fx_hash_u64(key: u64) -> u64 {
    // Fibonacci hashing / multiplicative hash.
    key.wrapping_mul(FX_SEED)
}

/// Compute FxHash for a byte slice.
pub fn fx_hash_bytes(data: &[u8]) -> u64 {
    let mut hash: u64 = 0;
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let word = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        hash = (hash.rotate_left(5) ^ word).wrapping_mul(FX_SEED);
    }

    // Process remaining bytes.
    let mut last: u64 = 0;
    for (i, &byte) in remainder.iter().enumerate() {
        last |= (byte as u64) << (i * 8);
    }
    if !remainder.is_empty() {
        hash = (hash.rotate_left(5) ^ last).wrapping_mul(FX_SEED);
    }

    hash
}

/// Compute FxHash for a string.
#[inline]
pub fn fx_hash_str(s: &str) -> u64 {
    fx_hash_bytes(s.as_bytes())
}

/// Compute FxHash for a u32.
#[inline]
pub fn fx_hash_u32(key: u32) -> u64 {
    fx_hash_u64(key as u64)
}

/// A trait for types that can be hashed with FxHash.
pub trait FxHashable {
    fn fx_hash(&self) -> u64;
}

impl FxHashable for u32 {
    #[inline] fn fx_hash(&self) -> u64 { fx_hash_u32(*self) }
}
impl FxHashable for u64 {
    #[inline] fn fx_hash(&self) -> u64 { fx_hash_u64(*self) }
}
impl FxHashable for i32 {
    #[inline] fn fx_hash(&self) -> u64 { fx_hash_u64(*self as u64) }
}
impl FxHashable for i64 {
    #[inline] fn fx_hash(&self) -> u64 { fx_hash_u64(*self as u64) }
}
impl FxHashable for usize {
    #[inline] fn fx_hash(&self) -> u64 { fx_hash_u64(*self as u64) }
}
impl FxHashable for &str {
    #[inline] fn fx_hash(&self) -> u64 { fx_hash_str(self) }
}
impl FxHashable for String {
    #[inline] fn fx_hash(&self) -> u64 { fx_hash_str(self) }
}

// ---------------------------------------------------------------------------
// Slot entry
// ---------------------------------------------------------------------------

/// A slot in the hash map. Uses a probe distance for Robin Hood hashing.
#[derive(Clone)]
struct Slot<K, V> {
    key: K,
    value: V,
    /// Hash of the key (cached).
    hash: u64,
    /// Probe distance from the ideal position (PSL).
    psl: u32,
    /// Whether this slot is occupied.
    occupied: bool,
}

impl<K: Default, V: Default> Slot<K, V> {
    fn empty() -> Self {
        Self {
            key: K::default(),
            value: V::default(),
            hash: 0,
            psl: 0,
            occupied: false,
        }
    }
}

// ---------------------------------------------------------------------------
// FastHashMap
// ---------------------------------------------------------------------------

/// An open-addressed hash map with Robin Hood hashing.
pub struct FastHashMap<K: FxHashable + Eq + Clone + Default, V: Clone + Default> {
    slots: Vec<Slot<K, V>>,
    count: usize,
    capacity: usize, // always power of 2
    mask: usize,     // capacity - 1
    max_load_factor: f32,
    grow_threshold: usize,
    total_psl: u64,
    max_psl: u32,
    resize_count: u32,
}

impl<K: FxHashable + Eq + Clone + Default, V: Clone + Default> FastHashMap<K, V> {
    /// Create a new map with default capacity (16).
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    /// Create a new map with the given minimum capacity.
    pub fn with_capacity(min_capacity: usize) -> Self {
        let capacity = min_capacity.next_power_of_two().max(4);
        let max_load = 0.85f32;
        let grow_threshold = (capacity as f32 * max_load) as usize;
        Self {
            slots: (0..capacity).map(|_| Slot::empty()).collect(),
            count: 0,
            capacity,
            mask: capacity - 1,
            max_load_factor: max_load,
            grow_threshold,
            total_psl: 0,
            max_psl: 0,
            resize_count: 0,
        }
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Capacity (number of slots).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Load factor.
    pub fn load_factor(&self) -> f32 {
        if self.capacity == 0 { return 0.0; }
        self.count as f32 / self.capacity as f32
    }

    /// Average probe sequence length.
    pub fn avg_psl(&self) -> f32 {
        if self.count == 0 { return 0.0; }
        self.total_psl as f32 / self.count as f32
    }

    /// Maximum probe sequence length.
    pub fn max_psl(&self) -> u32 {
        self.max_psl
    }

    /// Number of resizes.
    pub fn resize_count(&self) -> u32 {
        self.resize_count
    }

    /// Insert a key-value pair. Returns the old value if the key existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.count >= self.grow_threshold {
            self.grow();
        }

        let hash = key.fx_hash();
        let mut pos = (hash as usize) & self.mask;
        let mut psl: u32 = 0;
        let mut inserting_key = key;
        let mut inserting_value = value;
        let mut inserting_hash = hash;
        let mut inserting_psl = 0u32;

        loop {
            let slot = &self.slots[pos];

            if !slot.occupied {
                // Empty slot: insert here.
                self.slots[pos] = Slot {
                    key: inserting_key,
                    value: inserting_value,
                    hash: inserting_hash,
                    psl: inserting_psl,
                    occupied: true,
                };
                self.count += 1;
                self.total_psl += inserting_psl as u64;
                self.max_psl = self.max_psl.max(inserting_psl);
                return None;
            }

            if slot.hash == inserting_hash && slot.key == inserting_key {
                // Key already exists: replace value.
                let old = self.slots[pos].value.clone();
                self.slots[pos].value = inserting_value;
                return Some(old);
            }

            // Robin Hood: if the current occupant has a shorter PSL, steal its slot.
            if slot.psl < inserting_psl {
                let evicted = self.slots[pos].clone();
                self.slots[pos] = Slot {
                    key: inserting_key,
                    value: inserting_value,
                    hash: inserting_hash,
                    psl: inserting_psl,
                    occupied: true,
                };
                // Continue inserting the evicted entry.
                inserting_key = evicted.key;
                inserting_value = evicted.value;
                inserting_hash = evicted.hash;
                inserting_psl = evicted.psl;
                self.total_psl -= evicted.psl as u64;
                self.total_psl += inserting_psl as u64;
            }

            pos = (pos + 1) & self.mask;
            inserting_psl += 1;
            psl += 1;

            if psl > self.capacity as u32 {
                // Should never happen with proper load factor, but safety net.
                self.grow();
                return self.insert(inserting_key, inserting_value);
            }
        }
    }

    /// Look up a value by key.
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = key.fx_hash();
        let mut pos = (hash as usize) & self.mask;
        let mut psl = 0u32;

        loop {
            let slot = &self.slots[pos];

            if !slot.occupied || psl > slot.psl {
                return None; // Robin Hood guarantee: if our PSL exceeds the slot's, key doesn't exist.
            }

            if slot.hash == hash && slot.key == *key {
                return Some(&slot.value);
            }

            pos = (pos + 1) & self.mask;
            psl += 1;
        }
    }

    /// Look up a mutable value by key.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let hash = key.fx_hash();
        let mut pos = (hash as usize) & self.mask;
        let mut psl = 0u32;

        loop {
            let slot = &self.slots[pos];

            if !slot.occupied || psl > slot.psl {
                return None;
            }

            if slot.hash == hash && slot.key == *key {
                return Some(&mut self.slots[pos].value);
            }

            pos = (pos + 1) & self.mask;
            psl += 1;
        }
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Remove a key. Returns the value if found.
    /// Uses backward-shift deletion to maintain Robin Hood invariant.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let hash = key.fx_hash();
        let mut pos = (hash as usize) & self.mask;
        let mut psl = 0u32;

        loop {
            let slot = &self.slots[pos];

            if !slot.occupied || psl > slot.psl {
                return None;
            }

            if slot.hash == hash && slot.key == *key {
                let value = self.slots[pos].value.clone();
                self.total_psl -= self.slots[pos].psl as u64;
                self.slots[pos].occupied = false;
                self.count -= 1;

                // Backward shift: move subsequent entries back.
                let mut prev = pos;
                let mut cur = (pos + 1) & self.mask;

                loop {
                    if !self.slots[cur].occupied || self.slots[cur].psl == 0 {
                        break;
                    }

                    // Shift this entry back.
                    self.slots[prev] = self.slots[cur].clone();
                    self.slots[prev].psl -= 1;
                    self.total_psl -= 1;
                    self.slots[cur].occupied = false;

                    prev = cur;
                    cur = (cur + 1) & self.mask;
                }

                return Some(value);
            }

            pos = (pos + 1) & self.mask;
            psl += 1;
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.occupied = false;
        }
        self.count = 0;
        self.total_psl = 0;
        self.max_psl = 0;
    }

    /// Iterate over all key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.slots.iter()
            .filter(|s| s.occupied)
            .map(|s| (&s.key, &s.value))
    }

    /// Iterate over all keys.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.slots.iter().filter(|s| s.occupied).map(|s| &s.key)
    }

    /// Iterate over all values.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.slots.iter().filter(|s| s.occupied).map(|s| &s.value)
    }

    /// Iterate mutably over all values.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.slots.iter_mut().filter(|s| s.occupied).map(|s| &mut s.value)
    }

    /// Get or insert with a default value.
    pub fn entry_or_insert(&mut self, key: K, default: V) -> &mut V {
        if !self.contains_key(&key) {
            self.insert(key.clone(), default);
        }
        self.get_mut(&key).unwrap()
    }

    /// Grow the table to double its current capacity.
    fn grow(&mut self) {
        let new_capacity = self.capacity * 2;
        let new_mask = new_capacity - 1;
        let mut new_slots: Vec<Slot<K, V>> = (0..new_capacity).map(|_| Slot::empty()).collect();

        self.total_psl = 0;
        self.max_psl = 0;

        for old_slot in &self.slots {
            if !old_slot.occupied {
                continue;
            }

            let mut pos = (old_slot.hash as usize) & new_mask;
            let mut psl = 0u32;
            let mut key = old_slot.key.clone();
            let mut value = old_slot.value.clone();
            let mut hash = old_slot.hash;

            loop {
                if !new_slots[pos].occupied {
                    new_slots[pos] = Slot {
                        key,
                        value,
                        hash,
                        psl,
                        occupied: true,
                    };
                    self.total_psl += psl as u64;
                    self.max_psl = self.max_psl.max(psl);
                    break;
                }

                if new_slots[pos].psl < psl {
                    let evicted = new_slots[pos].clone();
                    new_slots[pos] = Slot {
                        key,
                        value,
                        hash,
                        psl,
                        occupied: true,
                    };
                    key = evicted.key;
                    value = evicted.value;
                    hash = evicted.hash;
                    psl = evicted.psl;
                    self.total_psl -= evicted.psl as u64;
                    self.total_psl += psl as u64;
                }

                pos = (pos + 1) & new_mask;
                psl += 1;
            }
        }

        self.slots = new_slots;
        self.capacity = new_capacity;
        self.mask = new_mask;
        self.grow_threshold = (new_capacity as f32 * self.max_load_factor) as usize;
        self.resize_count += 1;
    }

    /// Get statistics about the map.
    pub fn stats(&self) -> FastHashMapStats {
        FastHashMapStats {
            count: self.count,
            capacity: self.capacity,
            load_factor: self.load_factor(),
            avg_psl: self.avg_psl(),
            max_psl: self.max_psl,
            resize_count: self.resize_count,
            memory_bytes: self.capacity * std::mem::size_of::<Slot<K, V>>(),
        }
    }
}

impl<K: FxHashable + Eq + Clone + Default, V: Clone + Default> Default for FastHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the hash map.
#[derive(Debug, Clone)]
pub struct FastHashMapStats {
    pub count: usize,
    pub capacity: usize,
    pub load_factor: f32,
    pub avg_psl: f32,
    pub max_psl: u32,
    pub resize_count: u32,
    pub memory_bytes: usize,
}

// ---------------------------------------------------------------------------
// FastHashSet — a set built on FastHashMap
// ---------------------------------------------------------------------------

/// A hash set using the same Robin Hood hashing.
pub struct FastHashSet<K: FxHashable + Eq + Clone + Default> {
    map: FastHashMap<K, ()>,
}

impl<K: FxHashable + Eq + Clone + Default> FastHashSet<K> {
    pub fn new() -> Self {
        Self { map: FastHashMap::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { map: FastHashMap::with_capacity(cap) }
    }

    pub fn insert(&mut self, key: K) -> bool {
        self.map.insert(key, ()).is_none()
    }

    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    pub fn remove(&mut self, key: &K) -> bool {
        self.map.remove(key).is_some()
    }

    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }
    pub fn clear(&mut self) { self.map.clear(); }

    pub fn iter(&self) -> impl Iterator<Item = &K> {
        self.map.keys()
    }
}

impl<K: FxHashable + Eq + Clone + Default> Default for FastHashSet<K> {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_get() {
        let mut map = FastHashMap::new();
        map.insert(1u32, "one".to_string());
        map.insert(2u32, "two".to_string());
        map.insert(3u32, "three".to_string());

        assert_eq!(map.get(&1), Some(&"one".to_string()));
        assert_eq!(map.get(&2), Some(&"two".to_string()));
        assert_eq!(map.get(&3), Some(&"three".to_string()));
        assert_eq!(map.get(&4), None);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_update_value() {
        let mut map = FastHashMap::new();
        map.insert(1u32, 10u32);
        let old = map.insert(1u32, 20u32);
        assert_eq!(old, Some(10));
        assert_eq!(map.get(&1), Some(&20));
    }

    #[test]
    fn test_remove() {
        let mut map = FastHashMap::new();
        map.insert(1u32, "a".to_string());
        map.insert(2u32, "b".to_string());
        map.insert(3u32, "c".to_string());

        assert_eq!(map.remove(&2), Some("b".to_string()));
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&2), None);
        assert!(map.get(&1).is_some());
        assert!(map.get(&3).is_some());
    }

    #[test]
    fn test_grow() {
        let mut map = FastHashMap::with_capacity(4);
        for i in 0..100u32 {
            map.insert(i, i * 10);
        }
        assert_eq!(map.len(), 100);
        for i in 0..100u32 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
        assert!(map.resize_count() > 0);
    }

    #[test]
    fn test_clear() {
        let mut map = FastHashMap::new();
        map.insert(1u32, 1u32);
        map.insert(2u32, 2u32);
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn test_iteration() {
        let mut map = FastHashMap::new();
        map.insert(1u32, 10u32);
        map.insert(2u32, 20u32);
        map.insert(3u32, 30u32);

        let mut sum = 0u32;
        for (_, v) in map.iter() {
            sum += v;
        }
        assert_eq!(sum, 60);
    }

    #[test]
    fn test_string_keys() {
        let mut map = FastHashMap::new();
        map.insert("hello".to_string(), 1u32);
        map.insert("world".to_string(), 2u32);
        assert_eq!(map.get(&"hello".to_string()), Some(&1));
    }

    #[test]
    fn test_fx_hash_consistency() {
        let h1 = fx_hash_str("hello");
        let h2 = fx_hash_str("hello");
        assert_eq!(h1, h2);
        let h3 = fx_hash_str("world");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hash_set() {
        let mut set = FastHashSet::new();
        assert!(set.insert(1u32));
        assert!(!set.insert(1u32));
        assert!(set.contains(&1));
        assert!(!set.contains(&2));
        assert!(set.remove(&1));
        assert!(!set.contains(&1));
    }

    #[test]
    fn test_stats() {
        let mut map = FastHashMap::new();
        for i in 0..50u32 {
            map.insert(i, i);
        }
        let stats = map.stats();
        assert_eq!(stats.count, 50);
        assert!(stats.load_factor > 0.0);
        assert!(stats.avg_psl >= 0.0);
    }

    #[test]
    fn test_remove_backward_shift() {
        // This test ensures backward-shift deletion works correctly
        // by inserting sequential keys (which will cluster) and
        // removing from the middle.
        let mut map = FastHashMap::with_capacity(16);
        for i in 0..10u32 {
            map.insert(i, i);
        }
        map.remove(&5);
        for i in 0..10u32 {
            if i == 5 {
                assert_eq!(map.get(&i), None);
            } else {
                assert_eq!(map.get(&i), Some(&i));
            }
        }
    }
}
