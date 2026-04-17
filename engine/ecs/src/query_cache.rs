//! Query result caching for the Genovo ECS.
//!
//! Caches query results across frames so repeated queries with the same
//! parameters avoid redundant iteration. Results are invalidated when
//! archetype changes occur (entity spawn/despawn, component add/remove).
//!
//! # Architecture
//!
//! ```text
//!   Query("Transform, Velocity")
//!       │
//!       ▼
//!   ┌─────────────┐    hit
//!   │ Query Cache  │──────────► cached entity list
//!   └─────┬───────┘
//!         │ miss
//!         ▼
//!   ┌─────────────┐
//!   │ Full query   │──────────► fresh entity list → store in cache
//!   └─────────────┘
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// QueryKey
// ---------------------------------------------------------------------------

/// A key that uniquely identifies a query (set of required/excluded components).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryKey {
    /// Component type names that must be present.
    pub required: Vec<String>,
    /// Component type names that must be absent.
    pub excluded: Vec<String>,
    /// Optional tag filter.
    pub filter_tag: Option<String>,
}

impl QueryKey {
    /// Create a new query key.
    pub fn new(required: &[&str], excluded: &[&str]) -> Self {
        let mut req: Vec<String> = required.iter().map(|s| s.to_string()).collect();
        req.sort();
        let mut exc: Vec<String> = excluded.iter().map(|s| s.to_string()).collect();
        exc.sort();
        Self {
            required: req,
            excluded: exc,
            filter_tag: None,
        }
    }

    /// Create with required components only.
    pub fn with_components(components: &[&str]) -> Self {
        Self::new(components, &[])
    }

    /// Add a tag filter.
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.filter_tag = Some(tag.to_string());
        self
    }

    /// Returns a stable string representation for debugging.
    pub fn fingerprint(&self) -> String {
        let req = self.required.join(",");
        let exc = self.excluded.join(",");
        let tag = self
            .filter_tag
            .as_deref()
            .unwrap_or("");
        format!("req=[{}] exc=[{}] tag={}", req, exc, tag)
    }
}

impl fmt::Display for QueryKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.fingerprint())
    }
}

// ---------------------------------------------------------------------------
// CachedQueryResult
// ---------------------------------------------------------------------------

/// A cached set of entity IDs matching a query.
#[derive(Debug, Clone)]
pub struct CachedQueryResult {
    /// The query key.
    pub key: QueryKey,
    /// Matched entity IDs.
    pub entities: Vec<u32>,
    /// The archetype generation at which this result was computed.
    pub archetype_generation: u64,
    /// When this result was last accessed.
    pub last_access: Instant,
    /// When this result was created.
    pub created_at: Instant,
    /// Number of times this cached result has been accessed.
    pub access_count: u64,
    /// How long the query took to evaluate (for profiling).
    pub evaluation_time_us: u64,
}

impl CachedQueryResult {
    /// Create a new cached result.
    pub fn new(
        key: QueryKey,
        entities: Vec<u32>,
        archetype_generation: u64,
        evaluation_time_us: u64,
    ) -> Self {
        let now = Instant::now();
        Self {
            key,
            entities,
            archetype_generation,
            last_access: now,
            created_at: now,
            access_count: 0,
            evaluation_time_us,
        }
    }

    /// Mark this result as accessed.
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;
    }

    /// Returns the age of this cached result.
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Returns time since last access.
    pub fn idle_time(&self) -> std::time::Duration {
        self.last_access.elapsed()
    }

    /// Returns the number of entities in this result.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}

// ---------------------------------------------------------------------------
// CacheConfig
// ---------------------------------------------------------------------------

/// Configuration for the query cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of cached queries.
    pub max_entries: usize,
    /// Maximum age before eviction (in seconds).
    pub max_age_secs: f64,
    /// Maximum idle time before eviction (in seconds).
    pub max_idle_secs: f64,
    /// Whether to collect hit/miss statistics.
    pub track_stats: bool,
    /// Minimum evaluation time (microseconds) to qualify for caching.
    /// Queries faster than this threshold won't be cached.
    pub min_cache_time_us: u64,
    /// Whether lazy re-evaluation is enabled.
    pub lazy_evaluation: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_age_secs: 60.0,
            max_idle_secs: 10.0,
            track_stats: true,
            min_cache_time_us: 10,
            lazy_evaluation: true,
        }
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Statistics about query cache performance.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of cache lookups.
    pub total_lookups: u64,
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of invalidations.
    pub invalidations: u64,
    /// Number of evictions (due to capacity or age).
    pub evictions: u64,
    /// Total time saved by cache hits (microseconds).
    pub time_saved_us: u64,
    /// Current number of cached entries.
    pub current_entries: usize,
    /// Peak number of cached entries.
    pub peak_entries: usize,
    /// Total number of insertions.
    pub insertions: u64,
}

impl CacheStats {
    /// Returns the hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_lookups as f64
        }
    }

    /// Returns the miss rate (0.0 to 1.0).
    pub fn miss_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.misses as f64 / self.total_lookups as f64
        }
    }

    /// Returns the estimated time saved in milliseconds.
    pub fn time_saved_ms(&self) -> f64 {
        self.time_saved_us as f64 / 1000.0
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = CacheStats::default();
    }
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Query Cache Statistics:")?;
        writeln!(f, "  lookups:      {}", self.total_lookups)?;
        writeln!(f, "  hits:         {} ({:.1}%)", self.hits, self.hit_rate() * 100.0)?;
        writeln!(f, "  misses:       {} ({:.1}%)", self.misses, self.miss_rate() * 100.0)?;
        writeln!(f, "  invalidations:{}", self.invalidations)?;
        writeln!(f, "  evictions:    {}", self.evictions)?;
        writeln!(f, "  insertions:   {}", self.insertions)?;
        writeln!(f, "  entries:      {}", self.current_entries)?;
        writeln!(f, "  peak entries: {}", self.peak_entries)?;
        writeln!(
            f,
            "  time saved:   {:.2}ms",
            self.time_saved_ms()
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// QueryCache
// ---------------------------------------------------------------------------

/// Caches query results to avoid repeated iteration over archetypes.
pub struct QueryCache {
    /// Cached results, keyed by QueryKey.
    entries: HashMap<QueryKey, CachedQueryResult>,
    /// Current archetype generation (incremented on structural changes).
    current_generation: u64,
    /// Configuration.
    config: CacheConfig,
    /// Performance statistics.
    stats: CacheStats,
    /// LRU ordering: most recently accessed key at the end.
    lru_order: Vec<QueryKey>,
}

impl QueryCache {
    /// Create a new query cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new query cache with custom configuration.
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            current_generation: 0,
            config,
            stats: CacheStats::default(),
            lru_order: Vec::new(),
        }
    }

    /// Notify the cache that an archetype change has occurred.
    /// This increments the generation counter, causing stale entries to
    /// be lazily invalidated on next access.
    pub fn notify_archetype_change(&mut self) {
        self.current_generation += 1;
        self.stats.invalidations += 1;
    }

    /// Bulk invalidation: mark the entire cache as stale.
    pub fn invalidate_all(&mut self) {
        self.current_generation += 1;
        self.stats.invalidations += 1;
    }

    /// Invalidate entries that involve specific components.
    pub fn invalidate_component(&mut self, component: &str) {
        let stale_keys: Vec<QueryKey> = self
            .entries
            .iter()
            .filter(|(key, _)| {
                key.required.iter().any(|c| c == component)
                    || key.excluded.iter().any(|c| c == component)
            })
            .map(|(key, _)| key.clone())
            .collect();

        for key in stale_keys {
            self.entries.remove(&key);
            self.lru_order.retain(|k| k != &key);
        }
    }

    /// Look up a cached query result.
    ///
    /// Returns `Some` if the result is valid (same generation), `None` on miss.
    pub fn get(&mut self, key: &QueryKey) -> Option<&[u32]> {
        self.stats.total_lookups += 1;

        if let Some(entry) = self.entries.get_mut(key) {
            if entry.archetype_generation == self.current_generation {
                // Cache hit.
                entry.touch();
                self.stats.hits += 1;
                self.stats.time_saved_us += entry.evaluation_time_us;

                // Update LRU order.
                self.lru_order.retain(|k| k != key);
                self.lru_order.push(key.clone());

                return Some(&entry.entities);
            }
            // Stale entry.
        }

        self.stats.misses += 1;
        None
    }

    /// Insert a query result into the cache.
    pub fn insert(
        &mut self,
        key: QueryKey,
        entities: Vec<u32>,
        evaluation_time_us: u64,
    ) {
        // Don't cache trivially fast queries.
        if evaluation_time_us < self.config.min_cache_time_us {
            return;
        }

        // Evict if at capacity.
        while self.entries.len() >= self.config.max_entries {
            self.evict_lru();
        }

        let result = CachedQueryResult::new(
            key.clone(),
            entities,
            self.current_generation,
            evaluation_time_us,
        );

        self.entries.insert(key.clone(), result);
        self.lru_order.retain(|k| k != &key);
        self.lru_order.push(key);

        self.stats.insertions += 1;
        self.stats.current_entries = self.entries.len();
        if self.stats.current_entries > self.stats.peak_entries {
            self.stats.peak_entries = self.stats.current_entries;
        }
    }

    /// Get a result, inserting it if missing using the provided evaluator.
    pub fn get_or_insert<F>(&mut self, key: &QueryKey, evaluator: F) -> &[u32]
    where
        F: FnOnce(&QueryKey) -> (Vec<u32>, u64),
    {
        // Check cache first.
        if self.get(key).is_some() {
            // We just called get() which recorded the hit.
            return &self.entries.get(key).unwrap().entities;
        }

        // Evaluate and insert.
        let (entities, time_us) = evaluator(key);
        self.insert(key.clone(), entities, time_us);

        &self.entries.get(key).unwrap().entities
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if let Some(oldest_key) = self.lru_order.first().cloned() {
            self.entries.remove(&oldest_key);
            self.lru_order.remove(0);
            self.stats.evictions += 1;
            self.stats.current_entries = self.entries.len();
        }
    }

    /// Evict entries older than `max_age_secs`.
    pub fn evict_stale(&mut self) {
        let max_age = std::time::Duration::from_secs_f64(self.config.max_age_secs);
        let max_idle = std::time::Duration::from_secs_f64(self.config.max_idle_secs);

        let stale_keys: Vec<QueryKey> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.age() > max_age || entry.idle_time() > max_idle)
            .map(|(key, _)| key.clone())
            .collect();

        for key in stale_keys {
            self.entries.remove(&key);
            self.lru_order.retain(|k| k != &key);
            self.stats.evictions += 1;
        }

        self.stats.current_entries = self.entries.len();
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
        self.stats.current_entries = 0;
    }

    /// Returns a reference to the cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
        self.stats.current_entries = self.entries.len();
    }

    /// Returns the current number of cached entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the current archetype generation.
    pub fn generation(&self) -> u64 {
        self.current_generation
    }

    /// Returns the configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: CacheConfig) {
        self.config = config;
    }

    /// Get information about all cached queries.
    pub fn cached_queries(&self) -> Vec<CacheEntryInfo> {
        self.entries
            .iter()
            .map(|(key, entry)| CacheEntryInfo {
                key: key.fingerprint(),
                entity_count: entry.entity_count(),
                access_count: entry.access_count,
                age_ms: entry.age().as_millis() as u64,
                idle_ms: entry.idle_time().as_millis() as u64,
                evaluation_time_us: entry.evaluation_time_us,
                is_valid: entry.archetype_generation == self.current_generation,
            })
            .collect()
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for QueryCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryCache")
            .field("entries", &self.entries.len())
            .field("generation", &self.current_generation)
            .field("hit_rate", &format!("{:.1}%", self.stats.hit_rate() * 100.0))
            .finish()
    }
}

/// Information about a single cache entry.
#[derive(Debug, Clone)]
pub struct CacheEntryInfo {
    pub key: String,
    pub entity_count: usize,
    pub access_count: u64,
    pub age_ms: u64,
    pub idle_ms: u64,
    pub evaluation_time_us: u64,
    pub is_valid: bool,
}

impl fmt::Display for CacheEntryInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} entities, {} accesses, age={}ms, eval={}us, valid={}",
            self.key,
            self.entity_count,
            self.access_count,
            self.age_ms,
            self.evaluation_time_us,
            self.is_valid
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit() {
        let mut cache = QueryCache::new();
        let key = QueryKey::with_components(&["Transform", "Velocity"]);

        cache.insert(key.clone(), vec![1, 2, 3], 100);

        let result = cache.get(&key);
        assert_eq!(result, Some([1u32, 2, 3].as_slice()));
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = QueryCache::new();
        let key = QueryKey::with_components(&["Transform"]);

        let result = cache.get(&key);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_invalidation() {
        let mut cache = QueryCache::new();
        let key = QueryKey::with_components(&["Transform"]);

        cache.insert(key.clone(), vec![1, 2], 100);
        assert!(cache.get(&key).is_some());

        cache.notify_archetype_change();
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_component_invalidation() {
        let mut cache = QueryCache::new();
        let key1 = QueryKey::with_components(&["Transform"]);
        let key2 = QueryKey::with_components(&["Velocity"]);

        cache.insert(key1.clone(), vec![1], 100);
        cache.insert(key2.clone(), vec![2], 100);

        cache.invalidate_component("Transform");

        // key1 should be gone, key2 should remain.
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = QueryCache::with_config(CacheConfig {
            max_entries: 3,
            ..Default::default()
        });

        let keys: Vec<QueryKey> = (0..4)
            .map(|i| QueryKey::with_components(&[&format!("Comp{}", i)]))
            .collect();

        for (i, key) in keys.iter().enumerate() {
            cache.insert(key.clone(), vec![i as u32], 100);
        }

        // Should have evicted the first entry.
        assert_eq!(cache.entry_count(), 3);
        assert!(cache.get(&keys[0]).is_none());
        assert!(cache.get(&keys[3]).is_some());
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = QueryCache::new();
        let key = QueryKey::with_components(&["Transform"]);

        cache.insert(key.clone(), vec![1], 100);

        // 3 hits, 1 miss.
        cache.get(&key);
        cache.get(&key);
        cache.get(&key);
        let missing_key = QueryKey::with_components(&["Missing"]);
        cache.get(&missing_key);

        assert!((cache.stats().hit_rate() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_get_or_insert() {
        let mut cache = QueryCache::new();
        let key = QueryKey::with_components(&["Transform"]);

        let result = cache.get_or_insert(&key, |_| (vec![10, 20, 30], 50));
        assert_eq!(result, &[10, 20, 30]);

        // Second call should be a cache hit.
        let result = cache.get_or_insert(&key, |_| panic!("should not evaluate"));
        assert_eq!(result, &[10, 20, 30]);
    }

    #[test]
    fn test_clear() {
        let mut cache = QueryCache::new();
        let key = QueryKey::with_components(&["A"]);
        cache.insert(key, vec![1], 100);
        assert_eq!(cache.entry_count(), 1);

        cache.clear();
        assert_eq!(cache.entry_count(), 0);
    }
}
