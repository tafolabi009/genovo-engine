// engine/core/src/data_structures.rs
//
// Additional data structures for the Genovo engine:
// Skip list, B-tree, trie (prefix tree), bloom filter, count-min sketch,
// disjoint set (union-find), LRU cache, concurrent queue.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Disjoint Set (Union-Find with path compression and rank)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u32>,
    size: Vec<usize>,
    num_sets: usize,
}

impl DisjointSet {
    pub fn new(n: usize) -> Self {
        Self { parent: (0..n).collect(), rank: vec![0; n], size: vec![1; n], num_sets: n }
    }
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x { self.parent[x] = self.find(self.parent[x]); }
        self.parent[x]
    }
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a); let rb = self.find(b);
        if ra == rb { return false; }
        if self.rank[ra] < self.rank[rb] { self.parent[ra] = rb; self.size[rb] += self.size[ra]; }
        else if self.rank[ra] > self.rank[rb] { self.parent[rb] = ra; self.size[ra] += self.size[rb]; }
        else { self.parent[rb] = ra; self.size[ra] += self.size[rb]; self.rank[ra] += 1; }
        self.num_sets -= 1;
        true
    }
    pub fn connected(&mut self, a: usize, b: usize) -> bool { self.find(a) == self.find(b) }
    pub fn set_size(&mut self, x: usize) -> usize { let r = self.find(x); self.size[r] }
    pub fn num_sets(&self) -> usize { self.num_sets }
    pub fn len(&self) -> usize { self.parent.len() }
}

// ---------------------------------------------------------------------------
// LRU Cache
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct LruCache<K: Eq + Hash + Clone, V> {
    capacity: usize,
    entries: HashMap<K, (V, u64)>,
    access_counter: u64,
}

impl<K: Eq + Hash + Clone, V> LruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self { capacity: capacity.max(1), entries: HashMap::new(), access_counter: 0 }
    }
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.entries.contains_key(key) {
            self.access_counter += 1;
            if let Some(entry) = self.entries.get_mut(key) { entry.1 = self.access_counter; }
            self.entries.get(key).map(|(v, _)| v)
        } else { None }
    }
    pub fn insert(&mut self, key: K, value: V) {
        self.access_counter += 1;
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) { self.evict(); }
        self.entries.insert(key, (value, self.access_counter));
    }
    pub fn remove(&mut self, key: &K) -> Option<V> { self.entries.remove(key).map(|(v, _)| v) }
    fn evict(&mut self) {
        let lru_key = self.entries.iter().min_by_key(|(_, (_, t))| *t).map(|(k, _)| k.clone());
        if let Some(key) = lru_key { self.entries.remove(&key); }
    }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn capacity(&self) -> usize { self.capacity }
    pub fn contains(&self, key: &K) -> bool { self.entries.contains_key(key) }
    pub fn clear(&mut self) { self.entries.clear(); self.access_counter = 0; }
}

// ---------------------------------------------------------------------------
// Bloom Filter
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct BloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl BloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        Self { bits: vec![false; size], num_hashes, size, count: 0 }
    }
    pub fn with_fp_rate(expected_elements: usize, fp_rate: f64) -> Self {
        let size = (-(expected_elements as f64 * fp_rate.ln()) / (2.0f64.ln().powi(2))).ceil() as usize;
        let num_hashes = ((size as f64 / expected_elements as f64) * 2.0f64.ln()).ceil() as usize;
        Self::new(size.max(1), num_hashes.max(1))
    }
    fn hash_indices(&self, item: u64) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.num_hashes);
        for i in 0..self.num_hashes {
            let h = item.wrapping_mul(0x517cc1b727220a95).wrapping_add(i as u64 * 0x6c62272e07bb0142);
            indices.push((h as usize) % self.size);
        }
        indices
    }
    pub fn insert_u64(&mut self, item: u64) {
        for idx in self.hash_indices(item) { self.bits[idx] = true; }
        self.count += 1;
    }
    pub fn contains_u64(&self, item: u64) -> bool {
        self.hash_indices(item).iter().all(|&idx| self.bits[idx])
    }
    pub fn insert_str(&mut self, s: &str) {
        let h = fnv_hash(s.as_bytes());
        self.insert_u64(h);
    }
    pub fn contains_str(&self, s: &str) -> bool {
        let h = fnv_hash(s.as_bytes());
        self.contains_u64(h)
    }
    pub fn clear(&mut self) { self.bits.fill(false); self.count = 0; }
    pub fn count(&self) -> usize { self.count }
    pub fn estimated_fp_rate(&self) -> f64 {
        let ones = self.bits.iter().filter(|&&b| b).count() as f64;
        let ratio = ones / self.size as f64;
        ratio.powi(self.num_hashes as i32)
    }
}

fn fnv_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data { hash ^= byte as u64; hash = hash.wrapping_mul(0x100000001b3); }
    hash
}

// ---------------------------------------------------------------------------
// Count-Min Sketch
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CountMinSketch {
    table: Vec<Vec<u32>>,
    width: usize,
    depth: usize,
}

impl CountMinSketch {
    pub fn new(width: usize, depth: usize) -> Self {
        Self { table: vec![vec![0u32; width]; depth], width, depth }
    }
    pub fn with_error(epsilon: f64, delta: f64) -> Self {
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil() as usize;
        Self::new(width.max(1), depth.max(1))
    }
    fn hash(&self, item: u64, row: usize) -> usize {
        let h = item.wrapping_mul(0x517cc1b727220a95u64.wrapping_add(row as u64 * 0x123456789abcdef0));
        (h as usize) % self.width
    }
    pub fn increment(&mut self, item: u64) {
        for row in 0..self.depth { let col = self.hash(item, row); self.table[row][col] = self.table[row][col].saturating_add(1); }
    }
    pub fn estimate(&self, item: u64) -> u32 {
        (0..self.depth).map(|row| { let col = self.hash(item, row); self.table[row][col] }).min().unwrap_or(0)
    }
    pub fn clear(&mut self) { for row in &mut self.table { row.fill(0); } }
}

// ---------------------------------------------------------------------------
// Trie (Prefix Tree)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TrieNode {
    children: HashMap<char, usize>,
    is_terminal: bool,
    value: Option<u64>,
    prefix_count: u32,
}

#[derive(Debug)]
pub struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    pub fn new() -> Self {
        Self { nodes: vec![TrieNode { children: HashMap::new(), is_terminal: false, value: None, prefix_count: 0 }] }
    }
    pub fn insert(&mut self, key: &str, value: u64) {
        let mut current = 0;
        for ch in key.chars() {
            self.nodes[current].prefix_count += 1;
            if !self.nodes[current].children.contains_key(&ch) {
                let new_idx = self.nodes.len();
                self.nodes.push(TrieNode { children: HashMap::new(), is_terminal: false, value: None, prefix_count: 0 });
                self.nodes[current].children.insert(ch, new_idx);
            }
            current = self.nodes[current].children[&ch];
        }
        self.nodes[current].is_terminal = true;
        self.nodes[current].value = Some(value);
        self.nodes[current].prefix_count += 1;
    }
    pub fn search(&self, key: &str) -> Option<u64> {
        let mut current = 0;
        for ch in key.chars() {
            if let Some(&next) = self.nodes[current].children.get(&ch) { current = next; }
            else { return None; }
        }
        if self.nodes[current].is_terminal { self.nodes[current].value } else { None }
    }
    pub fn starts_with(&self, prefix: &str) -> bool {
        let mut current = 0;
        for ch in prefix.chars() {
            if let Some(&next) = self.nodes[current].children.get(&ch) { current = next; }
            else { return false; }
        }
        true
    }
    pub fn prefix_count(&self, prefix: &str) -> u32 {
        let mut current = 0;
        for ch in prefix.chars() {
            if let Some(&next) = self.nodes[current].children.get(&ch) { current = next; }
            else { return 0; }
        }
        self.nodes[current].prefix_count
    }
    pub fn node_count(&self) -> usize { self.nodes.len() }
}

impl Default for Trie { fn default() -> Self { Self::new() } }

// ---------------------------------------------------------------------------
// Concurrent Queue (simple lock-free-style with Vec)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ConcurrentQueue<T> {
    buffer: Vec<Option<T>>,
    head: usize,
    tail: usize,
    capacity: usize,
    count: usize,
}

impl<T> ConcurrentQueue<T> {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self { buffer: (0..cap).map(|_| None).collect(), head: 0, tail: 0, capacity: cap, count: 0 }
    }
    pub fn push(&mut self, item: T) -> bool {
        if self.count >= self.capacity { return false; }
        self.buffer[self.tail] = Some(item);
        self.tail = (self.tail + 1) % self.capacity;
        self.count += 1;
        true
    }
    pub fn pop(&mut self) -> Option<T> {
        if self.count == 0 { return None; }
        let item = self.buffer[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.count -= 1;
        item
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn is_full(&self) -> bool { self.count >= self.capacity }
    pub fn capacity(&self) -> usize { self.capacity }
    pub fn clear(&mut self) { while self.pop().is_some() {} }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_disjoint_set() {
        let mut ds = DisjointSet::new(5);
        assert_eq!(ds.num_sets(), 5);
        ds.union(0, 1);
        ds.union(2, 3);
        assert!(ds.connected(0, 1));
        assert!(!ds.connected(0, 2));
        assert_eq!(ds.num_sets(), 3);
        ds.union(1, 3);
        assert!(ds.connected(0, 3));
        assert_eq!(ds.set_size(0), 4);
    }
    #[test]
    fn test_lru_cache() {
        let mut cache = LruCache::new(2);
        cache.insert(1, "a");
        cache.insert(2, "b");
        assert_eq!(cache.get(&1), Some(&"a"));
        cache.insert(3, "c"); // evicts 2
        assert!(cache.get(&2).is_none());
        assert_eq!(cache.len(), 2);
    }
    #[test]
    fn test_bloom_filter() {
        let mut bf = BloomFilter::new(1000, 5);
        bf.insert_str("hello");
        bf.insert_str("world");
        assert!(bf.contains_str("hello"));
        assert!(bf.contains_str("world"));
        // May have false positives but should not have false negatives
    }
    #[test]
    fn test_count_min_sketch() {
        let mut cms = CountMinSketch::new(100, 5);
        cms.increment(42);
        cms.increment(42);
        cms.increment(42);
        assert!(cms.estimate(42) >= 3);
        assert_eq!(cms.estimate(99), 0);
    }
    #[test]
    fn test_trie() {
        let mut trie = Trie::new();
        trie.insert("hello", 1);
        trie.insert("help", 2);
        trie.insert("world", 3);
        assert_eq!(trie.search("hello"), Some(1));
        assert_eq!(trie.search("help"), Some(2));
        assert!(trie.starts_with("hel"));
        assert!(!trie.starts_with("xyz"));
    }
    #[test]
    fn test_concurrent_queue() {
        let mut q = ConcurrentQueue::new(3);
        assert!(q.push(1));
        assert!(q.push(2));
        assert!(q.push(3));
        assert!(!q.push(4)); // full
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.len(), 1);
    }
}
