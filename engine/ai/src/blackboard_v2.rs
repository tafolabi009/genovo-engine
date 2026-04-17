// engine/ai/src/blackboard_v2.rs
//
// Enhanced blackboard system for game AI. Features:
//   - Typed keys with runtime type checking
//   - Blackboard sharing between AI agents
//   - Blackboard observers (notify on change)
//   - Blackboard persistence (serialization)
//   - Time-stamped entries with automatic expiry
//   - Hierarchical blackboard (parent/child)

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Blackboard value types
// ---------------------------------------------------------------------------

/// A strongly-typed blackboard value.
#[derive(Debug, Clone, PartialEq)]
pub enum BlackboardValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Vec3([f32; 3]),
    EntityId(u64),
    /// Arbitrary tagged data.
    Custom { type_name: String, data: Vec<u8> },
}

impl BlackboardValue {
    pub fn type_name(&self) -> &str {
        match self {
            Self::Bool(_) => "Bool",
            Self::Int(_) => "Int",
            Self::Float(_) => "Float",
            Self::String(_) => "String",
            Self::Vec3(_) => "Vec3",
            Self::EntityId(_) => "EntityId",
            Self::Custom { type_name, .. } => type_name,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self { Self::Bool(v) => Some(*v), _ => None }
    }
    pub fn as_int(&self) -> Option<i64> {
        match self { Self::Int(v) => Some(*v), _ => None }
    }
    pub fn as_float(&self) -> Option<f64> {
        match self { Self::Float(v) => Some(*v), _ => None }
    }
    pub fn as_string(&self) -> Option<&str> {
        match self { Self::String(v) => Some(v), _ => None }
    }
    pub fn as_vec3(&self) -> Option<[f32; 3]> {
        match self { Self::Vec3(v) => Some(*v), _ => None }
    }
    pub fn as_entity_id(&self) -> Option<u64> {
        match self { Self::EntityId(v) => Some(*v), _ => None }
    }

    /// Convert to f64 if numeric.
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Self::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }
}

impl std::fmt::Display for BlackboardValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{v}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v:.3}"),
            Self::String(v) => write!(f, "\"{v}\""),
            Self::Vec3(v) => write!(f, "({:.2}, {:.2}, {:.2})", v[0], v[1], v[2]),
            Self::EntityId(v) => write!(f, "entity({v})"),
            Self::Custom { type_name, data } => write!(f, "custom<{type_name}>({} bytes)", data.len()),
        }
    }
}

// ---------------------------------------------------------------------------
// Timestamped entry
// ---------------------------------------------------------------------------

/// A blackboard entry with metadata.
#[derive(Debug, Clone)]
pub struct BlackboardEntry {
    pub value: BlackboardValue,
    /// Time when this entry was last set (seconds since epoch or game time).
    pub timestamp: f64,
    /// Optional expiry time (seconds). If `None`, the entry never expires.
    pub ttl: Option<f64>,
    /// The source that set this value (e.g. agent ID).
    pub source: u64,
    /// Number of times this entry has been written.
    pub write_count: u32,
}

impl BlackboardEntry {
    pub fn is_expired(&self, current_time: f64) -> bool {
        if let Some(ttl) = self.ttl {
            current_time - self.timestamp > ttl
        } else {
            false
        }
    }

    pub fn age(&self, current_time: f64) -> f64 {
        current_time - self.timestamp
    }
}

// ---------------------------------------------------------------------------
// Observer
// ---------------------------------------------------------------------------

/// An observer ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObserverId(pub u64);

/// What kind of change to observe.
#[derive(Debug, Clone, PartialEq)]
pub enum ObserverFilter {
    /// Observe any change to a specific key.
    Key(String),
    /// Observe any key that starts with a prefix.
    Prefix(String),
    /// Observe all changes.
    All,
}

/// A change notification.
#[derive(Debug, Clone)]
pub struct ChangeNotification {
    pub key: String,
    pub old_value: Option<BlackboardValue>,
    pub new_value: BlackboardValue,
    pub timestamp: f64,
    pub source: u64,
}

/// An observer registration.
#[derive(Debug, Clone)]
struct ObserverRegistration {
    id: ObserverId,
    filter: ObserverFilter,
}

// ---------------------------------------------------------------------------
// Blackboard
// ---------------------------------------------------------------------------

/// A typed blackboard for AI agent state.
pub struct Blackboard {
    entries: HashMap<String, BlackboardEntry>,
    observers: Vec<ObserverRegistration>,
    pending_notifications: Vec<(ObserverId, ChangeNotification)>,
    next_observer_id: u64,
    current_time: f64,
    owner_id: u64,
    parent: Option<Box<Blackboard>>,
    name: String,
}

impl Blackboard {
    pub fn new(name: &str, owner_id: u64) -> Self {
        Self {
            entries: HashMap::new(),
            observers: Vec::new(),
            pending_notifications: Vec::new(),
            next_observer_id: 1,
            current_time: 0.0,
            owner_id,
            parent: None,
            name: name.to_string(),
        }
    }

    pub fn with_parent(mut self, parent: Blackboard) -> Self {
        self.parent = Some(Box::new(parent));
        self
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn owner_id(&self) -> u64 { self.owner_id }
    pub fn entry_count(&self) -> usize { self.entries.len() }

    /// Set the current time for expiry checks.
    pub fn set_time(&mut self, time: f64) {
        self.current_time = time;
    }

    // -----------------------------------------------------------------------
    // Typed setters
    // -----------------------------------------------------------------------

    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.set(key, BlackboardValue::Bool(value), self.owner_id, None);
    }

    pub fn set_int(&mut self, key: &str, value: i64) {
        self.set(key, BlackboardValue::Int(value), self.owner_id, None);
    }

    pub fn set_float(&mut self, key: &str, value: f64) {
        self.set(key, BlackboardValue::Float(value), self.owner_id, None);
    }

    pub fn set_string(&mut self, key: &str, value: &str) {
        self.set(key, BlackboardValue::String(value.to_string()), self.owner_id, None);
    }

    pub fn set_vec3(&mut self, key: &str, value: [f32; 3]) {
        self.set(key, BlackboardValue::Vec3(value), self.owner_id, None);
    }

    pub fn set_entity(&mut self, key: &str, entity_id: u64) {
        self.set(key, BlackboardValue::EntityId(entity_id), self.owner_id, None);
    }

    /// Set with source and TTL.
    pub fn set_with_ttl(&mut self, key: &str, value: BlackboardValue, source: u64, ttl: f64) {
        self.set(key, value, source, Some(ttl));
    }

    /// Core set operation.
    pub fn set(&mut self, key: &str, value: BlackboardValue, source: u64, ttl: Option<f64>) {
        let old = self.entries.get(key).map(|e| e.value.clone());

        let entry = self.entries.entry(key.to_string()).or_insert_with(|| BlackboardEntry {
            value: value.clone(),
            timestamp: self.current_time,
            ttl,
            source,
            write_count: 0,
        });
        entry.value = value.clone();
        entry.timestamp = self.current_time;
        entry.ttl = ttl;
        entry.source = source;
        entry.write_count += 1;

        // Notify observers.
        self.notify_observers(key, old, value);
    }

    // -----------------------------------------------------------------------
    // Typed getters
    // -----------------------------------------------------------------------

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.value.as_bool()
    }

    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.get(key)?.value.as_int()
    }

    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.get(key)?.value.as_float()
    }

    pub fn get_string(&self, key: &str) -> Option<String> {
        self.get(key)?.value.as_string().map(|s| s.to_string())
    }

    pub fn get_vec3(&self, key: &str) -> Option<[f32; 3]> {
        self.get(key)?.value.as_vec3()
    }

    pub fn get_entity(&self, key: &str) -> Option<u64> {
        self.get(key)?.value.as_entity_id()
    }

    /// Get with default value.
    pub fn get_bool_or(&self, key: &str, default: bool) -> bool {
        self.get_bool(key).unwrap_or(default)
    }

    pub fn get_float_or(&self, key: &str, default: f64) -> f64 {
        self.get_float(key).unwrap_or(default)
    }

    pub fn get_int_or(&self, key: &str, default: i64) -> i64 {
        self.get_int(key).unwrap_or(default)
    }

    /// Get an entry by key, checking expiry and parent.
    pub fn get(&self, key: &str) -> Option<&BlackboardEntry> {
        if let Some(entry) = self.entries.get(key) {
            if !entry.is_expired(self.current_time) {
                return Some(entry);
            }
        }
        // Check parent blackboard.
        if let Some(ref parent) = self.parent {
            return parent.get(key);
        }
        None
    }

    /// Check if a key exists (and is not expired).
    pub fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// Remove a key.
    pub fn remove(&mut self, key: &str) -> Option<BlackboardEntry> {
        self.entries.remove(key)
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Remove expired entries.
    pub fn cleanup_expired(&mut self) -> usize {
        let time = self.current_time;
        let before = self.entries.len();
        self.entries.retain(|_, entry| !entry.is_expired(time));
        before - self.entries.len()
    }

    /// Get all keys.
    pub fn keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Get all keys matching a prefix.
    pub fn keys_with_prefix(&self, prefix: &str) -> Vec<String> {
        self.entries.keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect()
    }

    /// Get all key-value pairs as a snapshot.
    pub fn snapshot(&self) -> Vec<(String, BlackboardValue)> {
        self.entries.iter()
            .filter(|(_, e)| !e.is_expired(self.current_time))
            .map(|(k, e)| (k.clone(), e.value.clone()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Observers
    // -----------------------------------------------------------------------

    /// Register an observer. Returns an ID for unregistration.
    pub fn add_observer(&mut self, filter: ObserverFilter) -> ObserverId {
        let id = ObserverId(self.next_observer_id);
        self.next_observer_id += 1;
        self.observers.push(ObserverRegistration { id, filter });
        id
    }

    /// Remove an observer.
    pub fn remove_observer(&mut self, id: ObserverId) {
        self.observers.retain(|o| o.id != id);
    }

    /// Drain pending notifications.
    pub fn drain_notifications(&mut self) -> Vec<(ObserverId, ChangeNotification)> {
        std::mem::take(&mut self.pending_notifications)
    }

    fn notify_observers(&mut self, key: &str, old: Option<BlackboardValue>, new: BlackboardValue) {
        let notification = ChangeNotification {
            key: key.to_string(),
            old_value: old,
            new_value: new,
            timestamp: self.current_time,
            source: self.owner_id,
        };

        for obs in &self.observers {
            let matches = match &obs.filter {
                ObserverFilter::Key(k) => k == key,
                ObserverFilter::Prefix(p) => key.starts_with(p),
                ObserverFilter::All => true,
            };
            if matches {
                self.pending_notifications.push((obs.id, notification.clone()));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Serialize the blackboard to a portable format.
    pub fn serialize(&self) -> SerializedBlackboard {
        let entries = self.entries.iter()
            .filter(|(_, e)| !e.is_expired(self.current_time))
            .map(|(k, e)| {
                SerializedEntry {
                    key: k.clone(),
                    value: e.value.clone(),
                    timestamp: e.timestamp,
                    ttl: e.ttl,
                    source: e.source,
                }
            })
            .collect();
        SerializedBlackboard {
            name: self.name.clone(),
            owner_id: self.owner_id,
            entries,
        }
    }

    /// Deserialize from a portable format, replacing current contents.
    pub fn deserialize(&mut self, data: &SerializedBlackboard) {
        self.entries.clear();
        for entry in &data.entries {
            self.entries.insert(entry.key.clone(), BlackboardEntry {
                value: entry.value.clone(),
                timestamp: entry.timestamp,
                ttl: entry.ttl,
                source: entry.source,
                write_count: 1,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Merge / compare
    // -----------------------------------------------------------------------

    /// Merge another blackboard into this one (other's values overwrite on conflict).
    pub fn merge_from(&mut self, other: &Blackboard) {
        for (key, entry) in &other.entries {
            if entry.is_expired(other.current_time) {
                continue;
            }
            self.set(key, entry.value.clone(), entry.source, entry.ttl);
        }
    }

    /// Compare two blackboards, returning keys that differ.
    pub fn diff(&self, other: &Blackboard) -> Vec<String> {
        let mut diffs = Vec::new();
        for key in self.entries.keys() {
            match (self.get(key), other.get(key)) {
                (Some(a), Some(b)) => {
                    if a.value != b.value {
                        diffs.push(key.clone());
                    }
                }
                (Some(_), None) | (None, Some(_)) => {
                    diffs.push(key.clone());
                }
                _ => {}
            }
        }
        for key in other.entries.keys() {
            if !self.entries.contains_key(key) {
                diffs.push(key.clone());
            }
        }
        diffs.sort();
        diffs.dedup();
        diffs
    }
}

impl std::fmt::Debug for Blackboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Blackboard")
            .field("name", &self.name)
            .field("owner_id", &self.owner_id)
            .field("entries", &self.entries.len())
            .field("observers", &self.observers.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Shared blackboard
// ---------------------------------------------------------------------------

/// A shared blackboard that multiple agents can read/write.
pub struct SharedBlackboard {
    inner: Blackboard,
    reader_ids: Vec<u64>,
    writer_ids: Vec<u64>,
}

impl SharedBlackboard {
    pub fn new(name: &str) -> Self {
        Self {
            inner: Blackboard::new(name, 0),
            reader_ids: Vec::new(),
            writer_ids: Vec::new(),
        }
    }

    pub fn add_reader(&mut self, agent_id: u64) {
        if !self.reader_ids.contains(&agent_id) {
            self.reader_ids.push(agent_id);
        }
    }

    pub fn add_writer(&mut self, agent_id: u64) {
        if !self.writer_ids.contains(&agent_id) {
            self.writer_ids.push(agent_id);
        }
    }

    pub fn remove_reader(&mut self, agent_id: u64) {
        self.reader_ids.retain(|&id| id != agent_id);
    }

    pub fn remove_writer(&mut self, agent_id: u64) {
        self.writer_ids.retain(|&id| id != agent_id);
    }

    pub fn can_write(&self, agent_id: u64) -> bool {
        self.writer_ids.contains(&agent_id)
    }

    pub fn set(&mut self, key: &str, value: BlackboardValue, writer_id: u64) -> bool {
        if !self.can_write(writer_id) {
            return false;
        }
        self.inner.set(key, value, writer_id, None);
        true
    }

    pub fn get(&self, key: &str) -> Option<&BlackboardEntry> {
        self.inner.get(key)
    }

    pub fn blackboard(&self) -> &Blackboard { &self.inner }
    pub fn blackboard_mut(&mut self) -> &mut Blackboard { &mut self.inner }
}

// ---------------------------------------------------------------------------
// Serialization types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SerializedEntry {
    pub key: String,
    pub value: BlackboardValue,
    pub timestamp: f64,
    pub ttl: Option<f64>,
    pub source: u64,
}

#[derive(Debug, Clone)]
pub struct SerializedBlackboard {
    pub name: String,
    pub owner_id: u64,
    pub entries: Vec<SerializedEntry>,
}

// ---------------------------------------------------------------------------
// Blackboard manager (manages multiple blackboards)
// ---------------------------------------------------------------------------

/// Manages multiple blackboards and shared blackboards.
pub struct BlackboardManager {
    blackboards: HashMap<u64, Blackboard>,
    shared: HashMap<String, SharedBlackboard>,
    current_time: f64,
}

impl BlackboardManager {
    pub fn new() -> Self {
        Self {
            blackboards: HashMap::new(),
            shared: HashMap::new(),
            current_time: 0.0,
        }
    }

    pub fn set_time(&mut self, time: f64) {
        self.current_time = time;
        for bb in self.blackboards.values_mut() {
            bb.set_time(time);
        }
        for sbb in self.shared.values_mut() {
            sbb.inner.set_time(time);
        }
    }

    /// Create a blackboard for an agent.
    pub fn create_blackboard(&mut self, agent_id: u64, name: &str) {
        let mut bb = Blackboard::new(name, agent_id);
        bb.set_time(self.current_time);
        self.blackboards.insert(agent_id, bb);
    }

    /// Get an agent's blackboard.
    pub fn get(&self, agent_id: u64) -> Option<&Blackboard> {
        self.blackboards.get(&agent_id)
    }

    pub fn get_mut(&mut self, agent_id: u64) -> Option<&mut Blackboard> {
        self.blackboards.get_mut(&agent_id)
    }

    /// Remove an agent's blackboard.
    pub fn remove_blackboard(&mut self, agent_id: u64) {
        self.blackboards.remove(&agent_id);
    }

    /// Create a shared blackboard.
    pub fn create_shared(&mut self, name: &str) {
        self.shared.insert(name.to_string(), SharedBlackboard::new(name));
    }

    /// Get a shared blackboard.
    pub fn get_shared(&self, name: &str) -> Option<&SharedBlackboard> {
        self.shared.get(name)
    }

    pub fn get_shared_mut(&mut self, name: &str) -> Option<&mut SharedBlackboard> {
        self.shared.get_mut(name)
    }

    /// Cleanup expired entries in all blackboards.
    pub fn cleanup_all(&mut self) -> usize {
        let mut total = 0;
        for bb in self.blackboards.values_mut() {
            total += bb.cleanup_expired();
        }
        for sbb in self.shared.values_mut() {
            total += sbb.inner.cleanup_expired();
        }
        total
    }
}

impl Default for BlackboardManager {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_set_get() {
        let mut bb = Blackboard::new("test", 1);
        bb.set_bool("alive", true);
        bb.set_float("health", 100.0);
        bb.set_string("name", "Agent 47");
        bb.set_vec3("position", [1.0, 2.0, 3.0]);

        assert_eq!(bb.get_bool("alive"), Some(true));
        assert_eq!(bb.get_float("health"), Some(100.0));
        assert_eq!(bb.get_string("name"), Some("Agent 47".to_string()));
        assert_eq!(bb.get_vec3("position"), Some([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_expiry() {
        let mut bb = Blackboard::new("test", 1);
        bb.set_time(0.0);
        bb.set_with_ttl("temp", BlackboardValue::Bool(true), 1, 5.0);
        assert!(bb.has("temp"));

        bb.set_time(6.0);
        assert!(!bb.has("temp"));
    }

    #[test]
    fn test_observer() {
        let mut bb = Blackboard::new("test", 1);
        let obs_id = bb.add_observer(ObserverFilter::Key("health".to_string()));
        bb.set_float("health", 50.0);

        let notifs = bb.drain_notifications();
        assert_eq!(notifs.len(), 1);
        assert_eq!(notifs[0].0, obs_id);
        assert_eq!(notifs[0].1.key, "health");
    }

    #[test]
    fn test_prefix_observer() {
        let mut bb = Blackboard::new("test", 1);
        let _obs = bb.add_observer(ObserverFilter::Prefix("enemy.".to_string()));
        bb.set_int("enemy.count", 5);
        bb.set_int("friend.count", 3);

        let notifs = bb.drain_notifications();
        assert_eq!(notifs.len(), 1);
        assert_eq!(notifs[0].1.key, "enemy.count");
    }

    #[test]
    fn test_serialization() {
        let mut bb = Blackboard::new("test", 1);
        bb.set_float("x", 42.0);
        bb.set_bool("flag", true);

        let serialized = bb.serialize();
        let mut bb2 = Blackboard::new("test2", 2);
        bb2.deserialize(&serialized);

        assert_eq!(bb2.get_float("x"), Some(42.0));
        assert_eq!(bb2.get_bool("flag"), Some(true));
    }

    #[test]
    fn test_shared_blackboard() {
        let mut shared = SharedBlackboard::new("squad");
        shared.add_writer(1);
        shared.add_reader(2);

        assert!(shared.set("target", BlackboardValue::EntityId(99), 1));
        assert!(!shared.set("target", BlackboardValue::EntityId(100), 2)); // not a writer

        assert_eq!(shared.get("target").unwrap().value.as_entity_id(), Some(99));
    }

    #[test]
    fn test_merge() {
        let mut bb1 = Blackboard::new("a", 1);
        bb1.set_int("x", 1);
        bb1.set_int("y", 2);

        let mut bb2 = Blackboard::new("b", 2);
        bb2.set_int("x", 10);
        bb2.set_int("z", 3);

        bb1.merge_from(&bb2);
        assert_eq!(bb1.get_int("x"), Some(10)); // overwritten
        assert_eq!(bb1.get_int("y"), Some(2));  // kept
        assert_eq!(bb1.get_int("z"), Some(3));  // added
    }

    #[test]
    fn test_diff() {
        let mut bb1 = Blackboard::new("a", 1);
        bb1.set_int("x", 1);
        bb1.set_int("y", 2);

        let mut bb2 = Blackboard::new("b", 2);
        bb2.set_int("x", 1);
        bb2.set_int("y", 99);
        bb2.set_int("z", 3);

        let diffs = bb1.diff(&bb2);
        assert!(diffs.contains(&"y".to_string()));
        assert!(diffs.contains(&"z".to_string()));
        assert!(!diffs.contains(&"x".to_string()));
    }

    #[test]
    fn test_parent_blackboard() {
        let mut parent = Blackboard::new("parent", 0);
        parent.set_int("global_difficulty", 3);

        let child = Blackboard::new("child", 1).with_parent(parent);
        // Child should find parent's value.
        assert_eq!(child.get_int("global_difficulty"), Some(3));
    }
}
