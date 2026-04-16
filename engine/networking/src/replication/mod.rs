//! Network state replication.
//!
//! Manages synchronization of ECS component state across the network.
//! Components implementing [`Replicated`] are automatically tracked by the
//! [`ReplicationManager`], which handles serialization, delta compression,
//! and bandwidth allocation.
//!
//! ## Wire format for state updates
//!
//! A replication frame is a sequence of entity updates:
//!
//! ```text
//! [u32 tick][u16 entity_count]
//!   for each entity:
//!     [u64 network_id][u8 flags][u16 data_len][...data...]
//! ```
//!
//! Flags:
//!   bit 0: full state (vs delta)
//!   bit 1: entity spawn (new entity)
//!   bit 2: entity despawn
//!   bit 3: authority change

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use genovo_core::EngineResult;
use genovo_ecs::{Component, Entity};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum replication payload per frame (bytes).
const MAX_REPLICATION_FRAME_SIZE: usize = 4096;

/// Flag: this update contains the full component state (not a delta).
const FLAG_FULL_STATE: u8 = 1 << 0;
/// Flag: this entity is being spawned on the remote side.
const FLAG_ENTITY_SPAWN: u8 = 1 << 1;
/// Flag: this entity is being despawned on the remote side.
const FLAG_ENTITY_DESPAWN: u8 = 1 << 2;
/// Flag: the authority for this entity has changed.
const FLAG_AUTHORITY_CHANGE: u8 = 1 << 3;

// ---------------------------------------------------------------------------
// NetworkId
// ---------------------------------------------------------------------------

/// A network-wide unique identifier for a replicated entity.
///
/// Unlike [`Entity`], which is only valid within a single [`World`](genovo_ecs::World),
/// a `NetworkId` is consistent across all connected peers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NetworkId(pub u64);

impl NetworkId {
    /// Creates a new network ID from a raw value.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw numeric value.
    pub const fn raw(&self) -> u64 {
        self.0
    }

    /// Encode as 8 bytes (big-endian).
    pub fn encode(&self) -> [u8; 8] {
        self.0.to_be_bytes()
    }

    /// Decode from 8 bytes (big-endian).
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }
        let arr: [u8; 8] = bytes[..8].try_into().ok()?;
        Some(Self(u64::from_be_bytes(arr)))
    }
}

impl Component for NetworkId {}

// ---------------------------------------------------------------------------
// Authority
// ---------------------------------------------------------------------------

/// Determines which peer has authoritative control over a replicated entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Authority {
    /// The server is authoritative; clients receive state updates.
    Server,
    /// The owning client is authoritative; server relays to other clients.
    Client {
        /// The peer ID of the owning client.
        owner: u32,
    },
    /// Authority is shared (e.g., physics objects with cooperative simulation).
    Shared,
}

impl Authority {
    /// Encode the authority to bytes.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Authority::Server => vec![0],
            Authority::Client { owner } => {
                let mut buf = vec![1];
                buf.extend_from_slice(&owner.to_be_bytes());
                buf
            }
            Authority::Shared => vec![2],
        }
    }

    /// Decode authority from bytes. Returns the authority and bytes consumed.
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        if data.is_empty() {
            return None;
        }
        match data[0] {
            0 => Some((Authority::Server, 1)),
            1 => {
                if data.len() < 5 {
                    return None;
                }
                let owner =
                    u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
                Some((Authority::Client { owner }, 5))
            }
            2 => Some((Authority::Shared, 1)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ReplicationPolicy
// ---------------------------------------------------------------------------

/// Controls how a component is replicated over the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReplicationPolicy {
    /// State is sent reliably and in order (guaranteed delivery).
    Reliable,
    /// State is sent unreliably (latest-wins, may be dropped).
    Unreliable,
    /// State is only sent when the component value changes (delta updates).
    OnChange,
}

// ---------------------------------------------------------------------------
// ReplicationPriority
// ---------------------------------------------------------------------------

/// Priority level for replication bandwidth allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReplicationPriority {
    /// Low priority: ambient world state, distant objects.
    Low = 0,
    /// Medium priority: nearby NPCs, projectiles.
    Medium = 1,
    /// High priority: player state, critical game events.
    High = 2,
}

// ---------------------------------------------------------------------------
// ReplicationChannel
// ---------------------------------------------------------------------------

/// A named replication channel with priority and bandwidth settings.
///
/// Channels allow grouping replicated data by importance. For example, player
/// positions might use a high-priority, high-bandwidth channel, while ambient
/// NPC state uses a lower-priority channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationChannel {
    /// Human-readable channel name (e.g., `"player_state"`, `"world_updates"`).
    pub name: String,
    /// Channel priority (higher = more bandwidth allocation).
    pub priority: u8,
    /// Maximum bytes per second allocated to this channel.
    pub max_bandwidth_bps: u32,
    /// Default replication policy for data on this channel.
    pub default_policy: ReplicationPolicy,
    /// How many bytes have been sent this frame.
    #[serde(skip)]
    bytes_this_frame: u32,
    /// Timestamp of the last frame reset.
    #[serde(skip)]
    last_reset_tick: u64,
}

impl ReplicationChannel {
    /// Creates a new replication channel with the given settings.
    pub fn new(
        name: impl Into<String>,
        priority: u8,
        max_bandwidth_bps: u32,
        default_policy: ReplicationPolicy,
    ) -> Self {
        Self {
            name: name.into(),
            priority,
            max_bandwidth_bps,
            default_policy,
            bytes_this_frame: 0,
            last_reset_tick: 0,
        }
    }

    /// Returns the bytes remaining in this frame's bandwidth budget.
    /// Assumes a tick rate of 60 Hz for budget calculation.
    pub fn bytes_remaining(&self) -> u32 {
        let budget_per_frame = self.max_bandwidth_bps / 60; // 60 fps
        budget_per_frame.saturating_sub(self.bytes_this_frame)
    }

    /// Consume bytes from this frame's budget. Returns `true` if within budget.
    pub fn consume(&mut self, bytes: u32) -> bool {
        let budget_per_frame = self.max_bandwidth_bps / 60;
        if self.bytes_this_frame + bytes <= budget_per_frame {
            self.bytes_this_frame += bytes;
            true
        } else {
            false
        }
    }

    /// Reset per-frame counters for a new tick.
    pub fn reset_frame(&mut self, tick: u64) {
        if tick != self.last_reset_tick {
            self.bytes_this_frame = 0;
            self.last_reset_tick = tick;
        }
    }
}

// ---------------------------------------------------------------------------
// Replicated trait
// ---------------------------------------------------------------------------

/// Marker trait for components that should be replicated across the network.
///
/// Types implementing this trait must also implement [`Serialize`] and
/// [`Deserialize`] so they can be transmitted over the wire.
pub trait Replicated: Component + Serialize + for<'de> Deserialize<'de> {
    /// Returns the replication policy for this component type.
    fn replication_policy() -> ReplicationPolicy;

    /// Returns the channel name this component should be replicated on.
    fn channel_name() -> &'static str {
        "default"
    }

    /// Returns the replication priority for this component type.
    fn replication_priority() -> ReplicationPriority {
        ReplicationPriority::Medium
    }

    /// Serialize this component to bytes.
    fn serialize_component(&self) -> Vec<u8>
    where
        Self: Sized,
    {
        // Use a simple length-prefixed binary format.
        let json = serde_json::to_vec(self).unwrap_or_default();
        let mut buf = Vec::with_capacity(4 + json.len());
        buf.extend_from_slice(&(json.len() as u32).to_be_bytes());
        buf.extend_from_slice(&json);
        buf
    }

    /// Deserialize a component from bytes.
    fn deserialize_component(data: &[u8]) -> Option<Self>
    where
        Self: Sized,
    {
        if data.len() < 4 {
            return None;
        }
        let len =
            u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + len {
            return None;
        }
        serde_json::from_slice(&data[4..4 + len]).ok()
    }

    /// Computes a delta between `old` and `new` state for bandwidth optimization.
    ///
    /// Returns `None` if the states are identical (no update needed).
    fn compute_delta(old: &Self, new: &Self) -> Option<Vec<u8>>
    where
        Self: Sized,
    {
        let old_bytes = old.serialize_component();
        let new_bytes = new.serialize_component();
        if old_bytes == new_bytes {
            None // no change
        } else {
            // Simple delta: send the full new state with a diff marker.
            // A real implementation could use a binary diff algorithm.
            Some(new_bytes)
        }
    }

    /// Applies a delta to the current state.
    fn apply_delta(&mut self, delta: &[u8]) -> EngineResult<()>
    where
        Self: Sized,
    {
        if let Some(new_val) = Self::deserialize_component(delta) {
            *self = new_val;
            Ok(())
        } else {
            Err(genovo_core::EngineError::InvalidArgument(
                "failed to apply delta".into(),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// ComponentSnapshot
// ---------------------------------------------------------------------------

/// Serialized snapshot of a single component's state.
#[derive(Debug, Clone)]
pub struct ComponentSnapshot {
    /// Type identifier (string name for cross-platform stability).
    pub type_name: String,
    /// Serialized component data.
    pub data: Vec<u8>,
    /// CRC32 checksum for fast comparison.
    pub checksum: u32,
}

impl ComponentSnapshot {
    /// Create a new component snapshot.
    pub fn new(type_name: impl Into<String>, data: Vec<u8>) -> Self {
        let checksum = Self::compute_crc32(&data);
        Self {
            type_name: type_name.into(),
            data,
            checksum,
        }
    }

    /// Simple CRC32 computation (non-standard but deterministic).
    fn compute_crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFFFFFF;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
        }
        !crc
    }

    /// Returns true if two snapshots are identical (by checksum).
    pub fn matches(&self, other: &ComponentSnapshot) -> bool {
        self.type_name == other.type_name && self.checksum == other.checksum
    }

    /// Encode the snapshot to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let name_bytes = self.type_name.as_bytes();
        let mut buf = Vec::with_capacity(2 + name_bytes.len() + 4 + 2 + self.data.len());
        // type_name length (u16) + type_name + checksum (u32) + data length (u16) + data
        buf.extend_from_slice(&(name_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&self.checksum.to_be_bytes());
        buf.extend_from_slice(&(self.data.len() as u16).to_be_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Decode a snapshot from bytes. Returns the snapshot and bytes consumed.
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 2 {
            return None;
        }
        let name_len = u16::from_be_bytes([data[0], data[1]]) as usize;
        let offset = 2 + name_len;
        if data.len() < offset + 6 {
            return None;
        }
        let type_name =
            String::from_utf8(data[2..2 + name_len].to_vec()).ok()?;
        let checksum = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let data_len = u16::from_be_bytes([data[offset + 4], data[offset + 5]]) as usize;
        let total = offset + 6 + data_len;
        if data.len() < total {
            return None;
        }
        let snap_data = data[offset + 6..total].to_vec();
        Some((
            Self {
                type_name,
                data: snap_data,
                checksum,
            },
            total,
        ))
    }
}

// ---------------------------------------------------------------------------
// EntityUpdate
// ---------------------------------------------------------------------------

/// A replication update for a single entity.
#[derive(Debug, Clone)]
pub struct EntityUpdate {
    /// Network ID of the entity.
    pub network_id: NetworkId,
    /// Flags describing the type of update.
    pub flags: u8,
    /// Authority (included if FLAG_AUTHORITY_CHANGE or FLAG_ENTITY_SPAWN is set).
    pub authority: Option<Authority>,
    /// Component snapshots.
    pub components: Vec<ComponentSnapshot>,
}

impl EntityUpdate {
    /// Create a state update for an existing entity.
    pub fn state(network_id: NetworkId, components: Vec<ComponentSnapshot>) -> Self {
        Self {
            network_id,
            flags: FLAG_FULL_STATE,
            authority: None,
            components,
        }
    }

    /// Create a spawn update for a new entity.
    pub fn spawn(
        network_id: NetworkId,
        authority: Authority,
        components: Vec<ComponentSnapshot>,
    ) -> Self {
        Self {
            network_id,
            flags: FLAG_ENTITY_SPAWN | FLAG_FULL_STATE,
            authority: Some(authority),
            components,
        }
    }

    /// Create a despawn update.
    pub fn despawn(network_id: NetworkId) -> Self {
        Self {
            network_id,
            flags: FLAG_ENTITY_DESPAWN,
            authority: None,
            components: Vec::new(),
        }
    }

    /// Create a delta update (only changed components).
    pub fn delta(network_id: NetworkId, components: Vec<ComponentSnapshot>) -> Self {
        Self {
            network_id,
            flags: 0, // no FLAG_FULL_STATE
            authority: None,
            components,
        }
    }

    /// Returns true if this is a full-state update.
    pub fn is_full_state(&self) -> bool {
        self.flags & FLAG_FULL_STATE != 0
    }

    /// Returns true if this is a spawn update.
    pub fn is_spawn(&self) -> bool {
        self.flags & FLAG_ENTITY_SPAWN != 0
    }

    /// Returns true if this is a despawn update.
    pub fn is_despawn(&self) -> bool {
        self.flags & FLAG_ENTITY_DESPAWN != 0
    }

    /// Encode the entity update to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // network_id (8) + flags (1)
        buf.extend_from_slice(&self.network_id.encode());
        buf.push(self.flags);

        // Authority if spawn or authority_change.
        if self.flags & (FLAG_ENTITY_SPAWN | FLAG_AUTHORITY_CHANGE) != 0 {
            if let Some(ref auth) = self.authority {
                let auth_bytes = auth.encode();
                buf.extend_from_slice(&auth_bytes);
            } else {
                buf.push(0); // default Server
            }
        }

        // Component count (u16).
        buf.extend_from_slice(&(self.components.len() as u16).to_be_bytes());

        // Components.
        for comp in &self.components {
            let comp_bytes = comp.encode();
            buf.extend_from_slice(&comp_bytes);
        }

        buf
    }

    /// Decode an entity update from bytes. Returns the update and bytes consumed.
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 9 {
            return None;
        }
        let network_id = NetworkId::decode(&data[0..8])?;
        let flags = data[8];
        let mut offset = 9;

        let authority = if flags & (FLAG_ENTITY_SPAWN | FLAG_AUTHORITY_CHANGE) != 0 {
            let (auth, consumed) = Authority::decode(&data[offset..])?;
            offset += consumed;
            Some(auth)
        } else {
            None
        };

        if data.len() < offset + 2 {
            return None;
        }
        let comp_count =
            u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        let mut components = Vec::with_capacity(comp_count);
        for _ in 0..comp_count {
            let (comp, consumed) = ComponentSnapshot::decode(&data[offset..])?;
            offset += consumed;
            components.push(comp);
        }

        Some((
            Self {
                network_id,
                flags,
                authority,
                components,
            },
            offset,
        ))
    }
}

// ---------------------------------------------------------------------------
// ReplicationFrame
// ---------------------------------------------------------------------------

/// A complete replication frame containing updates for multiple entities.
#[derive(Debug, Clone)]
pub struct ReplicationFrame {
    /// The tick number this frame represents.
    pub tick: u64,
    /// Entity updates in this frame.
    pub updates: Vec<EntityUpdate>,
}

impl ReplicationFrame {
    /// Create a new empty frame for the given tick.
    pub fn new(tick: u64) -> Self {
        Self {
            tick,
            updates: Vec::new(),
        }
    }

    /// Add an entity update to the frame.
    pub fn add_update(&mut self, update: EntityUpdate) {
        self.updates.push(update);
    }

    /// Encode the frame to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // tick (8) + entity_count (u16)
        buf.extend_from_slice(&self.tick.to_be_bytes());
        buf.extend_from_slice(&(self.updates.len() as u16).to_be_bytes());

        for update in &self.updates {
            let update_bytes = update.encode();
            let update_len = update_bytes.len() as u16;
            buf.extend_from_slice(&update_len.to_be_bytes());
            buf.extend_from_slice(&update_bytes);
        }

        buf
    }

    /// Decode a frame from bytes.
    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 10 {
            return None;
        }
        let tick = u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let entity_count =
            u16::from_be_bytes([data[8], data[9]]) as usize;
        let mut offset = 10;

        let mut updates = Vec::with_capacity(entity_count);
        for _ in 0..entity_count {
            if data.len() < offset + 2 {
                return None;
            }
            let update_len =
                u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if data.len() < offset + update_len {
                return None;
            }
            let (update, _consumed) =
                EntityUpdate::decode(&data[offset..offset + update_len])?;
            offset += update_len;
            updates.push(update);
        }

        Some(Self { tick, updates })
    }

    /// Returns the total encoded size in bytes.
    pub fn encoded_size(&self) -> usize {
        self.encode().len()
    }

    /// Returns true if the frame is empty (no updates).
    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ReplicatedEntityState
// ---------------------------------------------------------------------------

/// Internal tracking data for a single replicated entity.
#[derive(Debug)]
struct ReplicatedEntityState {
    /// The local ECS entity.
    entity: Entity,
    /// Network-wide identifier.
    network_id: NetworkId,
    /// Who has authority over this entity.
    authority: Authority,
    /// Last known serialized state per component type (for delta computation).
    last_state: HashMap<String, ComponentSnapshot>,
    /// Tick number of the last replication update.
    last_replicated_tick: u64,
    /// Whether this entity needs a full state sync (e.g., just spawned).
    needs_full_sync: bool,
    /// Whether this entity has been marked for despawn.
    pending_despawn: bool,
    /// Per-component dirty flags (type_name -> dirty).
    dirty_components: HashMap<String, bool>,
}

// ---------------------------------------------------------------------------
// ReplicationManager
// ---------------------------------------------------------------------------

/// Manages the replication of entity/component state across the network.
///
/// Tracks which entities are replicated, their authority, and handles
/// serialization/delta compression for network transmission.
pub struct ReplicationManager {
    /// Mapping from network ID to tracking state.
    entities: HashMap<NetworkId, ReplicatedEntityState>,
    /// Mapping from local entity to network ID.
    entity_to_network_id: HashMap<Entity, NetworkId>,
    /// Registered replication channels.
    channels: Vec<ReplicationChannel>,
    /// Counter for assigning network IDs.
    next_network_id: u64,
    /// Current simulation tick.
    current_tick: u64,
    /// Pending spawn notifications to send.
    pending_spawns: Vec<NetworkId>,
    /// Pending despawn notifications to send.
    pending_despawns: Vec<NetworkId>,
    /// Last acknowledged tick per peer (peer_id -> tick).
    peer_acked_ticks: HashMap<u32, u64>,
    /// State snapshots keyed by (network_id, tick) for delta reference.
    state_history: HashMap<(NetworkId, u64), Vec<ComponentSnapshot>>,
    /// Maximum state history depth per entity.
    max_history_depth: usize,
}

impl ReplicationManager {
    /// Creates a new replication manager.
    pub fn new() -> Self {
        let default_channel = ReplicationChannel::new(
            "default",
            128,
            1_000_000, // 1 MB/s
            ReplicationPolicy::Reliable,
        );

        let high_channel = ReplicationChannel::new(
            "player_state",
            255,
            2_000_000, // 2 MB/s
            ReplicationPolicy::Unreliable,
        );

        let low_channel = ReplicationChannel::new(
            "world_updates",
            64,
            500_000, // 500 KB/s
            ReplicationPolicy::OnChange,
        );

        Self {
            entities: HashMap::new(),
            entity_to_network_id: HashMap::new(),
            channels: vec![default_channel, high_channel, low_channel],
            next_network_id: 1,
            current_tick: 0,
            pending_spawns: Vec::new(),
            pending_despawns: Vec::new(),
            peer_acked_ticks: HashMap::new(),
            state_history: HashMap::new(),
            max_history_depth: 64,
        }
    }

    /// Registers an entity for replication with the given authority.
    ///
    /// Returns the assigned [`NetworkId`].
    pub fn register_entity(&mut self, entity: Entity, authority: Authority) -> NetworkId {
        let network_id = NetworkId::new(self.next_network_id);
        self.next_network_id += 1;

        let state = ReplicatedEntityState {
            entity,
            network_id,
            authority,
            last_state: HashMap::new(),
            last_replicated_tick: 0,
            needs_full_sync: true,
            pending_despawn: false,
            dirty_components: HashMap::new(),
        };

        self.entities.insert(network_id, state);
        self.entity_to_network_id.insert(entity, network_id);
        self.pending_spawns.push(network_id);

        log::debug!(
            "Registered entity {:?} for replication as {:?} with authority {:?}",
            entity,
            network_id,
            authority
        );
        network_id
    }

    /// Registers an entity with a specific network ID (used on the receiving side).
    pub fn register_entity_with_id(
        &mut self,
        entity: Entity,
        network_id: NetworkId,
        authority: Authority,
    ) {
        let state = ReplicatedEntityState {
            entity,
            network_id,
            authority,
            last_state: HashMap::new(),
            last_replicated_tick: 0,
            needs_full_sync: false,
            pending_despawn: false,
            dirty_components: HashMap::new(),
        };

        self.entities.insert(network_id, state);
        self.entity_to_network_id.insert(entity, network_id);

        // Ensure our next_network_id stays ahead.
        if network_id.raw() >= self.next_network_id {
            self.next_network_id = network_id.raw() + 1;
        }
    }

    /// Removes an entity from replication.
    pub fn unregister_entity(&mut self, entity: Entity) -> Option<NetworkId> {
        if let Some(network_id) = self.entity_to_network_id.remove(&entity) {
            if let Some(state) = self.entities.get_mut(&network_id) {
                state.pending_despawn = true;
            }
            self.pending_despawns.push(network_id);
            log::debug!("Unregistered entity {:?} from replication", entity);
            Some(network_id)
        } else {
            None
        }
    }

    /// Returns the network ID for a local entity, if it is replicated.
    pub fn get_network_id(&self, entity: Entity) -> Option<NetworkId> {
        self.entity_to_network_id.get(&entity).copied()
    }

    /// Returns the local entity for a network ID, if known.
    pub fn get_entity(&self, network_id: NetworkId) -> Option<Entity> {
        self.entities.get(&network_id).map(|s| s.entity)
    }

    /// Returns the authority for a replicated entity.
    pub fn get_authority(&self, network_id: NetworkId) -> Option<Authority> {
        self.entities.get(&network_id).map(|s| s.authority)
    }

    /// Sets the authority for a replicated entity.
    pub fn set_authority(&mut self, network_id: NetworkId, authority: Authority) {
        if let Some(state) = self.entities.get_mut(&network_id) {
            state.authority = authority;
        }
    }

    /// Mark a component as dirty for a given entity (it has changed and needs replication).
    pub fn mark_dirty(&mut self, entity: Entity, component_type: &str) {
        if let Some(network_id) = self.entity_to_network_id.get(&entity) {
            if let Some(state) = self.entities.get_mut(network_id) {
                state.dirty_components.insert(component_type.to_string(), true);
            }
        }
    }

    /// Update the component snapshot for an entity.
    pub fn update_component_state(
        &mut self,
        entity: Entity,
        type_name: &str,
        data: Vec<u8>,
    ) {
        if let Some(network_id) = self.entity_to_network_id.get(&entity) {
            if let Some(state) = self.entities.get_mut(network_id) {
                let snapshot = ComponentSnapshot::new(type_name, data);
                state.last_state.insert(type_name.to_string(), snapshot);
                state
                    .dirty_components
                    .insert(type_name.to_string(), true);
            }
        }
    }

    /// Adds a replication channel.
    pub fn add_channel(&mut self, channel: ReplicationChannel) {
        self.channels.push(channel);
    }

    /// Returns a channel by name.
    pub fn get_channel(&self, name: &str) -> Option<&ReplicationChannel> {
        self.channels.iter().find(|c| c.name == name)
    }

    /// Record that a peer has acknowledged up to a given tick.
    pub fn acknowledge_peer(&mut self, peer_id: u32, tick: u64) {
        let entry = self.peer_acked_ticks.entry(peer_id).or_insert(0);
        if tick > *entry {
            *entry = tick;
        }
    }

    /// Get the oldest unacknowledged tick across all peers.
    pub fn oldest_unacked_tick(&self) -> u64 {
        self.peer_acked_ticks
            .values()
            .copied()
            .min()
            .unwrap_or(0)
    }

    /// Build a state update frame containing all dirty entities.
    ///
    /// This is the main serialization entry point. It examines all tracked
    /// entities, computes deltas where possible, and produces a compact
    /// replication frame.
    pub fn build_state_update(&mut self) -> ReplicationFrame {
        let tick = self.current_tick;
        let mut frame = ReplicationFrame::new(tick);

        // Reset channel budgets.
        for ch in &mut self.channels {
            ch.reset_frame(tick);
        }

        // Process pending despawns.
        let despawns: Vec<NetworkId> = self.pending_despawns.drain(..).collect();
        for nid in &despawns {
            frame.add_update(EntityUpdate::despawn(*nid));
        }

        // Remove despawned entities from tracking.
        for nid in &despawns {
            if let Some(state) = self.entities.remove(nid) {
                self.entity_to_network_id.remove(&state.entity);
            }
            // Clean up state history.
            self.state_history
                .retain(|(id, _), _| id != nid);
        }

        // Collect updates sorted by priority (high first).
        let mut entity_ids: Vec<NetworkId> = self.entities.keys().copied().collect();
        // Sort by whether they need full sync (spawns first), then by network ID for determinism.
        entity_ids.sort_by_key(|nid| {
            let state = &self.entities[nid];
            let priority = if state.needs_full_sync { 0 } else { 1 };
            (priority, nid.raw())
        });

        for nid in entity_ids {
            let state = match self.entities.get(&nid) {
                Some(s) => s,
                None => continue,
            };

            // Skip entities with no changes unless they need a full sync.
            let has_dirty = state.dirty_components.values().any(|&d| d);
            if !has_dirty && !state.needs_full_sync {
                continue;
            }

            // Build component snapshots.
            let components: Vec<ComponentSnapshot> = state
                .last_state
                .values()
                .cloned()
                .collect();

            if components.is_empty() && !state.is_spawn_pending() {
                continue;
            }

            let update = if state.needs_full_sync {
                EntityUpdate::spawn(nid, state.authority, components.clone())
            } else if has_dirty {
                // Delta: only include dirty components.
                let dirty_components: Vec<ComponentSnapshot> = state
                    .dirty_components
                    .iter()
                    .filter(|(_, dirty)| **dirty)
                    .filter_map(|(type_name, _)| {
                        state.last_state.get(type_name).cloned()
                    })
                    .collect();

                if dirty_components.is_empty() {
                    continue;
                }
                EntityUpdate::delta(nid, dirty_components)
            } else {
                continue;
            };

            // Check bandwidth budget.
            let update_size = update.encode().len();
            let within_budget = if let Some(ch) = self.channels.first_mut() {
                ch.consume(update_size as u32)
            } else {
                true
            };

            if within_budget {
                frame.add_update(update);

                // Store in history for delta reference.
                self.state_history
                    .insert((nid, tick), components);
            }
        }

        // Clear dirty flags and needs_full_sync for entities we included.
        for update in &frame.updates {
            if let Some(state) = self.entities.get_mut(&update.network_id) {
                state.needs_full_sync = false;
                state.last_replicated_tick = tick;
                for value in state.dirty_components.values_mut() {
                    *value = false;
                }
            }
        }

        // Prune old state history.
        let oldest_needed = self.oldest_unacked_tick();
        self.state_history
            .retain(|(_, t), _| *t >= oldest_needed);

        frame
    }

    /// Serialize the current state update as bytes for network transmission.
    pub fn build_state_update_bytes(&mut self) -> Vec<u8> {
        let frame = self.build_state_update();
        frame.encode()
    }

    /// Apply an incoming replication frame from a remote peer.
    ///
    /// Returns a list of (NetworkId, action) pairs describing what happened.
    pub fn apply_state_update(
        &mut self,
        data: &[u8],
    ) -> EngineResult<Vec<ReplicationAction>> {
        let frame = ReplicationFrame::decode(data).ok_or_else(|| {
            genovo_core::EngineError::InvalidArgument(
                "failed to decode replication frame".into(),
            )
        })?;

        let mut actions = Vec::new();

        for update in &frame.updates {
            if update.is_despawn() {
                // Remove the entity.
                if let Some(state) = self.entities.remove(&update.network_id) {
                    self.entity_to_network_id.remove(&state.entity);
                    actions.push(ReplicationAction::Despawn {
                        network_id: update.network_id,
                        entity: state.entity,
                    });
                }
                continue;
            }

            if update.is_spawn() {
                // Check if we already know about this entity.
                if self.entities.contains_key(&update.network_id) {
                    // Already exists, just update components.
                    if let Some(state) =
                        self.entities.get_mut(&update.network_id)
                    {
                        for comp in &update.components {
                            state.last_state.insert(
                                comp.type_name.clone(),
                                comp.clone(),
                            );
                        }
                        if let Some(ref auth) = update.authority {
                            state.authority = *auth;
                        }
                    }
                    actions.push(ReplicationAction::Update {
                        network_id: update.network_id,
                        components: update.components.clone(),
                    });
                } else {
                    // Need to spawn a new entity.
                    let authority = update.authority.unwrap_or(Authority::Server);
                    actions.push(ReplicationAction::Spawn {
                        network_id: update.network_id,
                        authority,
                        components: update.components.clone(),
                    });
                }
                continue;
            }

            // Regular state update (full or delta).
            if let Some(state) = self.entities.get_mut(&update.network_id) {
                for comp in &update.components {
                    state
                        .last_state
                        .insert(comp.type_name.clone(), comp.clone());
                }
                actions.push(ReplicationAction::Update {
                    network_id: update.network_id,
                    components: update.components.clone(),
                });
            }
        }

        Ok(actions)
    }

    /// Advances the replication tick.
    pub fn advance_tick(&mut self) {
        self.current_tick += 1;
    }

    /// Returns the number of currently replicated entities.
    pub fn replicated_entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns the current replication tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Returns all network IDs currently tracked.
    pub fn all_network_ids(&self) -> Vec<NetworkId> {
        self.entities.keys().copied().collect()
    }

    /// Returns the total number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }
}

impl ReplicatedEntityState {
    /// Helper: returns true if this entity has a pending spawn.
    fn is_spawn_pending(&self) -> bool {
        self.needs_full_sync
    }
}

impl Default for ReplicationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ReplicationAction
// ---------------------------------------------------------------------------

/// Describes an action that the application should take after applying a
/// replication update.
#[derive(Debug, Clone)]
pub enum ReplicationAction {
    /// A new entity needs to be spawned locally.
    Spawn {
        /// The network ID for the new entity.
        network_id: NetworkId,
        /// The authority.
        authority: Authority,
        /// Initial component data.
        components: Vec<ComponentSnapshot>,
    },
    /// An existing entity's components have been updated.
    Update {
        /// The network ID of the updated entity.
        network_id: NetworkId,
        /// Updated component data.
        components: Vec<ComponentSnapshot>,
    },
    /// An entity should be despawned locally.
    Despawn {
        /// The network ID of the entity to remove.
        network_id: NetworkId,
        /// The local entity to despawn.
        entity: Entity,
    },
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // NetworkId
    // -----------------------------------------------------------------------

    #[test]
    fn test_network_id_roundtrip() {
        let id = NetworkId::new(0xDEADBEEF_CAFEBABE);
        let bytes = id.encode();
        let decoded = NetworkId::decode(&bytes).unwrap();
        assert_eq!(id, decoded);
    }

    #[test]
    fn test_network_id_decode_too_short() {
        assert!(NetworkId::decode(&[1, 2, 3]).is_none());
    }

    // -----------------------------------------------------------------------
    // Authority
    // -----------------------------------------------------------------------

    #[test]
    fn test_authority_encode_decode_server() {
        let auth = Authority::Server;
        let bytes = auth.encode();
        let (decoded, consumed) = Authority::decode(&bytes).unwrap();
        assert_eq!(decoded, Authority::Server);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_authority_encode_decode_client() {
        let auth = Authority::Client { owner: 42 };
        let bytes = auth.encode();
        let (decoded, consumed) = Authority::decode(&bytes).unwrap();
        assert_eq!(decoded, auth);
        assert_eq!(consumed, 5);
    }

    #[test]
    fn test_authority_encode_decode_shared() {
        let auth = Authority::Shared;
        let bytes = auth.encode();
        let (decoded, consumed) = Authority::decode(&bytes).unwrap();
        assert_eq!(decoded, Authority::Shared);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_authority_decode_empty() {
        assert!(Authority::decode(&[]).is_none());
    }

    #[test]
    fn test_authority_decode_invalid() {
        assert!(Authority::decode(&[99]).is_none());
    }

    // -----------------------------------------------------------------------
    // ComponentSnapshot
    // -----------------------------------------------------------------------

    #[test]
    fn test_component_snapshot_roundtrip() {
        let snap = ComponentSnapshot::new("Position", vec![1, 2, 3, 4, 5]);
        let encoded = snap.encode();
        let (decoded, consumed) = ComponentSnapshot::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.type_name, "Position");
        assert_eq!(decoded.data, vec![1, 2, 3, 4, 5]);
        assert_eq!(decoded.checksum, snap.checksum);
    }

    #[test]
    fn test_component_snapshot_checksum_differs() {
        let snap1 = ComponentSnapshot::new("Pos", vec![1, 2, 3]);
        let snap2 = ComponentSnapshot::new("Pos", vec![1, 2, 4]);
        assert_ne!(snap1.checksum, snap2.checksum);
        assert!(!snap1.matches(&snap2));
    }

    #[test]
    fn test_component_snapshot_checksum_matches() {
        let snap1 = ComponentSnapshot::new("Pos", vec![1, 2, 3]);
        let snap2 = ComponentSnapshot::new("Pos", vec![1, 2, 3]);
        assert_eq!(snap1.checksum, snap2.checksum);
        assert!(snap1.matches(&snap2));
    }

    #[test]
    fn test_component_snapshot_empty_data() {
        let snap = ComponentSnapshot::new("Empty", vec![]);
        let encoded = snap.encode();
        let (decoded, _) = ComponentSnapshot::decode(&encoded).unwrap();
        assert!(decoded.data.is_empty());
    }

    // -----------------------------------------------------------------------
    // EntityUpdate
    // -----------------------------------------------------------------------

    #[test]
    fn test_entity_update_spawn_roundtrip() {
        let nid = NetworkId::new(42);
        let comps = vec![
            ComponentSnapshot::new("Position", vec![10, 20, 30]),
            ComponentSnapshot::new("Velocity", vec![1, 2]),
        ];
        let update = EntityUpdate::spawn(nid, Authority::Server, comps);
        assert!(update.is_spawn());
        assert!(update.is_full_state());

        let encoded = update.encode();
        let (decoded, _) = EntityUpdate::decode(&encoded).unwrap();
        assert_eq!(decoded.network_id, nid);
        assert!(decoded.is_spawn());
        assert_eq!(decoded.components.len(), 2);
        assert_eq!(decoded.authority, Some(Authority::Server));
    }

    #[test]
    fn test_entity_update_despawn_roundtrip() {
        let nid = NetworkId::new(99);
        let update = EntityUpdate::despawn(nid);
        assert!(update.is_despawn());

        let encoded = update.encode();
        let (decoded, _) = EntityUpdate::decode(&encoded).unwrap();
        assert_eq!(decoded.network_id, nid);
        assert!(decoded.is_despawn());
        assert!(decoded.components.is_empty());
    }

    #[test]
    fn test_entity_update_delta_roundtrip() {
        let nid = NetworkId::new(7);
        let comps = vec![ComponentSnapshot::new("Health", vec![100])];
        let update = EntityUpdate::delta(nid, comps);
        assert!(!update.is_full_state());
        assert!(!update.is_spawn());

        let encoded = update.encode();
        let (decoded, _) = EntityUpdate::decode(&encoded).unwrap();
        assert_eq!(decoded.network_id, nid);
        assert!(!decoded.is_full_state());
        assert_eq!(decoded.components.len(), 1);
        assert_eq!(decoded.components[0].type_name, "Health");
    }

    // -----------------------------------------------------------------------
    // ReplicationFrame
    // -----------------------------------------------------------------------

    #[test]
    fn test_replication_frame_roundtrip() {
        let mut frame = ReplicationFrame::new(100);
        frame.add_update(EntityUpdate::spawn(
            NetworkId::new(1),
            Authority::Server,
            vec![ComponentSnapshot::new("Pos", vec![1, 2, 3])],
        ));
        frame.add_update(EntityUpdate::delta(
            NetworkId::new(2),
            vec![ComponentSnapshot::new("Vel", vec![4, 5])],
        ));
        frame.add_update(EntityUpdate::despawn(NetworkId::new(3)));

        let encoded = frame.encode();
        let decoded = ReplicationFrame::decode(&encoded).unwrap();

        assert_eq!(decoded.tick, 100);
        assert_eq!(decoded.updates.len(), 3);
        assert!(decoded.updates[0].is_spawn());
        assert!(!decoded.updates[1].is_spawn());
        assert!(decoded.updates[2].is_despawn());
    }

    #[test]
    fn test_replication_frame_empty() {
        let frame = ReplicationFrame::new(0);
        assert!(frame.is_empty());
        let encoded = frame.encode();
        let decoded = ReplicationFrame::decode(&encoded).unwrap();
        assert_eq!(decoded.tick, 0);
        assert!(decoded.updates.is_empty());
    }

    #[test]
    fn test_replication_frame_decode_too_short() {
        assert!(ReplicationFrame::decode(&[0; 5]).is_none());
    }

    // -----------------------------------------------------------------------
    // ReplicationManager
    // -----------------------------------------------------------------------

    #[test]
    fn test_manager_register_entity() {
        let mut mgr = ReplicationManager::new();
        let entity = Entity::new(0, 0);
        let nid = mgr.register_entity(entity, Authority::Server);

        assert_eq!(nid, NetworkId::new(1));
        assert_eq!(mgr.replicated_entity_count(), 1);
        assert_eq!(mgr.get_network_id(entity), Some(nid));
        assert_eq!(mgr.get_entity(nid), Some(entity));
        assert_eq!(mgr.get_authority(nid), Some(Authority::Server));
    }

    #[test]
    fn test_manager_register_multiple() {
        let mut mgr = ReplicationManager::new();
        let e1 = Entity::new(0, 0);
        let e2 = Entity::new(1, 0);
        let e3 = Entity::new(2, 0);

        let n1 = mgr.register_entity(e1, Authority::Server);
        let n2 = mgr.register_entity(e2, Authority::Client { owner: 1 });
        let n3 = mgr.register_entity(e3, Authority::Shared);

        assert_eq!(n1, NetworkId::new(1));
        assert_eq!(n2, NetworkId::new(2));
        assert_eq!(n3, NetworkId::new(3));
        assert_eq!(mgr.replicated_entity_count(), 3);
    }

    #[test]
    fn test_manager_unregister_entity() {
        let mut mgr = ReplicationManager::new();
        let entity = Entity::new(0, 0);
        let nid = mgr.register_entity(entity, Authority::Server);

        // Entity is registered and pending despawn still counts.
        let removed = mgr.unregister_entity(entity);
        assert_eq!(removed, Some(nid));
        // The entity is still in the map but marked for despawn.
        // It will be removed when build_state_update() processes it.
        assert!(mgr.pending_despawns.len() == 1);
    }

    #[test]
    fn test_manager_set_authority() {
        let mut mgr = ReplicationManager::new();
        let entity = Entity::new(0, 0);
        let nid = mgr.register_entity(entity, Authority::Server);

        mgr.set_authority(nid, Authority::Client { owner: 5 });
        assert_eq!(
            mgr.get_authority(nid),
            Some(Authority::Client { owner: 5 })
        );
    }

    #[test]
    fn test_manager_build_empty_update() {
        let mut mgr = ReplicationManager::new();
        let frame = mgr.build_state_update();
        assert!(frame.is_empty());
    }

    #[test]
    fn test_manager_build_spawn_update() {
        let mut mgr = ReplicationManager::new();
        let entity = Entity::new(0, 0);
        let nid = mgr.register_entity(entity, Authority::Server);

        // Add some component data.
        mgr.update_component_state(entity, "Position", vec![1, 2, 3]);

        let frame = mgr.build_state_update();
        assert_eq!(frame.updates.len(), 1);
        assert!(frame.updates[0].is_spawn());
        assert_eq!(frame.updates[0].network_id, nid);
    }

    #[test]
    fn test_manager_build_delta_update() {
        let mut mgr = ReplicationManager::new();
        let entity = Entity::new(0, 0);
        let _nid = mgr.register_entity(entity, Authority::Server);

        // First update: spawn with data.
        mgr.update_component_state(entity, "Position", vec![1, 2, 3]);
        let _ = mgr.build_state_update(); // consumes the spawn

        // Second update: dirty the component.
        mgr.update_component_state(entity, "Position", vec![4, 5, 6]);
        let frame = mgr.build_state_update();

        assert_eq!(frame.updates.len(), 1);
        assert!(!frame.updates[0].is_spawn()); // delta, not spawn
    }

    #[test]
    fn test_manager_build_despawn_update() {
        let mut mgr = ReplicationManager::new();
        let entity = Entity::new(0, 0);
        let _nid = mgr.register_entity(entity, Authority::Server);

        // Consume the spawn.
        let _ = mgr.build_state_update();

        // Unregister.
        mgr.unregister_entity(entity);

        let frame = mgr.build_state_update();
        assert_eq!(frame.updates.len(), 1);
        assert!(frame.updates[0].is_despawn());

        // Entity should now be gone.
        assert_eq!(mgr.replicated_entity_count(), 0);
    }

    #[test]
    fn test_manager_roundtrip_state_update() {
        let mut sender = ReplicationManager::new();
        let entity = Entity::new(0, 0);
        sender.register_entity(entity, Authority::Server);
        sender.update_component_state(entity, "Pos", vec![10, 20, 30]);

        let bytes = sender.build_state_update_bytes();

        let mut receiver = ReplicationManager::new();
        let actions = receiver.apply_state_update(&bytes).unwrap();

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ReplicationAction::Spawn {
                network_id,
                authority,
                components,
            } => {
                assert_eq!(*network_id, NetworkId::new(1));
                assert_eq!(*authority, Authority::Server);
                assert_eq!(components.len(), 1);
                assert_eq!(components[0].type_name, "Pos");
                assert_eq!(components[0].data, vec![10, 20, 30]);
            }
            _ => panic!("Expected Spawn action"),
        }
    }

    #[test]
    fn test_manager_apply_despawn() {
        let mut mgr = ReplicationManager::new();
        let entity = Entity::new(0, 0);
        let nid = mgr.register_entity(entity, Authority::Server);
        // Consume the spawn.
        let _ = mgr.build_state_update();

        // Register on the receiver side.
        let mut receiver = ReplicationManager::new();
        let recv_entity = Entity::new(100, 0);
        receiver.register_entity_with_id(recv_entity, nid, Authority::Server);
        assert_eq!(receiver.replicated_entity_count(), 1);

        // Build a despawn.
        mgr.unregister_entity(entity);
        let bytes = mgr.build_state_update_bytes();

        let actions = receiver.apply_state_update(&bytes).unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ReplicationAction::Despawn {
                network_id,
                entity,
            } => {
                assert_eq!(*network_id, nid);
                assert_eq!(*entity, recv_entity);
            }
            _ => panic!("Expected Despawn action"),
        }
        assert_eq!(receiver.replicated_entity_count(), 0);
    }

    #[test]
    fn test_manager_mark_dirty() {
        let mut mgr = ReplicationManager::new();
        let entity = Entity::new(0, 0);
        mgr.register_entity(entity, Authority::Server);
        mgr.update_component_state(entity, "Pos", vec![1, 2, 3]);
        // Consume spawn.
        let _ = mgr.build_state_update();

        // Initially no dirty components.
        let frame = mgr.build_state_update();
        assert!(frame.is_empty());

        // Mark dirty.
        mgr.mark_dirty(entity, "Pos");
        let frame = mgr.build_state_update();
        assert_eq!(frame.updates.len(), 1);
    }

    #[test]
    fn test_manager_peer_ack_tracking() {
        let mut mgr = ReplicationManager::new();
        mgr.acknowledge_peer(1, 10);
        mgr.acknowledge_peer(2, 5);
        mgr.acknowledge_peer(3, 15);

        assert_eq!(mgr.oldest_unacked_tick(), 5);

        mgr.acknowledge_peer(2, 20);
        assert_eq!(mgr.oldest_unacked_tick(), 10);
    }

    #[test]
    fn test_manager_advance_tick() {
        let mut mgr = ReplicationManager::new();
        assert_eq!(mgr.current_tick(), 0);
        mgr.advance_tick();
        assert_eq!(mgr.current_tick(), 1);
        mgr.advance_tick();
        assert_eq!(mgr.current_tick(), 2);
    }

    // -----------------------------------------------------------------------
    // ReplicationChannel
    // -----------------------------------------------------------------------

    #[test]
    fn test_channel_bandwidth_budget() {
        let mut ch = ReplicationChannel::new(
            "test",
            128,
            60_000, // 60KB/s => 1000 bytes per frame at 60fps
            ReplicationPolicy::Reliable,
        );

        ch.reset_frame(1);
        assert_eq!(ch.bytes_remaining(), 1000);

        assert!(ch.consume(500));
        assert_eq!(ch.bytes_remaining(), 500);

        assert!(ch.consume(500));
        assert_eq!(ch.bytes_remaining(), 0);

        assert!(!ch.consume(1)); // over budget
    }

    #[test]
    fn test_channel_reset_per_tick() {
        let mut ch = ReplicationChannel::new(
            "test",
            128,
            60_000,
            ReplicationPolicy::Reliable,
        );

        ch.reset_frame(1);
        ch.consume(500);
        assert_eq!(ch.bytes_remaining(), 500);

        ch.reset_frame(2); // new tick
        assert_eq!(ch.bytes_remaining(), 1000); // budget reset
    }

    #[test]
    fn test_channel_same_tick_no_reset() {
        let mut ch = ReplicationChannel::new(
            "test",
            128,
            60_000,
            ReplicationPolicy::Reliable,
        );

        ch.reset_frame(1);
        ch.consume(500);
        ch.reset_frame(1); // same tick
        assert_eq!(ch.bytes_remaining(), 500); // NOT reset
    }

    // -----------------------------------------------------------------------
    // CRC32 consistency
    // -----------------------------------------------------------------------

    #[test]
    fn test_crc32_deterministic() {
        let data = b"hello world";
        let c1 = ComponentSnapshot::compute_crc32(data);
        let c2 = ComponentSnapshot::compute_crc32(data);
        assert_eq!(c1, c2);

        let c3 = ComponentSnapshot::compute_crc32(b"hello worlD");
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_crc32_empty() {
        let c = ComponentSnapshot::compute_crc32(b"");
        // CRC32 of empty data should be 0x00000000 (complement of 0xFFFFFFFF).
        assert_eq!(c, 0x00000000);
    }
}
