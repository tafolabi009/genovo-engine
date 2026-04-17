// engine/networking/src/replication_v2.rs
// Enhanced replication: property-level, conditional, priority-based bandwidth, interest management.
use std::collections::{HashMap, BTreeMap, VecDeque, HashSet};

pub type NetworkId = u32;
pub type ClientId = u32;
pub type PropertyId = u16;
pub type ComponentTypeId = u16;

#[derive(Debug, Clone)]
pub enum ReplicatedValue { Bool(bool), U8(u8), U16(u16), U32(u32), I32(i32), F32(f32), Vec3([f32; 3]), Quat([f32; 4]), String(String), Bytes(Vec<u8>) }

impl ReplicatedValue {
    pub fn size_bytes(&self) -> usize {
        match self { Self::Bool(_)=>1, Self::U8(_)=>1, Self::U16(_)=>2, Self::U32(_)|Self::I32(_)|Self::F32(_)=>4, Self::Vec3(_)=>12, Self::Quat(_)=>16, Self::String(s)=>2+s.len(), Self::Bytes(b)=>2+b.len() }
    }
    pub fn differs(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(a), Self::Bool(b)) => a != b,
            (Self::F32(a), Self::F32(b)) => (a - b).abs() > 0.001,
            (Self::Vec3(a), Self::Vec3(b)) => (a[0]-b[0]).abs() > 0.001 || (a[1]-b[1]).abs() > 0.001 || (a[2]-b[2]).abs() > 0.001,
            _ => true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReplicatedProperty { pub id: PropertyId, pub name: String, pub value: ReplicatedValue, pub dirty: bool, pub priority: f32, pub condition: ReplicationCondition, pub interpolation: InterpolationMode, pub compress: bool }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicationCondition { Always, OwnerOnly, SkipOwner, InitialOnly, OnChange }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode { None, Linear, Hermite, Snap }

#[derive(Debug, Clone)]
pub struct ReplicatedEntity {
    pub network_id: NetworkId,
    pub owner: Option<ClientId>,
    pub authority: ClientId,
    pub entity_type: String,
    pub properties: HashMap<PropertyId, ReplicatedProperty>,
    pub relevance_position: [f32; 3],
    pub relevance_radius: f32,
    pub priority: f32,
    pub is_dormant: bool,
    pub last_replicated: HashMap<ClientId, u64>,
}

impl ReplicatedEntity {
    pub fn new(net_id: NetworkId, entity_type: &str) -> Self {
        Self { network_id: net_id, owner: None, authority: 0, entity_type: entity_type.to_string(), properties: HashMap::new(), relevance_position: [0.0; 3], relevance_radius: 100.0, priority: 1.0, is_dormant: false, last_replicated: HashMap::new() }
    }
    pub fn add_property(&mut self, id: PropertyId, name: &str, value: ReplicatedValue, condition: ReplicationCondition) {
        self.properties.insert(id, ReplicatedProperty { id, name: name.to_string(), value, dirty: true, priority: 1.0, condition, interpolation: InterpolationMode::Linear, compress: false });
    }
    pub fn set_property(&mut self, id: PropertyId, value: ReplicatedValue) {
        if let Some(prop) = self.properties.get_mut(&id) {
            if prop.value.differs(&value) { prop.value = value; prop.dirty = true; }
        }
    }
    pub fn dirty_properties(&self) -> Vec<PropertyId> { self.properties.iter().filter(|(_, p)| p.dirty).map(|(&id, _)| id).collect() }
    pub fn clear_dirty(&mut self) { for p in self.properties.values_mut() { p.dirty = false; } }
}

pub struct InterestManager { pub client_positions: HashMap<ClientId, [f32; 3]>, pub relevance_radius: f32 }
impl InterestManager {
    pub fn new(radius: f32) -> Self { Self { client_positions: HashMap::new(), relevance_radius: radius } }
    pub fn update_client_position(&mut self, client: ClientId, pos: [f32; 3]) { self.client_positions.insert(client, pos); }
    pub fn is_relevant(&self, client: ClientId, entity: &ReplicatedEntity) -> bool {
        if let Some(cpos) = self.client_positions.get(&client) {
            let dx = cpos[0] - entity.relevance_position[0];
            let dy = cpos[1] - entity.relevance_position[1];
            let dz = cpos[2] - entity.relevance_position[2];
            let dist_sq = dx*dx + dy*dy + dz*dz;
            let max_dist = self.relevance_radius + entity.relevance_radius;
            dist_sq <= max_dist * max_dist
        } else { true }
    }
}

pub struct BandwidthAllocator { pub budget_bytes_per_frame: u32, pub used_bytes: u32, pub allocations: Vec<(NetworkId, u32)> }
impl BandwidthAllocator {
    pub fn new(budget: u32) -> Self { Self { budget_bytes_per_frame: budget, used_bytes: 0, allocations: Vec::new() } }
    pub fn can_allocate(&self, bytes: u32) -> bool { self.used_bytes + bytes <= self.budget_bytes_per_frame }
    pub fn allocate(&mut self, entity: NetworkId, bytes: u32) -> bool {
        if self.can_allocate(bytes) { self.used_bytes += bytes; self.allocations.push((entity, bytes)); true } else { false }
    }
    pub fn reset(&mut self) { self.used_bytes = 0; self.allocations.clear(); }
    pub fn utilization(&self) -> f32 { self.used_bytes as f32 / self.budget_bytes_per_frame.max(1) as f32 }
}

pub struct ReplicationManagerV2 {
    pub entities: HashMap<NetworkId, ReplicatedEntity>,
    pub interest: InterestManager,
    pub bandwidth: BandwidthAllocator,
    pub clients: HashSet<ClientId>,
    next_network_id: NetworkId,
    pub frame: u64,
    pub stats: ReplicationStats,
}

#[derive(Debug, Clone, Default)]
pub struct ReplicationStats { pub entities_replicated: u32, pub properties_sent: u32, pub bytes_sent: u32, pub entities_skipped: u32, pub bandwidth_utilization: f32 }

impl ReplicationManagerV2 {
    pub fn new(bandwidth_budget: u32) -> Self {
        Self { entities: HashMap::new(), interest: InterestManager::new(100.0), bandwidth: BandwidthAllocator::new(bandwidth_budget), clients: HashSet::new(), next_network_id: 1, frame: 0, stats: ReplicationStats::default() }
    }
    pub fn register_entity(&mut self, entity_type: &str) -> NetworkId {
        let id = self.next_network_id; self.next_network_id += 1;
        self.entities.insert(id, ReplicatedEntity::new(id, entity_type));
        id
    }
    pub fn add_client(&mut self, client: ClientId) { self.clients.insert(client); }
    pub fn remove_client(&mut self, client: ClientId) { self.clients.remove(&client); }
    pub fn replicate_frame(&mut self) {
        self.frame += 1; self.bandwidth.reset(); self.stats = ReplicationStats::default();
        let mut updates: Vec<(NetworkId, f32)> = Vec::new();
        for (id, entity) in &self.entities {
            if entity.is_dormant { self.stats.entities_skipped += 1; continue; }
            let dirty_count = entity.dirty_properties().len();
            if dirty_count == 0 { continue; }
            updates.push((*id, entity.priority * dirty_count as f32));
        }
        updates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (net_id, _) in updates {
            let entity = match self.entities.get(&net_id) { Some(e) => e, None => continue };
            let dirty = entity.dirty_properties();
            let estimated_bytes: u32 = dirty.iter().map(|pid| entity.properties.get(pid).map(|p| p.value.size_bytes() as u32 + 4).unwrap_or(0)).sum();
            if self.bandwidth.allocate(net_id, estimated_bytes) {
                self.stats.entities_replicated += 1;
                self.stats.properties_sent += dirty.len() as u32;
                self.stats.bytes_sent += estimated_bytes;
            } else { self.stats.entities_skipped += 1; }
        }
        for entity in self.entities.values_mut() { entity.clear_dirty(); }
        self.stats.bandwidth_utilization = self.bandwidth.utilization();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_replication() {
        let mut mgr = ReplicationManagerV2::new(10000);
        mgr.add_client(0);
        let id = mgr.register_entity("player");
        if let Some(e) = mgr.entities.get_mut(&id) {
            e.add_property(0, "position", ReplicatedValue::Vec3([0.0; 3]), ReplicationCondition::Always);
            e.set_property(0, ReplicatedValue::Vec3([1.0, 2.0, 3.0]));
        }
        mgr.replicate_frame();
        assert!(mgr.stats.entities_replicated > 0);
    }
    #[test]
    fn test_bandwidth() {
        let mut ba = BandwidthAllocator::new(1000);
        assert!(ba.allocate(1, 500));
        assert!(ba.allocate(2, 400));
        assert!(!ba.allocate(3, 200));
    }
}
