// engine/networking/src/delta_compression.rs
//
// Delta state compression for the Genovo engine networking.
//
// Per-field dirty bits, baseline acking, quantized encoding, and
// bandwidth tracking per entity for efficient state replication.

use std::collections::HashMap;

pub const MAX_FIELDS: usize = 64;
pub const MAX_BASELINES: usize = 32;
pub const DEFAULT_QUANTIZE_POSITION: f32 = 100.0;
pub const DEFAULT_QUANTIZE_ANGLE: f32 = 10000.0;

pub type EntityNetId = u32;
pub type FieldIndex = u8;
pub type Tick = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldType { U8, U16, U32, I8, I16, I32, F16, F32, Bool, Vec3F16, Vec3F32, QuatF16, String8 }

impl FieldType {
    pub fn size_bytes(&self) -> usize {
        match self { Self::U8|Self::I8|Self::Bool => 1, Self::U16|Self::I16|Self::F16 => 2, Self::U32|Self::I32|Self::F32 => 4, Self::Vec3F16 => 6, Self::Vec3F32 => 12, Self::QuatF16 => 8, Self::String8 => 9 }
    }
}

#[derive(Debug, Clone)]
pub struct FieldDef { pub index: FieldIndex, pub name: String, pub field_type: FieldType, pub quantize_scale: f32, pub priority: f32 }

#[derive(Debug, Clone)]
pub struct EntitySchema { pub schema_id: u16, pub fields: Vec<FieldDef>, pub total_size: usize }

impl EntitySchema {
    pub fn new(schema_id: u16) -> Self { Self { schema_id, fields: Vec::new(), total_size: 0 } }
    pub fn add_field(&mut self, name: &str, ft: FieldType, quantize_scale: f32, priority: f32) {
        let idx = self.fields.len() as FieldIndex;
        self.total_size += ft.size_bytes();
        self.fields.push(FieldDef { index: idx, name: name.to_string(), field_type: ft, quantize_scale, priority });
    }
    pub fn field_count(&self) -> usize { self.fields.len() }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue { U8(u8), U16(u16), U32(u32), I8(i8), I16(i16), I32(i32), F16(u16), F32(f32), Bool(bool), Vec3([f32;3]), Quat([f32;4]), Str(String) }

impl FieldValue {
    pub fn quantize_f32(val: f32, scale: f32) -> u16 { ((val * scale).round().clamp(-32768.0, 32767.0) as i16) as u16 }
    pub fn dequantize_f32(q: u16, scale: f32) -> f32 { (q as i16) as f32 / scale }
    pub fn encode_size(&self) -> usize {
        match self { Self::U8(_)|Self::I8(_)|Self::Bool(_) => 1, Self::U16(_)|Self::I16(_)|Self::F16(_) => 2,
            Self::U32(_)|Self::I32(_)|Self::F32(_) => 4, Self::Vec3(_) => 6, Self::Quat(_) => 8, Self::Str(s) => 1 + s.len().min(255) }
    }
}

#[derive(Debug, Clone)]
pub struct EntityState { pub entity_id: EntityNetId, pub tick: Tick, pub fields: Vec<FieldValue> }

#[derive(Debug, Clone)]
pub struct Baseline { pub tick: Tick, pub state: EntityState, pub acked: bool }

#[derive(Debug, Clone)]
pub struct DirtyBits { bits: u64 }
impl DirtyBits {
    pub fn new() -> Self { Self { bits: 0 } }
    pub fn set(&mut self, idx: usize) { self.bits |= 1u64 << idx; }
    pub fn clear(&mut self, idx: usize) { self.bits &= !(1u64 << idx); }
    pub fn is_set(&self, idx: usize) -> bool { (self.bits >> idx) & 1 == 1 }
    pub fn clear_all(&mut self) { self.bits = 0; }
    pub fn any(&self) -> bool { self.bits != 0 }
    pub fn count(&self) -> u32 { self.bits.count_ones() }
    pub fn raw(&self) -> u64 { self.bits }
    pub fn set_all(&mut self, count: usize) { self.bits = (1u64 << count.min(64)) - 1; }
}

#[derive(Debug, Clone)]
pub struct DeltaPacket {
    pub entity_id: EntityNetId,
    pub base_tick: Tick,
    pub current_tick: Tick,
    pub dirty_bits: DirtyBits,
    pub field_data: Vec<u8>,
    pub is_full_state: bool,
}

impl DeltaPacket {
    pub fn encoded_size(&self) -> usize { 4 + 4 + 4 + 8 + self.field_data.len() + 1 }
}

#[derive(Debug, Clone, Default)]
pub struct BandwidthTracker {
    pub bytes_sent: u64,
    pub bytes_per_second: f32,
    pub packets_sent: u64,
    pub full_states_sent: u64,
    pub delta_states_sent: u64,
    window_bytes: Vec<(f32, u32)>,
}

impl BandwidthTracker {
    pub fn record(&mut self, bytes: u32, time: f32, is_full: bool) {
        self.bytes_sent += bytes as u64;
        self.packets_sent += 1;
        if is_full { self.full_states_sent += 1; } else { self.delta_states_sent += 1; }
        self.window_bytes.push((time, bytes));
        self.window_bytes.retain(|&(t, _)| time - t < 1.0);
        self.bytes_per_second = self.window_bytes.iter().map(|(_, b)| *b as f32).sum();
    }
    pub fn compression_ratio(&self) -> f32 {
        if self.full_states_sent == 0 { return 0.0; }
        self.delta_states_sent as f32 / (self.full_states_sent + self.delta_states_sent) as f32
    }
}

/// Per-entity delta compression state.
#[derive(Debug)]
pub struct EntityDeltaState {
    pub entity_id: EntityNetId,
    pub schema_id: u16,
    pub current: EntityState,
    pub baselines: Vec<Baseline>,
    pub dirty: DirtyBits,
    pub bandwidth: BandwidthTracker,
    pub priority: f32,
    pub last_sent_tick: Tick,
    pub send_interval: f32,
    pub time_since_send: f32,
}

impl EntityDeltaState {
    pub fn new(entity_id: EntityNetId, schema: &EntitySchema) -> Self {
        let fields = vec![FieldValue::U8(0); schema.field_count()];
        let state = EntityState { entity_id, tick: 0, fields };
        Self { entity_id, schema_id: schema.schema_id, current: state, baselines: Vec::new(),
            dirty: DirtyBits::new(), bandwidth: BandwidthTracker::default(), priority: 1.0,
            last_sent_tick: 0, send_interval: 0.05, time_since_send: 0.0 }
    }

    pub fn set_field(&mut self, index: usize, value: FieldValue) {
        if index < self.current.fields.len() && self.current.fields[index] != value {
            self.current.fields[index] = value;
            self.dirty.set(index);
        }
    }

    pub fn acknowledge_baseline(&mut self, tick: Tick) {
        for b in &mut self.baselines { if b.tick == tick { b.acked = true; } }
        self.baselines.retain(|b| b.acked || b.tick > tick.saturating_sub(MAX_BASELINES as u32));
    }

    pub fn latest_acked_baseline(&self) -> Option<&Baseline> {
        self.baselines.iter().rev().find(|b| b.acked)
    }

    pub fn build_delta(&mut self, current_tick: Tick, schema: &EntitySchema) -> DeltaPacket {
        self.current.tick = current_tick;
        let baseline = self.latest_acked_baseline();

        let (base_tick, dirty, is_full) = match baseline {
            Some(b) => {
                let mut delta_dirty = DirtyBits::new();
                for (i, field) in self.current.fields.iter().enumerate() {
                    if i < b.state.fields.len() && *field != b.state.fields[i] { delta_dirty.set(i); }
                }
                (b.tick, delta_dirty, false)
            }
            None => { let mut d = DirtyBits::new(); d.set_all(schema.field_count()); (0, d, true) }
        };

        let mut field_data = Vec::new();
        for (i, field) in self.current.fields.iter().enumerate() {
            if dirty.is_set(i) {
                // Simplified encoding: just track size.
                let size = field.encode_size();
                field_data.extend(vec![0u8; size]);
            }
        }

        // Store new baseline.
        self.baselines.push(Baseline { tick: current_tick, state: self.current.clone(), acked: false });
        while self.baselines.len() > MAX_BASELINES { self.baselines.remove(0); }
        self.dirty.clear_all();
        self.last_sent_tick = current_tick;

        DeltaPacket { entity_id: self.entity_id, base_tick, current_tick, dirty_bits: dirty, field_data, is_full_state: is_full }
    }

    pub fn should_send(&self, time: f32) -> bool { self.dirty.any() || self.time_since_send >= self.send_interval }
    pub fn update_timer(&mut self, dt: f32) { self.time_since_send += dt; }
}

/// DeltaCompression manager for all entities.
#[derive(Debug)]
pub struct DeltaCompressionManager {
    pub entities: HashMap<EntityNetId, EntityDeltaState>,
    pub schemas: HashMap<u16, EntitySchema>,
    pub current_tick: Tick,
    pub total_bandwidth: BandwidthTracker,
}

impl DeltaCompressionManager {
    pub fn new() -> Self { Self { entities: HashMap::new(), schemas: HashMap::new(), current_tick: 0, total_bandwidth: BandwidthTracker::default() } }

    pub fn register_schema(&mut self, schema: EntitySchema) { self.schemas.insert(schema.schema_id, schema); }

    pub fn add_entity(&mut self, entity_id: EntityNetId, schema_id: u16) {
        if let Some(schema) = self.schemas.get(&schema_id) {
            self.entities.insert(entity_id, EntityDeltaState::new(entity_id, schema));
        }
    }

    pub fn remove_entity(&mut self, entity_id: EntityNetId) { self.entities.remove(&entity_id); }

    pub fn set_field(&mut self, entity_id: EntityNetId, field_index: usize, value: FieldValue) {
        if let Some(state) = self.entities.get_mut(&entity_id) { state.set_field(field_index, value); }
    }

    pub fn acknowledge(&mut self, entity_id: EntityNetId, tick: Tick) {
        if let Some(state) = self.entities.get_mut(&entity_id) { state.acknowledge_baseline(tick); }
    }

    pub fn build_deltas(&mut self, dt: f32, time: f32) -> Vec<DeltaPacket> {
        self.current_tick += 1;
        let mut packets = Vec::new();
        let tick = self.current_tick;
        let schemas = &self.schemas;
        for state in self.entities.values_mut() {
            state.update_timer(dt);
            if state.should_send(time) {
                if let Some(schema) = schemas.get(&state.schema_id) {
                    let packet = state.build_delta(tick, schema);
                    let size = packet.encoded_size() as u32;
                    state.bandwidth.record(size, time, packet.is_full_state);
                    state.time_since_send = 0.0;
                    packets.push(packet);
                }
            }
        }
        packets
    }

    pub fn entity_count(&self) -> usize { self.entities.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirty_bits() {
        let mut db = DirtyBits::new();
        db.set(0); db.set(5);
        assert!(db.is_set(0)); assert!(!db.is_set(1)); assert!(db.is_set(5));
        assert_eq!(db.count(), 2);
    }

    #[test]
    fn test_quantize() {
        let q = FieldValue::quantize_f32(1.5, 100.0);
        let d = FieldValue::dequantize_f32(q, 100.0);
        assert!((d - 1.5).abs() < 0.02);
    }

    #[test]
    fn test_delta_compression() {
        let mut schema = EntitySchema::new(1);
        schema.add_field("x", FieldType::F32, 100.0, 1.0);
        schema.add_field("y", FieldType::F32, 100.0, 1.0);

        let mut mgr = DeltaCompressionManager::new();
        mgr.register_schema(schema);
        mgr.add_entity(1, 1);
        mgr.set_field(1, 0, FieldValue::F32(5.0));

        let packets = mgr.build_deltas(0.016, 0.0);
        assert_eq!(packets.len(), 1);
        assert!(packets[0].is_full_state); // First packet is always full.
    }

    #[test]
    fn test_baseline_ack() {
        let mut schema = EntitySchema::new(1);
        schema.add_field("x", FieldType::F32, 100.0, 1.0);
        let mut state = EntityDeltaState::new(1, &schema);
        state.set_field(0, FieldValue::F32(1.0));
        let packet = state.build_delta(1, &schema);
        assert!(packet.is_full_state);
        state.acknowledge_baseline(1);
        assert!(state.latest_acked_baseline().is_some());
        state.set_field(0, FieldValue::F32(2.0));
        let packet2 = state.build_delta(2, &schema);
        assert!(!packet2.is_full_state);
    }
}
