// engine/physics/src/collision_events.rs
//
// Collision event system for the Genovo physics engine.
//
// Provides collision begin/stay/end events, trigger enter/exit events,
// event filtering by layer, event callbacks, contact info (points, normals,
// impulses), and event history buffer.
//
// # Architecture
//
// The `CollisionEventSystem` tracks which collision pairs are active each
// frame. By comparing against the previous frame's active pairs, it
// generates begin/end events. Pairs that persist across frames produce
// stay events. Events are dispatched through registered callbacks and
// also stored in a ring buffer for deferred processing.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Unique body handle.
pub type BodyId = u64;

/// Collision layer mask (32 layers).
pub type LayerMask = u32;

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    #[inline]
    pub fn add(self, o: Self) -> Self { Self::new(self.x + o.x, self.y + o.y, self.z + o.z) }

    #[inline]
    pub fn sub(self, o: Self) -> Self { Self::new(self.x - o.x, self.y - o.y, self.z - o.z) }

    #[inline]
    pub fn scale(self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) }

    #[inline]
    pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }

    #[inline]
    pub fn length(self) -> f32 { self.dot(self).sqrt() }

    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l > 1e-7 { self.scale(1.0 / l) } else { Self::ZERO }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of contact points per collision pair.
pub const MAX_CONTACTS_PER_PAIR: usize = 8;

/// Maximum number of events in the history buffer.
pub const MAX_EVENT_HISTORY: usize = 4096;

/// Maximum number of callbacks registered per event type.
pub const MAX_CALLBACKS_PER_TYPE: usize = 64;

/// Default event buffer capacity.
pub const DEFAULT_EVENT_BUFFER_CAPACITY: usize = 1024;

/// Layer mask that matches all layers.
pub const ALL_LAYERS: LayerMask = 0xFFFF_FFFF;

// ---------------------------------------------------------------------------
// Contact info
// ---------------------------------------------------------------------------

/// A single contact point between two colliding bodies.
#[derive(Debug, Clone, Copy)]
pub struct ContactPoint {
    /// World-space position of the contact.
    pub position: Vec3,
    /// Contact normal (pointing from body A to body B).
    pub normal: Vec3,
    /// Penetration depth (positive = overlapping).
    pub depth: f32,
    /// Normal impulse applied to resolve this contact.
    pub impulse_normal: f32,
    /// Tangential (friction) impulse.
    pub impulse_tangent: f32,
    /// Local position on body A.
    pub local_a: Vec3,
    /// Local position on body B.
    pub local_b: Vec3,
}

impl Default for ContactPoint {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            normal: Vec3::new(0.0, 1.0, 0.0),
            depth: 0.0,
            impulse_normal: 0.0,
            impulse_tangent: 0.0,
            local_a: Vec3::ZERO,
            local_b: Vec3::ZERO,
        }
    }
}

/// Contact manifold: all contact points between a pair of bodies.
#[derive(Debug, Clone)]
pub struct ContactInfo {
    /// Contact points.
    pub points: Vec<ContactPoint>,
    /// Average contact normal.
    pub average_normal: Vec3,
    /// Average contact position.
    pub average_position: Vec3,
    /// Total normal impulse across all contacts.
    pub total_impulse_normal: f32,
    /// Total tangential impulse across all contacts.
    pub total_impulse_tangent: f32,
    /// Relative velocity at the contact (body A velocity - body B velocity).
    pub relative_velocity: Vec3,
    /// Whether the contact has separation velocity (separating).
    pub separating: bool,
}

impl Default for ContactInfo {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            average_normal: Vec3::new(0.0, 1.0, 0.0),
            average_position: Vec3::ZERO,
            total_impulse_normal: 0.0,
            total_impulse_tangent: 0.0,
            relative_velocity: Vec3::ZERO,
            separating: false,
        }
    }
}

impl ContactInfo {
    /// Create contact info from a set of contact points.
    pub fn from_points(points: Vec<ContactPoint>) -> Self {
        if points.is_empty() {
            return Self::default();
        }

        let n = points.len() as f32;
        let mut avg_normal = Vec3::ZERO;
        let mut avg_pos = Vec3::ZERO;
        let mut total_normal = 0.0f32;
        let mut total_tangent = 0.0f32;

        for p in &points {
            avg_normal = avg_normal.add(p.normal);
            avg_pos = avg_pos.add(p.position);
            total_normal += p.impulse_normal;
            total_tangent += p.impulse_tangent;
        }

        Self {
            average_normal: avg_normal.scale(1.0 / n).normalize(),
            average_position: avg_pos.scale(1.0 / n),
            total_impulse_normal: total_normal,
            total_impulse_tangent: total_tangent,
            relative_velocity: Vec3::ZERO,
            separating: false,
            points,
        }
    }

    /// Get the strongest contact point (highest normal impulse).
    pub fn strongest_contact(&self) -> Option<&ContactPoint> {
        self.points
            .iter()
            .max_by(|a, b| a.impulse_normal.partial_cmp(&b.impulse_normal).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the deepest contact point.
    pub fn deepest_contact(&self) -> Option<&ContactPoint> {
        self.points
            .iter()
            .max_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Number of contact points.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Impact speed (magnitude of relative velocity along the normal).
    pub fn impact_speed(&self) -> f32 {
        self.relative_velocity.dot(self.average_normal).abs()
    }
}

// ---------------------------------------------------------------------------
// Collision pair
// ---------------------------------------------------------------------------

/// An unordered pair of body IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollisionPair {
    /// Body with the smaller ID.
    pub body_a: BodyId,
    /// Body with the larger ID.
    pub body_b: BodyId,
}

impl CollisionPair {
    /// Create a collision pair (canonicalized with smaller ID first).
    pub fn new(a: BodyId, b: BodyId) -> Self {
        if a <= b {
            Self { body_a: a, body_b: b }
        } else {
            Self { body_a: b, body_b: a }
        }
    }

    /// Check if this pair involves a specific body.
    pub fn involves(&self, body: BodyId) -> bool {
        self.body_a == body || self.body_b == body
    }

    /// Get the other body in the pair.
    pub fn other(&self, body: BodyId) -> Option<BodyId> {
        if self.body_a == body {
            Some(self.body_b)
        } else if self.body_b == body {
            Some(self.body_a)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

/// The type of collision event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollisionEventType {
    /// Two bodies started colliding this frame.
    Begin,
    /// Two bodies are still colliding (ongoing).
    Stay,
    /// Two bodies stopped colliding this frame.
    End,
}

impl fmt::Display for CollisionEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Begin => write!(f, "Begin"),
            Self::Stay => write!(f, "Stay"),
            Self::End => write!(f, "End"),
        }
    }
}

/// The type of trigger event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriggerEventType {
    /// A body entered a trigger volume.
    Enter,
    /// A body is inside a trigger volume (ongoing).
    Stay,
    /// A body exited a trigger volume.
    Exit,
}

impl fmt::Display for TriggerEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Enter => write!(f, "Enter"),
            Self::Stay => write!(f, "Stay"),
            Self::Exit => write!(f, "Exit"),
        }
    }
}

/// A collision event.
#[derive(Debug, Clone)]
pub struct CollisionEvent {
    /// The pair of colliding bodies.
    pub pair: CollisionPair,
    /// Event type.
    pub event_type: CollisionEventType,
    /// Contact information (empty for End events).
    pub contact_info: ContactInfo,
    /// Frame number when this event occurred.
    pub frame: u64,
    /// Time within the physics step (0..dt).
    pub time: f32,
    /// Layer of body A.
    pub layer_a: LayerMask,
    /// Layer of body B.
    pub layer_b: LayerMask,
}

/// A trigger event.
#[derive(Debug, Clone)]
pub struct TriggerEvent {
    /// The trigger body.
    pub trigger_body: BodyId,
    /// The other body that entered/exited the trigger.
    pub other_body: BodyId,
    /// Event type.
    pub event_type: TriggerEventType,
    /// Frame number.
    pub frame: u64,
    /// Layer of the trigger.
    pub trigger_layer: LayerMask,
    /// Layer of the other body.
    pub other_layer: LayerMask,
}

// ---------------------------------------------------------------------------
// Event filter
// ---------------------------------------------------------------------------

/// Filter for which events to receive.
#[derive(Debug, Clone)]
pub struct EventFilter {
    /// Only report events involving bodies on these layers.
    pub layer_mask: LayerMask,
    /// Only report events for specific bodies (empty = all bodies).
    pub body_filter: HashSet<BodyId>,
    /// Which collision event types to report.
    pub collision_types: HashSet<CollisionEventType>,
    /// Which trigger event types to report.
    pub trigger_types: HashSet<TriggerEventType>,
    /// Minimum impact speed to report (for collision events).
    pub min_impact_speed: f32,
    /// Whether to include trigger events.
    pub include_triggers: bool,
    /// Whether to include collision events.
    pub include_collisions: bool,
}

impl Default for EventFilter {
    fn default() -> Self {
        let mut collision_types = HashSet::new();
        collision_types.insert(CollisionEventType::Begin);
        collision_types.insert(CollisionEventType::Stay);
        collision_types.insert(CollisionEventType::End);

        let mut trigger_types = HashSet::new();
        trigger_types.insert(TriggerEventType::Enter);
        trigger_types.insert(TriggerEventType::Stay);
        trigger_types.insert(TriggerEventType::Exit);

        Self {
            layer_mask: ALL_LAYERS,
            body_filter: HashSet::new(),
            collision_types,
            trigger_types,
            min_impact_speed: 0.0,
            include_triggers: true,
            include_collisions: true,
        }
    }
}

impl EventFilter {
    /// Create a filter for a specific body.
    pub fn for_body(body: BodyId) -> Self {
        let mut filter = Self::default();
        filter.body_filter.insert(body);
        filter
    }

    /// Create a filter for a specific layer.
    pub fn for_layer(layer: LayerMask) -> Self {
        Self {
            layer_mask: layer,
            ..Default::default()
        }
    }

    /// Create a filter for collision begin events only.
    pub fn begin_only() -> Self {
        let mut collision_types = HashSet::new();
        collision_types.insert(CollisionEventType::Begin);
        Self {
            collision_types,
            ..Default::default()
        }
    }

    /// Test if a collision event passes this filter.
    pub fn accepts_collision(&self, event: &CollisionEvent) -> bool {
        if !self.include_collisions {
            return false;
        }
        if !self.collision_types.contains(&event.event_type) {
            return false;
        }
        if (event.layer_a & self.layer_mask) == 0 && (event.layer_b & self.layer_mask) == 0 {
            return false;
        }
        if !self.body_filter.is_empty()
            && !self.body_filter.contains(&event.pair.body_a)
            && !self.body_filter.contains(&event.pair.body_b)
        {
            return false;
        }
        if event.event_type == CollisionEventType::Begin || event.event_type == CollisionEventType::Stay {
            if event.contact_info.impact_speed() < self.min_impact_speed {
                return false;
            }
        }
        true
    }

    /// Test if a trigger event passes this filter.
    pub fn accepts_trigger(&self, event: &TriggerEvent) -> bool {
        if !self.include_triggers {
            return false;
        }
        if !self.trigger_types.contains(&event.event_type) {
            return false;
        }
        if (event.trigger_layer & self.layer_mask) == 0 && (event.other_layer & self.layer_mask) == 0 {
            return false;
        }
        if !self.body_filter.is_empty()
            && !self.body_filter.contains(&event.trigger_body)
            && !self.body_filter.contains(&event.other_body)
        {
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Callback types
// ---------------------------------------------------------------------------

/// Callback ID for unregistration.
pub type CallbackId = u64;

/// Boxed collision event callback.
type CollisionCallback = Box<dyn Fn(&CollisionEvent) + Send + Sync>;

/// Boxed trigger event callback.
type TriggerCallback = Box<dyn Fn(&TriggerEvent) + Send + Sync>;

/// Registered callback with filter and ID.
struct RegisteredCollisionCallback {
    id: CallbackId,
    filter: EventFilter,
    callback: CollisionCallback,
}

struct RegisteredTriggerCallback {
    id: CallbackId,
    filter: EventFilter,
    callback: TriggerCallback,
}

// ---------------------------------------------------------------------------
// Event history
// ---------------------------------------------------------------------------

/// A record in the event history.
#[derive(Debug, Clone)]
pub enum EventRecord {
    Collision(CollisionEvent),
    Trigger(TriggerEvent),
}

impl EventRecord {
    /// Get the frame number.
    pub fn frame(&self) -> u64 {
        match self {
            Self::Collision(e) => e.frame,
            Self::Trigger(e) => e.frame,
        }
    }

    /// Whether this is a collision event.
    pub fn is_collision(&self) -> bool {
        matches!(self, Self::Collision(_))
    }

    /// Whether this is a trigger event.
    pub fn is_trigger(&self) -> bool {
        matches!(self, Self::Trigger(_))
    }
}

/// Ring buffer for event history.
pub struct EventHistory {
    records: VecDeque<EventRecord>,
    max_records: usize,
    total_recorded: u64,
}

impl EventHistory {
    /// Create a new event history buffer.
    pub fn new(max_records: usize) -> Self {
        Self {
            records: VecDeque::with_capacity(max_records),
            max_records,
            total_recorded: 0,
        }
    }

    /// Record an event.
    pub fn record(&mut self, event: EventRecord) {
        if self.records.len() >= self.max_records {
            self.records.pop_front();
        }
        self.records.push_back(event);
        self.total_recorded += 1;
    }

    /// Get all records.
    pub fn records(&self) -> &VecDeque<EventRecord> {
        &self.records
    }

    /// Query events for a specific body.
    pub fn query_body(&self, body: BodyId) -> Vec<&EventRecord> {
        self.records
            .iter()
            .filter(|r| match r {
                EventRecord::Collision(e) => e.pair.involves(body),
                EventRecord::Trigger(e) => e.trigger_body == body || e.other_body == body,
            })
            .collect()
    }

    /// Query events in a frame range.
    pub fn query_frame_range(&self, start_frame: u64, end_frame: u64) -> Vec<&EventRecord> {
        self.records
            .iter()
            .filter(|r| {
                let f = r.frame();
                f >= start_frame && f <= end_frame
            })
            .collect()
    }

    /// Get the most recent N events.
    pub fn recent(&self, count: usize) -> Vec<&EventRecord> {
        self.records.iter().rev().take(count).collect()
    }

    /// Clear the history.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Total number of events ever recorded.
    pub fn total_recorded(&self) -> u64 {
        self.total_recorded
    }

    /// Current number of events in the buffer.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics for the collision event system.
#[derive(Debug, Clone, Copy, Default)]
pub struct CollisionEventStats {
    /// Number of collision begin events this frame.
    pub collision_begins: u32,
    /// Number of collision stay events this frame.
    pub collision_stays: u32,
    /// Number of collision end events this frame.
    pub collision_ends: u32,
    /// Number of trigger enter events this frame.
    pub trigger_enters: u32,
    /// Number of trigger stay events this frame.
    pub trigger_stays: u32,
    /// Number of trigger exits this frame.
    pub trigger_exits: u32,
    /// Total active collision pairs.
    pub active_pairs: u32,
    /// Total active trigger pairs.
    pub active_trigger_pairs: u32,
    /// Number of callbacks invoked.
    pub callbacks_invoked: u32,
    /// Total events dispatched.
    pub total_events: u32,
}

// ---------------------------------------------------------------------------
// Collision event system
// ---------------------------------------------------------------------------

/// Main collision event system.
///
/// Tracks active collision and trigger pairs across frames, generates
/// begin/stay/end events, dispatches callbacks, and stores event history.
pub struct CollisionEventSystem {
    /// Active collision pairs from the current frame.
    current_collision_pairs: HashMap<CollisionPair, ContactInfo>,
    /// Active collision pairs from the previous frame.
    previous_collision_pairs: HashSet<CollisionPair>,
    /// Active trigger pairs from the current frame.
    current_trigger_pairs: HashSet<CollisionPair>,
    /// Active trigger pairs from the previous frame.
    previous_trigger_pairs: HashSet<CollisionPair>,
    /// Registered collision callbacks.
    collision_callbacks: Vec<RegisteredCollisionCallback>,
    /// Registered trigger callbacks.
    trigger_callbacks: Vec<RegisteredTriggerCallback>,
    /// Event history buffer.
    history: EventHistory,
    /// Pending collision events (dispatched at end of frame).
    pending_collision_events: Vec<CollisionEvent>,
    /// Pending trigger events.
    pending_trigger_events: Vec<TriggerEvent>,
    /// Statistics for the current frame.
    stats: CollisionEventStats,
    /// Current frame number.
    frame: u64,
    /// Next callback ID.
    next_callback_id: CallbackId,
    /// Body layer map.
    body_layers: HashMap<BodyId, LayerMask>,
}

impl CollisionEventSystem {
    /// Create a new collision event system.
    pub fn new() -> Self {
        Self {
            current_collision_pairs: HashMap::new(),
            previous_collision_pairs: HashSet::new(),
            current_trigger_pairs: HashSet::new(),
            previous_trigger_pairs: HashSet::new(),
            collision_callbacks: Vec::new(),
            trigger_callbacks: Vec::new(),
            history: EventHistory::new(MAX_EVENT_HISTORY),
            pending_collision_events: Vec::new(),
            pending_trigger_events: Vec::new(),
            stats: CollisionEventStats::default(),
            frame: 0,
            next_callback_id: 0,
            body_layers: HashMap::new(),
        }
    }

    /// Set the layer for a body.
    pub fn set_body_layer(&mut self, body: BodyId, layer: LayerMask) {
        self.body_layers.insert(body, layer);
    }

    /// Get the layer for a body.
    pub fn body_layer(&self, body: BodyId) -> LayerMask {
        *self.body_layers.get(&body).unwrap_or(&ALL_LAYERS)
    }

    /// Begin a new physics frame. Call before reporting any contacts.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.stats = CollisionEventStats::default();

        // Move current pairs to previous.
        self.previous_collision_pairs = self
            .current_collision_pairs
            .keys()
            .cloned()
            .collect();
        self.current_collision_pairs.clear();

        self.previous_trigger_pairs = self.current_trigger_pairs.clone();
        self.current_trigger_pairs.clear();

        self.pending_collision_events.clear();
        self.pending_trigger_events.clear();
    }

    /// Report a collision contact between two bodies for this frame.
    pub fn report_collision(&mut self, body_a: BodyId, body_b: BodyId, contact: ContactInfo) {
        let pair = CollisionPair::new(body_a, body_b);
        self.current_collision_pairs.insert(pair, contact);
    }

    /// Report a trigger overlap between two bodies for this frame.
    pub fn report_trigger(&mut self, trigger_body: BodyId, other_body: BodyId) {
        let pair = CollisionPair::new(trigger_body, other_body);
        self.current_trigger_pairs.insert(pair);
    }

    /// Process all reported contacts and generate events.
    /// Call after all contacts have been reported for this frame.
    pub fn process_events(&mut self) {
        // --- Collision events ---

        // Begin: in current but not in previous.
        for (pair, contact) in &self.current_collision_pairs {
            if !self.previous_collision_pairs.contains(pair) {
                let event = CollisionEvent {
                    pair: *pair,
                    event_type: CollisionEventType::Begin,
                    contact_info: contact.clone(),
                    frame: self.frame,
                    time: 0.0,
                    layer_a: self.body_layer(pair.body_a),
                    layer_b: self.body_layer(pair.body_b),
                };
                self.pending_collision_events.push(event);
                self.stats.collision_begins += 1;
            } else {
                // Stay: in both current and previous.
                let event = CollisionEvent {
                    pair: *pair,
                    event_type: CollisionEventType::Stay,
                    contact_info: contact.clone(),
                    frame: self.frame,
                    time: 0.0,
                    layer_a: self.body_layer(pair.body_a),
                    layer_b: self.body_layer(pair.body_b),
                };
                self.pending_collision_events.push(event);
                self.stats.collision_stays += 1;
            }
        }

        // End: in previous but not in current.
        for pair in &self.previous_collision_pairs {
            if !self.current_collision_pairs.contains_key(pair) {
                let event = CollisionEvent {
                    pair: *pair,
                    event_type: CollisionEventType::End,
                    contact_info: ContactInfo::default(),
                    frame: self.frame,
                    time: 0.0,
                    layer_a: self.body_layer(pair.body_a),
                    layer_b: self.body_layer(pair.body_b),
                };
                self.pending_collision_events.push(event);
                self.stats.collision_ends += 1;
            }
        }

        // --- Trigger events ---

        for pair in &self.current_trigger_pairs {
            if !self.previous_trigger_pairs.contains(pair) {
                let event = TriggerEvent {
                    trigger_body: pair.body_a,
                    other_body: pair.body_b,
                    event_type: TriggerEventType::Enter,
                    frame: self.frame,
                    trigger_layer: self.body_layer(pair.body_a),
                    other_layer: self.body_layer(pair.body_b),
                };
                self.pending_trigger_events.push(event);
                self.stats.trigger_enters += 1;
            } else {
                let event = TriggerEvent {
                    trigger_body: pair.body_a,
                    other_body: pair.body_b,
                    event_type: TriggerEventType::Stay,
                    frame: self.frame,
                    trigger_layer: self.body_layer(pair.body_a),
                    other_layer: self.body_layer(pair.body_b),
                };
                self.pending_trigger_events.push(event);
                self.stats.trigger_stays += 1;
            }
        }

        for pair in &self.previous_trigger_pairs {
            if !self.current_trigger_pairs.contains(pair) {
                let event = TriggerEvent {
                    trigger_body: pair.body_a,
                    other_body: pair.body_b,
                    event_type: TriggerEventType::Exit,
                    frame: self.frame,
                    trigger_layer: self.body_layer(pair.body_a),
                    other_layer: self.body_layer(pair.body_b),
                };
                self.pending_trigger_events.push(event);
                self.stats.trigger_exits += 1;
            }
        }

        self.stats.active_pairs = self.current_collision_pairs.len() as u32;
        self.stats.active_trigger_pairs = self.current_trigger_pairs.len() as u32;
        self.stats.total_events = self.pending_collision_events.len() as u32
            + self.pending_trigger_events.len() as u32;
    }

    /// Dispatch all pending events to registered callbacks and record in history.
    pub fn dispatch_events(&mut self) {
        // Dispatch collision events.
        for event in &self.pending_collision_events {
            for cb in &self.collision_callbacks {
                if cb.filter.accepts_collision(event) {
                    (cb.callback)(event);
                    self.stats.callbacks_invoked += 1;
                }
            }
            self.history.record(EventRecord::Collision(event.clone()));
        }

        // Dispatch trigger events.
        for event in &self.pending_trigger_events {
            for cb in &self.trigger_callbacks {
                if cb.filter.accepts_trigger(event) {
                    (cb.callback)(event);
                    self.stats.callbacks_invoked += 1;
                }
            }
            self.history.record(EventRecord::Trigger(event.clone()));
        }
    }

    /// Register a collision event callback. Returns a callback ID for unregistration.
    pub fn on_collision<F>(&mut self, filter: EventFilter, callback: F) -> CallbackId
    where
        F: Fn(&CollisionEvent) + Send + Sync + 'static,
    {
        let id = self.next_callback_id;
        self.next_callback_id += 1;
        self.collision_callbacks.push(RegisteredCollisionCallback {
            id,
            filter,
            callback: Box::new(callback),
        });
        id
    }

    /// Register a trigger event callback.
    pub fn on_trigger<F>(&mut self, filter: EventFilter, callback: F) -> CallbackId
    where
        F: Fn(&TriggerEvent) + Send + Sync + 'static,
    {
        let id = self.next_callback_id;
        self.next_callback_id += 1;
        self.trigger_callbacks.push(RegisteredTriggerCallback {
            id,
            filter,
            callback: Box::new(callback),
        });
        id
    }

    /// Unregister a collision callback.
    pub fn remove_collision_callback(&mut self, id: CallbackId) {
        self.collision_callbacks.retain(|cb| cb.id != id);
    }

    /// Unregister a trigger callback.
    pub fn remove_trigger_callback(&mut self, id: CallbackId) {
        self.trigger_callbacks.retain(|cb| cb.id != id);
    }

    /// Get pending collision events (after process_events, before dispatch).
    pub fn pending_collision_events(&self) -> &[CollisionEvent] {
        &self.pending_collision_events
    }

    /// Get pending trigger events.
    pub fn pending_trigger_events(&self) -> &[TriggerEvent] {
        &self.pending_trigger_events
    }

    /// Get the event history buffer.
    pub fn history(&self) -> &EventHistory {
        &self.history
    }

    /// Get stats for the current frame.
    pub fn stats(&self) -> &CollisionEventStats {
        &self.stats
    }

    /// Get the current frame number.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Check if two bodies are currently colliding.
    pub fn are_colliding(&self, a: BodyId, b: BodyId) -> bool {
        let pair = CollisionPair::new(a, b);
        self.current_collision_pairs.contains_key(&pair)
    }

    /// Get the contact info between two bodies (if colliding).
    pub fn get_contact_info(&self, a: BodyId, b: BodyId) -> Option<&ContactInfo> {
        let pair = CollisionPair::new(a, b);
        self.current_collision_pairs.get(&pair)
    }

    /// Get all bodies currently colliding with a specific body.
    pub fn colliding_with(&self, body: BodyId) -> Vec<BodyId> {
        self.current_collision_pairs
            .keys()
            .filter_map(|pair| pair.other(body))
            .collect()
    }

    /// Get all bodies currently inside a trigger.
    pub fn bodies_in_trigger(&self, trigger: BodyId) -> Vec<BodyId> {
        self.current_trigger_pairs
            .iter()
            .filter_map(|pair| pair.other(trigger))
            .collect()
    }

    /// Remove all tracking for a body (when destroyed).
    pub fn remove_body(&mut self, body: BodyId) {
        self.current_collision_pairs.retain(|pair, _| !pair.involves(body));
        self.previous_collision_pairs.retain(|pair| !pair.involves(body));
        self.current_trigger_pairs.retain(|pair| !pair.involves(body));
        self.previous_trigger_pairs.retain(|pair| !pair.involves(body));
        self.body_layers.remove(&body);
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.current_collision_pairs.clear();
        self.previous_collision_pairs.clear();
        self.current_trigger_pairs.clear();
        self.previous_trigger_pairs.clear();
        self.pending_collision_events.clear();
        self.pending_trigger_events.clear();
        self.history.clear();
        self.body_layers.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_collision_pair_canonical() {
        let p1 = CollisionPair::new(5, 3);
        let p2 = CollisionPair::new(3, 5);
        assert_eq!(p1, p2);
        assert_eq!(p1.body_a, 3);
        assert_eq!(p1.body_b, 5);
    }

    #[test]
    fn test_collision_begin_end() {
        let mut system = CollisionEventSystem::new();

        // Frame 1: bodies 1 and 2 collide.
        system.begin_frame();
        system.report_collision(1, 2, ContactInfo::default());
        system.process_events();
        assert_eq!(system.stats().collision_begins, 1);

        // Frame 2: still colliding.
        system.begin_frame();
        system.report_collision(1, 2, ContactInfo::default());
        system.process_events();
        assert_eq!(system.stats().collision_stays, 1);
        assert_eq!(system.stats().collision_begins, 0);

        // Frame 3: no longer colliding.
        system.begin_frame();
        system.process_events();
        assert_eq!(system.stats().collision_ends, 1);
    }

    #[test]
    fn test_trigger_enter_exit() {
        let mut system = CollisionEventSystem::new();

        system.begin_frame();
        system.report_trigger(10, 20);
        system.process_events();
        assert_eq!(system.stats().trigger_enters, 1);

        system.begin_frame();
        system.process_events();
        assert_eq!(system.stats().trigger_exits, 1);
    }

    #[test]
    fn test_event_callback() {
        let mut system = CollisionEventSystem::new();

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        system.on_collision(EventFilter::default(), move |_event| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        system.begin_frame();
        system.report_collision(1, 2, ContactInfo::default());
        system.process_events();
        system.dispatch_events();

        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_event_filter() {
        let filter = EventFilter::for_body(42);
        assert!(filter.body_filter.contains(&42));
        assert!(!filter.body_filter.contains(&99));
    }

    #[test]
    fn test_event_history() {
        let mut history = EventHistory::new(10);
        assert!(history.is_empty());

        for i in 0..15 {
            history.record(EventRecord::Trigger(TriggerEvent {
                trigger_body: 1,
                other_body: 2,
                event_type: TriggerEventType::Enter,
                frame: i,
                trigger_layer: ALL_LAYERS,
                other_layer: ALL_LAYERS,
            }));
        }

        // Buffer should cap at 10.
        assert_eq!(history.len(), 10);
        assert_eq!(history.total_recorded(), 15);
    }

    #[test]
    fn test_colliding_with() {
        let mut system = CollisionEventSystem::new();
        system.begin_frame();
        system.report_collision(1, 2, ContactInfo::default());
        system.report_collision(1, 3, ContactInfo::default());

        let neighbors = system.colliding_with(1);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&3));
    }

    #[test]
    fn test_contact_info_from_points() {
        let points = vec![
            ContactPoint {
                position: Vec3::new(1.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                depth: 0.1,
                impulse_normal: 5.0,
                ..Default::default()
            },
            ContactPoint {
                position: Vec3::new(-1.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                depth: 0.2,
                impulse_normal: 3.0,
                ..Default::default()
            },
        ];

        let info = ContactInfo::from_points(points);
        assert_eq!(info.point_count(), 2);
        assert!((info.total_impulse_normal - 8.0).abs() < 1e-6);
        assert!(info.deepest_contact().unwrap().depth > 0.15);
    }
}
