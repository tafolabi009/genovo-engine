// engine/gameplay/src/event_system.rs
// Gameplay event system: typed events, channels, history, replay, filtering, statistics.
use std::collections::{HashMap, VecDeque};
use std::any::{Any, TypeId};

pub type EventId = u64;

#[derive(Debug, Clone)]
pub struct EventHeader {
    pub id: EventId,
    pub timestamp: f64,
    pub frame: u64,
    pub source: String,
    pub priority: u8,
}

pub trait GameEvent: std::fmt::Debug + Send + Sync + 'static {
    fn event_name(&self) -> &'static str;
    fn clone_event(&self) -> Box<dyn GameEvent>;
}

#[derive(Debug, Clone)]
pub struct DamageEvent { pub source: u32, pub target: u32, pub amount: f32, pub damage_type: String }
impl GameEvent for DamageEvent {
    fn event_name(&self) -> &'static str { "DamageEvent" }
    fn clone_event(&self) -> Box<dyn GameEvent> { Box::new(self.clone()) }
}

#[derive(Debug, Clone)]
pub struct DeathEvent { pub entity: u32, pub killer: Option<u32> }
impl GameEvent for DeathEvent {
    fn event_name(&self) -> &'static str { "DeathEvent" }
    fn clone_event(&self) -> Box<dyn GameEvent> { Box::new(self.clone()) }
}

#[derive(Debug, Clone)]
pub struct ItemPickupEvent { pub entity: u32, pub item_id: u32, pub quantity: u32 }
impl GameEvent for ItemPickupEvent {
    fn event_name(&self) -> &'static str { "ItemPickupEvent" }
    fn clone_event(&self) -> Box<dyn GameEvent> { Box::new(self.clone()) }
}

#[derive(Debug, Clone)]
pub struct QuestEvent { pub quest_id: u32, pub event_type: String }
impl GameEvent for QuestEvent {
    fn event_name(&self) -> &'static str { "QuestEvent" }
    fn clone_event(&self) -> Box<dyn GameEvent> { Box::new(self.clone()) }
}

struct EventEntry {
    header: EventHeader,
    event: Box<dyn GameEvent>,
}

type EventCallback = Box<dyn Fn(&dyn GameEvent) + Send + Sync>;

pub struct EventChannel {
    name: String,
    callbacks: Vec<(String, EventCallback)>,
    filter: Option<Box<dyn Fn(&dyn GameEvent) -> bool + Send + Sync>>,
}

impl EventChannel {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), callbacks: Vec::new(), filter: None }
    }
    pub fn subscribe(&mut self, id: &str, callback: EventCallback) {
        self.callbacks.push((id.to_string(), callback));
    }
    pub fn unsubscribe(&mut self, id: &str) {
        self.callbacks.retain(|(cid, _)| cid != id);
    }
    pub fn dispatch(&self, event: &dyn GameEvent) {
        if let Some(ref f) = self.filter {
            if !f(event) { return; }
        }
        for (_, cb) in &self.callbacks { cb(event); }
    }
}

#[derive(Debug, Clone, Default)]
pub struct EventStats {
    pub total_events: u64,
    pub events_per_type: HashMap<String, u64>,
    pub events_this_frame: u32,
    pub peak_events_per_frame: u32,
    pub total_channels: usize,
    pub total_subscribers: usize,
}

pub struct EventSystem {
    channels: HashMap<String, EventChannel>,
    history: VecDeque<EventHeader>,
    history_max: usize,
    next_id: EventId,
    current_frame: u64,
    current_time: f64,
    pub stats: EventStats,
    pending: Vec<(String, Box<dyn GameEvent>)>,
    replay_buffer: Vec<(f64, String, Box<dyn GameEvent>)>,
    is_recording: bool,
    is_replaying: bool,
    replay_index: usize,
    replay_speed: f32,
}

impl EventSystem {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(), history: VecDeque::new(), history_max: 1000,
            next_id: 1, current_frame: 0, current_time: 0.0,
            stats: EventStats::default(), pending: Vec::new(),
            replay_buffer: Vec::new(), is_recording: false, is_replaying: false,
            replay_index: 0, replay_speed: 1.0,
        }
    }

    pub fn create_channel(&mut self, name: &str) -> &mut EventChannel {
        self.channels.entry(name.to_string()).or_insert_with(|| EventChannel::new(name))
    }

    pub fn emit(&mut self, channel: &str, event: Box<dyn GameEvent>) {
        let id = self.next_id;
        self.next_id += 1;
        let header = EventHeader {
            id, timestamp: self.current_time, frame: self.current_frame,
            source: channel.to_string(), priority: 0,
        };

        // Record
        if self.is_recording {
            self.replay_buffer.push((self.current_time, channel.to_string(), event.clone_event()));
        }

        // Stats
        self.stats.total_events += 1;
        self.stats.events_this_frame += 1;
        *self.stats.events_per_type.entry(event.event_name().to_string()).or_insert(0) += 1;

        // History
        self.history.push_back(header);
        if self.history.len() > self.history_max { self.history.pop_front(); }

        // Dispatch
        if let Some(ch) = self.channels.get(channel) { ch.dispatch(event.as_ref()); }
    }

    pub fn update(&mut self, dt: f32) {
        self.current_time += dt as f64;
        self.current_frame += 1;
        self.stats.peak_events_per_frame = self.stats.peak_events_per_frame.max(self.stats.events_this_frame);
        self.stats.events_this_frame = 0;
        self.stats.total_channels = self.channels.len();
        self.stats.total_subscribers = self.channels.values().map(|c| c.callbacks.len()).sum();

        // Process pending
        let pending: Vec<_> = self.pending.drain(..).collect();
        for (ch, evt) in pending { self.emit(&ch, evt); }
    }

    pub fn start_recording(&mut self) { self.is_recording = true; self.replay_buffer.clear(); }
    pub fn stop_recording(&mut self) { self.is_recording = false; }
    pub fn start_replay(&mut self) { self.is_replaying = true; self.replay_index = 0; }
    pub fn stop_replay(&mut self) { self.is_replaying = false; }
    pub fn replay_event_count(&self) -> usize { self.replay_buffer.len() }
    pub fn history_len(&self) -> usize { self.history.len() }

    pub fn emit_deferred(&mut self, channel: &str, event: Box<dyn GameEvent>) {
        self.pending.push((channel.to_string(), event));
    }

    pub fn clear_history(&mut self) { self.history.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::{AtomicU32, Ordering}};

    #[test]
    fn test_event_system() {
        let mut sys = EventSystem::new();
        sys.create_channel("combat");
        sys.emit("combat", Box::new(DamageEvent {
            source: 0, target: 1, amount: 10.0, damage_type: "physical".into()
        }));
        assert_eq!(sys.stats.total_events, 1);
    }

    #[test]
    fn test_history() {
        let mut sys = EventSystem::new();
        sys.create_channel("game");
        for i in 0..5 {
            sys.emit("game", Box::new(DeathEvent { entity: i, killer: None }));
        }
        assert_eq!(sys.history_len(), 5);
    }

    #[test]
    fn test_recording() {
        let mut sys = EventSystem::new();
        sys.create_channel("items");
        sys.start_recording();
        sys.emit("items", Box::new(ItemPickupEvent { entity: 0, item_id: 1, quantity: 3 }));
        sys.stop_recording();
        assert_eq!(sys.replay_event_count(), 1);
    }
}
