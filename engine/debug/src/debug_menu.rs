//! In-game debug menu: category tree, cvar display/edit, performance graphs,
//! memory breakdown, entity inspector, and cheat commands.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum CvarValue { Bool(bool), Int(i32), Float(f32), String(String) }
impl CvarValue {
    pub fn as_string(&self) -> String { match self { Self::Bool(v) => v.to_string(), Self::Int(v) => v.to_string(), Self::Float(v) => format!("{:.3}", v), Self::String(v) => v.clone() } }
}
impl std::fmt::Display for CvarValue { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.as_string()) } }

#[derive(Debug, Clone)]
pub struct Cvar { pub name: String, pub value: CvarValue, pub default: CvarValue, pub description: String, pub category: String, pub min: Option<f64>, pub max: Option<f64>, pub read_only: bool, pub flags: u32 }
impl Cvar {
    pub fn new(name: impl Into<String>, value: CvarValue, cat: impl Into<String>) -> Self {
        let v = value.clone();
        Self { name: name.into(), value, default: v, description: String::new(), category: cat.into(), min: None, max: None, read_only: false, flags: 0 }
    }
    pub fn is_modified(&self) -> bool { self.value != self.default }
    pub fn reset(&mut self) { self.value = self.default.clone(); }
}

#[derive(Debug, Clone)]
pub struct PerfGraph { pub name: String, pub values: Vec<f32>, pub max_samples: usize, pub min_val: f32, pub max_val: f32, pub color: [f32; 4], pub unit: String }
impl PerfGraph {
    pub fn new(name: impl Into<String>, max_samples: usize) -> Self {
        Self { name: name.into(), values: Vec::with_capacity(max_samples), max_samples, min_val: 0.0, max_val: 100.0, color: [0.2, 0.8, 0.2, 1.0], unit: "ms".to_string() }
    }
    pub fn push(&mut self, value: f32) { if self.values.len() >= self.max_samples { self.values.remove(0); } self.values.push(value); }
    pub fn average(&self) -> f32 { if self.values.is_empty() { 0.0 } else { self.values.iter().sum::<f32>() / self.values.len() as f32 } }
    pub fn peak(&self) -> f32 { self.values.iter().cloned().fold(0.0f32, f32::max) }
    pub fn current(&self) -> f32 { self.values.last().copied().unwrap_or(0.0) }
}

#[derive(Debug, Clone)]
pub struct MemoryCategory { pub name: String, pub allocated: u64, pub peak: u64, pub count: u32, pub color: [f32; 4] }

#[derive(Debug, Clone)]
pub struct EntityInspector { pub selected_entity: Option<u64>, pub components: Vec<(String, Vec<(String, String)>)>, pub expanded_components: Vec<String> }
impl EntityInspector {
    pub fn new() -> Self { Self { selected_entity: None, components: Vec::new(), expanded_components: Vec::new() } }
    pub fn select(&mut self, entity: u64) { self.selected_entity = Some(entity); }
    pub fn clear(&mut self) { self.selected_entity = None; self.components.clear(); }
}
impl Default for EntityInspector { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone)]
pub struct CheatCommand { pub name: String, pub description: String, pub handler_id: u32, pub args: Vec<String>, pub category: String }
impl CheatCommand {
    pub fn new(name: impl Into<String>, desc: impl Into<String>, handler: u32) -> Self {
        Self { name: name.into(), description: desc.into(), handler_id: handler, args: Vec::new(), category: "General".to_string() }
    }
}

#[derive(Debug, Clone)]
pub enum DebugMenuEvent { CvarChanged(String, CvarValue), CheatExecuted(String, Vec<String>), EntitySelected(u64), CategoryExpanded(String), MenuToggled(bool) }

pub struct DebugMenuState {
    pub visible: bool, pub cvars: HashMap<String, Cvar>, pub categories: Vec<String>,
    pub selected_category: Option<String>, pub perf_graphs: Vec<PerfGraph>,
    pub memory_categories: Vec<MemoryCategory>, pub entity_inspector: EntityInspector,
    pub cheats: Vec<CheatCommand>, pub events: Vec<DebugMenuEvent>,
    pub search_text: String, pub show_perf: bool, pub show_memory: bool,
    pub show_entities: bool, pub show_cheats: bool, pub console_history: Vec<String>,
    pub console_input: String, pub opacity: f32,
}

impl DebugMenuState {
    pub fn new() -> Self {
        Self {
            visible: false, cvars: HashMap::new(), categories: Vec::new(),
            selected_category: None, perf_graphs: Vec::new(),
            memory_categories: Vec::new(), entity_inspector: EntityInspector::new(),
            cheats: Vec::new(), events: Vec::new(), search_text: String::new(),
            show_perf: true, show_memory: false, show_entities: false,
            show_cheats: false, console_history: Vec::new(),
            console_input: String::new(), opacity: 0.9,
        }
    }
    pub fn toggle(&mut self) { self.visible = !self.visible; self.events.push(DebugMenuEvent::MenuToggled(self.visible)); }
    pub fn register_cvar(&mut self, cvar: Cvar) { let cat = cvar.category.clone(); if !self.categories.contains(&cat) { self.categories.push(cat); } self.cvars.insert(cvar.name.clone(), cvar); }
    pub fn set_cvar(&mut self, name: &str, value: CvarValue) { if let Some(c) = self.cvars.get_mut(name) { if !c.read_only { c.value = value.clone(); self.events.push(DebugMenuEvent::CvarChanged(name.to_string(), value)); } } }
    pub fn get_cvar(&self, name: &str) -> Option<&CvarValue> { self.cvars.get(name).map(|c| &c.value) }
    pub fn register_cheat(&mut self, cheat: CheatCommand) { self.cheats.push(cheat); }
    pub fn execute_cheat(&mut self, name: &str, args: Vec<String>) { self.events.push(DebugMenuEvent::CheatExecuted(name.to_string(), args)); self.console_history.push(format!("> {}", name)); }
    pub fn add_perf_graph(&mut self, graph: PerfGraph) { self.perf_graphs.push(graph); }
    pub fn update_perf(&mut self, name: &str, value: f32) { if let Some(g) = self.perf_graphs.iter_mut().find(|g| g.name == name) { g.push(value); } }
    pub fn cvar_count(&self) -> usize { self.cvars.len() }
    pub fn drain_events(&mut self) -> Vec<DebugMenuEvent> { std::mem::take(&mut self.events) }
}
impl Default for DebugMenuState { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cvar_operations() {
        let mut menu = DebugMenuState::new();
        menu.register_cvar(Cvar::new("r_fov", CvarValue::Float(90.0), "Rendering"));
        menu.set_cvar("r_fov", CvarValue::Float(100.0));
        assert_eq!(menu.get_cvar("r_fov"), Some(&CvarValue::Float(100.0)));
    }
    #[test]
    fn perf_graph() {
        let mut g = PerfGraph::new("FPS", 100);
        for i in 0..50 { g.push(i as f32); }
        assert!((g.average() - 24.5).abs() < 0.1);
    }

    #[test]
    fn perf_graph_peak() {
        let mut g = PerfGraph::new("Frame Time", 60);
        g.push(10.0);
        g.push(50.0);
        g.push(20.0);
        assert!((g.peak() - 50.0).abs() < 0.01);
        assert!((g.current() - 20.0).abs() < 0.01);
    }

    #[test]
    fn toggle_menu() {
        let mut menu = DebugMenuState::new();
        assert!(!menu.visible);
        menu.toggle();
        assert!(menu.visible);
        menu.toggle();
        assert!(!menu.visible);
    }

    #[test]
    fn entity_inspector() {
        let mut inspector = EntityInspector::new();
        inspector.select(42);
        assert_eq!(inspector.selected_entity, Some(42));
        inspector.clear();
        assert_eq!(inspector.selected_entity, None);
    }

    #[test]
    fn cheat_command() {
        let mut menu = DebugMenuState::new();
        menu.register_cheat(CheatCommand::new("god", "Toggle god mode", 1));
        menu.register_cheat(CheatCommand::new("noclip", "Toggle noclip", 2));
        assert_eq!(menu.cheats.len(), 2);
        menu.execute_cheat("god", vec![]);
        assert!(!menu.events.is_empty());
    }

    #[test]
    fn cvar_reset() {
        let mut cvar = Cvar::new("test", CvarValue::Int(10), "Test");
        cvar.value = CvarValue::Int(20);
        assert!(cvar.is_modified());
        cvar.reset();
        assert!(!cvar.is_modified());
        assert_eq!(cvar.value, CvarValue::Int(10));
    }

    #[test]
    fn cvar_read_only() {
        let mut menu = DebugMenuState::new();
        let mut cvar = Cvar::new("version", CvarValue::String("1.0".to_string()), "System");
        cvar.read_only = true;
        menu.register_cvar(cvar);
        menu.set_cvar("version", CvarValue::String("2.0".to_string()));
        assert_eq!(menu.get_cvar("version"), Some(&CvarValue::String("1.0".to_string())));
    }
}

// ---------------------------------------------------------------------------
// Debug draw helpers
// ---------------------------------------------------------------------------

/// Represents a debug text entry to render on screen.
#[derive(Debug, Clone)]
pub struct DebugTextEntry {
    /// The text to display.
    pub text: String,
    /// Screen position (normalized 0..1).
    pub position: [f32; 2],
    /// Text color.
    pub color: [f32; 4],
    /// Font size in pixels.
    pub font_size: f32,
    /// Duration to display (0 = permanent until cleared).
    pub duration: f32,
    /// Time remaining.
    pub time_remaining: f32,
    /// Whether to show a background behind the text.
    pub show_background: bool,
    /// Background color.
    pub background_color: [f32; 4],
}

impl DebugTextEntry {
    /// Create a new debug text entry.
    pub fn new(text: impl Into<String>, position: [f32; 2]) -> Self {
        Self {
            text: text.into(),
            position,
            color: [1.0, 1.0, 1.0, 1.0],
            font_size: 14.0,
            duration: 0.0,
            time_remaining: 0.0,
            show_background: true,
            background_color: [0.0, 0.0, 0.0, 0.5],
        }
    }

    /// Create a temporary text entry that fades after a duration.
    pub fn temporary(text: impl Into<String>, position: [f32; 2], duration: f32) -> Self {
        let mut entry = Self::new(text, position);
        entry.duration = duration;
        entry.time_remaining = duration;
        entry
    }

    /// Set the color.
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Set font size.
    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Update the entry, returning false if it should be removed.
    pub fn update(&mut self, dt: f32) -> bool {
        if self.duration > 0.0 {
            self.time_remaining -= dt;
            if self.time_remaining <= 0.0 {
                return false;
            }
            // Fade out in the last 0.5 seconds.
            if self.time_remaining < 0.5 {
                self.color[3] = self.time_remaining / 0.5;
            }
        }
        true
    }
}

/// Manages on-screen debug text overlays.
pub struct DebugTextOverlay {
    /// Active text entries.
    pub entries: Vec<DebugTextEntry>,
    /// Whether the overlay is enabled.
    pub enabled: bool,
    /// Default text color.
    pub default_color: [f32; 4],
    /// Maximum number of entries.
    pub max_entries: usize,
}

impl DebugTextOverlay {
    /// Create a new overlay.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            enabled: true,
            default_color: [1.0, 1.0, 1.0, 1.0],
            max_entries: 100,
        }
    }

    /// Add a text entry.
    pub fn add(&mut self, entry: DebugTextEntry) {
        if self.entries.len() < self.max_entries {
            self.entries.push(entry);
        }
    }

    /// Add a simple text message at the next available position.
    pub fn print(&mut self, text: impl Into<String>) {
        let y = 0.02 + self.entries.len() as f32 * 0.025;
        self.add(DebugTextEntry::new(text, [0.02, y]));
    }

    /// Add a temporary message.
    pub fn flash(&mut self, text: impl Into<String>, duration: f32) {
        let y = 0.5 + self.entries.len() as f32 * 0.03;
        self.add(DebugTextEntry::temporary(text, [0.5, y], duration));
    }

    /// Update all entries, removing expired ones.
    pub fn update(&mut self, dt: f32) {
        self.entries.retain_mut(|e| e.update(dt));
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get the number of active entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for DebugTextOverlay {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Performance counter
// ---------------------------------------------------------------------------

/// A named performance counter that tracks min/max/avg over a sliding window.
#[derive(Debug, Clone)]
pub struct PerfCounter {
    /// Counter name.
    pub name: String,
    /// Current value.
    pub current: f64,
    /// Minimum value in window.
    pub min: f64,
    /// Maximum value in window.
    pub max: f64,
    /// Running sum for average.
    pub sum: f64,
    /// Number of samples.
    pub count: u64,
    /// Unit string.
    pub unit: String,
    /// History for graphing.
    pub history: Vec<f64>,
    /// Maximum history size.
    pub max_history: usize,
}

impl PerfCounter {
    /// Create a new counter.
    pub fn new(name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            current: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            sum: 0.0,
            count: 0,
            unit: unit.into(),
            history: Vec::new(),
            max_history: 300,
        }
    }

    /// Record a value.
    pub fn record(&mut self, value: f64) {
        self.current = value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
        self.count += 1;
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(value);
    }

    /// Get the average value.
    pub fn average(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum / self.count as f64 }
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.current = 0.0;
        self.min = f64::MAX;
        self.max = f64::MIN;
        self.sum = 0.0;
        self.count = 0;
        self.history.clear();
    }

    /// Format as a display string.
    pub fn format(&self) -> String {
        format!(
            "{}: {:.2} {} (min={:.2}, max={:.2}, avg={:.2})",
            self.name, self.current, self.unit, self.min, self.max, self.average()
        )
    }
}

/// Collection of named performance counters.
pub struct PerfCounterSet {
    /// All counters.
    pub counters: HashMap<String, PerfCounter>,
}

impl PerfCounterSet {
    /// Create a new set.
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
        }
    }

    /// Get or create a counter.
    pub fn counter(&mut self, name: &str, unit: &str) -> &mut PerfCounter {
        self.counters
            .entry(name.to_string())
            .or_insert_with(|| PerfCounter::new(name, unit))
    }

    /// Record a value for a named counter.
    pub fn record(&mut self, name: &str, value: f64) {
        if let Some(counter) = self.counters.get_mut(name) {
            counter.record(value);
        }
    }

    /// Reset all counters.
    pub fn reset_all(&mut self) {
        for counter in self.counters.values_mut() {
            counter.reset();
        }
    }

    /// Get a summary of all counters.
    pub fn summary(&self) -> Vec<String> {
        let mut lines: Vec<String> = self.counters.values().map(|c| c.format()).collect();
        lines.sort();
        lines
    }
}

impl Default for PerfCounterSet {
    fn default() -> Self {
        Self::new()
    }
}
