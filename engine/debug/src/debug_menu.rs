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
}
